"""Native PyTorch LoRA for the last language attention blocks only."""
import math
import re
import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base, rank=4, alpha=8):
        super().__init__()
        if rank <= 0 or alpha <= 0:
            raise ValueError('Rank and alpha must be positive')
        self.base = base
        self.base.requires_grad_(False)
        self.rank, self.alpha = rank, alpha
        self.A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.enabled = True

    def forward(self, x):
        result = self.base(x)
        if not self.enabled:
            return result
        # Keep adapter math/parameters in FP32 even under CUDA autocast.
        with torch.autocast(device_type=x.device.type, enabled=False):
            delta = (x.float() @ self.A.t()) @ self.B.t()
        return result + (delta * (self.alpha / self.rank)).to(result.dtype)


def inject_lora(model, rank=4, alpha=8, last_layers=4):
    if last_layers <= 0:
        raise ValueError('last_layers must be positive')
    model.requires_grad_(False)
    candidates = []
    for name, module in model.named_modules():
        match = re.search(r'(?:^|\.)language_model\.layers\.(\d+)\.self_attn\.(q_proj|v_proj)$', name)
        if match and isinstance(module, nn.Linear):
            candidates.append((int(match.group(1)), name, module))
    layers = sorted({layer for layer, _, _ in candidates})
    if len(layers) < last_layers:
        raise ValueError('Unexpected Qwen language attention structure')
    chosen = set(layers[-last_layers:])
    targets = []
    for layer, name, module in candidates:
        if layer in chosen:
            parent_name, child = name.rsplit('.', 1)
            setattr(model.get_submodule(parent_name), child, LoRALinear(module, rank, alpha))
            targets.append(name)
    if len(targets) != 2 * last_layers:
        raise ValueError('Expected exactly q/v per selected language layer')
    return sorted(targets)


def adapters(model):
    return {name: module for name, module in model.named_modules() if isinstance(module, LoRALinear)}


def set_enabled(model, enabled):
    for module in adapters(model).values():
        module.enabled = bool(enabled)


def adapter_state(model):
    return {name+'.'+key: getattr(module, key).detach().cpu().clone()
            for name, module in adapters(model).items() for key in ('A', 'B')}


def load_adapter_state(model, state):
    current = adapter_state(model)
    if current.keys() != state.keys() or any(current[k].shape != state[k].shape for k in current):
        raise ValueError('Adapter targets/shapes do not match')
    with torch.no_grad():
        for name, module in adapters(model).items():
            for key in ('A', 'B'):
                value = state[name+'.'+key]
                if not torch.isfinite(value).all():
                    raise ValueError('Nonfinite adapter')
                getattr(module, key).copy_(value)
