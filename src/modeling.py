"""Official Transformers Qwen3-VL with explicit PyTorch device and dtype."""
import time
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


class QwenJudge:
    def __init__(self, directory, device='cuda', dtype=None):
        if device == 'mps' and not torch.backends.mps.is_available():
            raise RuntimeError('MPS unavailable in this process; run from VS Code or explicitly select CPU')
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable')
        self.device = torch.device(device)
        dtype = dtype or (torch.float32 if device == 'cpu' else torch.float16)
        if dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError('Unsupported dtype')
        self.processor = AutoProcessor.from_pretrained(directory, local_files_only=True)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            directory, local_files_only=True, dtype=dtype, attn_implementation='eager').to(self.device).eval()

    def synchronize(self):
        if self.device.type == 'mps':
            torch.mps.synchronize()
        elif self.device.type == 'cuda':
            torch.cuda.synchronize()

    @torch.inference_mode()
    def judge(self, image_path, prompt, max_new_tokens=192):
        content = []
        if image_path is not None:
            with Image.open(image_path) as source:
                picture = source.convert('RGB')
            picture.thumbnail((448, 448))
            content.append({'type': 'image', 'image': picture})
        content.append({'type': 'text', 'text': prompt})
        self.synchronize()
        start = time.perf_counter()
        inputs = self.processor.apply_chat_template([{'role':'user','content':content}],
            tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt').to(self.device)
        prompt_tokens = inputs['input_ids'].shape[-1]
        outputs = self.model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
        self.synchronize()
        completion = outputs[0, prompt_tokens:]
        return {'raw': self.processor.decode(completion, skip_special_tokens=True),
                'seconds': time.perf_counter()-start, 'input_tokens': prompt_tokens,
                'output_tokens': len(completion), 'reached_token_limit': len(completion) >= max_new_tokens}
