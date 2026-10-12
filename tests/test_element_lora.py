"""Behavioral tests for frozen-base LoRA, leakage guards and vote metrics."""
import copy
import math
import sys
import unittest
from pathlib import Path
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from element_lora import LoRALinear, inject_lora, adapter_state, load_adapter_state, set_enabled
from element_data import vote_probability, split_development, assert_disjoint, element_prompt
from element_training import human_vote_loss, binary_metrics


def tiny_model():
    model = nn.Module()
    model.language_model = nn.Module()
    model.language_model.layers = nn.ModuleList()
    for _ in range(3):
        block = nn.Module()
        block.self_attn = nn.Module()
        block.self_attn.q_proj = nn.Linear(4,4)
        block.self_attn.v_proj = nn.Linear(4,4)
        model.language_model.layers.append(block)
    model.visual = nn.Linear(4,4)
    return model


class ElementLoRATests(unittest.TestCase):
    def test_zero_init_freeze_update_and_disable(self):
        torch.manual_seed(7)
        base = nn.Linear(4,3)
        x = torch.randn(2,4)
        reference = base(x).detach()
        frozen = base.weight.detach().clone()
        layer = LoRALinear(base,rank=2,alpha=4)
        self.assertTrue(torch.equal(layer(x),reference))
        optimizer = torch.optim.SGD([layer.A,layer.B],lr=.1)
        layer(x).square().mean().backward()
        self.assertIsNone(base.weight.grad)
        self.assertGreater(layer.B.grad.abs().sum().item(),0)
        optimizer.step()
        self.assertTrue(torch.equal(base.weight,frozen))
        self.assertFalse(torch.equal(layer(x),reference))
        layer.enabled = False
        self.assertTrue(torch.equal(layer(x),reference))

    def test_only_last_language_qv_targets(self):
        model = tiny_model()
        names = inject_lora(model,last_layers=2)
        self.assertEqual(len(names),4)
        self.assertTrue(all('.layers.0.' not in n for n in names))
        self.assertFalse(model.visual.weight.requires_grad)
        trainable = {n for n,p in model.named_parameters() if p.requires_grad}
        self.assertEqual(trainable,{n+'.'+k for n in names for k in ('A','B')})

    def test_exact_adapter_reload_and_wrong_targets_rejected(self):
        model = tiny_model()
        inject_lora(model,last_layers=1)
        initial = adapter_state(model)
        trained = {k:v+0.1 for k,v in initial.items()}
        load_adapter_state(model,trained)
        for k,v in adapter_state(model).items(): self.assertTrue(torch.equal(v,trained[k]))
        with self.assertRaises(ValueError): load_adapter_state(model,{})
        invalid = copy.deepcopy(trained)
        invalid[next(iter(invalid))].fill_(float('nan'))
        with self.assertRaises(ValueError): load_adapter_state(model,invalid)

    def test_full_vocabulary_soft_target_loss(self):
        logits = torch.tensor([1.,2.,3.,4.],requires_grad=True)
        loss = human_vote_loss(logits,[1,3],.25)
        expected = -.25*logits.log_softmax(-1)[1]-.75*logits.log_softmax(-1)[3]
        self.assertAlmostEqual(loss.item(),expected.item())
        loss.backward()
        self.assertGreater(logits.grad[0].item(),0)
        with self.assertRaises(ValueError): human_vote_loss(logits,[1,3],1.1)

    def test_votes_are_not_pseudo_labels(self):
        self.assertEqual(vote_probability([1,0,1,0]),.5)
        for value in ([],[True],[2],None,[.5]):
            with self.assertRaises(ValueError): vote_probability(value)

    def test_no_holdout_or_prompt_leakage(self):
        rows = []
        for index in range(4):
            rows.append(dict(id=str(index),image='image.png',image_sha256=str(index),instruction='request',
                prompt_group=str(index),prompt_id=str(index),split='development',elements=['e'],
                human_elements={'e':[0,1]},source_revision='fixed'))
        rows.append(dict(rows[0],id='never',prompt_group='holdout',prompt_id='holdout',image_sha256='holdout',split='heldout'))
        split = split_development(rows,1)
        self.assertEqual(len(split['train']),3)
        self.assertEqual(len(split['validation']),1)
        self.assertNotIn('never',{r['image_id'] for values in split.values() for r in values})
        overlapping = copy.deepcopy(split['validation'])
        overlapping[0]['image_sha256'] = split['train'][0]['image_sha256']
        with self.assertRaises(ValueError): assert_disjoint(split['train'],overlapping)

    def test_metric_ties_and_soft_votes(self):
        result = binary_metrics([dict(p_pass_model=.5,p_pass_human=.5),
            dict(p_pass_model=.75,p_pass_human=1),dict(p_pass_model=.25,p_pass_human=0)])
        self.assertEqual(result['human_ties'],1)
        self.assertEqual(result['balanced_accuracy'],1)
        self.assertAlmostEqual(result['vote_brier'],.25)
        self.assertAlmostEqual(result['vote_nll'],(-math.log(.5)-2*math.log(.75))/3)

    def test_prompt_never_includes_human_labels(self):
        text = element_prompt('two red cups','red (color)')
        self.assertIn('two red cups',text)
        self.assertIn('red (color)',text)
        self.assertNotIn('human',text.lower())

if __name__ == '__main__': unittest.main()
