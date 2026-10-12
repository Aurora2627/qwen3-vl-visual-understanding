import unittest
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from constrained_choice import FixedChoice, resolve_choice_ids


class TokenizerStub:
    all_special_ids = [9]
    def encode(self, text, add_special_tokens=False):
        return {'1':[1],'2':[2],'split':[1,2],'special':[9],'alias':[1]}[text]
    def decode(self, ids, clean_up_tokenization_spaces=False):
        return {1:'1',2:'2',9:'special'}[ids[0]]


class FixedChoiceTests(unittest.TestCase):
    def test_disallowed_largest_logit_is_removed(self):
        source = torch.tensor([[100.,2.,3.,-1.]])
        processor = FixedChoice(['1','2'],[1,2],3)
        masked = processor(torch.zeros((1,3),dtype=torch.long),source)
        self.assertEqual(masked.argmax(-1).item(),2)
        self.assertTrue(torch.isneginf(masked[0,0]))
        self.assertEqual(source[0,0].item(),100.)
        self.assertEqual(processor.decode([2]),'2')
        self.assertAlmostEqual(sum(processor.statistics['candidate_probabilities']),1.,places=6)
        self.assertLess(processor.statistics['candidate_mass_full_vocabulary'],1e-30)
        self.assertEqual(processor.statistics['unconstrained_top_token_id'],0)

    def test_nan_candidate_is_rejected(self):
        processor = FixedChoice(['0','1'],[0,1],1)
        with self.assertRaises(ValueError):
            processor(torch.zeros((1,1),dtype=torch.long),torch.tensor([[0.,float('nan')]]))

    def test_wrong_length_and_reused_processor_are_rejected(self):
        processor = FixedChoice(['0','1'],[0,1],2)
        with self.assertRaises(ValueError):
            processor(torch.zeros((1,3),dtype=torch.long),torch.zeros((1,2)))
        processor(torch.zeros((1,2),dtype=torch.long),torch.zeros((1,2)))
        with self.assertRaises(ValueError):
            processor(torch.zeros((1,2),dtype=torch.long),torch.zeros((1,2)))
        with self.assertRaises(ValueError):
            processor.decode([0,1])

    def test_only_exact_single_ordinary_tokens_are_accepted(self):
        tokenizer = TokenizerStub()
        self.assertEqual(resolve_choice_ids(tokenizer,['1','2']),[1,2])
        for choices in [['split','2'],['special','2'],['alias','2'],['1','1']]:
            with self.assertRaises(ValueError):
                resolve_choice_ids(tokenizer,choices)


if __name__ == '__main__':
    unittest.main()
