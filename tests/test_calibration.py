import unittest
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from temperature_calibration import vote_distribution, calibration_metrics, fit_temperature


class CalibrationTests(unittest.TestCase):
    def test_vote_distribution_preserves_disagreement(self):
        self.assertEqual(vote_distribution([1,1,5]).tolist(),[2/3,0,0,0,1/3])
        for votes in [[],[True],[0],[6]]:
            with self.assertRaises(ValueError):
                vote_distribution(votes)

    def test_heldout_training_is_rejected(self):
        with self.assertRaises(ValueError):
            fit_temperature([{'split':'heldout'}])

    def test_uniform_scores_and_known_cross_entropy(self):
        result = calibration_metrics([[0,0,0,0,0]],[[0,0,1,0,0]])
        self.assertAlmostEqual(result['nll'],torch.tensor(5.).log().item(),places=6)
        self.assertAlmostEqual(result['soft_target_brier'],.8)
        self.assertEqual(result['expected_scores'],[3.])

    def test_overconfidence_can_be_softened_without_model_weights(self):
        rows = [{'id':'a','prompt_group':'x','split':'development',
                 'choices':['1','2','3','4','5'],
                 'candidate_logits':[12.,0.,0.,0.,0.],'human_overall':[1,1,2,3,4,5]},
                {'id':'b','prompt_group':'y','split':'development',
                 'choices':['1','2','3','4','5'],
                 'candidate_logits':[0.,0.,0.,0.,12.],'human_overall':[1,2,3,4,5,5]}]
        original = [list(r['candidate_logits']) for r in rows]
        logits = original
        targets = torch.stack([vote_distribution(r['human_overall']) for r in rows])
        fitted = fit_temperature(rows,steps=80)
        before = calibration_metrics(logits,targets)['nll']
        after = calibration_metrics(logits,targets,fitted['temperature'])['nll']
        self.assertLess(after,before)
        self.assertGreater(fitted['temperature'],1.)
        self.assertEqual([r['candidate_logits'] for r in rows],original)


if __name__ == '__main__':
    unittest.main()
