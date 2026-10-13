import sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from element_vote_temperature import fit_vote_temperature,apply_vote_temperature
from element_training import binary_metrics
class VoteTemperatureTests(unittest.TestCase):
 def test_validation_calibration_reduces_overconfidence_without_changing_decisions(self):
  rows=[{'choice_logits':[8.,0.],'p_pass_human':.75,'p_pass_model':.9996646499,'id':'a'},
   {'choice_logits':[0.,8.],'p_pass_human':.25,'p_pass_model':.0003353501,'id':'b'}]
  fit=fit_vote_temperature(rows);cal=apply_vote_temperature(rows,fit['temperature'])
  self.assertGreater(fit['temperature'],1)
  self.assertLess(binary_metrics(cal)['vote_nll'],binary_metrics(rows)['vote_nll'])
  self.assertEqual([r['p_pass_model']>=.5 for r in rows],[r['p_pass_model']>=.5 for r in cal])
 def test_rejects_invalid_temperature_targets_and_logits(self):
  for t in [0,-1,float('nan')]:
   with self.assertRaises(ValueError):apply_vote_temperature([{'choice_logits':[1,0]}],t)
  for rows in [[],[{'choice_logits':[float('nan'),0],'p_pass_human':.5}],[{'choice_logits':[1,0],'p_pass_human':2}]]:
   with self.assertRaises(ValueError):fit_vote_temperature(rows)
if __name__=='__main__':unittest.main()
