import sys,unittest
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from element_training import binary_vote_loss
from element_diagnostics import diagnostic_summary,split_corpus

class ExpandedElementTests(unittest.TestCase):
 def test_binary_loss_matches_vote_target_and_excludes_other_tokens(self):
  logits=torch.tensor([10.,0.,20.,0.],requires_grad=True)
  loss=binary_vote_loss(logits,[1,3],.75);loss.backward()
  self.assertAlmostEqual(loss.item(),.69314718,places=6)
  torch.testing.assert_close(logits.grad,torch.tensor([0.,-.25,0.,.25]))
 def test_binary_loss_rejects_invalid_choices_and_votes(self):
  for ids,target in [([1,1],.5),([1,9],.5),([1,2],float('nan')),([1,2],1.1)]:
   with self.assertRaises(ValueError):binary_vote_loss(torch.zeros(4),ids,target)
 def rows(self):
  return [{'id':str(i),'image':str(i)+'.png','image_sha256':str(i),'instruction':'prompt'+str(i),
   'prompt_group':str(i),'prompt_id':i,'elements':['cup'],'human_elements':{'cup':[0,1]},'source_revision':'r'} for i in range(3)]
 def test_group_split_assigns_every_element_without_leakage(self):
  result=split_corpus(self.rows(),1,1,1)
  self.assertEqual({k:len(v) for k,v in result.items()},{'train':1,'validation':1,'sealed_test':1})
  self.assertEqual(result['sealed_test'][0]['image_id'],'2')
 def test_group_split_rejects_cross_partition_image_duplicates(self):
  rows=self.rows();rows[2]['image_sha256']=rows[0]['image_sha256']
  with self.assertRaises(ValueError):split_corpus(rows,1,1,1)
 def test_confusion_ties_calibration_and_confident_errors(self):
  records=[{'p_pass_model':.95,'p_pass_human':0.,'prompt_group':'a'},
   {'p_pass_model':.1,'p_pass_human':0.,'prompt_group':'b'},
   {'p_pass_model':.8,'p_pass_human':1.,'prompt_group':'b'},
   {'p_pass_model':.99,'p_pass_human':.5,'prompt_group':'a'}]
  r=diagnostic_summary(records)
  self.assertEqual(r['wrong_at_confidence_ge_0_9'],1)
  self.assertEqual(r['confusion'],{'true_fail_pred_fail':1,'true_fail_pred_pass':1,'true_pass_pred_fail':0,'true_pass_pred_pass':1})
  self.assertEqual(sum(b['count'] for b in r['bins']),4)
  self.assertEqual(r['metrics']['human_ties'],1)
if __name__=='__main__':unittest.main()
