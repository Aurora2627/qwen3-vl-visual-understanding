import sys
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from fresh_evaluation import select_fresh,assert_fresh,paired_bootstrap,expand_elements
from evaluate_fresh_elements import messages_for

class FreshEvaluationTests(unittest.TestCase):
    def test_sampling_excludes_history_and_ignores_label_values(self):
        rows=[]
        for index in range(6):
            for j in range(2):
                rows.append({'prompt':'Prompt '+str(index),'prompt_id':index,'img_path':str(index)+'/'+str(j),
                    'element_score':{'object':[0,1]}})
        history=[{'instruction':'  PROMPT   0 ','prompt_id':1,'source_image':'unused'}]
        first=select_fresh(rows,history,2,5)
        self.assertTrue(all(r['prompt_id'] not in (0,1) for r,g in first))
        changed=[dict(r,element_score={'object':[1,1]}) for r in reversed(rows)]
        self.assertEqual([r['img_path'] for r,g in first],[r['img_path'] for r,g in select_fresh(changed,history,2,5)])
        with self.assertRaises(ValueError):select_fresh(rows,history,10,5)

    def test_leakage_detects_text_identity_and_image_hash(self):
        a={'id':'a','instruction':'Hello world','prompt_id':1,'image_sha256':'x'}
        b={'id':'b','instruction':' HELLO  WORLD ','prompt_id':2,'image_sha256':'y'}
        with self.assertRaisesRegex(ValueError,'instruction'):assert_fresh([b],[a])
        b.update(instruction='different',image_sha256='x')
        with self.assertRaisesRegex(ValueError,'image_sha256'):assert_fresh([b],[a])
        b.update(image_sha256='y',prompt_id=1)
        with self.assertRaisesRegex(ValueError,'prompt_id'):assert_fresh([b],[a])

    def test_paired_group_bootstrap_aligns_ids_and_keeps_pairs(self):
        records=[{'id':str(i),'prompt_group':str(i//2),'p_pass_human':float(i%2),'p_pass_model':.8 if i%2 else .2} for i in range(8)]
        result=paired_bootstrap(records,list(reversed(records)),100)
        self.assertEqual(result['groups'],4)
        for m in result['metrics'].values():
            self.assertEqual(m['right_minus_left'],0)
            self.assertEqual(m['ci95'],[0,0])
        with self.assertRaises(ValueError):paired_bootstrap(records,records[:-1],100)
        bad=[dict(r) for r in records];bad[0]['p_pass_human']=1
        with self.assertRaises(ValueError):paired_bootstrap(records,bad,100)

    def test_text_control_has_identical_text_and_no_image(self):
        row={'instruction':'A red cup','element':'red cup'}
        text=messages_for(row);image=messages_for(row,'image-object')
        self.assertEqual(text[0]['content'],[image[0]['content'][-1]])
        self.assertEqual([x['type'] for x in text[0]['content']],['text'])
        self.assertEqual([x['type'] for x in image[0]['content']],['image','text'])

    def test_expand_retains_votes_without_leaking_into_prompt(self):
        row={'id':'x','image':'a.png','image_sha256':'hash','instruction':'request','prompt_group':'g','prompt_id':1,
            'elements':['cup'],'human_elements':{'cup':[1,0,1]},'source_revision':'rev'}
        expanded=expand_elements([row])[0]
        self.assertAlmostEqual(expanded['p_pass'],2/3)
        self.assertNotIn('human_votes',str(messages_for(expanded)))
        self.assertEqual(expanded['split'],'fresh_evaluation')
if __name__=='__main__':unittest.main()
