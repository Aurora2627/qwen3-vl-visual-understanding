import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from judging import aggregate, make_prompt, parse_answer
from metrics import summarize


class JudgmentTests(unittest.TestCase):
    def test_invalid_is_not_a_negative(self):
        self.assertEqual(parse_answer('{"verdict":"FAIL"','direct',1)['verdict'],'INVALID')

    def test_missing_condition_rejected(self):
        text='{"checks":[{"verdict":"PASS","reason":"red"}]}'
        self.assertEqual(parse_answer(text,'decomposed',2)['verdict'],'INVALID')

    def test_uncertain_and_contradiction(self):
        self.assertEqual(aggregate(['PASS','UNCERTAIN']),'UNCERTAIN')
        self.assertEqual(aggregate(['FAIL','UNCERTAIN']),'FAIL')

    def test_prompt_has_no_reference(self):
        p=make_prompt('red square',['square is red'],'direct')
        self.assertNotIn('expected',p)
        self.assertIn('not evidence',p)

    def test_metric_denominator_and_pairs(self):
        rows=[{'expected':True,'verdict':'PASS','image_group':'a'},
              {'expected':False,'verdict':'INVALID','image_group':'a'},
              {'expected':True,'verdict':'UNCERTAIN','image_group':'b'},
              {'expected':False,'verdict':'FAIL','image_group':'b'}]
        m,_=summarize(rows)
        self.assertEqual(m['accuracy_all'],0.5)
        self.assertEqual(m['coverage'],0.5)
        self.assertEqual(m['accuracy_answered'],1.0)
        self.assertEqual(m['pair_accuracy'],0.0)

    def test_all_abstentions(self):
        m,_=summarize([{'expected':True,'verdict':'UNCERTAIN','image_group':'a'}])
        self.assertEqual(m['accuracy_all'],0)
        self.assertIsNone(m['accuracy_answered'])


if __name__=='__main__':
    unittest.main()
