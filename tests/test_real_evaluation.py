import io
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from correlation import ranks, pearson, spearman, real_summary
from real_judging import make_real_prompt, parse_real
from remote_zip import RangeZip


class RealEvaluationTests(unittest.TestCase):
    def test_tied_ranks_and_undefined_constant(self):
        self.assertEqual(ranks([2,1,2,4]).tolist(),[1.5,0,1.5,3])
        self.assertAlmostEqual(spearman([1,2,3],[3,2,1]),-1)
        self.assertIsNone(pearson([1,1],[2,3]))

    def test_no_reference_fields_in_prompt(self):
        text = make_real_prompt('A red apple',['apple (object)'],'elements')
        self.assertNotIn('human',text)
        self.assertNotIn('generator',text)

    def test_reordered_checks_rejected(self):
        self.assertFalse(parse_real('{"checks":[{"index":1,"verdict":"PASS","reason":"apple"}]}','elements',1)['valid'])
        self.assertFalse(parse_real('{"score":true,"reason":"apple"}','overall',1)['valid'])

    def test_uncertain_is_not_imputed_as_zero(self):
        result = parse_real('{"checks":[{"index":0,"verdict":"UNCERTAIN","reason":"blurred"}]}','elements',1)
        self.assertTrue(result['valid'])
        self.assertIsNone(result['score'])

    def test_invalid_pair_remains_in_denominator(self):
        rows = [{'id':'a','prompt_group':'x','human_overall':[1],'score':1,'valid':True,'protocol':'overall'},
                {'id':'b','prompt_group':'x','human_overall':[4],'score':None,'valid':False,'protocol':'overall'}]
        result = real_summary(rows)
        self.assertEqual(result['score_coverage'],.5)
        self.assertEqual(result['pair_accuracy_all_strict'],0)
        self.assertEqual(result['unscored_pairs'],1)

    def test_reader_crosses_parts_and_blocks(self):
        reader = RangeZip([{'rfilename':'a','size':5},{'rfilename':'b','size':4}],
                          'unused', '/tmp/qwen-range-unit',block_size=3)
        payload = {'a':b'abcde','b':b'fghi'}
        reader._block = lambda item,start:payload[item['rfilename']][start:start+3]
        reader.seek(2)
        self.assertEqual(reader.read(6),b'cdefgh')
        reader.seek(-2,io.SEEK_END)
        self.assertEqual(reader.read(),b'hi')

    def test_element_baseline_and_invalid_denominator(self):
        rows = [{'id':'a','prompt_group':'x','human_overall':[4],'score':1,'valid':True,
                 'protocol':'elements','elements':['red (color)'], 'human_elements':{'red (color)':[1,1,0]},
                 'parsed':{'checks':[{'verdict':'PASS'}]}},
                {'id':'b','prompt_group':'x','human_overall':[1],'score':None,'valid':False,
                 'protocol':'elements','elements':['red (color)'], 'human_elements':{'red (color)':[0,0,0]},'parsed':None}]
        result = real_summary(rows)['elements']
        self.assertEqual(result['accuracy_all'],.5)
        self.assertEqual(result['always_pass_accuracy'],.5)
        self.assertEqual(result['balanced_accuracy_all'],.5)
        self.assertEqual(result['human_disagreement_elements'],1)


if __name__ == '__main__':
    unittest.main()
