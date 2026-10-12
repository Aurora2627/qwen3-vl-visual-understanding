import unittest
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from report_choice_study import public_row


class PublicExportTests(unittest.TestCase):
    def test_removes_labels_and_nested_prompts(self):
        record = {'id':'sample','instruction':'PRIVATE_REQUEST','human_overall':[1,5],
            'human_elements':{'PRIVATE_ELEMENT':[0,1]},'score':None,
            'raw_choices':[{'index':0,'prompt':'PRIVATE_REQUEST','raw':'2','decision':'2'}],
            'parsed':{'checks':[{'index':0,'verdict':'UNCERTAIN','reference':'PRIVATE_LABEL'}]}}
        result = public_row(record)
        self.assertEqual(result,{'id':'sample','score':None,
            'raw_choices':[{'raw':'2','decision':'2','index':0}],
            'parsed':{'checks':[{'index':0,'verdict':'UNCERTAIN'}]}})
        self.assertNotIn('PRIVATE',str(result))

    def test_preserves_actual_free_failure_output(self):
        result = public_row({'raw':'The score is 3','valid':False,'score':None,
                             'error':'Not exactly one allowed digit','human_overall':[3]})
        self.assertEqual(result['raw'],'The score is 3')
        self.assertFalse(result['valid'])
        self.assertIsNone(result['score'])


if __name__ == '__main__':
    unittest.main()
