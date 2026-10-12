"""Frozen first-pass protocols for real generated images; no human labels in prompts."""
import json


def make_real_prompt(instruction, elements, protocol):
    prefix = ('Evaluate image-text alignment using only visible evidence, not aesthetic quality. '
              'The requested text is not evidence. Interpret attributes and relations in the full request.\n'
              'Request: '+instruction+'\n')
    if protocol == 'overall':
        return prefix+('Give an integer score 1-5: 1=none of the important content matches; '
            '2=few aspects match; 3=partially matches; 4=most content matches with some errors; '
            '5=all important content matches. Return only JSON '
            '{"score":3,"reason":"brief visible evidence, at most 30 words"}.')
    if protocol == 'elements':
        return prefix+('For each requested element, judge whether it is correctly realized in the context '
            'of the complete request, including bindings and relations. PASS=visible support; '
            'FAIL=visible contradiction or absent required content; UNCERTAIN=cannot determine. '
            'Category tags in parentheses are hints, not words to find in the picture.\n'
            'Elements in fixed order: '+json.dumps(elements)+'\nReturn only JSON '
            '{"checks":[{"index":0,"verdict":"PASS","reason":"brief evidence, at most 8 words"}]}. '
            'Include exactly '+str(len(elements))+' entries; indices start at zero and preserve order.')
    raise ValueError('Unknown protocol')


def parse_real(text, protocol, count):
    try:
        text = text.strip()
        if text.startswith('```') and text.endswith('```'):
            text = text[text.index('\n')+1:-3].strip()
        obj = json.loads(text)
        if not isinstance(obj, dict):
            raise ValueError('Expected object')
        if protocol == 'overall':
            if type(obj.get('score')) is not int or not 1 <= obj['score'] <= 5 or not isinstance(obj.get('reason'), str):
                raise ValueError('Invalid overall score')
            return {'valid':True, 'score':obj['score'], 'parsed':obj, 'error':None}
        if protocol != 'elements':
            raise ValueError('Unknown protocol')
        checks = obj['checks']
        if not isinstance(checks, list) or len(checks) != count or count == 0:
            raise ValueError('Wrong number of elements')
        for index, check in enumerate(checks):
            if (not isinstance(check, dict) or type(check.get('index')) is not int or check['index'] != index
                or check.get('verdict') not in ['PASS','FAIL','UNCERTAIN'] or not isinstance(check.get('reason'), str)):
                raise ValueError('Invalid or reordered element judgment')
        score = None if any(c['verdict']=='UNCERTAIN' for c in checks) else sum(c['verdict']=='PASS' for c in checks)/count
        return {'valid':True, 'score':score, 'parsed':obj, 'error':None}
    except (ValueError, TypeError, KeyError) as exc:
        return {'valid':False, 'score':None, 'parsed':None, 'error':str(exc)}
