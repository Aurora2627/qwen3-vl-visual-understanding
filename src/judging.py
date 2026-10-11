"""Prompt protocols and strict parsing, independent of the model backend."""
import json

VERDICTS = {'PASS', 'FAIL', 'UNCERTAIN'}


def make_prompt(instruction, claims, protocol):
    context = ('Judge whether the image satisfies the requested content. Treat the request as '
               'a specification, not evidence of what is present. Use only visible evidence. '
               'If no image is provided, answer UNCERTAIN. Do not assess aesthetics or invent objects.\n'
               'Request: ' + instruction + '\nConditions: ' + json.dumps(claims) + '\n')
    if protocol == 'direct':
        return context + ('Return only JSON: {"verdict":"PASS|FAIL|UNCERTAIN","reason":"short visible evidence"}. '
                          'PASS requires all conditions; FAIL requires at least one visible contradiction.')
    if protocol == 'decomposed':
        return context + ('Check every condition in its given order. Return only JSON: '
                          '{"checks":[{"verdict":"PASS|FAIL|UNCERTAIN","reason":"short visible evidence"}]}. '
                          'The checks array must have exactly ' + str(len(claims)) + ' entries.')
    raise ValueError('Unknown protocol: ' + protocol)


def aggregate(verdicts):
    if any(v not in VERDICTS for v in verdicts) or not verdicts:
        raise ValueError('Invalid condition verdicts')
    if 'FAIL' in verdicts:
        return 'FAIL'
    return 'PASS' if all(v == 'PASS' for v in verdicts) else 'UNCERTAIN'


def parse_answer(text, protocol, claim_count):
    """Malformed/truncated output stays INVALID; never silently convert to FAIL."""
    try:
        cleaned = text.strip()
        if cleaned.startswith('```json') and cleaned.endswith('```'):
            cleaned = cleaned[7:-3].strip()
        elif cleaned.startswith('```') and cleaned.endswith('```'):
            cleaned = cleaned[3:-3].strip()
        obj = json.loads(cleaned)
        if not isinstance(obj, dict):
            raise ValueError('Expected object')
        if protocol == 'direct':
            verdict = obj['verdict']
            if verdict not in VERDICTS or not isinstance(obj.get('reason'), str):
                raise ValueError('Invalid direct judgment')
        else:
            checks = obj['checks']
            if not isinstance(checks, list) or len(checks) != claim_count:
                raise ValueError('Condition count mismatch')
            if any(not isinstance(c, dict) or not isinstance(c.get('reason'), str) for c in checks):
                raise ValueError('Missing evidence')
            verdict = aggregate([c['verdict'] for c in checks])
        return {'verdict': verdict, 'parsed': obj, 'parse_error': None}
    except (ValueError, KeyError, TypeError) as exc:
        return {'verdict': 'INVALID', 'parsed': None, 'parse_error': str(exc)}
