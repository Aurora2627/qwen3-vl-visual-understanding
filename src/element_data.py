"""Development-only grouped splits and human-vote element targets."""
import hashlib
import json
from pathlib import Path


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def vote_probability(votes):
    if not isinstance(votes, list) or not votes or any(type(v) is not int or v not in (0, 1) for v in votes):
        raise ValueError('Element labels must be nonempty binary human votes')
    return sum(votes) / len(votes)


def split_development(rows, validation_groups=4, seed=20261012):
    development = [r for r in rows if r['split'] == 'development']
    groups = sorted({r['prompt_group'] for r in development}, key=lambda g: digest(str(seed)+':'+g))
    if not 0 < validation_groups < len(groups):
        raise ValueError('Need nonempty, prompt-disjoint train/validation')
    val = set(groups[:validation_groups])
    result = {'train': [], 'validation': []}
    for r in development:
        name = 'validation' if r['prompt_group'] in val else 'train'
        for index, element in enumerate(r['elements']):
            votes = r['human_elements'][element]
            result[name].append({'id': r['id']+':'+str(index), 'image_id': r['id'],
                'image': r['image'], 'image_sha256': r['image_sha256'],
                'instruction': r['instruction'], 'prompt_group': r['prompt_group'],
                'prompt_id': r['prompt_id'], 'element': element, 'element_index': index,
                'p_pass': vote_probability(votes), 'human_votes': votes, 'split': name,
                'source_revision': r['source_revision']})
    assert_disjoint(result['train'], result['validation'])
    historical = [r for r in rows if r['split'] != 'development']
    for key in ('prompt_group', 'prompt_id', 'image_sha256'):
        if {str(r[key]) for r in historical} & {str(r[key]) for r in development}:
            raise ValueError('Historical holdout overlaps development: '+key)
    return result


def assert_disjoint(train, validation):
    if not train or not validation:
        raise ValueError('Empty split')
    for key in ('prompt_group', 'prompt_id', 'image_sha256'):
        if {str(r[key]) for r in train} & {str(r[key]) for r in validation}:
            raise ValueError('Cross-split overlap: '+key)
    for name, rows in [('train', train), ('validation', validation)]:
        if any(r['split'] != name for r in rows):
            raise ValueError('Wrong split membership')
        if len({r['id'] for r in rows}) != len(rows):
            raise ValueError('Duplicate example id')


def element_prompt(instruction, element):
    return ('Judge one requested element using only visible image evidence. The request is not evidence. '
            'Check attributes and relations in the complete request. '
            'PASS=the element is correctly realized; FAIL=absent, contradicted or incorrectly realized. '
            'Category tags are hints. Return exactly PASS or FAIL.\nRequest: '+instruction+
            '\nElement: '+element+'\nDecision:')
