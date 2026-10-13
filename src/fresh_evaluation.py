"""Label-independent unseen-prompt selection and paired grouped uncertainty."""
import hashlib
import math
import random
from collections import defaultdict
from element_data import vote_probability
from element_training import binary_metrics

def canonical(text):
    return ' '.join(text.casefold().split())

def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def select_fresh(annotations, historical, groups=16, seed=20261013):
    excluded_text={canonical(r['instruction']) for r in historical}
    excluded_ids={str(r['prompt_id']) for r in historical}
    excluded_paths={r['source_image'] for r in historical if 'source_image' in r}
    pool=defaultdict(dict)
    for row in annotations:
        key=canonical(row['prompt'])
        if key in excluded_text or str(row['prompt_id']) in excluded_ids or row['img_path'] in excluded_paths:
            continue
        # Require available binary annotations, without selecting by their values.
        if not row.get('element_score'):
            continue
        for votes in row['element_score'].values():
            vote_probability(votes)
        pool[key].setdefault(row['img_path'],row)
    keys=sorted((k for k,v in pool.items() if len(v)>=2),key=lambda k:digest(str(seed)+':fresh:'+k))[:groups]
    if len(keys)!=groups:
        raise ValueError('Insufficient unused groups')
    return [(row,digest(key)) for key in keys for row in sorted(pool[key].values(),key=lambda r:digest(r['img_path']))[:2]]

def expand_elements(rows):
    result=[]
    for r in rows:
        for index,element in enumerate(r['elements']):
            result.append({'id':r['id']+':'+str(index),'image_id':r['id'],'image':r['image'],
                'image_sha256':r['image_sha256'],'instruction':r['instruction'],
                'prompt_group':r['prompt_group'],'prompt_id':r['prompt_id'],
                'element':element,'element_index':index,'p_pass':vote_probability(r['human_elements'][element]),
                'human_votes':r['human_elements'][element],'source_revision':r['source_revision'],'split':'fresh_evaluation'})
    return result

def assert_fresh(rows,historical):
    for field in ('prompt_id','image_sha256'):
        if {str(r[field]) for r in rows}&{str(r[field]) for r in historical}:
            raise ValueError('Historical overlap: '+field)
    if {canonical(r['instruction']) for r in rows}&{canonical(r['instruction']) for r in historical}:
        raise ValueError('Historical instruction overlap')
    if len({r['id'] for r in rows})!=len(rows):
        raise ValueError('Duplicate fresh IDs')

def paired_bootstrap(left,right,repetitions=2000,seed=20261013):
    a={r['id']:r for r in left};b={r['id']:r for r in right}
    if len(a)!=len(left) or len(b)!=len(right) or a.keys()!=b.keys():
        raise ValueError('Predictions must have identical unique IDs')
    groups=defaultdict(list)
    for key,r in a.items():
        if (r['prompt_group'],r['p_pass_human'])!=(b[key]['prompt_group'],b[key]['p_pass_human']):
            raise ValueError('Paired labels/groups mismatch')
        groups[r['prompt_group']].append(key)
    if len(groups)<2 or repetitions<100:
        raise ValueError('Need at least two groups and 100 replicates')
    names=sorted(groups);rng=random.Random(seed)
    metrics=('vote_nll','vote_brier','majority_accuracy','balanced_accuracy')
    deltas={k:[] for k in metrics}
    point_a=binary_metrics([a[k] for k in sorted(a)]);point_b=binary_metrics([b[k] for k in sorted(a)])
    for _ in range(repetitions):
        keys=[key for group in rng.choices(names,k=len(names)) for key in groups[group]]
        ma=binary_metrics([a[k] for k in keys]);mb=binary_metrics([b[k] for k in keys])
        for metric in metrics:
            if ma[metric] is not None and mb[metric] is not None:
                deltas[metric].append(mb[metric]-ma[metric])
    result={}
    for metric,values in deltas.items():
        values.sort()
        result[metric]={'right_minus_left':point_b[metric]-point_a[metric] if point_a[metric] is not None and point_b[metric] is not None else None,
            'ci95':[values[int(.025*(len(values)-1))],values[int(.975*(len(values)-1))]] if values else None,
            'valid_replicates':len(values)}
    return {'groups':len(groups),'repetitions':repetitions,'seed':seed,'metrics':result}
