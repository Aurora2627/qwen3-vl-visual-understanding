"""PyTorch-only correlation and prompt-cluster uncertainty for a small pilot."""
import torch


def ranks(values):
    values = torch.as_tensor(values, dtype=torch.float64)
    order = values.argsort(stable=True)
    result = torch.empty_like(values)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        stop = start+1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        result[order[start:stop]] = (start+stop-1)/2
        start = stop
    return result


def pearson(x, y):
    if len(x)<2:
        return None
    x,y = torch.as_tensor(x,dtype=torch.float64),torch.as_tensor(y,dtype=torch.float64)
    x,y = x-x.mean(), y-y.mean()
    denominator = (x.square().sum()*y.square().sum()).sqrt()
    return float((x*y).sum()/denominator) if denominator>0 else None


def spearman(x, y):
    return pearson(ranks(x),ranks(y))


def cluster_interval(rows, replicates=1000):
    groups = {}
    for row in rows:
        groups.setdefault(row['prompt_group'],[]).append(row)
    names = sorted(groups)
    if len(names)<2:
        return {'srcc_95_interval':None,'valid_replicates':0}
    generator = torch.Generator().manual_seed(20261011)
    scores = []
    for _ in range(replicates):
        selected = torch.randint(len(names),(len(names),),generator=generator).tolist()
        sample = [r for i in selected for r in groups[names[i]]]
        value = spearman([r['score'] for r in sample],[sum(r['human_overall'])/len(r['human_overall']) for r in sample])
        if value is not None:
            scores.append(value)
    interval = torch.quantile(torch.tensor(scores,dtype=torch.float64),torch.tensor([.025,.975],dtype=torch.float64)).tolist() if scores else None
    return {'srcc_95_interval':interval,'valid_replicates':len(scores),
            'bootstrap_replicates':replicates,'resampling_unit':'canonical prompt group'}


def real_summary(rows):
    answered = [r for r in rows if r['score'] is not None]
    groups = {}
    for row in rows:
        groups.setdefault(row['prompt_group'],[]).append(row)
    strict_pairs,correct,ties,invalid = 0,0,0,0
    for pair in groups.values():
        if len(pair)!=2:
            continue
        a,b = pair
        reference = sum(a['human_overall'])/len(a['human_overall'])-sum(b['human_overall'])/len(b['human_overall'])
        if abs(reference)<1e-12:
            ties += 1
            continue
        strict_pairs += 1
        if a['score'] is None or b['score'] is None:
            invalid += 1
        elif (a['score']-b['score'])*reference>0:
            correct += 1
    x = [r['score'] for r in answered]
    y = [sum(r['human_overall'])/len(r['human_overall']) for r in answered]
    result = {'n_images':len(rows),'n_prompt_groups':len(groups),'scored_images':len(answered),
        'score_coverage':len(answered)/len(rows),'invalid_outputs':sum(not r['valid'] for r in rows),
        'srcc_scored_only':spearman(x,y),'plcc_scored_only':pearson(x,y),
        'strict_human_pairs':strict_pairs,'pair_correct':correct,'human_tied_pairs':ties,
        'unscored_pairs':invalid,'pair_accuracy_all_strict':correct/strict_pairs if strict_pairs else None,
        'pair_treatment':'Human ties excluded; prediction ties, invalid and abstentions count as incorrect',
        'random_strict_pair_baseline':0.5 if strict_pairs else None,
        'constant_score_srcc':None,'constant_score_reason':'Zero variance makes correlation undefined'}
    result.update(cluster_interval(answered))
    if rows[0]['protocol'] == 'elements':
        total,good,abstain,disagreement,minority,positive,tp,tn = 0,0,0,0,0,0,0,0
        categories = {}
        for row in rows:
            checks = row['parsed']['checks'] if row['valid'] else None
            for index, name in enumerate(row['elements']):
                votes = row['human_elements'][name]
                if not votes or any(type(v) is not int or v not in [0,1] for v in votes):
                    raise ValueError('Unexpected binary element annotations')
                if sum(votes)*2 == len(votes):
                    continue
                expected = 'PASS' if sum(votes)*2>len(votes) else 'FAIL'
                prediction = checks[index]['verdict'] if checks is not None else 'INVALID'
                category = name.rsplit('(',1)[-1].rstrip(')') if '(' in name else 'unknown'
                counts = categories.setdefault(category,{'n':0,'correct':0,'false_accept':0,'false_reject':0,'unanswered':0})
                counts['n'] += 1
                counts['correct'] += int(prediction==expected)
                counts['false_accept'] += int(prediction=='PASS' and expected=='FAIL')
                counts['false_reject'] += int(prediction=='FAIL' and expected=='PASS')
                counts['unanswered'] += int(prediction in ['INVALID','UNCERTAIN'])
                total += 1
                good += int(prediction==expected)
                positive += int(expected=='PASS')
                tp += int(expected=='PASS' and prediction=='PASS')
                tn += int(expected=='FAIL' and prediction=='FAIL')
                abstain += int(prediction in ['INVALID','UNCERTAIN'])
                disagreement += int(len(set(votes))>1)
                minority += min(sum(votes),len(votes)-sum(votes))/len(votes)
        result['elements'] = {'n':total,'correct':good,'accuracy_all':good/total if total else None,
            'unanswered':abstain,'human_disagreement_elements':disagreement,
            'mean_human_minority_fraction':minority/total if total else None,
            'reference_positive':positive,
            'always_pass_accuracy':positive/total if total else None,
            'always_fail_accuracy':(total-positive)/total if total else None,
            'balanced_accuracy_all':(tp/positive+tn/(total-positive))/2 if 0<positive<total else None,
            'categories':categories,'reference':'strict majority of released binary human votes; tied votes excluded'}
    return result
