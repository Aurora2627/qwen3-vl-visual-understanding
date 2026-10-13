"""Observed-set diagnosis; never use these summaries as final-test evidence."""
import math
from collections import defaultdict
from element_training import binary_metrics

def diagnostic_summary(records):
    if not records:raise ValueError('Empty predictions')
    matrix={'true_fail_pred_fail':0,'true_fail_pred_pass':0,'true_pass_pred_fail':0,'true_pass_pred_pass':0}
    bins=[{'count':0,'sum_probability':0.,'sum_human_probability':0.} for _ in range(5)]
    wrong_confident=0;groups=defaultdict(list)
    for r in records:
        p,y=r['p_pass_model'],r['p_pass_human']
        if not all(math.isfinite(x) and 0<=x<=1 for x in (p,y)):raise ValueError('Invalid probability')
        b=bins[min(4,int(p*5))];b['count']+=1;b['sum_probability']+=p;b['sum_human_probability']+=y
        if y!=.5:
            truth='pass' if y>.5 else 'fail';pred='pass' if p>=.5 else 'fail'
            matrix['true_'+truth+'_pred_'+pred]+=1
            wrong_confident+=int(truth!=pred and max(p,1-p)>=.9)
        groups[r['prompt_group']].append(r)
    for index,b in enumerate(bins):
        n=b['count'];b.update(interval=[index/5,(index+1)/5],mean_probability=b.pop('sum_probability')/n if n else None,mean_human_probability=b.pop('sum_human_probability')/n if n else None)
    return {'metrics':binary_metrics(records),'confusion':matrix,'wrong_at_confidence_ge_0_9':wrong_confident,
        'bins':bins,'by_prompt_group':{g:binary_metrics(rows) for g,rows in sorted(groups.items())},
        'scope':'Observed v0.6 data diagnosis; not unseen-test performance'}

def split_corpus(image_rows,train_groups=48,validation_groups=16,test_groups=16):
    groups=list(dict.fromkeys(r['prompt_group'] for r in image_rows))
    if len(groups)!=train_groups+validation_groups+test_groups:raise ValueError('Unexpected group count')
    allocation={g:'train' if i<train_groups else 'validation' if i<train_groups+validation_groups else 'sealed_test' for i,g in enumerate(groups)}
    result={k:[] for k in ('train','validation','sealed_test')}
    from fresh_evaluation import expand_elements
    for row in image_rows:
        name=allocation[row['prompt_group']]
        for element in expand_elements([row]):element['split']=name;result[name].append(element)
    for name,rows in result.items():
        if not rows or any(r['split']!=name for r in rows) or len({r['id'] for r in rows})!=len(rows):
            raise ValueError('Invalid corpus split membership or IDs')
    for a,b in [('train','validation'),('train','sealed_test'),('validation','sealed_test')]:
        for field in ('prompt_group','prompt_id','image_sha256'):
            if {str(r[field]) for r in result[a]} & {str(r[field]) for r in result[b]}:
                raise ValueError('Cross-corpus overlap: '+field)
    return result
