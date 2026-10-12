"""Matched short-prompt ablation plus development-only scalar calibration.

Install under project/scripts before running so source snapshots include it.
v1 free-JSON results remain a historical diagnostic, not a matched ablation.
"""
import argparse
import contextlib
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    import torch
    from run_record import snapshot_run
    from correlation import real_summary
    from modeling import QwenJudge
    from choice_modeling import judge_choice
    from constrained_choice import overall_prompt, element_prompt
    from temperature_calibration import fit_temperature, calibration_metrics, vote_distribution
    from run_real_study import validate, Tee
    parser = argparse.ArgumentParser()
    parser.add_argument('--device',choices=['mps','cpu','cuda'],default='mps')
    parser.add_argument('--output',default=None)
    args = parser.parse_args()
    manifest = ROOT/'data/processed/evalmuse-pilot-v2/manifest.jsonl'
    records = [json.loads(line) for line in manifest.read_text().splitlines()]
    validate(records)
    if len(records)!=64 or {r['split'] for r in records}!={'development','heldout'}:
        raise ValueError('Expected full 64-image v2 pilot')
    historical = ROOT/'data/processed/evalmuse-pilot-v1/manifest.jsonl'
    historical_groups = {json.loads(line)['prompt_group'] for line in historical.read_text().splitlines()}
    if historical_groups & {r['prompt_group'] for r in records if r['split']=='heldout'}:
        raise ValueError('Heldout overlaps observed v1 data')
    model_record = ROOT/'reports/local-model.json'
    model = json.loads(model_record.read_text())
    config = {'model':model,'device':args.device,'generation':'greedy',
        'launch':'VS Code integrated terminal/debugger','image_long_edge':448,
        'protocols':['overall_free','overall_fixed','elements_fixed'],
        'free_max_new_tokens':16,'fixed_decision_tokens':1,
        'calibration_steps':200,'calibration_learning_rate':.05,
        'frozen_policy':'Freeze before first v2 output; calibration uses development only',
        'method_scope':'Fixed-choice logits masking and scalar calibration; no Qwen fine-tuning'}
    out = snapshot_run(ROOT/(args.output or ('runs/choice-'+datetime.now().strftime('%Y%m%d-%H%M%S'))),config,
        [manifest,manifest.parent/'dataset-record.json',historical,model_record])
    with (out/'console.log').open('x') as log, contextlib.redirect_stdout(Tee(sys.stdout,log)), contextlib.redirect_stderr(Tee(sys.stderr,log)):
        try:
            torch.manual_seed(42)
            judge = QwenJudge(str(ROOT/model['local_directory']),args.device)
            rows = []
            fitted = None
            with (out/'predictions.jsonl').open('x') as stream:
                for split in ['development','heldout']:
                    for protocol in config['protocols']:
                        for record in (r for r in records if r['split']==split):
                            if protocol=='overall_free':
                                prompt = overall_prompt(record['instruction'])
                                result = judge.judge(ROOT/record['image'],prompt,16)
                                digit = result['raw'].strip()
                                valid = digit in ['1','2','3','4','5']
                                result.update({'prompt':prompt,'score':int(digit) if valid else None,
                                    'valid':valid,'error':None if valid else 'Not exactly one allowed digit',
                                    'model_calls':1,'intentional_fixed_length':False})
                            elif protocol=='overall_fixed':
                                prompt = overall_prompt(record['instruction'])
                                result = judge_choice(judge,ROOT/record['image'],prompt,['1','2','3','4','5'])
                                result.update({'prompt':prompt,'score':int(result['decision']),
                                    'valid':True,'error':None,'model_calls':1,'intentional_fixed_length':True})
                            else:
                                answers,checks = [],[]
                                for index,element in enumerate(record['elements']):
                                    prompt = element_prompt(record['instruction'],element)
                                    answer = judge_choice(judge,ROOT/record['image'],prompt,['0','1','2'])
                                    answer.update({'index':index,'prompt':prompt})
                                    answers.append(answer)
                                    checks.append({'index':index,'verdict':{'0':'FAIL','1':'PASS','2':'UNCERTAIN'}[answer['decision']]})
                                score = None if any(c['verdict']=='UNCERTAIN' for c in checks) else sum(c['verdict']=='PASS' for c in checks)/len(checks)
                                result = {'raw_choices':answers,'parsed':{'checks':checks},'score':score,
                                    'valid':True,'error':None,'model_calls':len(answers),'intentional_fixed_length':True,
                                    'seconds':sum(a['seconds'] for a in answers),
                                    'input_tokens':sum(a['input_tokens'] for a in answers),
                                    'output_tokens':sum(a['output_tokens'] for a in answers)}
                            row = {**record,'protocol':protocol,**result}
                            stream.write(json.dumps(row,ensure_ascii=False)+'\n')
                            stream.flush()
                            rows.append(row)
                            print('CHOICE_JUDGED',len(rows),split,protocol,record['id'],result['valid'],flush=True)
                    if split=='development':
                        fitted = fit_temperature([r for r in rows if r['protocol']=='overall_fixed'])
                        torch.save({k:v for k,v in fitted.items() if k!='history'},out/'calibration.pt')
                        (out/'calibration-history.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in fitted['history']))
                        print('DEVELOPMENT_TEMPERATURE_FIT_COMPLETE',flush=True)
            groups,calibration = {},{}
            for split in ['development','heldout']:
                for protocol in config['protocols']:
                    subset = [r for r in rows if r['split']==split and r['protocol']==protocol]
                    # Shared fine-grained metric recognizes this protocol by the generic name.
                    metric_rows = [{**r,'protocol':'elements'} for r in subset] if protocol=='elements_fixed' else subset
                    groups[split+'/'+protocol] = real_summary(metric_rows)
                fixed = [r for r in rows if r['split']==split and r['protocol']=='overall_fixed']
                logits = [r['candidate_logits'] for r in fixed]
                targets = torch.stack([vote_distribution(r['human_overall']) for r in fixed])
                calibration[split] = {}
                for name,temperature in [('expected',1.),('calibrated',fitted['temperature'])]:
                    metrics = calibration_metrics(logits,targets,temperature)
                    derived = [{**r,'score':score} for r,score in zip(fixed,metrics['expected_scores'])]
                    groups[split+'/overall_'+name] = real_summary(derived)
                    calibration[split][name] = metrics
                torch.save({'candidate_logits':torch.tensor(logits,dtype=torch.float64),
                            'human_vote_distributions':targets},out/(split+'-ordinal-scores.pt'))
            summary = {'groups':groups,'calibration':calibration,'temperature':fitted['temperature'],
                'image_protocol_records':len(rows),'actual_generation_calls':sum(r['model_calls'] for r in rows),
                'scope':'v2 pilot; 32 reused development images, 32 images from unused heldout prompts',
                'comparison':'overall_free vs overall_fixed share the same prompt; JSON v1 is historical only'}
            (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
            (out/'runtime.json').write_text(json.dumps({'device':args.device,
                'dtype':str(next(judge.model.parameters()).dtype),
                'mps_available':torch.backends.mps.is_available()},indent=2))
            (out/'status.json').write_text(json.dumps({'status':'complete'}))
            print('CHOICE_STUDY_COMPLETE',out,flush=True)
        except Exception:
            (out/'status.json').write_text(json.dumps({'status':'failed','traceback':traceback.format_exc()}))
            raise


if __name__ == '__main__':
    main()
