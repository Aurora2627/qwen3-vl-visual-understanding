"""One development image: CUDA free/fixed-choice compatibility, not a new benchmark."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import torch
from modeling import QwenJudge
from choice_modeling import judge_choice
from constrained_choice import overall_prompt
from run_record import snapshot_run,sha
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    manifest=ROOT/'data/processed/evalmuse-pilot-v2/manifest.jsonl';meta=ROOT/'reports/local-model.json'
    row=next(json.loads(x) for x in manifest.read_text().splitlines() if json.loads(x)['split']=='development')
    info=json.loads(meta.read_text());image=ROOT/row['image'];assert sha(image)==row['image_sha256']
    out=snapshot_run(args.output,{'device':'cuda','scope':'one observed development image; hardware compatibility only','model':info},[manifest,meta])
    torch.cuda.reset_peak_memory_stats();judge=QwenJudge(str(ROOT/info['local_directory']))
    assert judge.device.type=='cuda'
    prompt=overall_prompt(row['instruction'])
    free=judge.judge(str(image),prompt,max_new_tokens=16)
    fixed=judge_choice(judge,str(image),prompt,['1','2','3','4','5'])
    assert fixed['decision'] in ['1','2','3','4','5'] and fixed['output_tokens']==1
    report={'passed':True,'device':'cuda','gpu':torch.cuda.get_device_name(),'dtype':str(next(judge.model.parameters()).dtype),'free':free,'fixed':fixed,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'scope':'one development image; no quality claim'}
    (out/'predictions.json').write_text(json.dumps(report,indent=2));print('QWEN_CUDA_SMOKE_PASSED',flush=True)
