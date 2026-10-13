"""Read existing predictions only; no additional model inference."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from element_diagnostics import diagnostic_summary
from run_record import snapshot_run,sha

def main():
 p=argparse.ArgumentParser();p.add_argument('--predictions',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
 out=snapshot_run(args.output,{'task':'observed-data diagnostics only'},[args.predictions])
 rows=[json.loads(line) for line in args.predictions.read_text(encoding='utf-8').splitlines()]
 result={'source_prediction_sha256':sha(args.predictions),'conditions':{name:diagnostic_summary([r for r in rows if r['condition']==name]) for name in sorted({r['condition'] for r in rows})}}
 (out/'diagnostics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
 for name,r in result['conditions'].items():print(name,json.dumps({'confusion':r['confusion'],'wrong_confident':r['wrong_at_confidence_ge_0_9'],'bins':r['bins']}),flush=True)
if __name__=='__main__':main()
