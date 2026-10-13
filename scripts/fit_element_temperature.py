"""Register and fit the calibration-only baseline before sealed-test inference."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from run_record import snapshot_run,sha
from element_data import read_jsonl
from element_training import binary_metrics
from element_vote_temperature import fit_vote_temperature,apply_vote_temperature

def main():
 p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--baseline-run',type=Path,required=True);p.add_argument('--validation-manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
 active=json.loads((args.study/'active-step.json').read_text(encoding='utf-8'))
 if active['stage']=='sealed-final-evaluation' or (args.study/'selection-lock.json').exists():raise ValueError('Calibration baseline must be fixed before test selection/execution')
 predictions=args.baseline_run/'baseline-predictions.json';rows=json.loads(predictions.read_text(encoding='utf-8'));validation=read_jsonl(args.validation_manifest)
 if any(r['split']!='validation' for r in validation):raise ValueError('Only development validation can fit temperature')
 targets={r['id']:r['p_pass'] for r in validation}
 if len(targets)!=len(rows) or {r['id'] for r in rows}!=targets.keys() or any(r['p_pass_human']!=targets[r['id']] for r in rows):raise ValueError('Calibration labels/IDs mismatch')
 config={'registered_before_final_test':time.time(),'baseline_predictions_sha256':sha(predictions),'validation_sha256':sha(args.validation_manifest),
  'scope':'Calibration-only comparator, no adapter selection or test labels used','policy':'Fit one positive temperature on development validation; apply fixed value once to sealed-test base-image logits'}
 out=snapshot_run(args.output,config,[predictions,args.validation_manifest])
 import torch;torch.set_num_threads(4)
 fitted=fit_vote_temperature(rows)
 result=dict(config,fit=fitted,validation_metrics=binary_metrics(apply_vote_temperature(rows,fitted['temperature'])),locked_at=time.time())
 (out/'calibration-lock.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
 print('CALIBRATION_LOCKED',json.dumps(result),flush=True)
if __name__=='__main__':main()
