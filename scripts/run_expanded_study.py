"""VS Code-launched Windows study: registered development comparison, then sealed test."""
import argparse,json,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from run_record import snapshot_run,sha
from element_data import read_jsonl

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
 stamp=args.output.name.replace('expanded-study-','')
 dataset=ROOT/'data/processed'/('element-corpus-v2-'+stamp)
 protocol={'dataset':str(dataset),'loss_modes':['full-vocabulary','binary-conditional'],'seed':20261013,'epochs':1,'learning_rate':0.00005,
  'selection':'minimum development validation vote NLL across exactly these two runs; ties prefer full-vocabulary',
  'test':'sealed prompt-disjoint test evaluated once after checkpoint SHA256 lock; no test-dependent retraining',
  'device':'Windows CUDA','source_revision':'89644892e4d85e24eaac8bacfd4f463576704203'}
 out=snapshot_run(args.output,protocol)
 (out/'pid.json').write_text(json.dumps({'pid':os.getpid(),'started':time.time()}),encoding='utf-8')
 env=os.environ.copy();env.update(PYTHONUTF8='1',PYTHONNOUSERSITE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
 steps=[]
 def step(name,command,network=False):
  config=dict(env)
  if network:config.update(HTTP_PROXY='http://127.0.0.1:7890',HTTPS_PROXY='http://127.0.0.1:7890')
  (out/'active-step.json').write_text(json.dumps({'stage':name,'started':time.time(),'command':command},indent=2),encoding='utf-8')
  print('STAGE_START',name,flush=True)
  with (out/(name+'.log')).open('w',encoding='utf-8') as log:
   r=subprocess.run(command,cwd=ROOT,env=config,stdout=log,stderr=subprocess.STDOUT)
  steps.append({'stage':name,'exit':r.returncode,'completed':time.time()})
  (out/'steps.json').write_text(json.dumps(steps,indent=2),encoding='utf-8')
  print('STAGE_END',name,r.returncode,flush=True)
  if r.returncode:raise RuntimeError('Stage failed: '+name)
 py=[sys.executable,'-X','utf8']
 model=ROOT/'data/cache/local-qwen3-vl'/protocol['source_revision']
 def train_command(mode,destination,reload=None,smoke=False):
  cmd=py+['scripts/train_element_lora.py','--dataset',str(dataset),'--model-dir',str(model),'--model-revision',protocol['source_revision'],
   '--output',str(destination),'--loss-mode',mode,'--learning-rate','0.00005','--seed','20261013','--precision','bf16']
  if smoke:cmd+=['--max-train','8','--max-validation','8']
  if reload:cmd+=['--reload-checkpoint',str(reload)]
  return cmd
 try:
  step('tests',py+['-m','unittest','discover','-s','tests','-v'])
  step('prepare',py+['scripts/prepare_element_corpus.py','--output',str(dataset)],network=True)
  step('smoke-conditional',train_command('binary-conditional',ROOT/'runs'/('expanded-smoke-'+stamp),smoke=True))
  models={}
  for mode in protocol['loss_modes']:
   run=ROOT/'runs'/('expanded-'+mode+'-'+stamp)
   step('train-'+mode,train_command(mode,run))
   step('reload-'+mode,train_command(mode,ROOT/'runs'/('expanded-reload-'+mode+'-'+stamp),reload=run/'adapter.pt'))
   summary=json.loads((run/'summary.json').read_text(encoding='utf-8'))
   models[mode]={'run':str(run),'checkpoint_sha256':sha(run/'adapter.pt'),'summary':summary}
  winner=min(protocol['loss_modes'],key=lambda mode:models[mode]['summary']['selected_validation']['vote_nll'])
  selected=models[winner];checkpoint=Path(selected['run'])/'adapter.pt'
  lock={'chosen_loss_mode':winner,'checkpoint':str(checkpoint),'checkpoint_sha256':selected['checkpoint_sha256'],
   'validation_comparison':{mode:value['summary']['selected_validation'] for mode,value in models.items()},
   'dataset_record_sha256':sha(dataset/'dataset-record.json'),'sealed_test_sha256':sha(dataset/'sealed_test.jsonl'),
   'criterion':protocol['selection'],'locked_before_test':time.time()}
  (out/'selection-lock.json').write_text(json.dumps(lock,indent=2),encoding='utf-8')
  test=dataset/'final-evaluation';test.mkdir(exist_ok=False)
  # Copy the preregistered sealed partition without selecting rows by score or model output.
  content=(dataset/'sealed_test.jsonl').read_bytes();(test/'elements.jsonl').write_bytes(content)
  (test/'dataset-record.json').write_text(json.dumps({'checkpoint_sha256':sha(checkpoint),'parent_record_sha256':sha(dataset/'dataset-record.json'),'official_test':False}),encoding='utf-8')
  (test/'selection-lock.json').write_text(json.dumps(lock,indent=2),encoding='utf-8')
  final_run=ROOT/'runs'/('expanded-final-evaluation-'+stamp)
  step('sealed-final-evaluation',py+['scripts/evaluate_fresh_elements.py','--dataset',str(test),'--checkpoint',str(checkpoint),'--model-dir',str(model),'--output',str(final_run)])
  assert sha(checkpoint)==lock['checkpoint_sha256']
  result={'protocol':protocol,'models':models,'selection':lock,'final_run':str(final_run),
   'dataset':str(dataset),'dataset_record':json.loads((dataset/'dataset-record.json').read_text(encoding='utf-8')),
   'final_results':json.loads((final_run/'results.json').read_text(encoding='utf-8')),'complete':True}
  (out/'completed.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
  print('EXPANDED_STUDY_COMPLETE',winner,flush=True)
 except Exception:
  (out/'FAILED.json').write_text(json.dumps({'traceback':traceback.format_exc(),'steps':steps},indent=2),encoding='utf-8')
  raise
if __name__=='__main__':main()
