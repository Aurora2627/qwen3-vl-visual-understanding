"""Prepare expanded development data and a sealed final-test partition."""
import argparse,io,json,sys,zipfile
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from run_record import snapshot_run,sha
from element_data import read_jsonl
from fresh_evaluation import select_fresh,assert_fresh,digest
from element_diagnostics import split_corpus

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
 raw=ROOT/'data/raw/evalmuse';info=json.loads((raw/'repository.json').read_text(encoding='utf-8'));ann=raw/'train_list.json'
 expected=next(x['lfs']['sha256'] for x in info['siblings'] if x['rfilename']=='train_list.json')
 if sha(ann)!=expected:raise ValueError('Annotation hash mismatch')
 paths=sorted((ROOT/'data/processed').glob('**/manifest.jsonl'))
 history=[r for path in paths for r in read_jsonl(path) if all(k in r for k in ('instruction','prompt_id','image_sha256'))]
 selected=select_fresh(json.loads(ann.read_text(encoding='utf-8')),history,80,20261014)
 config={'seed':20261014,'groups':80,'train_groups':48,'validation_groups':16,'sealed_test_groups':16,
  'revision':info['sha'],'annotation_sha256':sha(ann),'historical_manifests':{str(p.relative_to(ROOT)):sha(p) for p in paths},
  'plan':{'loss_modes':['full-vocabulary','binary-conditional'],'epochs':1,'learning_rate':0.00005,'seed':20261013,'rank':4,'alpha':8,'last_layers':4,'max_edge':224,'precision':'bf16',
   'selection':'lowest development validation vote NLL across the two fixed training runs; no test used for selection'},
  'test_policy':'No model execution or error inspection on sealed_test until one adapter is selected on validation and its SHA256 is locked',
  'official_test':False,'selection':'80 historical-prompt-disjoint groups ordered by seeded hash, two image paths per group; no filtering by score or model output'}
 out=snapshot_run(args.output,config,[ann]+paths)
 (out/'selection-lock.json').write_text(json.dumps({'config':config,'selected':[{'source_image':r['img_path'],'prompt_id':r['prompt_id'],'prompt_group':g} for r,g in selected]},indent=2),encoding='utf-8')
 parts=sorted((x for x in info['siblings'] if x['rfilename'].startswith('images.zip.part-')),key=lambda x:x['rfilename'])
 from remote_zip import RangeZip
 remote=RangeZip(parts,'https://huggingface.co/datasets/DY-Evalab/EvalMuse/resolve/'+info['sha'],ROOT/'data/cache/evalmuse-ranges'/info['sha'])
 records=[]
 with zipfile.ZipFile(remote) as archive,(out/'download.jsonl').open('w',encoding='utf-8') as log:
  members=archive.namelist()
  for row,group in selected:
   suffix=row['img_path'];matches=[n for n in members if n==suffix or n.endswith('/'+suffix)]
   if len(matches)!=1:raise ValueError('Ambiguous ZIP member')
   content=archive.read(matches[0])
   with Image.open(io.BytesIO(content)) as image:image.verify()
   identifier=digest(suffix)[:20];dest=out/(identifier+'.png');dest.write_bytes(content)
   r={'id':identifier,'image':str(dest.relative_to(ROOT)),'image_sha256':sha(dest),'instruction':row['prompt'],
    'prompt_group':group,'prompt_id':row['prompt_id'],'elements':list(row['element_score']),'human_elements':row['element_score'],
    'source_image':suffix,'source_revision':info['sha']}
   records.append(r);log.write(json.dumps({'id':identifier,'image_sha256':r['image_sha256']})+'\n');log.flush()
   print('CORPUS_IMAGE',len(records),'/ 160',flush=True)
 assert_fresh(records,history);splits=split_corpus(records)
 (out/'manifest.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),encoding='utf-8')
 for name,rows in splits.items():
  destination=out/(name+'.jsonl')
  destination.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
 config.update(split_counts={name:{'images':len({r['image_id'] for r in rows}),'groups':len({r['prompt_group'] for r in rows}),'elements':len(rows)} for name,rows in splits.items()},
  split_sha256={name:sha(out/(name+'.jsonl')) for name in splits},downloaded_range_bytes=remote.downloaded_bytes,
  integrity='Pinned revision and annotation SHA256; validated HTTP ranges and ZIP CRC32; extracted image SHA256; not whole archive SHA256')
 (out/'dataset-record.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
 print('CORPUS_READY',json.dumps(config['split_counts']),flush=True)
if __name__=='__main__':main()
