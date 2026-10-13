"""Lock unseen groups and the existing adapter before fetching images."""
import argparse
import io
import json
import sys
import zipfile
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from element_data import read_jsonl
from fresh_evaluation import select_fresh,expand_elements,assert_fresh,digest
from remote_zip import RangeZip
from run_record import snapshot_run,sha

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--groups',type=int,default=16)
    p.add_argument('--seed',type=int,default=20261013)
    args=p.parse_args()
    if args.groups<2:raise ValueError('At least two groups required')
    raw=ROOT/'data/raw/evalmuse'
    info=json.loads((raw/'repository.json').read_text(encoding='utf-8'))
    ann=raw/'train_list.json'
    expected=next(x['lfs']['sha256'] for x in info['siblings'] if x['rfilename']=='train_list.json')
    if sha(ann)!=expected:raise ValueError('Annotation hash mismatch')
    history_paths=sorted((ROOT/'data/processed').glob('evalmuse-*/manifest.jsonl'))
    if not history_paths:raise ValueError('Missing historical exclusion manifests')
    historical=[row for path in history_paths for row in read_jsonl(path)]
    selected=select_fresh(json.loads(ann.read_text(encoding='utf-8')),historical,args.groups,args.seed)
    config={'seed':args.seed,'groups':args.groups,'revision':info['sha'],'checkpoint_sha256':sha(args.checkpoint),
        'checkpoint':str(args.checkpoint.resolve()),'selection':'Canonical prompt and prompt_id exclusion of all historical manifests; seeded hash order; two paths per group; no score-dependent sampling',
        'historical_manifests':{str(path.relative_to(ROOT)):sha(path) for path in history_paths},
        'official_test':False,'scope':'Previously unused public training subset; fresh project evaluation, not guaranteed absent from model pretraining'}
    out=snapshot_run(args.output,config,[ann]+history_paths)
    (out/'selection-lock.json').write_text(json.dumps({'config':config,'selected':[{'source_image':r['img_path'],'prompt_group':g,'prompt_id':r['prompt_id']} for r,g in selected]},indent=2),encoding='utf-8')
    parts=sorted((x for x in info['siblings'] if x['rfilename'].startswith('images.zip.part-')),key=lambda x:x['rfilename'])
    remote=RangeZip(parts,'https://huggingface.co/datasets/DY-Evalab/EvalMuse/resolve/'+info['sha'],ROOT/'data/cache/evalmuse-ranges'/info['sha'])
    records=[]
    with zipfile.ZipFile(remote) as archive, (out/'download.jsonl').open('w',encoding='utf-8') as log:
        members=archive.namelist()
        for row,group in selected:
            suffix=row['img_path'];matches=[n for n in members if n==suffix or n.endswith('/'+suffix)]
            if len(matches)!=1:raise ValueError('Ambiguous ZIP member')
            content=archive.read(matches[0])
            with Image.open(io.BytesIO(content)) as image:image.verify()
            identifier=digest(suffix)[:20];dest=out/(identifier+'.png');dest.write_bytes(content)
            r={'id':identifier,'image':str(dest.relative_to(ROOT)),'image_sha256':sha(dest),
                'instruction':row['prompt'],'prompt_group':group,'prompt_id':row['prompt_id'],
                'elements':list(row['element_score']),'human_elements':row['element_score'],
                'source_image':suffix,'source_revision':info['sha'],'split':'fresh_evaluation'}
            records.append(r)
            log.write(json.dumps({'id':identifier,'image_sha256':r['image_sha256']})+'\n');log.flush()
            print('IMAGE_READY',len(records),flush=True)
    assert_fresh(records,historical)
    elements=expand_elements(records)
    for name,rows in [('manifest.jsonl',records),('elements.jsonl',elements)]:
        (out/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    config.update(images=len(records),elements=len(elements),downloaded_range_bytes=remote.downloaded_bytes,
        integrity='Pinned HTTPS revision; annotation SHA256; validated ranges; ZIP member CRC32; extracted SHA256; no whole archive SHA256')
    (out/'dataset-record.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    print('FRESH_READY',json.dumps(config),flush=True)
if __name__=='__main__':main()
