"""Real local Qwen3-VL inference; save raw judgments, provenance and PyTorch metrics."""
import argparse
import hashlib
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from judging import make_prompt, parse_answer
from metrics import summarize
from run_record import snapshot_run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device',choices=['mps','cpu','cuda'],default='cuda')
    parser.add_argument('--output',default=None)
    parser.add_argument('--limit',type=int,default=8)
    parser.add_argument('--groups',nargs='+',default=['direct','decomposed','mismatched_image','no_image'],
                        choices=['direct','decomposed','mismatched_image','no_image'])
    parser.add_argument('--max-new-tokens',type=int,default=192)
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error('limit must be positive')
    manifest = ROOT/'data/processed/diagnostics-v1/manifest.jsonl'
    model_record = ROOT/'reports/local-model.json'
    records = [json.loads(line) for line in manifest.read_text().splitlines()][:args.limit]
    for record in records:
        digest = hashlib.sha256((ROOT/record['image']).read_bytes()).hexdigest()
        if digest != record['image_sha256']:
            raise ValueError('Image hash mismatch: '+record['id'])
    metadata = json.loads(model_record.read_text())
    output = ROOT/(args.output or ('runs/study-'+datetime.now().strftime('%Y%m%d-%H%M%S')))
    config = {**vars(args),'model':metadata,'dataset':'procedural diagnostics, development only',
              'launch':'VS Code integrated terminal/debugger', 'generation':'greedy',
              'image_long_edge_limit':448, 'attention':'eager',
              'note':'Mismatch controls do not reuse original correctness labels for accuracy.'}
    out = snapshot_run(output,config,[manifest,model_record])
    (out/'predictions.jsonl').touch()
    try:
        import torch
        from modeling import QwenJudge
        torch.manual_seed(42)
        runtime = {'mps_available':torch.backends.mps.is_available(),'cuda_available':torch.cuda.is_available(),
                   'device':args.device, 'dtype':'float32' if args.device=='cpu' else 'float16'}
        (out/'runtime.json').write_text(json.dumps(runtime,indent=2))
        print('LOADING_MODEL',metadata['revision'],args.device,flush=True)
        judge = QwenJudge(str(ROOT/metadata['local_directory']),args.device)
        groups = {}
        unique_images = list(dict.fromkeys(r['image'] for r in records))
        for group in args.groups:
            rows = []
            for record in records:
                image = record['image']
                if group == 'mismatched_image':
                    if len(unique_images)<2:
                        raise ValueError('Mismatch control needs at least two distinct images')
                    image = unique_images[(unique_images.index(image)+1)%len(unique_images)]
                protocol = 'decomposed' if group=='decomposed' else 'direct'
                prompt = make_prompt(record['instruction'],record['claims'],protocol)
                result = judge.judge(None if group=='no_image' else ROOT/image,prompt,args.max_new_tokens)
                row = {**record, 'group':group,'image_used':None if group=='no_image' else image,
                       'prompt':prompt,**result,**parse_answer(result['raw'],protocol,len(record['claims']))}
                rows.append(row)
                with (out/'predictions.jsonl').open('a') as stream:
                    stream.write(json.dumps(row)+'\n')
                print('JUDGED',group,record['id'],row['verdict'],round(row['seconds'],2),flush=True)
            groups[group] = rows
        summary = {'scope':str(len(records))+' synthetic diagnostic cases; not benchmark or human agreement',
                   'groups':{},'controls':{}}
        for group, rows in groups.items():
            if group in ['direct','decomposed']:
                metrics,matrix = summarize(rows)
                summary['groups'][group] = metrics
                summary['groups'][group]['categories'] = {category:summarize([r for r in rows if r['category']==category])[0]
                    for category in sorted({r['category'] for r in rows})}
                torch.save(matrix,out/(group+'-confusion.pt'))
            else:
                summary['controls'][group] = {'n':len(rows),
                    'verdict_counts':{v:sum(r['verdict']==v for r in rows) for v in ['PASS','FAIL','UNCERTAIN','INVALID']},
                    'accuracy':None,'reason':'No image or different image; original labels are inapplicable'}
                if 'direct' in groups:
                    baseline = {r['id']:r for r in groups['direct']}
                    summary['controls'][group]['same_verdict_as_direct'] = sum(r['verdict']==baseline[r['id']]['verdict'] for r in rows)
        (out/'summary.json').write_text(json.dumps(summary,indent=2))
        (out/'status.json').write_text(json.dumps({'status':'complete'}))
        print('STUDY_COMPLETE',out,flush=True)
    except Exception:
        (out/'status.json').write_text(json.dumps({'status':'failed','traceback':traceback.format_exc()}))
        raise


if __name__ == '__main__':
    main()
