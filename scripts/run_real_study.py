"""Frozen two-protocol Qwen3-VL evaluation against released human annotations."""
import argparse
import contextlib
import hashlib
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from real_judging import make_real_prompt, parse_real
from run_record import snapshot_run
from correlation import real_summary


class Tee:
    def __init__(self, stream, log):
        self.stream,self.log = stream,log
    def write(self,value):
        self.stream.write(value)
        self.log.write(value)
        self.log.flush()
    def flush(self):
        self.stream.flush()
        self.log.flush()


def validate(records):
    for field in ['prompt_group','prompt_id','image_sha256']:
        development = {str(r[field]) for r in records if r['split']=='development'}
        heldout = {str(r[field]) for r in records if r['split']=='heldout'}
        if development & heldout:
            raise ValueError('Cross-split leakage: '+field)
    for row in records:
        if hashlib.sha256((ROOT/row['image']).read_bytes()).hexdigest() != row['image_sha256']:
            raise ValueError('Image SHA256 mismatch')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device',choices=['mps','cpu','cuda'],default='cuda')
    parser.add_argument('--output',default=None)
    parser.add_argument('--split',choices=['development','heldout','all'],default='all')
    parser.add_argument('--max-new-tokens',type=int,default=512)
    args = parser.parse_args()
    manifest = ROOT/'data/processed/evalmuse-pilot-v1/manifest.jsonl'
    dataset_record = manifest.parent/'dataset-record.json'
    records = [json.loads(line) for line in manifest.read_text().splitlines()]
    validate(records)
    if args.split!='all':
        records = [r for r in records if r['split']==args.split]
    if not records:
        raise ValueError('Empty data')
    model_record = ROOT/'reports/local-model.json'
    model = json.loads(model_record.read_text())
    out = snapshot_run(ROOT/(args.output or ('runs/real-'+datetime.now().strftime('%Y%m%d-%H%M%S'))),
        {**vars(args),'model':model,'protocols':['overall','elements'],'generation':'greedy',
         'seed':42,'image_long_edge_limit':448,'attention':'eager',
         'protocol_policy':'Frozen before first model output. Both splits evaluated once; no protocol tuning on heldout.',
         'launch':'VS Code integrated terminal/debugger'},[manifest,dataset_record,model_record])
    with (out/'console.log').open('w') as log, contextlib.redirect_stdout(Tee(sys.stdout,log)), contextlib.redirect_stderr(Tee(sys.stderr,log)):
        try:
            import torch
            from modeling import QwenJudge
            torch.manual_seed(42)
            print('LOADING_REAL_MODEL',args.device,flush=True)
            judge = QwenJudge(str(ROOT/model['local_directory']),args.device)
            runtime = {'device':args.device,'mps_available':torch.backends.mps.is_available(),
                       'dtype':str(next(judge.model.parameters()).dtype)}
            (out/'runtime.json').write_text(json.dumps(runtime,indent=2))
            predictions = []
            with (out/'predictions.jsonl').open('w') as stream:
                for split in ['development','heldout']:
                    for protocol in ['overall','elements']:
                        for record in (r for r in records if r['split']==split):
                            prompt = make_real_prompt(record['instruction'],record['elements'],protocol)
                            raw = judge.judge(ROOT/record['image'],prompt,args.max_new_tokens)
                            result = {**record,'protocol':protocol,'prompt':prompt,**raw,
                                      **parse_real(raw['raw'],protocol,len(record['elements']))}
                            stream.write(json.dumps(result,ensure_ascii=False)+'\n')
                            stream.flush()
                            predictions.append(result)
                            print('REAL_JUDGED',len(predictions),split,protocol,record['id'],result['valid'],flush=True)
            summary = {'scope':str(len(records))+'-image / '+str(len({r['prompt_group'] for r in records}))+'-prompt local pilot from public training annotations, not official test',
                       'groups':{}}
            for split in sorted({r['split'] for r in predictions}):
                for protocol in ['overall','elements']:
                    subset = [r for r in predictions if r['split']==split and r['protocol']==protocol]
                    summary['groups'][split+'/'+protocol] = real_summary(subset)
                    tensor = torch.tensor([[r['score'],sum(r['human_overall'])/len(r['human_overall'])]
                                           for r in subset if r['score'] is not None],dtype=torch.float64)
                    torch.save(tensor,out/(split+'-'+protocol+'-scores.pt'))
            if args.device=='mps':
                runtime.update({'mps_allocated_bytes_final':torch.mps.current_allocated_memory(),
                                'mps_driver_bytes_final':torch.mps.driver_allocated_memory(),
                                'memory_note':'End-of-run allocation, not peak memory'})
                (out/'runtime.json').write_text(json.dumps(runtime,indent=2))
            (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
            (out/'status.json').write_text(json.dumps({'status':'complete'}))
            print('REAL_STUDY_COMPLETE',out,flush=True)
        except Exception:
            (out/'status.json').write_text(json.dumps({'status':'failed','traceback':traceback.format_exc()}))
            raise


if __name__ == '__main__':
    main()
