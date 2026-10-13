"""Frozen v0.5 adapter vs base, true-image vs text-only, on locked fresh groups."""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from element_data import read_jsonl,element_prompt
from element_training import binary_metrics
from fresh_evaluation import assert_fresh,paired_bootstrap
from run_record import snapshot_run,sha

def messages_for(row,image=None):
    content=[] if image is None else [{'type':'image','image':image}]
    content.append({'type':'text','text':element_prompt(row['instruction'],row['element'])})
    return [{'role':'user','content':content}]

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--model-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--bootstrap',type=int,default=2000)
    args=p.parse_args()
    record=json.loads((args.dataset/'dataset-record.json').read_text(encoding='utf-8'))
    if sha(args.checkpoint)!=record['checkpoint_sha256']:raise ValueError('Checkpoint changed after data lock')
    rows=read_jsonl(args.dataset/'elements.jsonl')
    historical_paths=sorted((ROOT/'data/processed').glob('evalmuse-*/manifest.jsonl'))
    history=[r for path in historical_paths if path.parent!=args.dataset for r in read_jsonl(path)]
    assert_fresh(rows,history)
    for r in rows:
        if sha(ROOT/r['image'])!=r['image_sha256']:raise ValueError('Fresh image hash mismatch')
    config={'dataset':str(args.dataset.resolve()),'checkpoint_sha256':sha(args.checkpoint),
        'elements_sha256':sha(args.dataset/'elements.jsonl'),'protocol':'Binary PASS/FAIL; BF16 base/FP32 adapter; 224px; same prompt; text-only has no image token or pixels',
        'conditions':['base_image','lora_image','base_text','lora_text','train_prior'],
        'bootstrap':args.bootstrap,'seed':20261013,'checkpoint_fixed_before_download':True}
    out=snapshot_run(args.output,config,[args.dataset/'elements.jsonl',args.dataset/'selection-lock.json'])
    try:execute(args,record,rows,out)
    except Exception:
        (out/'FAILED.json').write_text(json.dumps({'error':traceback.format_exc()},indent=2),encoding='utf-8')
        raise

def execute(args,dataset_record,rows,out):
    import torch
    from PIL import Image
    from modeling import QwenJudge
    from train_element_lora import file_sha
    from constrained_choice import resolve_choice_ids
    from element_lora import inject_lora,load_adapter_state,set_enabled
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError('BF16 CUDA required')
    torch.manual_seed(20261013)
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    if checkpoint['format']!='native-element-lora-v1':raise ValueError('Unknown adapter format')
    if checkpoint['config']['max_edge']!=224 or checkpoint['config']['precision']!='bf16':raise ValueError('Protocol mismatch')
    identity=checkpoint['model_identity']
    for filename,expected in identity['weights_sha256'].items():
        if file_sha(args.model_dir/filename)!=expected:raise ValueError('Base weight mismatch')
    for filename,key in [('config.json','config_sha256'),('tokenizer.json','tokenizer_sha256')]:
        if sha(args.model_dir/filename)!=identity[key]:raise ValueError('Base metadata mismatch')
    train_path=Path(checkpoint['config']['dataset'])/'train.jsonl'
    if sha(train_path)!=checkpoint['train_sha256']:raise ValueError('Training prior source mismatch')
    train=read_jsonl(train_path)
    prior=sum(r['p_pass'] for r in train)/len(train)
    (out/'model-identity.json').write_text(json.dumps(identity,indent=2),encoding='utf-8')
    judge=QwenJudge(str(args.model_dir),'cuda',dtype=torch.bfloat16)
    model,processor=judge.model,judge.processor
    targets=inject_lora(model,checkpoint['rank'],checkpoint['alpha'],checkpoint['last_layers'])
    if targets!=checkpoint['targets']:raise ValueError('Adapter targets mismatch')
    load_adapter_state(model,checkpoint['adapter']);model.eval()
    for param in model.parameters():param.requires_grad_(False)
    choices=resolve_choice_ids(processor.tokenizer,['PASS','FAIL'])
    predictions={name:[] for name in ['base_image','lora_image','base_text','lora_text','train_prior']}
    torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    with (out/'predictions.jsonl').open('w',encoding='utf-8') as log,torch.inference_mode():
        for index,row in enumerate(sorted(rows,key=lambda r:r['id'])):
            with Image.open(ROOT/row['image']) as source:image=source.convert('RGB')
            image.thumbnail((224,224))
            for kind,pixels in [('image',image),('text',None)]:
                inputs=processor.apply_chat_template(messages_for(row,pixels),tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt').to('cuda')
                if inputs['input_ids'].shape[-1]>1024:raise ValueError('Input exceeds fixed protocol')
                if kind=='text' and any(k in inputs for k in ['pixel_values','image_grid_thw']):raise ValueError('Text control contains image input')
                for enabled,label in [(False,'base'),(True,'lora')]:
                    set_enabled(model,enabled)
                    values=model(**inputs,use_cache=False,logits_to_keep=1).logits[0,-1,choices].float()
                    if not torch.isfinite(values).all():raise ValueError('Non-finite logits')
                    probability=torch.softmax(values,dim=-1)[0].item()
                    r={'id':row['id'],'image_id':row['image_id'],'prompt_group':row['prompt_group'],
                        'element_index':row['element_index'],'condition':label+'_'+kind,
                        'p_pass_model':probability,'p_pass_human':row['p_pass'],'choice_logits':values.cpu().tolist()}
                    predictions[r['condition']].append(r);log.write(json.dumps(r)+'\n');log.flush()
                del inputs
            r={'id':row['id'],'image_id':row['image_id'],'prompt_group':row['prompt_group'],
                'condition':'train_prior','p_pass_model':prior,'p_pass_human':row['p_pass']}
            predictions['train_prior'].append(r);log.write(json.dumps(r)+'\n');log.flush()
            if (index+1)%16==0:print('ELEMENTS',index+1,'/',len(rows),flush=True)
    metrics={key:binary_metrics(value) for key,value in predictions.items()}
    pairs=[('base_image','lora_image'),('base_text','base_image'),('lora_text','lora_image'),('train_prior','base_image')]
    comparisons={a+'__vs__'+b:paired_bootstrap(predictions[a],predictions[b],args.bootstrap) for a,b in pairs}
    result={'scope':'Previously unused project-evaluation prompts from public train split; not official benchmark; no pretraining-contamination guarantee',
        'images':len({r['image_id'] for r in rows}),'groups':len({r['prompt_group'] for r in rows}),'elements':len(rows),
        'metrics':metrics,'paired_group_bootstrap':comparisons,'train_prior':prior,'train_manifest_sha256':sha(train_path),
        'checkpoint_sha256':sha(args.checkpoint),'dataset_sha256':sha(args.dataset/'elements.jsonl'),
        'seconds':time.perf_counter()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
        'precision':'BF16 frozen base / FP32 LoRA','complete':True}
    (out/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('EVALUATION_COMPLETE',json.dumps(result),flush=True)
if __name__=='__main__':main()
