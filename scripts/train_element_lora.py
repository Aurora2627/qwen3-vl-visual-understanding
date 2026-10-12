"""Windows CUDA human-vote answer-token LoRA; source and checkpoints preserved."""
import argparse
import contextlib
import hashlib
import json
import random
import sys
import time
import traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from run_record import snapshot_run, sha
from element_data import read_jsonl, assert_disjoint
from element_lora import inject_lora, adapters, set_enabled, adapter_state, load_adapter_state
from element_training import make_inputs, human_vote_loss, binary_metrics
from constrained_choice import resolve_choice_ids


def file_sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--model-revision', required=True)
    parser.add_argument('--dataset', type=Path, default=ROOT/'data/processed/element-sft-v1')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--rank', type=int, default=4)
    parser.add_argument('--alpha', type=int, default=8)
    parser.add_argument('--last-layers', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=0.0001)
    parser.add_argument('--accumulation', type=int, default=4)
    parser.add_argument('--max-train', type=int, default=0)
    parser.add_argument('--max-validation', type=int, default=0)
    parser.add_argument('--max-edge', type=int, default=224)
    parser.add_argument('--seed', type=int, default=20261012)
    parser.add_argument('--precision', choices=['bf16','fp16'], default='bf16')
    parser.add_argument('--reload-checkpoint', type=Path)
    args = parser.parse_args()
    if args.epochs < 1 or args.accumulation < 1 or args.max_edge < 112 or args.learning_rate <= 0 or args.max_train < 0 or args.max_validation < 0:
        raise ValueError('Invalid training configuration')
    train_path, val_path = args.dataset/'train.jsonl', args.dataset/'validation.jsonl'
    train, validation = read_jsonl(train_path), read_jsonl(val_path)
    assert_disjoint(train, validation)
    train = sorted(train, key=lambda r: hashlib.sha256((str(args.seed)+r['id']).encode()).hexdigest())
    validation = sorted(validation, key=lambda r: r['id'])
    if args.max_train:
        train = train[:args.max_train]
    if args.max_validation:
        validation = validation[:args.max_validation]
    if not train or not validation:
        raise ValueError('Empty selected split')
    for row in train+validation:
        if sha(ROOT/row['image']) != row['image_sha256']:
            raise ValueError('Image hash mismatch')
    config = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(device='cuda', task='human-vote answer-token LoRA', choices=['PASS','FAIL'],
        objective='Full-vocabulary next-token cross-entropy with soft binary human-vote target',
        scope='Observed development train/validation only; no independent-test or generalization claim',
        train_examples=len(train), validation_examples=len(validation))
    manifests = [train_path, val_path, args.dataset/'dataset-record.json']
    if args.reload_checkpoint:
        manifests += [args.reload_checkpoint]
    out = snapshot_run(args.output, config, manifests)
    with (out/'run.log').open('w', encoding='utf-8') as log:
        class Tee:
            def __init__(self, original): self.original = original
            def write(self, text): self.original.write(text); log.write(text); log.flush()
            def flush(self): self.original.flush(); log.flush()
        with contextlib.redirect_stdout(Tee(sys.stdout)), contextlib.redirect_stderr(Tee(sys.stderr)):
            try:
                execute(args, config, train, validation, out)
            except Exception:
                traceback.print_exc()
                (out/'FAILED.json').write_text(json.dumps({'failed':True,'error':traceback.format_exc()}), encoding='utf-8')
                raise


def execute(args, config, train, validation, out):
    import torch
    from modeling import QwenJudge
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this training entry')
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.cuda.reset_peak_memory_stats()
    model_files = sorted(args.model_dir.glob('*.safetensors'))
    if not model_files:
        raise ValueError('No local model weights')
    model_identity = {'revision':args.model_revision,
        'config_sha256':sha(args.model_dir/'config.json'),
        'tokenizer_sha256':sha(args.model_dir/'tokenizer.json'),
        'weights_sha256':{p.name:file_sha(p) for p in model_files}}
    (out/'model-identity.json').write_text(json.dumps(model_identity,indent=2),encoding='utf-8')
    if args.precision == 'bf16' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16 unavailable; explicitly select fp16')
    compute_dtype = torch.bfloat16 if args.precision == 'bf16' else torch.float16
    judge = QwenJudge(str(args.model_dir), 'cuda', dtype=compute_dtype)
    model, processor = judge.model, judge.processor
    targets = inject_lora(model, args.rank, args.alpha, args.last_layers)
    choice_ids = resolve_choice_ids(processor.tokenizer, ['PASS', 'FAIL'])
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters or any('visual' in n for n,p in model.named_parameters() if p.requires_grad):
        raise ValueError('Unexpected trainable vision parameters')
    trainable_names = {n for n,p in model.named_parameters() if p.requires_grad}
    expected = {name+'.'+key for name in targets for key in ('A','B')}
    if trainable_names != expected:
        raise ValueError('Unexpected trainable parameters')
    info = {'targets':targets, 'trainable_parameters':sum(p.numel() for p in parameters),
        'total_parameters':sum(p.numel() for p in model.parameters()), 'choice_token_ids':choice_ids,
        'gpu':torch.cuda.get_device_name(0), 'dtype':args.precision+' frozen base, FP32 adapters, CUDA autocast',
        'gradient_checkpointing':True, 'image_max_edge':args.max_edge}
    (out/'architecture.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    print('MODEL_READY',json.dumps(info),flush=True)

    def logits(row):
        inputs = make_inputs(processor,row,ROOT,judge.device,args.max_edge)
        with torch.autocast('cuda',dtype=compute_dtype):
            result = model(**inputs, use_cache=False, logits_to_keep=1).logits[0,-1].float()
        return result

    @torch.no_grad()
    def evaluate(label):
        model.eval()
        records = []
        for row in validation:
            values = logits(row)[choice_ids]
            prob = values.softmax(-1)[0].item()
            records.append({'id':row['id'],'image_id':row['image_id'],'prompt_group':row['prompt_group'],
                'element':row['element'],'p_pass_model':prob,'p_pass_human':row['p_pass'],
                'choice_logits':values.cpu().tolist()})
        (out/(label+'-predictions.json')).write_text(json.dumps(records,indent=2),encoding='utf-8')
        metrics = binary_metrics(records)
        print('VALIDATION',label,json.dumps(metrics),flush=True)
        return metrics,records

    if args.reload_checkpoint:
        checkpoint = torch.load(args.reload_checkpoint,map_location='cpu',weights_only=True)
        if checkpoint['model_identity'] != model_identity or checkpoint['targets'] != targets:
            raise ValueError('Checkpoint base/targets mismatch')
        if checkpoint['train_sha256'] != sha(args.dataset/'train.jsonl') or checkpoint['validation_sha256'] != sha(args.dataset/'validation.jsonl'):
            raise ValueError('Checkpoint dataset fingerprint mismatch')
        for key in ('rank','alpha','last_layers','max_edge','seed','max_validation','precision'):
            if checkpoint['config'][key] != config[key]: raise ValueError('Reload configuration mismatch: '+key)
        load_adapter_state(model,checkpoint['adapter'])
        metrics,records = evaluate('fresh-reload')
        previous = json.loads((args.reload_checkpoint.parent/'selected-predictions.json').read_text(encoding='utf-8'))
        if [r['id'] for r in previous] != [r['id'] for r in records]:
            raise ValueError('Reload evaluation IDs differ')
        difference = max(abs(a-b) for x,y in zip(previous,records) for a,b in zip(x['choice_logits'],y['choice_logits']))
        if difference > .002:
            raise ValueError('Fresh-process reload logits differ: '+str(difference))
        summary = {'scope':config['scope'],'reload_passed':True,'max_logit_difference':difference,
            'metrics':metrics,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
    else:
        # Zero-init adapter must reproduce the same frozen-base outputs before optimization.
        probe = validation[0]
        model.eval()
        with torch.no_grad():
            set_enabled(model,False)
            before = logits(probe)[choice_ids]
            set_enabled(model,True)
            initial = logits(probe)[choice_ids]
        if not torch.allclose(before,initial,atol=1e-5,rtol=0):
            raise ValueError('Zero-init LoRA changed baseline')
        baseline,base_records = evaluate('baseline')
        optimizer = torch.optim.AdamW(parameters,lr=args.learning_rate,weight_decay=0)
        scaler = torch.amp.GradScaler('cuda', enabled=args.precision=='fp16', init_scale=128)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        model.config.use_cache = False
        best = float('inf')
        updates = 0
        history = []
        start = time.perf_counter()
        for epoch in range(args.epochs):
            model.train()
            if hasattr(model.model,'visual'): model.model.visual.eval()
            order = list(train)
            random.Random(args.seed+epoch).shuffle(order)
            for offset in range(0,len(order),args.accumulation):
                batch = order[offset:offset+args.accumulation]
                optimizer.zero_grad(set_to_none=True)
                total_loss = 0.
                for row in batch:
                    loss = human_vote_loss(logits(row),choice_ids,row['p_pass'])
                    if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
                    total_loss += loss.item()/len(batch)
                    scaler.scale(loss/len(batch)).backward()
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(parameters,1.0,error_if_nonfinite=True)
                if norm.item() == 0: raise ValueError('Zero adapter gradient')
                if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
                    raise ValueError('Frozen base received gradients')
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() < old_scale:
                    raise ValueError('Optimizer step skipped due to overflow')
                updates += 1
                item = {'epoch':epoch+1,'update':updates,'loss':total_loss,'gradient_norm':norm.item()}
                history.append(item)
                print('TRAIN',json.dumps(item),flush=True)
                with (out/'training.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(item)+'\n')
            metrics,records = evaluate('epoch-'+str(epoch+1))
            if metrics['vote_nll'] < best:
                best = metrics['vote_nll']
                state = adapter_state(model)
                checkpoint = {'format':'native-element-lora-v1','adapter':state,'targets':targets,
                    'model_identity':model_identity,'rank':args.rank,'alpha':args.alpha,'last_layers':args.last_layers,
                    'epoch':epoch+1,'updates':updates,'validation_vote_nll':best,
                    'train_sha256':sha(args.dataset/'train.jsonl'),'validation_sha256':sha(args.dataset/'validation.jsonl'),
                    'config':config}
                torch.save(checkpoint,out/'adapter.pt')
                (out/'selected-predictions.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
                (out/'selected-metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
        state = torch.load(out/'adapter.pt',map_location='cpu',weights_only=True)['adapter']
        if not any(k.endswith('.B') and torch.count_nonzero(v).item()>0 for k,v in state.items()):
            raise ValueError('No nonzero LoRA B update')
        load_adapter_state(model,state)
        reloaded,records = evaluate('in-process-reload')
        selected = json.loads((out/'selected-predictions.json').read_text(encoding='utf-8'))
        difference = max(abs(a-b) for x,y in zip(selected,records) for a,b in zip(x['choice_logits'],y['choice_logits']))
        if difference > .002: raise ValueError('Reload differs')
        summary = {'scope':config['scope'],'baseline':baseline,'selected_validation':reloaded,
            'optimizer_updates':updates,'epochs':args.epochs,'nonzero_B_matrices':sum(k.endswith('.B') and bool(v.count_nonzero()) for k,v in state.items()),
            'training_seconds':time.perf_counter()-start,'in_process_reload_max_logit_difference':difference,
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
            'independent_test_available':False,'algorithm_improvement_claim':False}
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('ELEMENT_LORA_COMPLETE',json.dumps(summary),flush=True)

if __name__ == '__main__':
    main()
