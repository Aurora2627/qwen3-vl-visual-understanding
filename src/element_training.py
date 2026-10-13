"""One-token human-vote supervised objective and evaluation helpers."""
import math
import torch
from PIL import Image
from element_data import element_prompt


def make_inputs(processor, row, root, device, max_edge=224):
    with Image.open(root / row['image']) as source:
        image = source.convert('RGB')
    image.thumbnail((max_edge, max_edge))
    messages = [{'role': 'user', 'content': [
        {'type': 'image', 'image': image},
        {'type': 'text', 'text': element_prompt(row['instruction'], row['element'])}]}]
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors='pt').to(device)
    if inputs['input_ids'].shape[-1] > 1024:
        raise ValueError('Prompt too long; refusing silent truncation')
    return inputs


def human_vote_loss(logits, choice_ids, p_pass):
    if not 0 <= p_pass <= 1:
        raise ValueError('Invalid human vote probability')
    # Full vocabulary answer-token supervision: no loss on instruction/image tokens.
    logp = torch.log_softmax(logits.float(), dim=-1)
    return -(p_pass * logp[choice_ids[0]] + (1-p_pass) * logp[choice_ids[1]])


def binary_metrics(records):
    if not records:
        raise ValueError('Empty evaluation')
    nll, brier, correct, counts = [], [], [0, 0], [0, 0]
    for row in records:
        p, y = row['p_pass_model'], row['p_pass_human']
        if not (math.isfinite(p) and 0 <= p <= 1 and 0 <= y <= 1):
            raise ValueError('Invalid probability')
        q = min(1-1e-7, max(1e-7, p))
        nll.append(-y*math.log(q)-(1-y)*math.log1p(-q))
        # Expected two-class Brier against the actual binary annotator votes.
        brier.append(2*(y*(1-p)**2+(1-y)*p**2))
        if y != .5:
            label = int(y > .5)
            counts[label] += 1
            correct[label] += int(int(p >= .5) == label)
    recalls = [correct[i]/counts[i] if counts[i] else None for i in (0, 1)]
    return {'examples': len(records), 'human_ties': sum(r['p_pass_human']==.5 for r in records),
        'vote_nll': sum(nll)/len(nll), 'vote_brier': sum(brier)/len(brier),
        'majority_accuracy': sum(correct)/sum(counts) if sum(counts) else None,
        'balanced_accuracy': sum(recalls)/2 if None not in recalls else None,
        'majority_class_counts_fail_pass': counts}


def binary_vote_loss(logits, choice_ids, p_pass):
    """Conditional binary soft-label CE, aligned with the PASS/FAIL evaluator."""
    if logits.ndim != 1 or len(choice_ids)!=2 or len(set(choice_ids))!=2:
        raise ValueError('Need a vocabulary vector and two distinct choices')
    if not 0 <= p_pass <= 1 or any(i<0 or i>=logits.numel() for i in choice_ids):
        raise ValueError('Invalid target or choice token')
    logp=torch.log_softmax(logits[choice_ids].float(),dim=-1)
    return -(p_pass*logp[0]+(1-p_pass)*logp[1])
