"""One-parameter ordinal score calibration using PyTorch and development labels.

This updates a scalar temperature, never Qwen weights. Fits the empirical
distribution of human 1-5 votes, not an invented single ground-truth class.
"""
import torch


def vote_distribution(votes):
    if not votes or any(type(v) is not int or not 1<=v<=5 for v in votes):
        raise ValueError('Expected nonempty integer human votes in 1..5')
    counts = torch.bincount(torch.tensor(votes,dtype=torch.long),minlength=6)[1:6].double()
    return counts/counts.sum()


def calibration_metrics(logits, targets, temperature=1.):
    logits,targets = torch.as_tensor(logits,dtype=torch.float64),torch.as_tensor(targets,dtype=torch.float64)
    if logits.ndim!=2 or logits.shape[1]!=5 or targets.shape!=logits.shape or not len(logits):
        raise ValueError('Expected nonempty N x 5 logits and vote distributions')
    if not torch.isfinite(logits).all() or not torch.isfinite(targets).all() or (targets<0).any():
        raise ValueError('Invalid logits or targets')
    if not torch.allclose(targets.sum(-1),torch.ones(len(targets),dtype=torch.float64)):
        raise ValueError('Target rows must sum to one')
    temperature = torch.as_tensor(temperature,dtype=torch.float64)
    if temperature.numel()!=1 or not torch.isfinite(temperature) or temperature<=0:
        raise ValueError('Temperature must be positive and finite')
    log_probabilities = (logits/temperature).log_softmax(-1)
    probabilities = log_probabilities.exp()
    return {'nll':float(-(targets*log_probabilities).sum(-1).mean()),
            'soft_target_brier':float((probabilities-targets).square().sum(-1).mean()),
            'expected_scores':(probabilities*torch.arange(1,6,dtype=torch.float64)).sum(-1).tolist(),
            'interpretation':'Human vote distribution conditional on this task and candidate set, not general correctness confidence'}


def fit_temperature(rows, steps=200, learning_rate=.05):
    if not rows or any(r['split']!='development' for r in rows):
        raise ValueError('Fit accepts only development records')
    if len({r['id'] for r in rows})!=len(rows):
        raise ValueError('Duplicate development samples')
    if any(r['choices']!=['1','2','3','4','5'] for r in rows):
        raise ValueError('Calibration logits must preserve the 1..5 class order')
    if steps<=0 or learning_rate<=0:
        raise ValueError('Invalid optimization settings')
    logits = torch.tensor([r['candidate_logits'] for r in rows],dtype=torch.float64)
    targets = torch.stack([vote_distribution(r['human_overall']) for r in rows])
    calibration_metrics(logits,targets)
    log_temperature = torch.nn.Parameter(torch.zeros((),dtype=torch.float64))
    optimizer = torch.optim.Adam([log_temperature],lr=learning_rate)
    history = []
    for step in range(steps):
        optimizer.zero_grad()
        temperature = log_temperature.exp()
        loss = -(targets*(logits/temperature).log_softmax(-1)).sum(-1).mean()
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite calibration loss')
        history.append({'step':step,'nll_before_update':float(loss.detach()),
                        'temperature_before_update':float(temperature.detach())})
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            log_temperature.clamp_(torch.tensor(.05).log(),torch.tensor(20.).log())
    return {'log_temperature':log_temperature.detach().clone(),
            'temperature':float(log_temperature.detach().exp()),'training_ids':[r['id'] for r in rows],
            'training_prompt_groups':sorted({r['prompt_group'] for r in rows}),
            'steps':steps,'learning_rate':learning_rate,'history':history,
            'scope':'Scalar calibration only; Qwen model weights unchanged'}
