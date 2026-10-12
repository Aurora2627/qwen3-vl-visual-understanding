"""PyTorch fixed-choice decoding draft; not part of the frozen v0.3 run.

Masks first-step vocabulary logits to verified single-token choices. It does
not train the model, fix visual reasoning, or produce calibrated confidence.
Use with greedy generation and max_new_tokens=1, not free-form explanation.
"""
import torch


def resolve_choice_ids(tokenizer, choices):
    if not choices or len(set(choices)) != len(choices):
        raise ValueError('Choices must be nonempty and unique')
    identifiers = []
    for choice in choices:
        encoded = tokenizer.encode(choice, add_special_tokens=False)
        if len(encoded) != 1 or encoded[0] in tokenizer.all_special_ids:
            raise ValueError('Choice must be exactly one ordinary token: '+choice)
        if tokenizer.decode(encoded, clean_up_tokenization_spaces=False) != choice:
            raise ValueError('Choice does not round-trip exactly: '+choice)
        identifiers.append(encoded[0])
    if len(set(identifiers)) != len(identifiers):
        raise ValueError('Choice token IDs collide')
    return identifiers


class FixedChoice:
    """Transformers-compatible logits processor, one request/one generation step."""
    def __init__(self, choices, token_ids, prompt_tokens):
        if len(choices)<2 or len(choices)!=len(token_ids) or len(set(token_ids))!=len(token_ids):
            raise ValueError('Need at least two distinct choices and IDs')
        if len(set(choices))!=len(choices) or prompt_tokens<=0:
            raise ValueError('Invalid labels or prompt length')
        if any(type(value) is not int or value<0 for value in token_ids):
            raise ValueError('Invalid vocabulary ID')
        self.choices, self.token_ids = list(choices), list(token_ids)
        self.prompt_tokens = prompt_tokens
        self.statistics = None

    def __call__(self, input_ids, scores):
        if self.statistics is not None or input_ids.ndim!=2 or scores.ndim!=2:
            raise ValueError('Processor supports exactly one generation step')
        if input_ids.shape[1]!=self.prompt_tokens or input_ids.shape[0]!=1 or scores.shape[0]!=1:
            raise ValueError('Expected one unexpanded input and first-step logits')
        if max(self.token_ids)>=scores.shape[1]:
            raise ValueError('Choice ID outside vocabulary')
        selected = scores[:,self.token_ids].float()
        if not torch.isfinite(selected).all() or torch.isnan(scores).any() or (scores==float('inf')).any():
            raise ValueError('Nonfinite choice logits; refuse to invent a score')
        probabilities = selected.softmax(dim=-1)
        top = probabilities.sort(descending=True,dim=-1).values
        entropy = -(probabilities * probabilities.clamp_min(torch.finfo(torch.float32).tiny).log()).sum()
        self.statistics = {'choices':self.choices,'token_ids':self.token_ids,
            'candidate_logits':selected.detach().cpu().tolist()[0],
            'candidate_probabilities':probabilities.detach().cpu().tolist()[0],
            'candidate_entropy':float(entropy),'candidate_margin':float(top[0,0]-top[0,1]),
            'candidate_mass_full_vocabulary':float((selected.logsumexp(-1)-scores.float().logsumexp(-1)).exp()[0]),
            'unconstrained_top_token_id':int(scores.argmax(-1)[0]),
            'interpretation':'Conditional on the fixed choices; not calibrated correctness probability'}
        masked = torch.full_like(scores,float('-inf'))
        masked[:,self.token_ids] = scores[:,self.token_ids]
        return masked

    def decode(self, completion_ids):
        values = completion_ids.tolist() if isinstance(completion_ids,torch.Tensor) else list(completion_ids)
        if self.statistics is None or len(values)!=1 or values[0] not in self.token_ids:
            raise ValueError('Expected one allowed generated token')
        return self.choices[self.token_ids.index(values[0])]


def overall_prompt(instruction):
    return ('Judge whether the visible image fulfills this image-generation request. '
            'The request describes desired image content; do not require its words to be printed. '
            'Ignore aesthetic quality.\nRequest: '+instruction+'\n'
            'Score: 1=none of the important content matches; 2=few aspects match; '
            '3=partially matches; 4=most content matches with some errors; '
            '5=all important content matches. Output exactly one digit from 1,2,3,4,5.')


def element_prompt(instruction, element):
    return ('Judge this element in the context of the complete image-generation request. '
            'Use only visible evidence. Do not require request words to appear as printed text. '
            'Category tags in parentheses are hints.\nRequest: '+instruction+
            '\nElement: '+element+'\nOutput exactly one digit: '
            '0=FAIL, required content absent or contradicted; '
            '1=PASS, visible support with correct attributes/bindings/relations; '
            '2=UNCERTAIN, cannot determine from this image.')
