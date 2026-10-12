"""Fixed-choice inference using the project's existing PyTorch QwenJudge.

No second model is created. Kept outside the sealed v0.3 source until release.
"""
import time
import torch
from PIL import Image
from transformers import LogitsProcessorList
from constrained_choice import FixedChoice, resolve_choice_ids


@torch.inference_mode()
def judge_choice(judge, image_path, prompt, choices):
    identifiers = resolve_choice_ids(judge.processor.tokenizer,choices)
    content = []
    if image_path is not None:
        with Image.open(image_path) as source:
            picture = source.convert('RGB')
        picture.thumbnail((448,448))
        content.append({'type':'image','image':picture})
    content.append({'type':'text','text':prompt})
    judge.synchronize()
    start = time.perf_counter()
    inputs = judge.processor.apply_chat_template([{'role':'user','content':content}],
        tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt').to(judge.device)
    prompt_tokens = inputs['input_ids'].shape[-1]
    processor = FixedChoice(choices,identifiers,prompt_tokens)
    generated = judge.model.generate(**inputs,do_sample=False,num_beams=1,
        num_return_sequences=1,max_new_tokens=1,logits_processor=LogitsProcessorList([processor]))
    judge.synchronize()
    completion = generated[0,prompt_tokens:]
    decision = processor.decode(completion)
    return {'raw':decision,'seconds':time.perf_counter()-start,'input_tokens':prompt_tokens,
            'output_tokens':len(completion),'decision':decision,**processor.statistics,
            'generation_length_policy':'Exactly one complete decision token; length limit is intentional, not a truncated explanation'}
