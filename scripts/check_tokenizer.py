"""Check official local Qwen tokenization and Transformers generation compatibility.

Tiny random CPU model is an API test only; it is not a Qwen quality experiment.
No downloads, model training, or human labels are used.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel, LogitsProcessorList
from constrained_choice import FixedChoice, resolve_choice_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_directory',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_directory),local_files_only=True)
    choice_sets = {}
    torch.manual_seed(42)
    for name, choices in [('overall',['1','2','3','4','5']),('elements',['0','1','2'])]:
        identifiers = resolve_choice_ids(tokenizer,choices)
        vocabulary_size = max(identifiers)+8
        # Small random language model only checks the generation API invokes masking.
        model = GPT2LMHeadModel(GPT2Config(vocab_size=vocabulary_size,n_positions=8,
            n_embd=16,n_layer=1,n_head=1,bos_token_id=0,eos_token_id=vocabulary_size-1,
            pad_token_id=vocabulary_size-1)).eval()
        processor = FixedChoice(choices,identifiers,3)
        inputs = torch.tensor([[0,1,2]],dtype=torch.long)
        with torch.inference_mode():
            generated = model.generate(inputs,attention_mask=torch.ones_like(inputs),
                max_new_tokens=1,do_sample=False,logits_processor=LogitsProcessorList([processor]))
        decision = processor.decode(generated[0,3:])
        assert decision in choices
        choice_sets[name] = {'choices':choices,'token_ids':identifiers,
                             'random_api_test_output':decision}
    source_dir = Path(__file__).resolve().parent
    result = {'status':'passed','scope':'tokenizer and random CPU generation API check only',
        'torch':torch.__version__,'model_directory_name':args.model_directory.name,
        'choice_sets':choice_sets,
        'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source_dir.glob('*.py')}}
    with args.output.open('x') as stream:
        json.dump(result,stream,indent=2)
    print('TOKENIZER_AND_GENERATION_API_PASSED',choice_sets,flush=True)


if __name__ == '__main__':
    main()
