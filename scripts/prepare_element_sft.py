"""Freeze train/validation from previously observed development images only."""
import argparse
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from element_data import read_jsonl, split_development
from run_record import snapshot_run, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, default=ROOT/'data/processed/evalmuse-pilot-v2/manifest.jsonl')
    parser.add_argument('--output', type=Path, default=ROOT/'data/processed/element-sft-v1')
    parser.add_argument('--validation-groups', type=int, default=4)
    parser.add_argument('--seed', type=int, default=20261012)
    args = parser.parse_args()
    rows = read_jsonl(args.manifest)
    splits = split_development(rows, args.validation_groups, args.seed)
    for name, examples in splits.items():
        for row in examples:
            if sha(ROOT/row['image']) != row['image_sha256']:
                raise ValueError('Image hash mismatch: '+row['image_id'])
    config = {'task': 'element human-vote supervised LoRA', 'seed': args.seed,
        'validation_groups': args.validation_groups, 'sampling_uses_labels': False,
        'source_manifest_sha256': sha(args.manifest), 'independent_test_available': False,
        'historical_holdouts': 'quarantined: not used for training, validation, or new effect claims',
        'scope': 'Training/validation engineering pilot, not an independent benchmark',
        'splits': {k: {'images':len({r['image_id'] for r in v}),
            'prompts':len({r['prompt_group'] for r in v}), 'elements':len(v)} for k,v in splits.items()}}
    out = snapshot_run(args.output, config, [args.manifest])
    for name, examples in splits.items():
        (out/(name+'.jsonl')).write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in examples), encoding='utf-8')
    (out/'dataset-record.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    print('ELEMENT_DATA_READY', json.dumps(config['splits']), flush=True)

if __name__ == '__main__':
    main()
