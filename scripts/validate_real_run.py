"""Verify a finished run's provenance, completeness and public data boundary."""
import argparse
import hashlib
import json
from pathlib import Path
import torch


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run, report = args.run, args.report
    if json.loads((run/'status.json').read_text())['status'] != 'complete':
        raise ValueError('Incomplete run')
    source = json.loads((run/'source-manifest.json').read_text())
    for name, expected in source['source_sha256'].items():
        if sha(run/'source_snapshot'/name) != expected:
            raise ValueError('Source snapshot changed: '+name)
    for index, (name, expected) in enumerate(source['input_sha256'].items()):
        if sha(run/'input_manifests'/(str(index)+'-'+Path(name).name)) != expected:
            raise ValueError('Input snapshot changed')
    config = json.loads((run/'config.json').read_text())
    manifest = next((run/'input_manifests').glob('*-manifest.jsonl'))
    inputs = [json.loads(line) for line in manifest.read_text().splitlines()]
    if config['split'] != 'all':
        inputs = [r for r in inputs if r['split'] == config['split']]
    predictions = [json.loads(line) for line in (run/'predictions.jsonl').read_text().splitlines()]
    expected = {(r['id'], protocol) for r in inputs for protocol in config['protocols']}
    actual = [(r['id'], r['protocol']) for r in predictions]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('Missing or duplicate predictions')
    by_id = {r['id']:r for r in inputs}
    for row in predictions:
        if any(row[key] != value for key,value in by_id[row['id']].items()):
            raise ValueError('Prediction input differs from saved manifest')
    summary = json.loads((run/'summary.json').read_text())
    for key, metrics in summary['groups'].items():
        split, protocol = key.split('/')
        subset = [r for r in predictions if r['split']==split and r['protocol']==protocol]
        scored = [r for r in subset if r['score'] is not None]
        if len(subset) != metrics['n_images'] or len(scored) != metrics['scored_images']:
            raise ValueError('Summary coverage mismatch')
        saved = torch.load(run/(split+'-'+protocol+'-scores.pt'), weights_only=True)
        values = torch.tensor([[r['score'], sum(r['human_overall'])/len(r['human_overall'])]
                               for r in scored], dtype=torch.float64)
        if not torch.equal(saved, values):
            raise ValueError('Saved score tensor differs from predictions')
    if json.loads((report/'summary.json').read_text()) != summary:
        raise ValueError('Public summary differs')
    public = [json.loads(line) for line in (report/'model-outputs.jsonl').read_text().splitlines()]
    allowed = {'id','split','protocol','image_sha256','raw','parsed','error','valid','score',
               'seconds','input_tokens','output_tokens','reached_token_limit'}
    if len(public) != len(predictions):
        raise ValueError('Public output count differs')
    for original, exported in zip(predictions,public):
        if set(exported) != allowed or any(exported[k] != original[k] for k in allowed):
            raise ValueError('Export includes unexpected fields or changed predictions')
    result = {'status':'passed', 'run_id':run.name, 'predictions':len(predictions),
              'source_files_verified':len(source['source_sha256']),
              'input_manifests_verified':len(source['input_sha256']),
              'public_outputs_sha256':sha(report/'model-outputs.jsonl'),
              'checks':['snapshot hashes','prediction completeness','input identity',
                        'PyTorch score tensors','public summary equality','export field boundary']}
    with args.output.open('x') as stream:
        json.dump(result,stream,indent=2)
    print('REAL_RUN_VALIDATED',len(predictions),flush=True)


if __name__ == '__main__':
    main()
