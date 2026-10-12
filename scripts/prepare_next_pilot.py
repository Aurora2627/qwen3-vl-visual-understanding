"""Keep v1 development images, add prompt-disjoint unused heldout images.

Pure data preparation; no model output or human score values influence sampling.
Reuses the pinned public annotation file and range cache. Existing v1 is unchanged.
"""
import argparse
import hashlib
import io
import json
import shutil
import sys
import zipfile
from pathlib import Path
from PIL import Image

REVISION = 'fd2087b4bde6486ee1358a4ee546e3ee6c1860a6'


def sha(content):
    return hashlib.sha256(content).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root',type=Path,required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    sys.path.insert(0,str(root/'src'))
    from remote_zip import RangeZip
    previous = root/'data/processed/evalmuse-pilot-v1/manifest.jsonl'
    history = [json.loads(line) for line in previous.read_text().splitlines()]
    excluded = {r['prompt_group'] for r in history}
    development = [r for r in history if r['split']=='development']
    if len(development)!=32 or len(excluded)!=32:
        raise ValueError('Unexpected historical pilot')
    raw = root/'data/raw/evalmuse'
    metadata = json.loads((raw/'repository.json').read_text())
    annotations = raw/'train_list.json'
    expected_sha = next(item['lfs']['sha256'] for item in metadata['siblings'] if item['rfilename']=='train_list.json')
    if metadata['sha']!=REVISION or sha(annotations.read_bytes())!=expected_sha:
        raise ValueError('Unexpected dataset revision or annotation hash')
    groups = {}
    for row in json.loads(annotations.read_text()):
        key = ' '.join(row['prompt'].casefold().split())
        if sha(key.encode()) in excluded or not row['total_score'] or not row['element_score']:
            continue
        groups.setdefault(key,{})[row['img_path']] = row
    keys = sorted((k for k,rows in groups.items() if len(rows)>=2),
                  key=lambda k:sha(('evalmuse-new-heldout-v2:'+k).encode()))[:16]
    selected = [(key,row) for key in keys for row in sorted(groups[key].values(),
                key=lambda r:sha(r['img_path'].encode()))[:2]]
    if len(selected)!=32:
        raise ValueError('Need 16 new prompt groups / 32 images')
    out = root/'data/processed/evalmuse-pilot-v2'
    out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.jsonl').exists():
        raise FileExistsError('Completed v2 dataset already exists')
    snapshot = out/'preparation-source'
    if not snapshot.exists():
        snapshot.mkdir()
        for source in Path(__file__).resolve().parent.glob('*.py'):
            shutil.copy2(source,snapshot/source.name)
    parts = sorted((item for item in metadata['siblings'] if item['rfilename'].startswith('images.zip.part-')),
                   key=lambda item:item['rfilename'])
    remote = RangeZip(parts,'https://huggingface.co/datasets/DY-Evalab/EvalMuse/resolve/'+REVISION,
                      root/'data/cache/evalmuse-ranges'/REVISION)
    records = list(development)
    with zipfile.ZipFile(remote) as archive:
        members = archive.namelist()
        for key,row in selected:
            suffix = row['img_path']
            names = [name for name in members if name==suffix or name.endswith('/'+suffix)]
            if len(names)!=1:
                raise ValueError('Missing or ambiguous ZIP member')
            identifier = sha(suffix.encode())[:20]
            destination = out/(identifier+'.png')
            if not destination.exists():
                image_bytes = archive.read(names[0])
                with Image.open(io.BytesIO(image_bytes)) as image:
                    image.verify()
                destination.write_bytes(image_bytes)
            if any(type(v) is not int or not 1<=v<=5 for v in row['total_score']):
                raise ValueError('Unexpected human rating scale')
            record = {'id':identifier,'image':str(destination.relative_to(root)),
                'image_sha256':sha(destination.read_bytes()),'instruction':row['prompt'],
                'prompt_group':sha(key.encode()),'prompt_id':row['prompt_id'],'split':'heldout',
                'generator':suffix.split('/')[0],'elements':list(row['element_score']),
                'human_overall':row['total_score'],'human_elements':row['element_score'],
                'annotation_flags':{k:row.get(k) for k in ['type','promt_meaningless','split_confidence','attribute_confidence']},
                'source_image':suffix,'source_member':names[0],'source_revision':REVISION}
            records.append(record)
            print('NEW_HELDOUT_IMAGE_READY',len(records)-32,identifier,flush=True)
    for field in ['prompt_group','prompt_id','image_sha256']:
        if {str(r[field]) for r in development} & {str(r[field]) for r in records[32:]}:
            raise ValueError('Cross-split overlap: '+field)
    if excluded & {r['prompt_group'] for r in records[32:]}:
        raise ValueError('New heldout overlaps any v1 prompt')
    (out/'manifest.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
    (out/'dataset-record.json').write_text(json.dumps({'source':'DY-Evalab/EvalMuse',
        'revision':REVISION,'annotation_sha256':expected_sha,'historical_manifest_sha256':sha(previous.read_bytes()),
        'development':'32 existing v1 development images; no v1 heldout used for calibration',
        'heldout':'16 unused canonical prompts, 2 images each; excludes all 32 v1 prompt groups',
        'sample_rule':'seeded SHA256 prompt ordering then SHA256 source-image-path ordering',
        'images':64,'development_prompts':16,'heldout_prompts':16,
        'label_based_filter':False,'official_test':False,
        'downloaded_range_bytes_this_invocation':remote.downloaded_bytes,
        'integrity':'Pinned revision, validated ranges, member CRC32, extracted image SHA256; not whole-archive SHA256'},indent=2))
    print('NEXT_PILOT_READY',64,flush=True)


if __name__ == '__main__':
    main()
