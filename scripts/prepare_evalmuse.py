"""Deterministic prompt-disjoint pilot from public EvalMuse training annotations."""
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from remote_zip import RangeZip

def digest(value):
    return hashlib.sha256(value).hexdigest()

def prompt_key(text):
    return ' '.join(text.casefold().split())

def main():
    raw = ROOT/'data/raw/evalmuse'
    info = json.loads((raw/'repository.json').read_text())
    annotations = raw/'train_list.json'
    expected = next(x['lfs']['sha256'] for x in info['siblings'] if x['rfilename']=='train_list.json')
    if digest(annotations.read_bytes()) != expected:
        raise ValueError('Annotation SHA256 mismatch')
    data = json.loads(annotations.read_text())
    groups = {}
    for row in data:
        if not row['total_score'] or not row['element_score']:
            continue
        if any(type(x) is not int or not 1 <= x <= 5 for x in row['total_score']):
            raise ValueError('Unexpected overall human rating scale')
        groups.setdefault(prompt_key(row['prompt']), []).append(row)
    # Label-independent order; no sampling by model output, score or agreement.
    keys = sorted((k for k,v in groups.items() if len({r['img_path'] for r in v})>=2),
                  key=lambda k:digest(('evalmuse-pilot-v1:'+k).encode()))[:32]
    selected = []
    for index, key in enumerate(keys):
        unique = {r['img_path']:r for r in groups[key]}
        images = sorted(unique.values(), key=lambda r:digest(r['img_path'].encode()))[:2]
        for row in images:
            selected.append((row, 'development' if index<16 else 'heldout', digest(key.encode())))
    if len(selected) != 64:
        raise ValueError('Unexpected sample size')
    out = ROOT/'data/processed/evalmuse-pilot-v1'
    out.mkdir(parents=True, exist_ok=True)
    if (out/'manifest.jsonl').exists():
        raise FileExistsError('Completed dataset exists; use its saved manifest')
    parts = sorted((x for x in info['siblings'] if x['rfilename'].startswith('images.zip.part-')),
                   key=lambda x:x['rfilename'])
    remote = RangeZip(parts, 'https://huggingface.co/datasets/DY-Evalab/EvalMuse/resolve/'+info['sha'],
                      ROOT/'data/cache/evalmuse-ranges'/info['sha'])
    records = []
    with zipfile.ZipFile(remote) as archive:
        members = archive.namelist()
        (raw/'archive-members.json').write_text(json.dumps(members))
        print('ZIP_INDEX_READY', len(members), flush=True)
        for row, split, group in selected:
            suffix = row['img_path']
            candidates = [name for name in members if name==suffix or name.endswith('/'+suffix)]
            if len(candidates) != 1:
                raise ValueError('Missing or ambiguous image member: '+suffix)
            identifier = digest(suffix.encode())[:20]
            destination = out/(identifier+'.png')
            if not destination.exists():
                content = archive.read(candidates[0])  # zipfile verifies member CRC32.
                with Image.open(io.BytesIO(content)) as image:
                    image.verify()
                destination.write_bytes(content)
            record = {'id':identifier, 'image':str(destination.relative_to(ROOT)),
                'image_sha256':digest(destination.read_bytes()), 'instruction':row['prompt'],
                'prompt_group':group, 'prompt_id':row['prompt_id'], 'split':split,
                'generator':suffix.split('/')[0], 'elements':list(row['element_score']),
                'human_overall':row['total_score'], 'human_elements':row['element_score'],
                'annotation_flags':{k:row.get(k) for k in ['type','promt_meaningless','split_confidence','attribute_confidence']},
                'source_image':suffix, 'source_member':candidates[0], 'source_revision':info['sha']}
            records.append(record)
            print('IMAGE_READY',len(records),split,identifier,flush=True)
    (out/'manifest.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
    (out/'dataset-record.json').write_text(json.dumps({'source':'DY-Evalab/EvalMuse',
        'revision':info['sha'], 'annotation_sha256':expected, 'source_split':'public train_list.json',
        'sample_rule':'32 canonical-prompt groups ordered by seeded SHA256; 2 image paths ordered by SHA256 each',
        'development_prompts':16, 'heldout_prompts':16, 'images':64,
        'label_based_filter':False, 'official_test':False,
        'integrity':'Pinned HTTPS + validated HTTP ranges + member CRC32 + extracted image SHA256; full archive SHA256 not checked',
        'downloaded_range_bytes_this_invocation':remote.downloaded_bytes},indent=2))
    print('EVALMUSE_READY',len(records),flush=True)

if __name__ == '__main__':
    main()
