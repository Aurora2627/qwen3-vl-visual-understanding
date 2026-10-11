"""Immutable source/input provenance captured before model execution."""
import hashlib,json,platform,shutil,sys
from pathlib import Path
from importlib.metadata import version,PackageNotFoundError
ROOT=Path(__file__).resolve().parents[1]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def snapshot_run(output,config,manifests=()):
    out=Path(output);out.mkdir(parents=True,exist_ok=False);snapshot=out/'source_snapshot';snapshot.mkdir()
    sources={}
    for folder in ['src','scripts','tests','configs','.vscode']:
        for source in sorted((ROOT/folder).rglob('*')):
            if not source.is_file() or '__pycache__' in source.parts:continue
            relative=source.relative_to(ROOT);target=snapshot/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target);sources[str(relative)]=sha(source)
    for name in ['VERSION','README.md','PROJECT_PLAN.md','requirements.txt','VSCODE_START.md']:
        if (ROOT/name).exists():shutil.copy2(ROOT/name,snapshot/name);sources[name]=sha(ROOT/name)
    inputs={};destination=out/'input_manifests';destination.mkdir()
    for i,manifest in enumerate(manifests):
        manifest=Path(manifest);inputs[str(manifest)]=sha(manifest);shutil.copy2(manifest,destination/(str(i)+'-'+manifest.name))
    packages={}
    for name in ['torch','transformers','Pillow','numpy']:
        try:packages[name]=version(name)
        except PackageNotFoundError:packages[name]=None
    (out/'source-manifest.json').write_text(json.dumps({'source_sha256':sources,'input_sha256':inputs},indent=2))
    (out/'config.json').write_text(json.dumps(config,indent=2))
    (out/'environment.json').write_text(json.dumps({'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'packages':packages},indent=2))
    return out
