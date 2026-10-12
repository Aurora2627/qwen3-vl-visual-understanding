"""Inspect official EvalMuse metadata before selecting or downloading image data."""
import json
import subprocess
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from platform_tools import curl_executable

ROOT = Path(__file__).resolve().parents[1]
REPO = 'DY-Evalab/EvalMuse'

def fetch(url, destination):
    subprocess.run([curl_executable(), '-fL', '--http1.1', '--connect-timeout', '20',
                    '--max-time', '180', '--retry', '3', '--retry-all-errors', url, '-o', str(destination)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def main():
    out = ROOT/'data/raw/evalmuse'
    out.mkdir(parents=True, exist_ok=True)
    metadata = out/'repository.json'
    fetch('https://huggingface.co/api/datasets/'+REPO+'?blobs=true', metadata)
    info = json.loads(metadata.read_text())
    print('REVISION', info['sha'], flush=True)
    for item in info['siblings']:
        print(item, flush=True)
        if item['rfilename'].endswith(('.json', '.md')) and '/' not in item['rfilename']:
            if (out/item['rfilename']).exists() and (out/item['rfilename']).stat().st_size == item['size']:
                continue
            fetch('https://huggingface.co/datasets/'+REPO+'/resolve/'+info['sha']+'/'+item['rfilename'],
                  out/item['rfilename'])

if __name__ == '__main__':
    main()
