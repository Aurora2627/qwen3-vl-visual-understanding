"""Download official Qwen3-VL weights with pinned revision and SHA256 checks."""
import concurrent.futures,hashlib,json,re,subprocess,time
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from platform_tools import curl_executable
ROOT=Path(__file__).resolve().parents[1]
REPO='Qwen/Qwen3-VL-2B-Instruct'
def curl(url,path,extra=()):
    subprocess.run([curl_executable(),'-f','-L','--http1.1','--connect-timeout','20','--max-time','120','--retry','2','--retry-delay','2',*extra,url,'-o',str(path)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=True)
def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()
if __name__=='__main__':
    cache=ROOT/'data/cache/local-qwen3-vl';cache.mkdir(parents=True,exist_ok=True)
    metadata=cache/'model-info.json';curl('https://huggingface.co/api/models/'+REPO+'?blobs=true',metadata)
    info=json.loads(metadata.read_text());revision=info['sha'];directory=cache/revision;directory.mkdir(exist_ok=True)
    selected=[f for f in info['siblings'] if f['rfilename'].endswith(('.json','.safetensors','.txt','.model','.jinja')) and '/' not in f['rfilename']]
    files=[]
    for item in selected:
        name=item['rfilename'];target=directory/name;url='https://huggingface.co/'+REPO+'/resolve/'+revision+'/'+name
        if name.endswith('.safetensors'):
            size=item['lfs']['size'];expected=item['lfs']['sha256'];parts=directory/(name+'.parts');parts.mkdir(exist_ok=True);chunk=16*1024*1024
            if target.exists() and target.stat().st_size==size and sha(target)==expected:
                files.append({'file':name,'size':size,'sha256':expected});continue
            def download(index):
                start=index*chunk;end=min(size-1,start+chunk-1);part=parts/('%05d'%index);header=parts/('%05d.headers'%index)
                if part.exists() and part.stat().st_size==end-start+1:return part
                for attempt in range(3):
                    temporary=parts/('%05d.tmp'%index)
                    try:
                        curl(url+'?download=true&segment='+str(index),temporary,['-H','Range: bytes=%d-%d'%(start,end),'-D',str(header)])
                        ranges=re.findall(r'content-range:\s*bytes\s+(\d+)-(\d+)/(\d+)',header.read_text().lower())
                        if not ranges or tuple(map(int,ranges[-1]))!=(start,end,size) or temporary.stat().st_size!=end-start+1:raise RuntimeError('Invalid HTTP range')
                        temporary.replace(part);return part
                    except (subprocess.CalledProcessError,RuntimeError):
                        if attempt==2:raise
                raise RuntimeError('Download failed')
            count=(size+chunk-1)//chunk
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                completed=0
                for part in pool.map(download,range(count)):
                    completed+=1
                    print('WEIGHT_PART',completed,'/',count,flush=True)
            temporary=directory/(name+'.incomplete')
            with temporary.open('wb') as out:
                for index in range(count):
                    with (parts/('%05d'%index)).open('rb') as source:
                        for block in iter(lambda:source.read(8*1024*1024),b''):out.write(block)
            if sha(temporary)!=expected:raise RuntimeError('Weight SHA256 mismatch')
            temporary.replace(target)
        else:
            if not target.exists():curl(url,target)
        files.append({'file':name,'size':target.stat().st_size,'sha256':sha(target)})
    report={'model_id':REPO,'revision':revision,'local_directory':str(directory.relative_to(ROOT)),'files':files,'status':'downloaded-sha256-verified','source':'official Qwen model repository'}
    (ROOT/'reports/local-model.json').write_text(json.dumps(report,indent=2));print('LOCAL_MODEL_READY',revision,flush=True)
