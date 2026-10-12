"""Saved VS Code CUDA preflight; does not claim model-quality evaluation."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from run_record import snapshot_run
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; select the CUDA virtual environment in VS Code')
    out=snapshot_run(args.output,{'scope':'CUDA arithmetic and checkpoint compatibility only','device':'cuda'})
    torch.cuda.reset_peak_memory_stats()
    x=torch.arange(16,device='cuda',dtype=torch.float32).reshape(4,4)
    y=x@x.T;torch.cuda.synchronize()
    torch.testing.assert_close(y.cpu(),x.cpu()@x.cpu().T)
    torch.save({'tensor':y.cpu()},out/'cuda-tensor.pt')
    torch.testing.assert_close(torch.load(out/'cuda-tensor.pt',weights_only=True)['tensor'].to('cuda'),y)
    report={'passed':True,'device':'cuda','gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'torch_cuda':torch.version.cuda,'bf16_supported':torch.cuda.is_bf16_supported(),'peak_allocated_bytes':torch.cuda.max_memory_allocated()}
    (out/'metrics.json').write_text(json.dumps(report,indent=2));print('CUDA_PREFLIGHT_PASSED',json.dumps(report),flush=True)
