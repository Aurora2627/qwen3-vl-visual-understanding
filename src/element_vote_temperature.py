"""Single positive temperature fitted only on development validation soft votes."""
import math
import torch

def fit_vote_temperature(records):
 if not records:raise ValueError('Empty calibration data')
 logits=torch.tensor([r['choice_logits'] for r in records],dtype=torch.float64)
 y=torch.tensor([r['p_pass_human'] for r in records],dtype=torch.float64)
 if logits.ndim!=2 or logits.shape[1]!=2 or not torch.isfinite(logits).all() or not torch.isfinite(y).all() or ((y<0)|(y>1)).any():
  raise ValueError('Invalid calibration logits or targets')
 grid=torch.unique(torch.cat([torch.logspace(math.log10(.25),math.log10(32),1001,dtype=torch.float64),torch.tensor([1.],dtype=torch.float64)])).sort().values
 logp=torch.log_softmax(logits.unsqueeze(0)/grid[:,None,None],dim=-1)
 losses=-(y[None,:]*logp[:,:,0]+(1-y[None,:])*logp[:,:,1]).mean(dim=-1)
 best=int(losses.argmin())
 return {'temperature':grid[best].item(),'validation_unclipped_nll':losses[best].item(),'grid_points':len(grid),'bounds':[.25,32.],
  'boundary_selected':best in (0,len(grid)-1),'criterion':'development soft-vote binary NLL only'}

def apply_vote_temperature(records,temperature):
 if not math.isfinite(temperature) or temperature<=0:raise ValueError('Positive finite temperature required')
 result=[]
 for r in records:
  logits=torch.tensor(r['choice_logits'],dtype=torch.float64)
  if logits.shape!=(2,) or not torch.isfinite(logits).all():raise ValueError('Invalid logits')
  result.append(dict(r,p_pass_model=torch.softmax(logits/temperature,dim=-1)[0].item(),condition='base_image_temperature'))
 return result
