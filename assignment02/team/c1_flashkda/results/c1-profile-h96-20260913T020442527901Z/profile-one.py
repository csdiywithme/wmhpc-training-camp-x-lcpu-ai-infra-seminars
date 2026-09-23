import sys; sys.path.insert(0, '/opt')
from run_experiments import inputs
import torch, flash_kda
a,k=inputs(96)
flash_kda.fwd(*a,**k)
torch.cuda.synchronize()
