import subprocess,sys
print(subprocess.run(['nvidia-smi','--query-gpu=name,compute_cap,driver_version','--format=csv,noheader'],capture_output=True,text=True).stdout, flush=True)
import torch
print('torch',torch.__version__,'| cuda_build',torch.version.cuda,'| device_count',torch.cuda.device_count(), flush=True)
try:
    cap=torch.cuda.get_device_capability(); print('device capability sm_%d%d'%cap, flush=True)
    print('arch_list', torch.cuda.get_arch_list(), flush=True)
except Exception as e: print('cap err',e, flush=True)
try:
    x=torch.zeros(4).cuda(); y=(x+1).sum().item(); print('CUDA TENSOR OK ->',y, flush=True)
except Exception as e: print('CUDA FAIL:', str(e)[:300], flush=True)
