"""Run the optional FlexSED detector in an isolated process, with local checkpoints."""
import json
import os
import sys
import time


def main():
    import torch
    from api import FlexSED
    data=json.load(sys.stdin);start=time.monotonic()
    if torch.cuda.is_available() and torch.cuda.mem_get_info()[0] < 3*1024**3:
        raise RuntimeError('Not enough GPU memory for optional sound detection.')
    model=FlexSED(ckpt_path=data['checkpoint'],device='cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():result=model.run_inference(data['audio'],[data['query']])
    scores=result.squeeze().detach().cpu().numpy().reshape(-1)
    print('SOUNDCANVAS_FLEXSED='+json.dumps({'scores':scores.tolist(),'fps':25,'seconds':round(time.monotonic()-start,2),'peak_gpu_bytes':torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0}))

if __name__=='__main__':main()
