"""Optional local-only visual scene captioner, executed in a separate process."""
from __future__ import annotations
import io
import json
import subprocess
import sys
import time
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def main():
    started=time.monotonic()
    data=json.load(sys.stdin);path=data['video'];duration=float(data['duration']);model_path=data['model']
    if not torch.cuda.is_available(): raise RuntimeError('CUDA is unavailable')
    free,_=torch.cuda.mem_get_info()
    if free<10*1024**3: raise RuntimeError('Not enough free GPU memory for visual analysis')
    torch.cuda.reset_peak_memory_stats()
    times=[min(max(0.0,duration-.1),max(0.0,duration*fraction)) for fraction in (.08,.32,.57,.82)]
    frames=[]
    for t in times:
        r=subprocess.run(['ffmpeg','-v','error','-ss',str(t),'-i',path,'-frames:v','1','-vf',"scale='min(448,iw)':-2",'-f','image2pipe','-vcodec','mjpeg','pipe:1'],capture_output=True,check=True,timeout=30)
        with Image.open(io.BytesIO(r.stdout)) as image:frames.append(image.convert('RGB'))
    processor=AutoProcessor.from_pretrained(model_path,local_files_only=True)
    model=Qwen3VLForConditionalGeneration.from_pretrained(model_path,torch_dtype=torch.float16,local_files_only=True).eval().to('cuda')
    content=[]
    for t,image in zip(times,frames):
        content.extend([{'type':'text','text':f'Frame at {t:.1f} seconds:'},{'type':'image','image':image}])
    content.append({'type':'text','text':'Describe the visible action and scene briefly, using one short sentence per frame. Do not infer or describe sounds, speech, or music.'})
    inputs=processor.apply_chat_template([{'role':'user','content':content}],add_generation_prompt=True,tokenize=True,return_dict=True,return_tensors='pt').to('cuda')
    with torch.inference_mode():output=model.generate(**inputs,max_new_tokens=150,do_sample=False)
    description=processor.batch_decode(output[:,inputs['input_ids'].shape[1]:],skip_special_tokens=True)[0].strip()[:1200]
    print(json.dumps({'description':description,'frame_times':[round(x,2) for x in times],'model':'Qwen3-VL-2B-Instruct','elapsed_seconds':round(time.monotonic()-started,2),'peak_gpu_bytes':torch.cuda.max_memory_allocated()}))

if __name__=='__main__': main()
