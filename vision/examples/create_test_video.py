"""Synthetic clip for testing video I/O ONLY; not proof of YOLO inference."""
from pathlib import Path
import cv2
import numpy as np
p=Path(__file__).resolve().parent/'synthetic_input.avi'
w,h,fps=320,180,10
writer=cv2.VideoWriter(str(p),cv2.VideoWriter_fourcc(*'MJPG'),fps,(w,h))
if not writer.isOpened(): raise RuntimeError('MJPG encoder unavailable')
for t in range(20):
    frame=np.full((h,w,3),(30,35,40),np.uint8)
    cv2.rectangle(frame,(30+3*t,30),(115+3*t,135),(70,150,230),-1)
    cv2.putText(frame,'Synthetic input (NOT real YOLO)',(10,165),cv2.FONT_HERSHEY_SIMPLEX,.45,(230,230,230),1)
    writer.write(frame)
writer.release()
print('Created',p)
