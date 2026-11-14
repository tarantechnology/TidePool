import { useEffect, useRef } from 'react'
import type { CSSProperties } from 'react'

export function MaskOverlay({mask,style,color=[74,222,128]}:{mask:unknown;style:CSSProperties;color?:[number,number,number]}) {
  const canvas=useRef<HTMLCanvasElement>(null)
  const [r,g,b]=color
  useEffect(()=>{
    const data=mask as {encoding?:string;order?:string;size?:number[];counts?:number[]}
    const el=canvas.current
    if(!el || data?.encoding!=='rle' || data.order!=='C' || !data.size || !data.counts)return
    const [height,width]=data.size
    if(!Number.isInteger(height)||!Number.isInteger(width)||height<=0||width<=0||height*width>16_000_000)return
    if(data.counts.some(n=>!Number.isInteger(n)||n<0)||data.counts.reduce((a,b)=>a+b,0)!==height*width)return
    el.width=width;el.height=height
    const ctx=el.getContext('2d');if(!ctx)return
    const image=ctx.createImageData(width,height);let index=0
    data.counts.forEach((count,run)=>{if(run%2)for(let i=index;i<index+count;i++){image.data[i*4]=r;image.data[i*4+1]=g;image.data[i*4+2]=b;image.data[i*4+3]=115}index+=count})
    ctx.putImageData(image,0,0)
  },[mask,r,g,b])
  return <canvas ref={canvas} aria-label="Selected object mask" className="pointer-events-none absolute" style={style}/>
}
