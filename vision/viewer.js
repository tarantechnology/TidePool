(() => {
  'use strict';
  const $=id=>document.getElementById(id), core=window.SoundCanvasCV;
  const video=$('video'),canvas=$('overlay'),ctx=canvas.getContext('2d');
  let metadata=null,selection=null,mediaUrl=null,sourceHash=null,loadId=0,animation=null;
  function clearSelection() {selection=null;$('export').disabled=true;$('intervalStart').disabled=true;$('intervalEnd').disabled=true;$('intervalStart').value='';$('intervalEnd').value='';$('intervalStatus').textContent='Select an object to choose an isolation interval.';$('selected').textContent='Nothing selected yet.';$('selectionJson').textContent='';}
  function compatible() {return metadata && sourceHash===metadata.source_sha256 &&
    video.videoWidth===metadata.width && video.videoHeight===metadata.height &&
    video.duration+0.05>=metadata.processed_duration_seconds;}
  function status() {
    $('status').textContent=compatible() ? `${metadata.analyzed_frames} samples, ${metadata.width}×${metadata.height}. Click a box; selection time is frozen. Boxes between samples are approximate.` :
      (metadata && sourceHash ? 'Video does not match metadata (SHA256, dimensions or duration). Load the original processed video.' : 'Choose the original video and schema 1.0 detections.json.');
  }
  function rect() {return core.contentRect(canvas.clientWidth,canvas.clientHeight,video.videoWidth||1,video.videoHeight||1);}
  function resize() {
    const r=video.getBoundingClientRect(),dpr=window.devicePixelRatio||1;
    canvas.width=Math.round(r.width*dpr);canvas.height=Math.round(r.height*dpr);ctx.setTransform(dpr,0,0,dpr,0,0);draw();
  }
  function draw() {
    ctx.clearRect(0,0,canvas.clientWidth,canvas.clientHeight);
    if(!compatible())return;
    const frame=core.frameAtTime(metadata,video.currentTime),r=rect();
    for(const o of frame?.objects??[]) {
      const [a,b,c,d]=o.bbox_normalized,x=r.x+a*r.w,y=r.y+b*r.h;
      const active=selection && (selection.track_id!==null ? selection.track_id===o.track_id : selection.detection_id===o.detection_id);
      ctx.strokeStyle=active?'#5d9dff':'#47ef9a';ctx.lineWidth=active?3.5:2;ctx.strokeRect(x,y,(c-a)*r.w,(d-b)*r.h);
      const text=`${o.label} ${Math.round(o.confidence*100)}%${o.track_id==null?'':' #'+o.track_id}`;
      ctx.font='bold 13px system-ui';ctx.fillStyle=active?'#346ac2':'#096944';ctx.fillRect(x,Math.max(0,y-24),ctx.measureText(text).width+12,22);
      ctx.fillStyle='#fff';ctx.fillText(text,x+6,Math.max(16,y-8));
    }
  }
  function tick(){draw();animation=!video.paused&&!video.ended?requestAnimationFrame(tick):null;}
  video.addEventListener('play',()=>{if(animation===null)tick();});
  for(const name of ['pause','seeked','timeupdate'])video.addEventListener(name,draw);
  video.addEventListener('loadedmetadata',()=>{resize();status();});
  video.addEventListener('error',()=>{$('status').textContent='Video decoding failed. Use an H.264 MP4 (see README).';});
  new ResizeObserver(resize).observe(video);
  $('videoFile').addEventListener('change',async()=>{
    const file=$('videoFile').files[0];if(!file)return;
    const current=++loadId;sourceHash=null;clearSelection();draw();
    if(mediaUrl)URL.revokeObjectURL(mediaUrl);
    mediaUrl=URL.createObjectURL(file);video.src=mediaUrl;status();
    try {
      const digest=await crypto.subtle.digest('SHA-256',await file.arrayBuffer());
      if(current!==loadId)return;
      sourceHash=Array.from(new Uint8Array(digest),b=>b.toString(16).padStart(2,'0')).join('');status();draw();
    }catch(e){$('status').textContent=`Could not verify video identity: ${e.message}`;}
  });
  $('jsonFile').addEventListener('change',async()=>{
    const file=$('jsonFile').files[0];if(!file)return;
    metadata=null;clearSelection();draw();
    try {metadata=core.validateMetadata(JSON.parse(await file.text()));status();draw();}
    catch(e){metadata=null;$('status').textContent=`Invalid JSON: ${e.message}`;}
  });
  function updateInterval() {
    if(!selection)return;
    try {
      const start=$('intervalStart').value, end=$('intervalEnd').value;
      if(start===''||end==='') throw Error('Enter both interval boundaries.');
      selection.interval=core.validateVisualInterval(metadata,selection,Number(start),Number(end));
      $('intervalStatus').textContent='Visual interval ready.';
      $('export').disabled=false;
    } catch(e) {
      $('intervalStatus').textContent=e.message;
      $('export').disabled=true;
    }
    $('selectionJson').textContent=JSON.stringify(selection,null,2);
  }
  for(const id of ['intervalStart','intervalEnd']) $(id).addEventListener('input',updateInterval);
  $('stage').addEventListener('click',event=>{
    const bounds=video.getBoundingClientRect();
    if(event.clientY>bounds.bottom-42||!compatible())return;
    const r=rect(),boundsCanvas=canvas.getBoundingClientRect();
    const x=(event.clientX-boundsCanvas.left-r.x)/r.w,y=(event.clientY-boundsCanvas.top-r.y)/r.h;
    selection=core.selectAtPoint(metadata,video.currentTime,x,y);
    if(selection)video.pause();
    if(selection) {
      $('intervalStart').disabled=false;$('intervalEnd').disabled=false;
      $('intervalStart').value=selection.interval.start_seconds;$('intervalEnd').value=selection.interval.end_seconds;
      updateInterval();
    } else clearSelection();
    $('selected').textContent=selection?`Selected ${selection.label} (${selection.object_id}), observed at ${selection.timestamp_seconds.toFixed(3)}s`:'No detected object at this point.';
    $('selectionJson').textContent=selection?JSON.stringify(selection,null,2):'';draw();
  });
  $('export').addEventListener('click',()=>{
    if(!selection)return;
    const url=URL.createObjectURL(new Blob([JSON.stringify(selection,null,2)],{type:'application/json'}));
    const a=document.createElement('a');a.href=url;a.download='soundcanvas_selection.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  });
})();
