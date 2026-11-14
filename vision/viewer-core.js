/* Shared deterministic selection logic; usable from a local HTML file and Node. */
(function(root) {
  'use strict';
  function validateMetadata(data) {
    if (data.schema_version !== '1.0' || data.media_type !== 'video' ||
        typeof data.analysis_id !== 'string' || !data.analysis_id || typeof data.video_id !== 'string' || !/^[a-f0-9]{64}$/.test(data.source_sha256) ||
        !Number.isInteger(data.width) || data.width <= 0 || !Number.isInteger(data.height) || data.height <= 0 ||
        !Number.isFinite(data.processed_duration_seconds) || data.processed_duration_seconds <= 0 ||
        !Number.isFinite(data.fps) || data.fps <= 0 || !Number.isInteger(data.processed_frames) || data.processed_frames <= 0 ||
        !Array.isArray(data.frames) || !data.frames.length) throw Error('Expected SoundCanvas video schema 1.0');
    let previous = -1, previousIndex = -1;
    const ids = new Set();
    for (const f of data.frames) {
      if (!Number.isFinite(f.timestamp_seconds) || f.timestamp_seconds <= previous ||
          !Number.isInteger(f.frame_index) || f.frame_index <= previousIndex ||
          !Number.isFinite(f.valid_until_seconds) || f.valid_until_seconds <= f.timestamp_seconds ||
          f.valid_until_seconds > data.processed_duration_seconds || !Array.isArray(f.objects)) throw Error('Invalid frame interval/order');
      previous = f.timestamp_seconds; previousIndex = f.frame_index;
      for (const o of f.objects) {
        const b=o.bbox_normalized, p=o.bbox_xyxy;
        if (!Array.isArray(b) || b.length !== 4 || b.some(v=>!Number.isFinite(v)||v<0||v>1) ||
            b[0]>=b[2] || b[1]>=b[3] || !Array.isArray(p) || p.length!==4 || p.some(v=>!Number.isInteger(v)) ||
            p[0]<0 || p[1]<0 || p[0]>=p[2] || p[1]>=p[3] || p[2]>data.width || p[3]>data.height ||
            b.some((v,i)=>Math.abs(v-p[i]/(i%2 ? data.height : data.width))>0.000002) ||
            !Number.isFinite(o.confidence) || o.confidence<0 || o.confidence>1 ||
            typeof o.label!=='string' || typeof o.detection_id!=='string' || ids.has(o.detection_id) ||
            !(o.track_id===null || Number.isInteger(o.track_id))) throw Error('Invalid object');
        ids.add(o.detection_id);
      }
    }
    for (let i=0;i<data.frames.length-1;i++) {
      if(data.frames[i].valid_until_seconds!==data.frames[i+1].timestamp_seconds) throw Error('Frame intervals must be contiguous');
    }
    return data;
  }
  function frameAtTime(metadata, time) {
    if (!metadata || !Number.isFinite(time) || time<0 || time>=metadata.processed_duration_seconds) return null;
    const frames=metadata.frames;
    let lo=0, hi=frames.length;
    while(lo<hi) { const mid=(lo+hi)>>1; if(frames[mid].timestamp_seconds<=time) lo=mid+1; else hi=mid; }
    const f=frames[lo-1];
    return f && time<f.valid_until_seconds ? f : null;
  }
  function contentRect(width,height,videoWidth,videoHeight) {
    const scale=Math.min(width/videoWidth,height/videoHeight);
    const w=videoWidth*scale,h=videoHeight*scale;
    return {x:(width-w)/2,y:(height-h)/2,w,h};
  }
  function rounded(value) { return Math.round(value * 1e9) / 1e9; }
  function defaultInterval(metadata,frame) {
    const fps=metadata.fps, half=Math.max(1,Math.floor(fps+.5));
    return {start_seconds:rounded(Math.max(0,frame.frame_index-half)/fps),
      end_seconds:Math.min(metadata.processed_duration_seconds,rounded(Math.min(metadata.processed_frames,frame.frame_index+half+1)/fps))};
  }
  function validateVisualInterval(metadata, selection, start, end) {
    const fps=metadata.fps, duration=metadata.processed_duration_seconds;
    if (!Number.isFinite(start)||!Number.isFinite(end)||!Number.isFinite(fps)||fps<=0 ||
        start<0 || end>duration+1e-8 || end<=start)
      throw Error('Choose a valid interval inside the processed video.');
    if (end-start < .1-1e-8 || end-start > 30+1e-8)
      throw Error('The visual interval must be between 0.1 and 30 seconds.');
    if (start>selection.timestamp_seconds+1e-8 || end<=selection.timestamp_seconds+1e-8)
      throw Error('The interval must contain the selected observation.');
    const first=Math.round(start*fps), last=Math.round(end*fps);
    if (Math.abs(first/fps-start)>1e-7 || Math.abs(last/fps-end)>1e-7)
      throw Error('Interval boundaries must align with native video frames.');
    if (selection.track_id===null && last-first>1)
      throw Error('A tracked object is required for a multi-frame interval. Rerun with tracking.');
    const records=new Map(metadata.frames.map(f=>[f.frame_index,f]));
    let hasMask=false;
    for(let i=first;i<last;i++) {
      const frame=records.get(i);
      if(!frame) throw Error('Sparse metadata: rerun segmentation at native FPS for this interval.');
      const objects=frame.objects.filter(o=>selection.track_id!==null ? o.track_id===selection.track_id : o.detection_id===selection.detection_id);
      if(objects.length>1) throw Error('Multiple observations share the selected track ID.');
      if(objects.length && !objects[0].mask) throw Error('Instance masks are required: rerun with a segmentation model.');
      if(objects.length) hasMask=true;
    }
    if(!hasMask) throw Error('The selected object has no mask in this interval.');
    return {start_seconds:start,end_seconds:end};
  }
  function selectAtPoint(metadata,time,x,y) {
    if(x<0||x>1||y<0||y>1||!Number.isFinite(x)||!Number.isFinite(y)) return null;
    const frame=frameAtTime(metadata,time);
    if(!frame) return null;
    const hits=frame.objects.filter(o=>{const [a,b,c,d]=o.bbox_normalized;return x>=a&&x<=c&&y>=b&&y<=d;});
    const area=o=>{const [a,b,c,d]=o.bbox_normalized;return(c-a)*(d-b);};
    hits.sort((a,b)=>area(a)-area(b)||b.confidence-a.confidence);
    if(!hits.length)return null;
    return JSON.parse(JSON.stringify({schema_version:'1.0',kind:'visual_selection',
      video_id:metadata.video_id,analysis_id:metadata.analysis_id,source:metadata.source,source_sha256:metadata.source_sha256,
      frame_width:metadata.width,frame_height:metadata.height,frame_index:frame.frame_index,
      timestamp_seconds:frame.timestamp_seconds,clicked_at_seconds:time,
      interval:defaultInterval(metadata,frame),
      ...hits[0],audio_association:'unverified'}));
  }
  const api={validateMetadata,frameAtTime,contentRect,selectAtPoint,defaultInterval,validateVisualInterval};
  if(typeof module!=='undefined'&&module.exports)module.exports=api;
  else root.SoundCanvasCV=api;
})(typeof globalThis!=='undefined'?globalThis:this);
