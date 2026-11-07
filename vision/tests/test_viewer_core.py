import json
import shutil
import subprocess
from pathlib import Path
import pytest
from soundcanvas_cv.pipeline import process_media
from soundcanvas_cv.selection import select_at_point
from test_contract import Detector, video


@pytest.mark.skipif(not shutil.which('node'), reason='Node required for JS parity tests')
def test_javascript_python_selection_parity(video, tmp_path):
    data = process_media(video, tmp_path/'out', Detector(), 5, .5)
    core = Path(__file__).resolve().parents[1]/'viewer-core.js'
    script = '''const fs=require('fs'),assert=require('assert'); const core=require(process.argv[1]);
const d=JSON.parse(fs.readFileSync(0,'utf8'));core.validateMetadata(d);
const results=[0,.1,.21,.5,1].map(t=>core.selectAtPoint(d,t,.1,.2));
assert.equal(core.frameAtTime(d,-1),null); assert.equal(core.frameAtTime(d,NaN),null);
assert.equal(core.selectAtPoint(d,0,-1,0),null);
assert.deepEqual(core.contentRect(100,100,200,100),{x:0,y:25,w:100,h:50});
const invalid=structuredClone(d);invalid.frames[0].objects[0].bbox_normalized[0]=2;
assert.throws(()=>core.validateMetadata(invalid));
console.log(JSON.stringify(results));'''
    result = subprocess.run(['node','-e',script,str(core)],input=json.dumps(data),text=True,capture_output=True,check=True)
    assert json.loads(result.stdout) == [select_at_point(data,t,.1,.2) for t in [0,.1,.21,.5,1]]

@pytest.mark.skipif(not shutil.which('node'), reason='Node required for JS validation')
def test_visual_interval_selection_and_coverage():
    core = Path(__file__).resolve().parents[1]/'viewer-core.js'
    script = r'''const assert=require('assert'), c=require(process.argv[1]);
const objects=i=>[{track_id:1,detection_id:`frame:${i}:det:0`,bbox_normalized:[0,0,1,1],confidence:1,mask:{encoding:'rle'}}];
const d={fps:30,processed_frames:90,processed_duration_seconds:3,frames:Array.from({length:90},(_,i)=>({frame_index:i,timestamp_seconds:i/30,valid_until_seconds:(i+1)/30,objects:objects(i)}))};
const selected=c.selectAtPoint(d,1,.5,.5);
assert.equal(selected.frame_index,30); assert.equal(selected.timestamp_seconds,1);
assert.deepEqual(c.validateVisualInterval(d,selected,selected.interval.start_seconds,selected.interval.end_seconds),selected.interval);
assert(selected.interval.end_seconds-selected.interval.start_seconds>=.1);
assert.throws(()=>c.validateVisualInterval(d,selected,1,1+1/30),/0.1/);
assert.throws(()=>c.validateVisualInterval(d,selected,0,.5),/contain/);
assert.throws(()=>c.validateVisualInterval(d,selected,.01,2),/align/);
const sparse=structuredClone(d);sparse.frames=sparse.frames.filter(f=>f.frame_index!==15);
assert.throws(()=>c.validateVisualInterval(sparse,selected,0,2),/Sparse/);
const noMask=structuredClone(d);noMask.frames[15].objects[0].mask=null;
assert.throws(()=>c.validateVisualInterval(noMask,selected,0,2),/Instance masks/);
const absent=structuredClone(d);absent.frames[15].objects=[];
assert.deepEqual(c.validateVisualInterval(absent,selected,0,2),{start_seconds:0,end_seconds:2});
const untracked={...selected,track_id:null};assert.throws(()=>c.validateVisualInterval(d,untracked,0,2),/tracked/);
const short={...d,processed_duration_seconds:0.066666667,processed_frames:2,frames:d.frames.slice(0,2)};
const early=c.selectAtPoint(short,0,.5,.5);assert.throws(()=>c.validateVisualInterval(short,early,early.interval.start_seconds,early.interval.end_seconds),/0.1/);
'''
    subprocess.run(['node','-e',script,str(core)],check=True)
