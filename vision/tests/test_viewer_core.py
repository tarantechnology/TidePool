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
