"""Optional cross-repository contract test when Person 1's service is on PYTHONPATH."""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('sam_audio_service', reason='Audio service branch is not present')
pytest.importorskip('torch', reason='Audio service visual preparation needs torch')
from sam_audio_service.config import Settings
from sam_audio_service.visual_prompts import translate_visual_prompt
from sam_audio_service.visual import prepare_masked_video
from soundcanvas_cv.masks import encode_mask
from soundcanvas_cv.pipeline import process_media
from soundcanvas_cv.selection import select_at_point
from test_contract import Detector, video


@pytest.mark.skipif(not shutil.which('node') or not shutil.which('ffmpeg'), reason='Node and FFmpeg required')
def test_selected_interval_prepares_audio_service_masks(video, tmp_path):
    class Segmented(Detector):
        track = True
        def detect(self, frame):
            obj = super().detect(frame)[0]
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            mask[4:24, 4:20] = 1
            obj.mask = encode_mask(mask)
            return [obj]
    data = process_media(video, tmp_path/'out', Segmented(), 1000, 1)
    selection = select_at_point(data, .5, .1, .2)
    core = Path(__file__).resolve().parents[1]/'viewer-core.js'
    script = "const c=require(process.argv[1]),d=JSON.parse(require('fs').readFileSync(0,'utf8')); c.validateVisualInterval(d.metadata,d.selection,d.selection.interval.start_seconds,d.selection.interval.end_seconds);"
    subprocess.run(['node', '-e', script, str(core)], input=json.dumps({'metadata':data,'selection':selection}), text=True, check=True)
    settings = Settings(media_root=tmp_path.resolve())
    translated = translate_visual_prompt({'selection':selection, 'detections':data, 'video_ref':str(video)},
        source=None, source_duration=data['processed_duration_seconds'], settings=settings, backend_supports_visual=True)
    assert translated.visual is not None
    assert translated.visual.interval == (selection['interval']['start_seconds'], selection['interval']['end_seconds'])
    frames, masks = prepare_masked_video(translated.visual, size=64)
    assert frames.shape[0] == data['processed_frames']
    assert masks.shape[0] == frames.shape[0]
    assert all((m == 0).any() for m in masks)  # target foreground survives in every frame
