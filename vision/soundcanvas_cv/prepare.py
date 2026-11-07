"""Prepare aligned CV tensors, without running audio inference.

python -m soundcanvas_cv.prepare --video clip.mp4 --metadata detections.json \
    --selection soundcanvas_selection.json --output outputs/visual_prompt.npz
"""
import argparse
import json
from pathlib import Path
import numpy as np
from .audio_adapter import prepare_visual_inputs, mocked_handoff


def main(argv=None):
    p=argparse.ArgumentParser(description='Prepare SAM Audio visual tensors (no audio inference)')
    p.add_argument('--video',required=True);p.add_argument('--metadata',required=True)
    p.add_argument('--selection',required=True);p.add_argument('--output',required=True)
    p.add_argument('--allow-box-masks',action='store_true',help='Coarse rectangular experiment; not an instance mask')
    a=p.parse_args(argv)
    try:
        data=json.loads(Path(a.metadata).read_text());selection=json.loads(Path(a.selection).read_text())
        prepared=prepare_visual_inputs(a.video,data,selection,allow_box_masks=a.allow_box_masks)
    except (ValueError,KeyError,FileNotFoundError) as exc:
        p.exit(2,f'Visual preparation failed: {exc}\n')
    output=Path(a.output)
    if output.suffix.lower()!='.npz':p.error('--output must end in .npz')
    output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,frames_rgb=prepared.frames_rgb,foreground_masks=prepared.foreground_masks,
                        fps=prepared.fps,start_seconds=prepared.start_seconds,end_seconds=prepared.end_seconds,
                        mask_kind=prepared.mask_kind)
    output.with_suffix('.mock.json').write_text(json.dumps(mocked_handoff(selection),indent=2))
    print(f'Prepared {prepared.frames_rgb.shape[0]} frames; masks: {prepared.mask_kind}. No audio inference performed.')


if __name__=='__main__':main()
