"""Real YOLO/segmentation validation. From vision/: python -m examples.validate_real.

--reuse validates existing outputs. Blank-frame tests are artificial disappearances.
No SAM Audio inference is performed.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import cv2
import numpy as np
from soundcanvas_cv.pipeline import process_media, file_sha256
from soundcanvas_cv.yolo_adapter import YOLODetector
from soundcanvas_cv.masks import decode_mask
from soundcanvas_cv.selection import select_at_point
from soundcanvas_cv.audio_adapter import prepare_visual_inputs, mocked_handoff

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'examples/media/pedestrians.mp4'


def validate(data, output):
    assert data['processed_frames'] == 120 and data['fps'] == 10
    assert data['source_sha256'] == file_sha256(SOURCE)
    ids = set(); masks = 0; classes = Counter(); tracks = defaultdict(list)
    for frame in data['frames']:
        assert frame['timestamp_seconds'] == frame['frame_index']/10
        for obj in frame['objects']:
            assert obj['detection_id'] not in ids; ids.add(obj['detection_id'])
            assert 0 <= obj['confidence'] <= 1
            x1,y1,x2,y2 = obj['bbox_xyxy']
            assert 0 <= x1 < x2 <= data['width'] and 0 <= y1 < y2 <= data['height']
            expected = [x1/data['width'],y1/data['height'],x2/data['width'],y2/data['height']]
            np.testing.assert_allclose(obj['bbox_normalized'], expected, atol=1e-6)
            classes[obj['label']] += 1
            if obj['track_id'] is not None: tracks[obj['track_id']].append((frame['frame_index'],obj))
            if obj['mask'] is not None:
                decoded = decode_mask(obj['mask'])
                assert decoded.shape == (data['height'],data['width']) and decoded.any()
                masks += 1
    assert classes['person'] > 0
    cap = cv2.VideoCapture(str(output/'annotated.mp4')); total = 0
    while cap.read()[0]: total += 1
    cap.release(); assert total == data['processed_frames']
    step = round(data['fps']/data['sampling_fps'])
    gaps = [tid for tid,obs in tracks.items() if any(b[0]-a[0]>step for a,b in zip(obs,obs[1:]))]
    people = {tid:obs for tid,obs in tracks.items() if obs[0][1]['label']=='person'}
    return {'runtime':data['runtime'], 'detector':data['detector'], 'processed_frames':total,
            'samples':data['analyzed_frames'], 'inference_frames':data['inference_frames'],
            'observations':dict(classes), 'mask_observations':masks,
            'unique_person_track_ids':len(people), 'tracks_with_observation_gaps':gaps,
            'person_tracks':{str(tid):{'observations':len(obs),'first_frame':obs[0][0],'last_frame':obs[-1][0]}
                             for tid,obs in people.items()}}


def controlled_tracking(model):
    cap=cv2.VideoCapture(str(SOURCE)); ok,frame=cap.read();cap.release();assert ok
    reports=[]
    for missing in [5,40]:
        model.reset()
        for _ in range(4): before=model.detect(frame)
        for _ in range(missing): empty=model.detect(np.zeros_like(frame))
        assert not empty
        for _ in range(4): after=model.detect(frame)
        before_ids={o.track_id for o in before if o.label=='person'}
        after_ids={o.track_id for o in after if o.label=='person'}
        reports.append({'experiment':'repeated real first frame with inserted black frames (artificial disappearance)',
                        'missing_frames':missing,'before_person_ids':sorted(before_ids),
                        'after_person_ids':sorted(after_ids),'retained_ids':sorted(before_ids & after_ids)})
    return reports


def prepare_demo(data):
    counts=Counter(o['track_id'] for f in data['frames'] for o in f['objects'] if o['label']=='person')
    tid=counts.most_common(1)[0][0]
    frame=next(f for f in data['frames'] if any(o['track_id']==tid for o in f['objects']))
    obj=next(o for o in frame['objects'] if o['track_id']==tid)
    x1,y1,x2,y2=obj['bbox_normalized']
    selection=select_at_point(data,frame['timestamp_seconds'],(x1+x2)/2,(y1+y2)/2)
    assert selection['track_id']==tid
    selection['interval']={'start_seconds':0.,'end_seconds':12.}
    prepared=prepare_visual_inputs(SOURCE,data,selection)
    output=ROOT/'outputs/real_seg'
    np.savez_compressed(output/'visual_prompt.npz',frames_rgb=prepared.frames_rgb,
                        foreground_masks=prepared.foreground_masks,fps=prepared.fps,
                        start_seconds=prepared.start_seconds,end_seconds=prepared.end_seconds)
    (output/'selection.json').write_text(json.dumps(selection,indent=2))
    (output/'mock_handoff.json').write_text(json.dumps(mocked_handoff(selection),indent=2))
    index=int(prepared.foreground_masks.sum(axis=(1,2,3)).argmax());rgb=prepared.frames_rgb[index].transpose(1,2,0)
    masked=rgb*prepared.foreground_masks[index].transpose(1,2,0)
    cv2.imwrite(str(output/'masked_preview.jpg'),cv2.cvtColor(masked,cv2.COLOR_RGB2BGR))
    return {'frames_shape':list(prepared.frames_rgb.shape),'masks_shape':list(prepared.foreground_masks.shape),
            'mask_kind':prepared.mask_kind,'selected_track_id':tid,'inference_performed':False}


def main():
    p=argparse.ArgumentParser();p.add_argument('--reuse',action='store_true');args=p.parse_args();report={}
    for name,weights,track,sample in [('real_yolo','yolo26n.pt',True,5),('real_detect','yolo26n.pt',False,5),('real_seg','yolo26n-seg.pt',True,10)]:
        output=ROOT/'outputs'/name
        if args.reuse and (output/'detections.json').exists(): data=json.loads((output/'detections.json').read_text())
        else: data=process_media(SOURCE,output,YOLODetector(str(ROOT/weights),device='cpu',track=track),sample,12)
        report[name]=validate(data,output)
    report['controlled_tracking']=controlled_tracking(YOLODetector(str(ROOT/'yolo26n.pt'),device='cpu'))
    report['visual_preparation']=prepare_demo(json.loads((ROOT/'outputs/real_seg/detections.json').read_text()))
    report['weight_sha256']={name:file_sha256(ROOT/name) for name in ['yolo26n.pt','yolo26n-seg.pt']}
    report['source_sha256']=file_sha256(SOURCE)
    (ROOT/'outputs/validation_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
