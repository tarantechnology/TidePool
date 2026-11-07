"""CV-only preparation for SAM Audio, NOT an audio separation implementation.

Verified source: facebookresearch/sam-audio@bb4c6999d2677c7402360e426afc01ddfad6dce0
sam_audio/processor.py. mask_videos multiplies video by mask.eq(0).
"""
from dataclasses import dataclass
import math
import cv2
import numpy as np
from .masks import decode_mask
from .pipeline import file_sha256


@dataclass
class PreparedVisual:
    frames_rgb: np.ndarray  # N,C,H,W uint8
    foreground_masks: np.ndarray  # N,1,H,W uint8, 1 keeps target
    fps: float
    start_seconds: float
    end_seconds: float
    mask_kind: str


def prepare_visual_inputs(video_path, metadata, selection, *, allow_box_masks=False):
    """Decode exact selected interval with one matching mask per native CFR frame.

    For model masks, run segmentation with --sample-fps >= native FPS. Missing track
    observations get empty foreground, never an invented mask. Sparse metadata is
    rejected. Box masks are explicit coarse experiments and disabled by default.
    Audio must be cropped to the same interval separately by Person 1.
    """
    if metadata['media_type'] != 'video':
        raise ValueError('Visual audio preparation requires video')
    if selection['analysis_id'] != metadata['analysis_id'] or selection['video_id'] != metadata['video_id'] or selection['source_sha256'] != metadata['source_sha256']:
        raise ValueError('Selection belongs to another video')
    if file_sha256(video_path) != metadata['source_sha256']:
        raise ValueError('Source bytes do not match metadata')
    width, height, fps = metadata['width'], metadata['height'], metadata['fps']
    if (selection['frame_width'], selection['frame_height']) != (width, height):
        raise ValueError('Selection dimensions do not match video')
    observed = next((f for f in metadata['frames'] if f['frame_index'] == selection['frame_index']), None)
    matched = next((o for o in observed['objects'] if o['detection_id'] == selection['detection_id']), None) if observed else None
    if matched is None or any(matched[k] != selection[k] for k in ('track_id', 'class_id', 'bbox_xyxy', 'bbox_normalized', 'label')) or observed['timestamp_seconds'] != selection['timestamp_seconds']:
        raise ValueError('Selection does not match its observed detection')
    start, end = selection['interval']['start_seconds'], selection['interval']['end_seconds']
    if not all(isinstance(t, (float, int)) and math.isfinite(t) for t in (start, end)) or not 0 <= start < end <= metadata['processed_duration_seconds']:
        raise ValueError('Invalid selection interval')
    if end - start < .1 - 1e-8 or end - start > 30 + 1e-8:
        raise ValueError('Visual interval must be between 0.1 and 30 seconds')
    if not start <= selection['timestamp_seconds'] < end:
        raise ValueError('Selected observation lies outside requested interval')
    first, stop = round(start*fps), round(end*fps)
    if abs(first/fps-start) > 1e-7 or abs(stop/fps-end) > 1e-7:
        raise ValueError('Interval must align to native CFR frame boundaries')
    records = {f['frame_index']: f for f in metadata['frames']}
    cap = cv2.VideoCapture(str(video_path))
    frames, masks = [], []
    coarse = False
    try:
        if not cap.isOpened():
            raise ValueError('Could not decode source video')
        # Sequential decode gives deterministic frame indices across codecs.
        for index in range(stop):
            ok, frame = cap.read()
            if not ok:
                raise ValueError('Source ended before requested interval')
            if index < first:
                continue
            if index not in records:
                raise ValueError('Sparse metadata: rerun segmentation at native FPS for aligned masks')
            if frame.shape[:2] != (height, width):
                raise ValueError('Decoded dimensions differ from metadata')
            candidates = [o for o in records[index]['objects'] if (
                o['track_id'] == selection['track_id'] if selection['track_id'] is not None
                else o['detection_id'] == selection['detection_id'])]
            if len(candidates) > 1:
                raise ValueError('Ambiguous track observation')
            foreground = np.zeros((height, width), dtype=np.uint8)
            if candidates:
                obj = candidates[0]
                if obj.get('mask') is not None:
                    foreground = decode_mask(obj['mask'])
                    if foreground.shape != (height, width):
                        raise ValueError('Mask dimensions differ from source')
                elif allow_box_masks:
                    x1, y1, x2, y2 = obj['bbox_xyxy']
                    foreground[y1:y2, x1:x2] = 1
                    coarse = True
                else:
                    raise ValueError('Instance mask required; bbox alone is not a SAM Audio visual prompt')
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).transpose(2, 0, 1))
            masks.append(foreground[None])
    finally:
        cap.release()
    if not masks or not any(mask.any() for mask in masks):
        raise ValueError('No foreground for selected object in interval')
    return PreparedVisual(np.stack(frames), np.stack(masks), fps, start, end,
                          'coarse_bbox_experiment' if coarse else 'instance_segmentation')


def build_processor_batch(prepared, audio_48khz, processor):
    """Call an injected official processor. No model load, separation or stem here.

    audio_48khz: torch tensor C,T already cropped/resampled to 48kHz by audio service.
    Frames cover the same full interval; official tensor load_video uniformly maps
    these frames to audio features. Our foreground=1 -> SAM removal mask=0.
    """
    import torch
    if audio_48khz.ndim != 2 or audio_48khz.shape[0] not in (1, 2):
        raise ValueError('Expected 48kHz audio tensor shaped channels,time')
    if processor.audio_sampling_rate != 48000:
        raise ValueError('Adapter verified for 48kHz SAM Audio processor only')
    expected = round((prepared.end_seconds-prepared.start_seconds)*48000)
    if audio_48khz.shape[1] != expected:
        raise ValueError('Audio must be cropped to exactly the visual interval at 48kHz')
    frames = torch.from_numpy(prepared.frames_rgb.copy())
    removal_masks = 1 - torch.from_numpy(prepared.foreground_masks.copy())
    masked = processor.mask_videos([frames], [removal_masks])
    return processor(audios=[audio_48khz], descriptions=[''], masked_videos=masked)


def mocked_handoff(selection):
    """JSON-only boundary demonstration. Never creates a stem or claims inference."""
    return {'mock': True, 'inference_performed': False, 'stems': [],
            'status': 'awaiting_real_audio_service', 'visual_prompt': selection,
            'requirements': ['original video resolver', 'aligned instance masks and 48kHz audio clip',
                             'Person 1 SAM Audio model/revision and inference service']}
