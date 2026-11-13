"""Official SAM 3/3.1 video predictor -> existing schema-1.0 Detection records.

One concept can identify multiple objects. Model IDs are hypotheses, never verified
physical identities. No action or acoustic labels are derived by this adapter.
"""
from contextlib import nullcontext
import gc
import math
from pathlib import Path

import cv2
import numpy as np

from .masks import encode_mask
from .pipeline import Detection, file_sha256


def _numpy(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


def extract_sam3(outputs, shape, label, threshold=.5):
    """Derive exclusive pixel boxes from native masks; reject geometry drift."""
    height, width = shape
    ids = _numpy(outputs['out_obj_ids']).reshape(-1)
    masks = _numpy(outputs['out_binary_masks'])
    scores = _numpy(outputs['out_probs']).reshape(-1)
    if len(ids) != len(masks) or len(ids) != len(scores):
        raise ValueError('SAM 3 output ID/mask/score counts disagree')
    detections = []
    seen = set()
    for oid, mask, score in zip(ids, masks, scores, strict=True):
        tid = int(oid)
        if tid < 0 or tid != oid or tid in seen:
            raise ValueError('SAM 3 output contains invalid or duplicate object IDs')
        seen.add(tid)
        # Official multiplex tracker uses -1e4 for removed tracks while retaining
        # their bookkeeping entry. This is absence, not a confidence to clamp.
        if float(score) == -1e4:
            continue
        if not math.isfinite(float(score)) or not 0 <= score <= 1:
            raise ValueError(f'SAM 3 produced invalid score {float(score)!r} for object {tid}')
        if mask.shape == (1, height, width):
            mask = mask[0]
        if mask.shape != (height, width):
            raise ValueError('SAM 3 masks must match native decoded frame dimensions')
        mask = mask.astype(bool)
        if score < threshold or not mask.any():
            continue
        yy, xx = np.nonzero(mask)
        box = [int(xx.min()), int(yy.min()), int(xx.max())+1, int(yy.max())+1]
        detections.append(Detection(
            object_id=f'track:{tid}', track_id=tid, label=label, class_id=0,
            confidence=float(score), bbox_xyxy=box,
            bbox_normalized=[box[0]/width, box[1]/height, box[2]/width, box[3]/height],
            mask=encode_mask(mask)))
    return detections


class SAM3VideoDetector:
    """Stream one native-frame result per detect(), with CPU-offloaded video/state.

    Explicit checkpoint path prevents accidental multi-GB downloads. A separate
    CLI process owns the model and exits before SAM Audio is loaded.
    """
    track = True

    def __init__(self, source, checkpoint, concept, version='sam3.1', confidence=.5,
                 max_seconds=30., prompt=None, predictor=None, grounding_batch_size=1):
        if version not in ('sam3', 'sam3.1'):
            raise ValueError('version must be sam3 or sam3.1')
        if not concept.strip() or not 0 < confidence <= 1 or not math.isfinite(max_seconds) or max_seconds <= 0:
            raise ValueError('concept, confidence and duration must be valid')
        if not Path(checkpoint).is_file():
            raise FileNotFoundError('SAM checkpoint missing; obtain approved official weights first')
        self.source, self.checkpoint = str(source), str(checkpoint)
        self.concept, self.version = concept.strip(), version
        self.confidence, self.max_seconds = confidence, max_seconds
        if type(grounding_batch_size) is not int or grounding_batch_size <= 0:
            raise ValueError('grounding_batch_size must be positive')
        self.grounding_batch_size = grounding_batch_size
        self.prompt = dict(prompt or {})
        # Only the documented prompt fields may be forwarded; no session overrides.
        if set(self.prompt) - {'frame_index', 'points', 'point_labels', 'obj_id'}:
            raise ValueError('Unsupported SAM point-prompt field')
        if self.prompt:
            if set(self.prompt) != {'frame_index', 'points', 'point_labels', 'obj_id'}:
                raise ValueError('SAM point prompt requires frame_index, points, point_labels and obj_id')
            points = np.asarray(self.prompt['points'], dtype=float)
            labels = np.asarray(self.prompt['point_labels'])
            if (type(self.prompt['frame_index']) is not int or self.prompt['frame_index'] < 0
                    or type(self.prompt['obj_id']) is not int or self.prompt['obj_id'] < 0
                    or points.ndim != 2 or points.shape[1] != 2 or len(points) == 0
                    or not np.isfinite(points).all() or np.any((points < 0) | (points > 1))
                    or labels.shape != (len(points),) or not np.isin(labels, [0, 1]).all()):
                raise ValueError('SAM point prompt must have finite normalized points and binary labels')
        self.predictor = predictor
        self.injected = predictor is not None
        self.session_id = self.stream = None
        self.index = 0
        self.torch = None
        self.peak = {}

    def reset(self):
        self.close()
        self.index = 0

    def _start(self):
        cap = cv2.VideoCapture(self.source)
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        finally:
            cap.release()
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError('SAM video adapter requires CFR video')
        if self.prompt and self.prompt['frame_index'] >= min(count, math.ceil(self.max_seconds*fps)):
            raise ValueError('SAM point prompt lies outside analyzed video')
        if not self.injected:
            import torch
            from sam3.model_builder import build_sam3_predictor
            if not torch.cuda.is_available():
                raise RuntimeError('Official SAM video inference requires CUDA')
            self.torch = torch
            torch.cuda.reset_peak_memory_stats()
            # FA3 is not assumed available on L4; compilation costs are avoided.
            self.predictor = build_sam3_predictor(
                checkpoint_path=self.checkpoint, version=self.version,
                use_fa3=False, compile=False, async_loading_frames=False)
            if self.version == 'sam3.1':
                # Official builder defaults to 16 full-resolution grounding frames
                # per batch, which OOMs on L4. Microbatching preserves model/size.
                self.predictor.model.batched_grounding_batch_size = self.grounding_batch_size
                self.predictor.model.postprocess_batch_size = 1
        response = self.predictor.handle_request(dict(
            type='start_session', resource_path=self.source,
            offload_video_to_cpu=True, offload_state_to_cpu=self.version == 'sam3'))
        self.session_id = response['session_id']
        request = dict(type='add_prompt', session_id=self.session_id, frame_index=0, text=self.concept,
                       output_prob_thresh=self.confidence)
        self.predictor.handle_request(request)
        if self.prompt:
            point_prompt = dict(self.prompt)
            if self.torch:
                point_prompt['points'] = self.torch.tensor(point_prompt['points'], dtype=self.torch.float32)
                point_prompt['point_labels'] = self.torch.tensor(point_prompt['point_labels'], dtype=self.torch.int32)
            self.predictor.handle_request(dict(type='add_prompt', session_id=self.session_id,
                                               output_prob_thresh=self.confidence, **point_prompt))
        self.stream = iter(self.predictor.handle_stream_request(dict(
            type='propagate_in_video', session_id=self.session_id, propagation_direction='forward',
            start_frame_index=0, max_frame_num_to_track=math.ceil(fps*self.max_seconds),
            output_prob_thresh=self.confidence, evict_cached_frame_outputs=True)))

    def detect(self, frame):
        if self.stream is None:
            self._start()
        context = self.torch.inference_mode() if self.torch else nullcontext()
        with context:
            try:
                response = next(self.stream)
            except StopIteration as exc:
                raise ValueError('SAM output ended before native video frames') from exc
            if response['frame_index'] != self.index:
                raise ValueError('SAM output missing/reordered native frames; refusing shifted masks')
            self.index += 1
            return extract_sam3(response['outputs'], frame.shape[:2], self.concept, self.confidence)

    def describe(self):
        if self.torch:
            self.torch.cuda.synchronize()
            self.peak = dict(cuda_peak_allocated_bytes=self.torch.cuda.max_memory_allocated(),
                             cuda_peak_reserved_bytes=self.torch.cuda.max_memory_reserved())
        return dict(name='meta-sam3', model=self.version, concept=self.concept,
                    class_id_semantics='0 = prompted concept; not a COCO class ID',
                    weights_sha256=file_sha256(self.checkpoint), tracking=True,
                    grounding_batch_size=self.grounding_batch_size,
                    tracking_cadence='every decoded frame', device='cuda',
                    confidence_threshold=self.confidence, **self.peak)

    def close(self):
        try:
            if self.stream is not None and hasattr(self.stream, 'close'):
                self.stream.close()
            if self.session_id is not None:
                self.predictor.handle_request(dict(type='close_session', session_id=self.session_id))
        finally:
            self.session_id = self.stream = None
            if self.torch:
                self.torch.cuda.synchronize()
                self.peak = dict(cuda_peak_allocated_bytes=self.torch.cuda.max_memory_allocated(),
                                 cuda_peak_reserved_bytes=self.torch.cuda.max_memory_reserved())
            if self.predictor is not None and not self.injected:
                self.predictor.shutdown()
                self.predictor = None
                gc.collect()
                self.torch.cuda.empty_cache()
