"""Explicit public SAM 2.1 alternative: user-seeded multi-object video masks.

SAM 2 does not discover objects from text. Labels and seed points are supplied by
the operator; propagation can drift and must be reviewed before audio attribution.
"""
import gc
from pathlib import Path

import cv2
import numpy as np

from .pipeline import file_sha256
from .sam3_adapter import extract_sam3


class SAM2VideoDetector:
    track = True

    def __init__(self, source, checkpoint, seeds, max_seconds=30.,
                 config='configs/sam2.1/sam2.1_hiera_l.yaml'):
        if not Path(checkpoint).is_file():
            raise FileNotFoundError('SAM 2.1 checkpoint does not exist')
        if not seeds or len({s['object_id'] for s in seeds}) != len(seeds):
            raise ValueError('SAM 2 requires uniquely identified user seeds')
        for seed in seeds:
            points = np.asarray(seed['points'], dtype=float)
            labels = np.asarray(seed['point_labels'])
            if (type(seed['object_id']) is not int or seed['object_id'] < 0 or not seed['label'].strip()
                    or type(seed['frame_index']) is not int or seed['frame_index'] < 0
                    or points.ndim != 2 or points.shape[1] != 2 or len(points) == 0
                    or not np.isfinite(points).all() or np.any((points < 0) | (points > 1))
                    or labels.shape != (len(points),) or not np.isin(labels, [0, 1]).all()):
                raise ValueError('Invalid SAM 2 seed: use native frame indices and normalized points')
        self.source, self.checkpoint = str(source), str(checkpoint)
        self.seeds, self.max_seconds, self.config = seeds, max_seconds, config
        self.labels = {s['object_id']: s['label'] for s in seeds}
        self.predictor = self.state = self.stream = self.torch = None
        self.index = 0

    def reset(self):
        self.close()
        self.index = 0

    def _start(self):
        import torch
        from sam2.build_sam import build_sam2_video_predictor
        if not torch.cuda.is_available():
            raise RuntimeError('SAM 2 experiment requires CUDA')
        self.torch = torch
        cap = cv2.VideoCapture(self.source)
        try:
            self.width, self.height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        finally:
            cap.release()
        if not fps > 0 or any(s['frame_index'] >= min(count, self.max_seconds*fps) for s in self.seeds):
            raise ValueError('SAM 2 seed lies outside analyzed video')
        self.predictor = build_sam2_video_predictor(self.config, self.checkpoint, device='cuda')
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            self.state = self.predictor.init_state(self.source, offload_video_to_cpu=True,
                                                   offload_state_to_cpu=True, async_loading_frames=False)
            for seed in self.seeds:
                points = np.asarray(seed['points'], np.float32)*[self.width, self.height]
                self.predictor.add_new_points_or_box(self.state, frame_idx=seed['frame_index'],
                                                     obj_id=seed['object_id'], points=points,
                                                     labels=np.asarray(seed['point_labels'], np.int32))
            self.stream = iter(self.predictor.propagate_in_video(self.state, start_frame_idx=0,
                                                                max_frame_num_to_track=int(np.ceil(fps*self.max_seconds))))

    def detect(self, frame):
        if self.stream is None:
            self._start()
        with self.torch.inference_mode(), self.torch.autocast('cuda', dtype=self.torch.bfloat16):
            try:
                index, ids, logits = next(self.stream)
            except StopIteration as exc:
                raise ValueError('SAM 2 ended before source frames') from exc
            if index != self.index:
                raise ValueError('SAM 2 frame order mismatch; refusing shifted masks')
            self.index += 1
            result = []
            for oid, logit in zip(ids, logits, strict=True):
                mask = (logit > 0).cpu().numpy()
                score = float(logit[logit > 0].sigmoid().mean()) if mask.any() else 0.
                result.extend(extract_sam3(dict(out_obj_ids=[oid], out_binary_masks=mask[None],
                                               out_probs=[score]), frame.shape[:2], self.labels[oid], 0.))
            return result

    def describe(self):
        return dict(name='meta-sam2', model='sam2.1-hiera-large', config=self.config,
                    weights_sha256=file_sha256(self.checkpoint), tracking=True, device='cuda',
                    tracking_cadence='every decoded frame', seed_provenance=self.seeds,
                    confidence_semantics='mean foreground sigmoid; not class or physical-identity confidence',
                    label_provenance='user seed; no automatic object detection')

    def close(self):
        try:
            if self.stream is not None:
                self.stream.close()
            if self.state is not None:
                self.predictor.reset_state(self.state)
        finally:
            self.state = self.stream = self.predictor = None
            gc.collect()
            if self.torch:
                self.torch.cuda.empty_cache()
