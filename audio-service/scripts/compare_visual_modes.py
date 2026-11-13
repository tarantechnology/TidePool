#!/usr/bin/env python3
"""Real SAM Audio A/B with identical source, model, seed and full-length outputs.

Run after vision has exited. No mock backend, automatic downsize, automatic model
switch, listening score, or unsupported accuracy claim. A fresh output directory
and explicit model choice are mandatory. Checkpoint access is checked before load.
"""
import argparse
import json
import re
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sam_audio_service.audio_io import decode, encode_wav_float32, to_mono
from sam_audio_service.backends.base import BackendOptions, BackendPrompt, VisualInput
from sam_audio_service.backends.sam_audio import SamAudioBackend
from sam_audio_service.visual import decode_rle_mask


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video', type=Path, required=True)
    p.add_argument('--detections', type=Path, required=True)
    p.add_argument('--selection', type=Path, required=True)
    p.add_argument('--prompt', required=True)
    p.add_argument('--anchors', type=Path, help='Experimental anchors [{token,start,end}] on source timeline')
    p.add_argument('--anchor-provenance', choices=['unverified_candidate', 'reviewed'], default='unverified_candidate')
    p.add_argument('--model-id', required=True, choices=['facebook/sam-audio-large-tv', 'facebook/sam-audio-base-tv', 'facebook/sam-audio-small-tv'])
    p.add_argument('--dtype', choices=['float32', 'bfloat16'], default='float32')
    p.add_argument('--mode', choices=['all', 'text', 'visual', 'text_visual', 'combined'], default='all')
    p.add_argument('--mask-mode', choices=['keep_object', 'notebook'], default='keep_object')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--visual-batch-size', type=int, default=8)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('Output exists; use a fresh experiment directory')
    if args.mode in ('all', 'combined') and not args.anchors:
        p.error('Combined comparison needs explicit temporal hypotheses in --anchors')
    # Reuse vision identity/coverage/frame geometry validation before touching a model.
    from soundcanvas_cv.audio_adapter import prepare_visual_inputs
    metadata, selection = json.loads(args.detections.read_text()), json.loads(args.selection.read_text())
    if selection.get('track_id') is None:
        p.error('Visual comparison requires a tracked object identity')
    # Validate without retaining native-sized RGB/mask tensors throughout model loading.
    prepared = prepare_visual_inputs(args.video, metadata, selection)
    start, end = prepared.start_seconds, prepared.end_seconds
    del prepared
    audio = decode(path=args.video, max_channels=2)
    mixture = to_mono(audio.samples)
    first, stop = round(start*48000), round(end*48000)
    if stop > len(mixture):
        p.error('Visual interval exceeds decoded audio')
    window = mixture[first:stop]
    anchors = json.loads(args.anchors.read_text()) if args.anchors else []
    relative_anchors = []
    for a in anchors:
        if a['token'] not in ('+', '-') or not start <= a['start'] < a['end'] <= end:
            p.error('Anchors must fit selected visual interval')
        relative_anchors.append((a['token'], a['start']-start, a['end']-start))
    masks = {f['frame_index']: decode_rle_mask(o['mask']) for f in metadata['frames']
             if start-1e-8 <= f['timestamp_seconds'] < end-1e-8
             for o in f['objects'] if o['track_id'] == selection['track_id']}
    visual = VisualInput(video_path=str(args.video.resolve()), interval=(start, end),
                         frame_masks=masks, frame_rate=metadata['fps'],
                         source_sha256=metadata['source_sha256'], object_id=selection['object_id'],
                         label=None)  # Blank text stays genuinely visual-only in that arm.
    modes = ['text', 'visual', 'text_visual', 'combined'] if args.mode == 'all' else [args.mode]
    from soundcanvas_cv.gpu_guard import acquire_gpu
    gpu_lease = acquire_gpu()
    args.output.mkdir(parents=True)
    report = dict(model_id=args.model_id, dtype=args.dtype, source_sha256=metadata['source_sha256'],
                  analysis_id=metadata['analysis_id'], object_id=selection['object_id'], seed=args.seed,
                  interval=[start, end], mask_mode=args.mask_mode, is_mock=False, results=[],
                  temporal_anchor_provenance=args.anchor_provenance,
                  inference_verified=False,
                  listening_evaluation='not performed', objective_separation_metrics='unavailable without reference stems')
    backend = SamAudioBackend(model_id=args.model_id, device='cuda', dtype=args.dtype,
                              load_span_predictor=False, load_rankers=False, mask_mode=args.mask_mode,
                              visual_batch_size=args.visual_batch_size)
    try:
        # Metadata request only: no weight download when official access is missing.
        from huggingface_hub import hf_hub_url, get_hf_file_metadata
        get_hf_file_metadata(hf_hub_url(args.model_id, 'config.json'))
        started = time.perf_counter()
        backend.load()
        report['load_seconds'] = time.perf_counter()-started
        report['model'] = backend.info()
        for mode in modes:
            import torch
            torch.cuda.reset_peak_memory_stats()
            prompt = BackendPrompt(text='' if mode == 'visual' else args.prompt.lower(),
                                   visual=None if mode == 'text' else visual,
                                   anchors=relative_anchors if mode == 'combined' else None)
            started = time.perf_counter()
            result = backend.separate(window, prompt, BackendOptions(seed=args.seed))
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-started
            target = np.zeros_like(mixture)
            target[first:stop] = result.target
            residual = mixture-target
            # Keep source stereo by subtracting the same mono target from each channel,
            # matching the service's duplicate strategy, not duplicating the residual.
            target_channels = np.broadcast_to(target, audio.samples.shape).copy()
            residual_channels = audio.samples-target_channels
            for name, samples in [('target', target_channels), ('residual', residual_channels)]:
                (args.output/f'{mode}-{name}.wav').write_bytes(encode_wav_float32(samples, 48000))
            subprocess.run(['ffmpeg', '-v', 'error', '-n', '-i', str(args.video), '-i', str(args.output/f'{mode}-residual.wav'),
                            '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'copy', '-c:a', 'aac',
                            str(args.output/f'{mode}-edited.mp4')], check=True)
            facts = dict(mode=mode, elapsed_seconds=elapsed, prompt=prompt.text,
                         temporal_anchors=prompt.anchors, cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                         cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(), sample_count=len(target),
                         target_rms=float(np.sqrt(np.mean(target**2))),
                         reconstruction_max_abs=float(np.max(np.abs(target_channels+residual_channels-audio.samples))),
                         extra=result.extra)
            facts['near_silent_target'] = facts['target_rms'] < 1e-5
            facts['export_probe'] = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams', '-show_format',
                                                                       '-of', 'json', str(args.output/f'{mode}-edited.mp4')]))
            report['results'].append(facts)
            report['inference_verified'] = True
            write_json(args.output/'report.json', report)
    except Exception as exc:
        # Retain a useful cause while redacting credentials and resource URLs.
        summary = str(exc)
        try:
            from huggingface_hub import get_token
            token = get_token()
        except ImportError:
            token = None
        if token:
            summary = summary.replace(token, '[redacted]')
        summary = re.sub(r'https?://\S+', '<model resource URL>', summary)
        summary = re.sub(r'hf_[A-Za-z0-9]+', '[redacted]', summary)[:1200]
        report['failure'] = dict(type=type(exc).__name__, summary=summary,
                                 stage='inference' if backend.loaded else 'model_access_or_load')
        if 'torch' in sys.modules:
            torch = sys.modules['torch']
            if torch.cuda.is_available():
                report['failure']['cuda_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
                report['failure']['cuda_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        write_json(args.output/'report.json', report)
        raise SystemExit(f"Experiment failed: {type(exc).__name__}: {summary}; see report.json") from None
    finally:
        backend.unload()
    print(args.output/'report.json')


if __name__ == '__main__':
    main()
