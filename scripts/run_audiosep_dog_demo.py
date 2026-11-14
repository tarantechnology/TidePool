#!/usr/bin/env python3
"""Real CUDA AudioSep demo through SoundCanvas's engine and edit/export code.

One fixed prompt, one pass, no reference-assisted processing or normalization of
individual outputs. References are used only for evaluation after inference.
"""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'audio-service'), str(ROOT/'vision')]

from app.media import export, read_wav, write_wav
from sam_audio_service.audio_io import decode
from sam_audio_service.backends.audiosep import AudioSepBackend
from sam_audio_service.config import Settings
from sam_audio_service.contracts import SeparationMetadata
from sam_audio_service.engine import Engine
from sam_audio_service.metrics import sdr, si_sdr_improvement


def json_write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def reference_metrics(estimate, mixture, sources):
    """Joint waveform projection: source gains are proxies, not perceptual scores.

    Joint fitting avoids attributing correlated speech to barking. Distorted or
    delayed remnants can evade projection, so also report aggregate speech SDR.
    """
    estimate = np.asarray(estimate, np.float64).reshape(-1)
    mixture = np.asarray(mixture, np.float64).reshape(-1)
    names = ['speaker_a', 'speaker_b', 'barking']
    matrix = np.stack([np.asarray(sources[n], np.float64).reshape(-1) for n in names], axis=1)
    if matrix.shape[0] != len(estimate) or len(mixture) != len(estimate):
        raise ValueError('All metric inputs must have identical sample counts')
    if not np.isfinite(matrix).all() or not np.isfinite(estimate).all():
        raise ValueError('Non-finite evaluation input')
    if np.linalg.matrix_rank(matrix) != 3:
        raise ValueError('Three independent, non-silent reference sources are required')
    gains = np.linalg.lstsq(matrix, estimate, rcond=None)[0]
    db = lambda gain: float(20*np.log10(max(abs(float(gain)), 1e-6)))
    speech = matrix[:, 0]+matrix[:, 1]
    return {
        'residual_source_gains': dict(zip(names, map(float, gains))),
        'speaker_a_gain_db': db(gains[0]), 'speaker_b_gain_db': db(gains[1]),
        'barking_reduction_db_projection': -db(gains[2]),
        'speech_sdr_db': float(sdr(estimate, speech)),
        'speech_si_sdri_db': float(si_sdr_improvement(estimate, speech, mixture)),
        'baseline_speech_sdr_db': float(sdr(mixture, speech)),
        'unexplained_residual_rms': float(np.sqrt(np.mean((estimate-matrix@gains)**2))),
        'caveat': 'Waveform projection misses distorted/delayed remnants. Listen to both turns and barking; no perceptual quality score is claimed.',
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--audiosep-root', type=Path, required=True)
    p.add_argument('--chunk-seconds', type=float, default=0., help='0 for one window; 10 for longer clips')
    args = p.parse_args()
    if args.output.exists():
        p.error('Use a fresh output directory')
    source = args.input_dir.resolve()
    audio = decode(path=source/'mixture.wav', max_channels=2)
    if audio.sample_rate != 48000 or not audio.frames:
        p.error('Expected a nonempty 48 kHz mixture.wav')
    if not np.isfinite(args.chunk_seconds) or args.chunk_seconds < 0:
        p.error('chunk-seconds must be finite and nonnegative')
    # Load the shared lease without importing vision's optional OpenCV package.
    guard_spec = importlib.util.spec_from_file_location('soundcanvas_gpu_guard', ROOT/'vision/soundcanvas_cv/gpu_guard.py')
    guard = importlib.util.module_from_spec(guard_spec)
    guard_spec.loader.exec_module(guard)
    lease = guard.acquire_gpu()
    import torch
    if not torch.cuda.is_available():
        p.error('CUDA is required; CPU fallback is forbidden')
    args.output.mkdir(parents=True)
    model_id = 'audiosep-base-4M'
    backend = AudioSepBackend(str(args.audiosep_root), model_id, device='cuda')
    settings = Settings(backend='audiosep', model_id=model_id, device='cuda',
                        chunk_seconds_env=args.chunk_seconds, chunk_batch_env=1, refine='off',
                        suppression='off',
                        load_span_predictor=False, load_rankers=False)
    engine = Engine(settings, backend)
    report = {
        'source_commit': '8a5cb33', 'backend':'audiosep', 'is_mock':False,
        'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'inference_verified':False, 'prompt':'dog barking', 'visual_guidance':False,
        'interval':[0,audio.duration], 'sample_count':audio.frames, 'sample_rate':audio.sample_rate,
        'input_sha256':hashlib.sha256((source/'mixture.wav').read_bytes()).hexdigest(),
        'gpu':torch.cuda.get_device_name(), 'torch':torch.__version__,
        'cuda_runtime':torch.version.cuda, 'dtype':'float32', 'seed':42,
        'passes':1, 'refine':'off', 'chunk_seconds':args.chunk_seconds, 'chunk_batch':1,
        'listening_evaluation':'Pending human review; no winner or sound-quality claim',
        'packages':{name:importlib.metadata.version(name) for name in
                    ['torch','torchaudio','pytorch-lightning','transformers','numpy']},
    }
    try:
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        engine.load()
        torch.cuda.synchronize()
        report['load_seconds'] = time.perf_counter()-start
        report['load_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        report['load_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        assert backend.device == 'cuda' and not backend.is_mock
        start = time.perf_counter()
        result = engine.separate(audio, SeparationMetadata.model_validate({
            'prompts':['dog barking'],
            'options':{'seed':42,'passes':1,'chunk_seconds':args.chunk_seconds,'chunk_batch':1,
                       'suppression':'off',
                       'refine':'off','residual':'mixture_minus_targets'},
        }))
        torch.cuda.synchronize()
        report.update(inference_seconds=time.perf_counter()-start, model=result.model,
                      warnings=result.warnings,
                      cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
        target = next(s.samples for s in result.stems if s.role == 'target')
        residual = next(s.samples for s in result.stems if s.role == 'residual')
        if target.shape != audio.samples.shape or residual.shape != audio.samples.shape:
            raise ValueError('Output dimensions do not match the exact mixture')
        if not np.isfinite(target).all() or not np.isfinite(residual).all():
            raise ValueError('Non-finite inference output')
        write_wav(args.output/'isolated-barking.wav', target.T, 48000)
        write_wav(args.output/'conversation-residual.wav', residual.T, 48000)
        # Existing editor export: mute target, retain the mixture-derived residual.
        stems = [dict(path=str(args.output/name), muted=muted, solo=False, volume=1.,
                      start_offset=0., automation='[]', mute_ranges='[]')
                 for name, muted in [('isolated-barking.wav',True),('conversation-residual.wav',False)]]
        start = time.perf_counter()
        export(stems, source/'original.mp4', args.output/'edited.mp4', 'mp4', True)
        report['export_seconds'] = time.perf_counter()-start
        report['export_probe'] = json.loads(subprocess.check_output([
            'ffprobe','-v','error','-show_streams','-show_format','-of','json',str(args.output/'edited.mp4')]))
        report['target_rms'] = float(np.sqrt(np.mean(target**2)))
        report['residual_rms'] = float(np.sqrt(np.mean(residual**2)))
        report['reconstruction_max_abs'] = float(np.max(np.abs(target+residual-audio.samples)))
        report['output_peak'] = float(np.max(np.abs(residual)))
        if all((source/'references'/f'{name}.wav').is_file() for name in ['speaker_a','speaker_b','barking']):
            refs = {name:read_wav(source/'references'/f'{name}.wav')[0][:,0]
                    for name in ['speaker_a','speaker_b','barking']}
            report['metrics'] = reference_metrics(residual.mean(axis=0), audio.samples.mean(axis=0), refs)
            report['barking_target_si_sdri_db'] = float(si_sdr_improvement(target, refs['barking'], audio.samples))
        else:
            report['objective_separation_metrics'] = 'Unavailable: real clip has no isolated source references'
        report['inference_verified'] = True
    except Exception as exc:
        # Avoid including resource URLs or authentication material in shared artifacts.
        summary = re.sub(r'https?://\S+|hf_[A-Za-z0-9]+', '[redacted]', str(exc))[:1000]
        report['failure'] = {'type':type(exc).__name__, 'summary':summary}
        raise
    finally:
        backend.unload()
        json_write(args.output/'report.json', report)
        lease.close()
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
