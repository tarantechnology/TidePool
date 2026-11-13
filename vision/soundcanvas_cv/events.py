"""Conservative event evidence, separate from detections/selection contract 1.0.

Centroid acceleration and spectral-flux peaks are observations of model outputs
and waveforms, not action recognition or proof that an object caused a sound.
Semantic labels are supplied hypotheses or independently annotated audio events.
"""
from collections import defaultdict
import math

import numpy as np


def audio_events(samples, sample_rate, hop_seconds=.02):
    """Local positive spectral-flux maxima, rejecting silence. No semantic labels."""
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim != 1 or not np.isfinite(x).all() or sample_rate <= 0:
        raise ValueError('Expected finite mono audio and a positive sample rate')
    hop = max(1, round(sample_rate*hop_seconds))
    size = 2*hop
    if len(x) < size or np.max(np.abs(x), initial=0) < 1e-5:
        return []
    spectra, rms = [], []
    for start in range(0, len(x)-size+1, hop):
        window = x[start:start+size]
        spectra.append(np.abs(np.fft.rfft(window*np.hanning(size))))
        rms.append(float(np.sqrt(np.mean(window**2))))
    flux = np.maximum(np.diff(spectra, axis=0, prepend=[spectra[0]]), 0).mean(axis=1)
    median = np.median(flux)
    threshold = max(float(median+3*np.median(np.abs(flux-median))), float(flux.max())*.15, 1e-6)
    events, previous = [], -math.inf
    for i in range(1, len(flux)-1):
        t = (i*hop+size/2)/sample_rate
        if flux[i] > threshold and flux[i] >= flux[i-1] and flux[i] > flux[i+1] and rms[i] > 1e-4 and t-previous >= .10:
            events.append(dict(id=f'audio:{len(events)}', start=max(0., t-.04),
                               end=min(len(x)/sample_rate, t+.08), timestamp=t,
                               label='unclassified transient', status='observed_signal_feature',
                               provenance={'component': 'spectral_flux_v1', 'flux': float(flux[i]),
                                           'rms': rms[i], 'semantic_classifier': None}))
            previous = t
    return events


def motion_events(detections, min_acceleration=.5):
    """Track-local acceleration maxima; never cross missing native observations."""
    fps = detections['fps']
    if not fps or fps <= 0:
        raise ValueError('Motion evidence requires video FPS')
    tracks = defaultdict(list)
    for frame in detections['frames']:
        for obj in frame['objects']:
            if obj['track_id'] is None or obj.get('mask') is None:
                continue
            x1, y1, x2, y2 = obj['bbox_normalized']
            tracks[obj['object_id']].append((frame, obj, np.array([(x1+x2)/2, (y1+y2)/2])))
    evidence = []
    for oid, entries in tracks.items():
        last = -math.inf
        for left, mid, right in zip(entries, entries[1:], entries[2:]):
            if mid[0]['frame_index']-left[0]['frame_index'] != 1 or right[0]['frame_index']-mid[0]['frame_index'] != 1:
                continue
            acceleration = float(np.linalg.norm(right[2]-2*mid[2]+left[2])*fps**2)
            timestamp = mid[0]['timestamp_seconds']
            if acceleration < min_acceleration or timestamp-last < .12:
                continue
            evidence.append(dict(id=f'motion:{len(evidence)}', object_id=oid, label=mid[1]['label'],
                                 timestamp=timestamp, start=left[0]['timestamp_seconds'],
                                 end=right[0]['timestamp_seconds']+1/fps,
                                 observed_action='mask_bbox_acceleration', status='observed_visual_feature',
                                 provenance={'component': 'track_motion_v1', 'acceleration': acceleration,
                                             'detection_ids': [e[1]['detection_id'] for e in (left, mid, right)],
                                             'limitation': 'camera motion, mask jitter and occlusion can cause this feature'}))
            last = timestamp
    return evidence


def associate(detections, acoustic_events, hypotheses=None, tolerance=.12):
    """Temporal support only. Ambiguous sources abstain; scores are uncalibrated."""
    if not math.isfinite(tolerance) or not 0 < tolerance <= 1:
        raise ValueError('Invalid association tolerance')
    duration = detections['processed_duration_seconds']
    for event in acoustic_events:
        if not all(math.isfinite(event[k]) for k in ('start', 'end', 'timestamp')) or not 0 <= event['start'] <= event['timestamp'] <= event['end'] <= duration+1e-8:
            raise ValueError('Audio evidence is outside the analyzed source timeline')
        if not event.get('provenance'):
            raise ValueError('Audio events require provenance')
    visual = motion_events(detections)
    associations = []
    for audio in acoustic_events:
        matches = [v for v in visual if abs(v['timestamp']-audio['timestamp']) <= tolerance]
        sources = {v['object_id'] for v in matches}
        for v in matches:
            ambiguous = len(sources) > 1
            associations.append(dict(object_id=v['object_id'], visual_event_id=v['id'],
                                     audio_event_id=audio['id'], interval=[audio['start'], audio['end']],
                                     candidate_acoustic_labels=(hypotheses or {}).get(v['label'], []),
                                     association_confidence=0. if ambiguous else .49*(1-abs(v['timestamp']-audio['timestamp'])/tolerance),
                                     confidence_kind='uncalibrated temporal heuristic; not a causal probability',
                                     status='ambiguous' if ambiguous else 'candidate_requires_review',
                                     provenance={'component': 'temporal_coincidence_v1',
                                                 'delta_seconds': abs(v['timestamp']-audio['timestamp']),
                                                 'causality_verified': False}))
    return dict(schema_version='object-sound-evidence/0.1', analysis_id=detections['analysis_id'],
                video_id=detections['video_id'], source_sha256=detections['source_sha256'],
                object_observations={'reference': 'detections.json', 'includes': ['identities', 'native_masks', 'timestamps']},
                visual_events=visual, audio_events=acoustic_events, associations=associations,
                limitations=['No semantic action recognizer installed', 'No causal sound attribution',
                             'Stationary or slowly scraping objects may have no motion event',
                             'No automatic destructive edit; review separated stem first'])


def reviewed_anchors(graph, object_id, approved_event_ids):
    """Convert explicitly reviewed, nonambiguous associations to existing SAM anchors."""
    approved = set(approved_event_ids)
    intervals = sorted({tuple(a['interval']) for a in graph['associations']
                        if a['object_id'] == object_id and a['audio_event_id'] in approved
                        and a['status'] == 'candidate_requires_review'})
    merged = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    return [dict(token='+', start=start, end=end) for start, end in merged]
