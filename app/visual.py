"""Project-owned media handoff for CV selection 1.0; no model dependencies."""
import hashlib
import json
import math
from pathlib import Path

from .core import probe, project_dir, run


def file_sha256(path):
    with Path(path).open('rb') as stream:
        if hasattr(hashlib, 'file_digest'):  # Python 3.11+
            return hashlib.file_digest(stream, 'sha256').hexdigest()
        sha = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            sha.update(block)
        return sha.hexdigest()


def reject_media_refs(value):
    """Only the backend may choose files for the privileged audio service to read."""
    if isinstance(value, dict):
        if any(key in value for key in ('video_ref', 'original_ref', 'mask_video_ref', 'detections_ref', 'audio_ref')):
            raise ValueError('Client media references are not accepted; use uploaded project media and stored detections')
        for child in value.values():
            reject_media_refs(child)
    elif isinstance(value, list):
        for child in value:
            reject_media_refs(child)


def project_source(project):
    """Resolve only this project's immutable upload, including after a queued job resumes."""
    path = Path(project['original']).resolve(strict=True)
    root = (project_dir(project['id']) / 'original').resolve()
    if not path.is_relative_to(root):
        raise ValueError('Original media is outside this project')
    info = json.loads(project['metadata'])
    if info.get('has_video') and not info.get('frame_rate'):
        info = {**info, **probe(path)}  # Uploads created before source metadata was added.
    sha = file_sha256(path)
    if info.get('source_sha256') and info['source_sha256'] != sha:
        raise ValueError('Uploaded media hash changed; upload and analyze the same original bytes again')
    return {'asset_id': project['id'], 'original_ref': path.as_uri(), 'sha256': sha,
            'duration': info['duration'], 'frame_rate': info.get('frame_rate')}


def validate_identity(detections, project, source):
    if not isinstance(detections, dict):
        raise ValueError('Stored detections must be a JSON object')
    if detections.get('schema_version') != '1.0' or not detections.get('analysis_id'):
        raise ValueError('Stored detections must use schema 1.0 and include analysis_id')
    if detections.get('source_sha256') != source['sha256']:
        raise ValueError('Detections source_sha256 does not match the uploaded video')
    if detections.get('video_id') not in (project['id'], 'sha256:' + source['sha256']):
        raise ValueError('Detections video_id does not identify this project upload')


def validate_timeline(path, fps):
    """The current CV/model contract supports unrotated CFR media starting at zero."""
    data = json.loads(run('ffprobe', '-v', 'error', '-show_streams', '-of', 'json', str(path)))
    for stream in data['streams']:
        if stream['codec_type'] not in ('audio', 'video'):
            continue
        if abs(float(stream.get('start_time', 0))) > 1 / 48000:
            raise ValueError('Visual prompting requires audio/video starting at zero; normalize the clip before upload')
        if stream['codec_type'] == 'video':
            rotation = float(stream.get('tags', {}).get('rotate', 0))
            rotation += sum(float(s.get('rotation', 0)) for s in stream.get('side_data_list', []))
            if rotation % 360:
                raise ValueError('Normalize video rotation before visual prompting')
    frames = json.loads(run('ffprobe', '-v', 'error', '-select_streams', 'v:0',
                            '-show_entries', 'frame=best_effort_timestamp_time', '-of', 'json', str(path)))['frames']
    if not frames or any(abs(float(frame.get('best_effort_timestamp_time', 'inf')) - i / fps) > 1e-4
                         for i, frame in enumerate(frames)):
        raise ValueError('Visual prompting requires constant frame rate; normalize the clip before upload')


def validate_selection(selection, detections, project, source):
    validate_identity(detections, project, source)
    if selection.get('schema_version') != '1.0':
        raise ValueError('Selection must use schema 1.0')
    for key in ('video_id', 'source_sha256', 'analysis_id'):
        if selection.get(key) != detections[key]:
            raise ValueError(f'Selection {key} does not match stored detections')
    info = json.loads(project['metadata'])
    if info.get('has_video') and not info.get('frame_rate'):
        info = {**info, **probe(project['original'])}
    if not info['has_video']:
        raise ValueError('Visual selection requires a video upload')
    width, height = detections.get('width'), detections.get('height')
    if (width, height) != (info.get('width'), info.get('height')):
        raise ValueError('Detection dimensions do not match uploaded video')
    if (selection.get('frame_width'), selection.get('frame_height')) != (width, height):
        raise ValueError('Selection dimensions do not match stored detections')
    fps = detections.get('fps')
    if type(fps) not in (int, float) or not math.isfinite(fps) or fps <= 0 or not math.isclose(fps, info.get('frame_rate') or 0, rel_tol=1e-5):
        raise ValueError('Detection FPS does not match uploaded video')
    track = selection.get('track_id')
    if not isinstance(track, int) or isinstance(track, bool):
        raise ValueError('Visual interval requires a tracked object')
    interval = selection.get('interval') or {}
    if not isinstance(interval, dict):
        raise ValueError('Selection interval must be a JSON object')
    start, end = interval.get('start_seconds'), interval.get('end_seconds')
    if any(type(t) not in (int, float) or not math.isfinite(t) for t in (start, end)):
        raise ValueError('Selection interval must contain finite start and end seconds')
    if start < 0 or not .1 - 1e-8 <= end - start <= 30 + 1e-8 or end > min(detections.get('processed_duration_seconds', 0), info['duration']) + 1e-6:
        raise ValueError('Selection interval must span 0.1–30 seconds inside processed media')
    audio_duration = probe(project['audio'])['duration']
    if end > audio_duration + 1e-6:
        raise ValueError('Selection interval extends beyond the extracted audio; choose a range within the audio duration')
    if any(abs(t * fps - round(t * fps)) > 1e-5 for t in (start, end)):
        raise ValueError('Selection interval must align to native frame boundaries')
    records = {}
    frames = detections.get('frames')
    if not isinstance(frames, list):
        raise ValueError('Detection frames must be a list')
    for frame in frames:
        if not isinstance(frame, dict):
            raise ValueError('Detection frames must be JSON objects')
        index = frame.get('frame_index')
        if not isinstance(index, int) or isinstance(index, bool) or index in records:
            raise ValueError('Detection frame indices must be unique integers')
        records[index] = frame
        if not isinstance(frame.get('objects'), list) or any(not isinstance(obj, dict) for obj in frame['objects']):
            raise ValueError('Frame objects must be a list of JSON objects')
    if type(selection.get('frame_index')) is not int or not isinstance(selection.get('detection_id'), str):
        raise ValueError('Selection requires an integer frame_index and a detection_id')
    observation = records.get(selection.get('frame_index'))
    if observation is None or not start <= observation['timestamp_seconds'] < end:
        raise ValueError('Selection interval must contain its observed frame')
    matches = [obj for obj in observation.get('objects', []) if obj.get('detection_id') == selection.get('detection_id')]
    if len(matches) != 1 or matches[0].get('track_id') != track:
        raise ValueError('Selected detection_id/track_id was not observed in stored metadata')
    if selection.get('timestamp_seconds') != observation['timestamp_seconds']:
        raise ValueError('Selection timestamp does not match its observed frame')
    for key, value in matches[0].items():
        if key in selection and selection[key] != value:
            raise ValueError(f'Selection {key} differs from the stored observation')
    visible = False
    for index in range(round(start * fps), round(end * fps)):
        frame = records.get(index)
        if frame is None:
            raise ValueError(f'Sparse detections: native frame {index} is missing; rerun segmentation at native FPS')
        if abs(frame.get('timestamp_seconds', -1) - index / fps) > 1e-6 or abs(frame.get('valid_until_seconds', -1) - (index + 1) / fps) > 1e-6:
            raise ValueError(f'Detection timestamps do not match native frame {index}')
        objects = [obj for obj in frame.get('objects', []) if obj.get('track_id') == track]
        if len(objects) > 1:
            raise ValueError(f'Ambiguous selected track at frame {index}')
        if not objects:
            continue  # An analyzed frame with no observation means empty foreground.
        mask = objects[0].get('mask') or {}
        if not isinstance(mask, dict):
            raise ValueError(f'Instance mask must be an RLE object at frame {index}')
        counts = mask.get('counts')
        if (mask.get('encoding'), mask.get('order'), mask.get('foreground_value'), mask.get('size')) != ('rle', 'C', 1, [height, width]):
            raise ValueError(f'Missing or invalid native instance mask at frame {index}')
        if not isinstance(counts, list) or not counts or any(type(c) is not int or c < 0 for c in counts) or sum(counts) != width * height:
            raise ValueError(f'Invalid mask RLE counts at frame {index}')
        visible |= sum(counts[1::2]) > 0
    if not visible:
        raise ValueError('Selected object masks are empty throughout the interval')


def prepare_payload(project, payload):
    """Snapshot stored masks with the frozen selection; clients never select a path."""
    raw = payload.get('visual_prompt')
    reject_media_refs(raw)
    stored = json.loads(project['visual']) if project['visual'] else None
    if raw is None:
        raw = stored
    if not raw:
        return payload
    reject_media_refs(raw)
    if 'selection' in raw and not isinstance(raw['selection'], dict):
        raise ValueError('Selection must be a JSON object')
    selection = raw.get('selection') if isinstance(raw.get('selection'), dict) else raw if raw.get('kind') == 'visual_selection' else None
    if selection is None:
        # Without a selected object nothing here can steer the model, and the stored
        # detections document (every frame's RLE) must not cross the wire for nothing.
        payload['visual_prompt'] = None
        return payload
    if not stored:
        raise ValueError('Save the matching detections with visual-metadata before selecting an object')
    detections = stored.get('detections', stored)
    if raw.get('detections') is not None and raw['detections'] != detections:
        raise ValueError('Inline detections differ from stored project detections')
    source = project_source(project)
    validate_selection(selection, detections, project, source)
    validate_timeline(project['original'], source['frame_rate'])
    payload['source'] = source
    payload['asset_id'] = project['id']  # Existing audio response correlation field.
    payload['visual_prompt'] = {'selection': selection, 'detections': detections}
    return payload
