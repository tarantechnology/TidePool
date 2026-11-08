"""Freeze a displayed detection and its time interval for integration."""
from bisect import bisect_right
from copy import deepcopy
import math
from .pipeline import pick_object_at_point


def frame_at_time(metadata, timestamp):
    if not math.isfinite(timestamp) or timestamp < 0:
        return None
    frames = metadata['frames']
    if metadata['media_type'] == 'image':
        return frames[0] if frames and timestamp == 0 else None
    if timestamp >= metadata['processed_duration_seconds']:
        return None
    i = bisect_right([f['timestamp_seconds'] for f in frames], timestamp) - 1
    return frames[i] if i >= 0 and timestamp < frames[i]['valid_until_seconds'] else None


def default_interval(metadata, frame):
    """Choose up to two seconds centered on the observation, on CFR boundaries."""
    if metadata['media_type'] == 'image':
        return {'start_seconds': 0., 'end_seconds': None}
    fps = metadata['fps']
    count = metadata['processed_frames']
    half = max(1, math.floor(fps + .5))
    first = max(0, frame['frame_index'] - half)
    last = min(count, frame['frame_index'] + half + 1)
    return {'start_seconds': round(first / fps, 9),
            'end_seconds': min(metadata['processed_duration_seconds'], round(last / fps, 9))}


def select_at_point(metadata, timestamp, x, y):
    frame = frame_at_time(metadata, timestamp)
    obj = pick_object_at_point(frame['objects'], x, y) if frame else None
    if obj is None:
        return None
    return deepcopy({'schema_version': '1.0', 'kind': 'visual_selection',
                     'video_id': metadata['video_id'], 'analysis_id': metadata['analysis_id'], 'source': metadata['source'],
                     'source_sha256': metadata['source_sha256'],
                     'frame_width': metadata['width'], 'frame_height': metadata['height'],
                     'frame_index': frame['frame_index'],
                     'timestamp_seconds': frame['timestamp_seconds'],
                     'clicked_at_seconds': timestamp,
                     'interval': default_interval(metadata, frame),
                     **obj, 'audio_association': 'unverified'})
