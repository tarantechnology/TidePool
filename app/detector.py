"""Object-detection (vision/ package) availability, invocation and error reporting.

The worker runs the detector as ``sys.executable -m soundcanvas_cv`` so a missing or broken
install must be reported in terms of *this* interpreter. Weights and ultralytics settings live
under ``STORAGE_ROOT/vision`` so they persist in the Docker volume and are found again by the
capabilities check without importing ultralytics into the API process.
"""
import importlib.util
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

from . import core

NOT_INSTALLED = 'Object detection is not installed in this backend. Run: pip install -e ./vision (see README).'
_lock = threading.Lock()
_import_error = ...  # Ellipsis = not checked yet; None = imports fine; str = last error line.


def model():
    return os.getenv('VISION_MODEL', 'yolo26n-seg.pt')


def device():
    return os.getenv('VISION_DEVICE', 'cpu')


def adapter():
    value = os.getenv('VISION_ADAPTER', 'yolo')
    if value not in ('yolo', 'sam3', 'sam3.1', 'sam2.1'):
        raise ValueError('VISION_ADAPTER must be yolo, sam3, sam3.1 or sam2.1')
    return value


def home():
    """Detector working directory: ultralytics downloads bare asset names into its cwd."""
    path = core.ROOT / 'vision'
    path.mkdir(parents=True, exist_ok=True)
    return path


def environment():
    env = dict(os.environ)
    # Keep settings.json next to the weights unless the operator chose a location.
    env.setdefault('YOLO_CONFIG_DIR', str(home() / 'config'))
    return env


def model_argument():
    """A path that exists relative to the backend's cwd keeps working from the detector's cwd."""
    name = model()
    local = Path(name).expanduser()
    return str(local.resolve()) if local.is_file() else name


def weights_cached():
    """Where ultralytics' attempt_download_asset looks: the path itself (cwd), then settings weights_dir."""
    local = Path(model()).expanduser()
    if local.is_file():
        return True
    if local.is_absolute():
        return False
    cwd = core.ROOT / 'vision'
    if (cwd / local).is_file():
        return True
    settings = Path(environment()['YOLO_CONFIG_DIR']) / 'Ultralytics' / 'settings.json'
    try:
        weights_dir = Path(json.loads(settings.read_text())['weights_dir'])
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return (cwd / weights_dir / local).is_file()  # An absolute weights_dir replaces cwd.


def _spawn(args, timeout, **kwargs):
    """Seam for tests: run the detector interpreter and capture its output."""
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout, **kwargs)


def _probe_import():
    """Import the detector stack in the worker's interpreter once, without loading weights."""
    try:
        imports = {'yolo': 'import soundcanvas_cv.yolo_adapter, ultralytics',
                   'sam2.1': 'import soundcanvas_cv.sam2_adapter; from sam2.build_sam import build_sam2_video_predictor'}.get(
                       adapter(), 'import soundcanvas_cv.sam3_adapter; from sam3.model_builder import build_sam3_predictor')
        result = _spawn([sys.executable, '-c', imports], 120,
                        cwd=home(), env=environment())
    except (subprocess.SubprocessError, OSError) as exc:
        return f'Object detection could not be checked: {exc}'
    if result.returncode == 0:
        return None
    return explain(result.stderr, result.returncode)


def status(refresh=False):
    """Capabilities ``vision`` block. Availability is cached; weights presence is re-checked."""
    global _import_error
    with _lock:
        if refresh:
            _import_error = ...
        if _import_error is ...:
            dependency = {'yolo': 'ultralytics', 'sam2.1': 'sam2'}.get(adapter(), 'sam3')
            if any(importlib.util.find_spec(name) is None for name in ('soundcanvas_cv', dependency)):
                _import_error = NOT_INSTALLED if adapter() == 'yolo' else 'Official SAM runtime is not installed in the detector interpreter'
            else:
                _import_error = _probe_import()
        reason = _import_error
    return {'available': reason is None, 'reason': reason, 'model': model(),
            'weights_cached': weights_cached(), 'device': device()}


def forget():
    """A failed run that says the package is missing invalidates a cached 'available'."""
    global _import_error
    with _lock:
        _import_error = ...


def last_line(text):
    lines = [line.strip() for line in (text or '').splitlines() if line.strip()]
    return lines[-1] if lines else ''


def explain(stderr, returncode=None):
    """Map detector stderr to one actionable sentence for the editor."""
    text = stderr or ''
    if any(f"No module named '{name}" in text or f'No module named {name}' in text
           for name in ('soundcanvas_cv', 'ultralytics', 'cv2', 'lap', 'torch')):
        return NOT_INSTALLED
    if 'Download failure' in text or 'Environment may be offline' in text or 'urlopen error' in text:
        return (f'Could not download detector weights (offline?); run `python -m soundcanvas_cv --model {model()} ...` '
                'once with network or set VISION_MODEL to a local path')
    line = last_line(text)
    if line:
        return line[-1000:]
    return f'Object detection exited with code {returncode}' if returncode is not None else 'Object detection failed'


class DetectorError(RuntimeError):
    pass


def detect(source, output_dir, video_id):
    args = [sys.executable, '-m', 'soundcanvas_cv', '--input', str(source), '--output-dir', str(output_dir),
            '--model', model_argument(), '--sample-fps', '1000', '--max-seconds', os.getenv('VISION_MAX_SECONDS', '30'),
            '--device', device(), '--video-id', video_id]
    if adapter() != 'yolo':
        args.extend(['--adapter', adapter(), '--concept', os.getenv('VISION_CONCEPT', 'knife')])
    if adapter() == 'sam2.1':
        if not os.getenv('VISION_SEED_OBJECTS'):
            raise DetectorError('SAM 2 requires VISION_SEED_OBJECTS with user-supplied points; it has no text detector')
        args.extend(['--seed-objects', str(Path(os.environ['VISION_SEED_OBJECTS']).expanduser().resolve())])
    if os.getenv('VISION_IMAGE_SIZE') and adapter() == 'yolo':
        args.extend(['--image-size', os.environ['VISION_IMAGE_SIZE']])
    timeout = int(os.getenv('VISION_TIMEOUT_SECONDS', os.getenv('FFMPEG_TIMEOUT', '600')))
    try:
        result = _spawn(args, timeout, cwd=home(), env=environment())
    except subprocess.TimeoutExpired as exc:
        raise DetectorError(f'Object detection timed out after {timeout} s; lower VISION_MAX_SECONDS or raise VISION_TIMEOUT_SECONDS') from exc
    except OSError as exc:
        raise DetectorError(f'Object detection could not start: {exc}') from exc
    if result.returncode != 0:
        message = explain(result.stderr, result.returncode)
        if message == NOT_INSTALLED:
            forget()
        raise DetectorError(message)
    return json.loads((Path(output_dir) / 'detections.json').read_text())


def prefetch():
    """Download the configured weights into the same place the worker will look (dev_setup.sh)."""
    if adapter() != 'yolo':
        print('SAM requires an approved official checkpoint at VISION_MODEL; automatic prefetch is disabled', file=sys.stderr)
        return 2
    result = _spawn([sys.executable, '-c', 'import sys; from ultralytics import YOLO; YOLO(sys.argv[1])', model_argument()],
                    None, cwd=home(), env=environment())
    if result.returncode != 0:
        print(explain(result.stderr, result.returncode), file=sys.stderr)
        return result.returncode
    print(f'Detector weights for {model()} are cached under {home()}')
    return 0


if __name__ == '__main__':
    if sys.argv[1:] != ['--prefetch']:
        sys.exit('usage: python -m app.detector --prefetch')
    sys.exit(prefetch())
