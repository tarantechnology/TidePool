import pytest

from soundcanvas_cv.gpu_guard import acquire_gpu


def test_experiments_cannot_share_gpu_lease(tmp_path, monkeypatch):
    monkeypatch.setenv('SOUNDCANVAS_GPU_LOCK', str(tmp_path/'gpu.lock'))
    monkeypatch.setattr('soundcanvas_cv.gpu_guard.subprocess.check_output', lambda *a, **k: '')
    first = acquire_gpu()
    try:
        with pytest.raises(BlockingIOError):
            acquire_gpu()
    finally:
        first.close()
    acquire_gpu().close()


def test_unrelated_gpu_work_causes_refusal(tmp_path, monkeypatch):
    monkeypatch.setenv('SOUNDCANVAS_GPU_LOCK', str(tmp_path/'gpu.lock'))
    monkeypatch.setattr('soundcanvas_cv.gpu_guard.subprocess.check_output', lambda *a, **k: '12345\n')
    with pytest.raises(RuntimeError, match='already exist'):
        acquire_gpu()
    # Refusal releases our own lock; it never signals the other process.
    monkeypatch.setattr('soundcanvas_cv.gpu_guard.subprocess.check_output', lambda *a, **k: '')
    acquire_gpu().close()
