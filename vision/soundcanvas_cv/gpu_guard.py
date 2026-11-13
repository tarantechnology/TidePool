"""Cooperative experiment lock plus a read-only check for unrelated GPU work."""
import fcntl
import os
from pathlib import Path
import subprocess


def acquire_gpu():
    """Return an open lock lease; caller keeps it alive for the entire experiment.

    This lock serializes our experiment scripts, not arbitrary other GPU programs.
    Existing compute processes cause refusal; none are signalled or terminated.
    """
    path = Path(os.getenv('SOUNDCANVAS_GPU_LOCK', '/workspace/locks/soundcanvas-gpu.lock'))
    path.parent.mkdir(parents=True, exist_ok=True)
    lease = path.open('a')
    try:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
        if processes:
            raise RuntimeError('GPU compute processes already exist; wait for them to finish')
        return lease
    except BaseException:
        lease.close()
        raise
