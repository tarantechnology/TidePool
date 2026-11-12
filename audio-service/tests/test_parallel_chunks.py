import threading
import time

import numpy as np

from sam_audio_service.audio_io import DecodedAudio
from sam_audio_service.backends.base import Backend, BackendResult
from sam_audio_service.config import Settings
from sam_audio_service.contracts import SeparationMetadata
from sam_audio_service.engine import Engine
from sam_audio_service.errors import InsufficientGpuMemory


class TrackingBackend(Backend):
    is_mock = True
    sample_rate = 100

    def __init__(self, fail_parallel=False):
        self.lock = threading.Lock()
        self.active = self.peak = self.calls = 0
        self.fail_parallel = fail_parallel

    def load(self): pass
    def info(self): return {}

    def separate(self, mixture, prompt, options):
        with self.lock:
            self.active += 1
            self.calls += 1
            self.peak = max(self.peak, self.active)
            fail = self.fail_parallel and self.active > 1
        try:
            time.sleep(.03)
            if fail:
                self.fail_parallel = False
                raise InsufficientGpuMemory('simulated OOM')
            return BackendResult(target=mixture*.25, residual=mixture*.75)
        finally:
            with self.lock: self.active -= 1


def run(backend, seed=None):
    settings=Settings(chunk_workers=2, chunk_overlap_seconds=.25)
    engine=Engine(settings,backend)
    signal=np.linspace(-1,1,337,dtype=np.float32)[None,:]
    metadata=SeparationMetadata.model_validate({'prompts':['voice'],'options':{'chunk_seconds':1,'seed':seed}})
    result=engine.separate(DecodedAudio(signal,100,100,1),metadata)
    return signal,result


def test_windows_run_concurrently_and_keep_sample_alignment():
    backend=TrackingBackend()
    signal,result=run(backend)
    assert backend.peak == 2 and backend.calls > 2
    np.testing.assert_allclose(result.stems[0].samples,signal*.25,atol=1e-6)
    np.testing.assert_allclose(sum(s.samples for s in result.stems),signal,atol=1e-6)


def test_seeded_windows_do_not_race_global_rng():
    backend=TrackingBackend()
    _,result=run(backend,seed=7)
    assert backend.peak == 1 and any('sequential' in w for w in result.warnings)


def test_parallel_oom_retries_without_partial_mix():
    backend=TrackingBackend(fail_parallel=True)
    signal,result=run(backend)
    assert backend.peak == 2
    assert any("retried sequentially" in w for w in result.warnings)
    np.testing.assert_allclose(result.stems[0].samples,signal*.25,atol=1e-6)
