"""Optional AudioSep text backend; model code and weights are installed separately."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

from .base import Backend, BackendOptions, BackendPrompt, BackendResult
from ..errors import UnsupportedOperation


class AudioSepBackend(Backend):
    name = 'audiosep'
    is_mock = False
    supports_visual = False
    supports_spans = False
    supports_span_prediction = False
    supports_independent_candidates = False
    sample_rate = 48_000

    def __init__(self, root: str, model_id: str, device: str = 'auto') -> None:
        self.root = Path(root).expanduser().resolve() if root else None
        self.model_id = model_id
        self.requested_device = device
        self.device = device
        self.model = None

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def info(self) -> dict:
        return {'model_id': self.model_id, 'device': self.device, 'dtype': 'float32', 'text_ranker_loaded': False}

    def load(self) -> None:
        if self.loaded:
            return
        if not self.root or not (self.root / 'models' / 'audiosep.py').is_file():
            raise RuntimeError('Set AUDIO_AUDIOSEP_ROOT to the prepared AudioSep checkout; see audio-service/README.md')
        config = self.root / 'config' / 'audiosep_base.yaml'
        checkpoint = Path(os.getenv('AUDIO_AUDIOSEP_CHECKPOINT', str(self.root / 'checkpoint' / 'audiosep_base_4M_steps.ckpt'))).expanduser().resolve()
        clap = Path(os.getenv('AUDIO_AUDIOSEP_CLAP_CHECKPOINT', str(self.root / 'checkpoint' / 'music_speech_audioset_epoch_15_esc_89.98.pt'))).expanduser().resolve()
        if not all(p.is_file() for p in (config, checkpoint, clap)):
            raise RuntimeError('AudioSep config or checkpoint is missing; see audio-service/README.md')
        import torch
        if self.requested_device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = self.requested_device
        # AudioSep is imported only when explicitly selected. Its official checkpoint contains
        # numpy scalar metadata; these allow-listed types do not disable PyTorch's safe loader.
        if hasattr(torch.serialization, 'add_safe_globals'):
            torch.serialization.add_safe_globals([(np.core.multiarray.scalar, 'numpy.core.multiarray.scalar'), np.dtype, type(np.dtype(np.float64)), type(np.dtype(np.float32)), type(np.dtype(np.int64))])
        if str(self.root) not in sys.path:
            sys.path.insert(0, str(self.root))
        from models.clap_encoder import CLAP_Encoder
        from utils import load_ss_model, parse_yaml
        encoder = CLAP_Encoder(pretrained_path=str(clap)).eval()
        model = load_ss_model(parse_yaml(str(config)), str(checkpoint), encoder)
        self.model = model.eval().to(torch.device(self.device))

    def separate(self, mixture: np.ndarray, prompt: BackendPrompt, options: BackendOptions) -> BackendResult:
        if not self.loaded:
            raise RuntimeError('AudioSep is not loaded')
        if prompt.visual or prompt.anchors:
            raise UnsupportedOperation('AudioSep supports text prompts only')
        if options.reranking_candidates != 1 or options.candidates != 1:
            raise UnsupportedOperation('AudioSep does not support ranked or independent candidates')
        import torch
        import torchaudio
        data = torch.as_tensor(np.asarray(mixture, dtype=np.float32))
        signal = torchaudio.functional.resample(data, self.sample_rate, 32_000).to(self.device)
        with torch.inference_mode():
            condition = self.model.query_encoder.get_query_embed(modality='text', text=[prompt.text], device=self.device)
            output = self.model.ss_model({'mixture': signal[None, None, :], 'condition': condition})['waveform'].reshape(-1)
            output = torchaudio.functional.resample(output, 32_000, self.sample_rate).float().cpu().numpy()
        output = np.pad(output, (0, max(0, len(mixture) - len(output))))[:len(mixture)]
        return BackendResult(target=output.astype(np.float32))

    def unload(self) -> None:
        self.model = None
        torch = sys.modules.get('torch')
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
