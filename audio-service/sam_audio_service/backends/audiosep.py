"""Optional AudioSep text backend; model code and weights are installed separately."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

from .base import Backend, BackendOptions, BackendPrompt, BackendResult
from ..errors import UnsupportedOperation

CHECKPOINT_REPO = 'Audio-AGI/AudioSep'
CHECKPOINT_REVISION = '0807a713283035d8b0d0628f1b8e3d6580459f7b'


def checkpoint_path(root: Path, variable: str, filename: str) -> Path:
    explicit = os.getenv(variable)
    local = Path(explicit).expanduser().resolve() if explicit else root/'checkpoint'/filename
    if local.is_file():
        return local
    if explicit:
        raise RuntimeError(f'{variable} points to a missing checkpoint')
    # Public official weights, pinned to the verified revision. Hub cache is reused.
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(CHECKPOINT_REPO, 'checkpoint/'+filename,
                               repo_type='space', revision=CHECKPOINT_REVISION))


class AudioSepBackend(Backend):
    name = 'audiosep'
    is_mock = False
    supports_visual = False
    supports_spans = False
    supports_span_prediction = False
    supports_independent_candidates = False
    supports_tagging = True
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
        if not config.is_file():
            raise RuntimeError('AudioSep config is missing; see audio-service/README.md')
        checkpoint = checkpoint_path(self.root, 'AUDIO_AUDIOSEP_CHECKPOINT', 'audiosep_base_4M_steps.ckpt')
        clap = checkpoint_path(self.root, 'AUDIO_AUDIOSEP_CLAP_CHECKPOINT', 'music_speech_audioset_epoch_15_esc_89.98.pt')
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

    def tag(self, windows: list[np.ndarray], labels: list[str]) -> tuple[np.ndarray, float]:
        """Zero-shot tagging with AudioSep's own CLAP query encoder: audio and text share one space."""
        if not self.loaded:
            raise RuntimeError('AudioSep is not loaded')
        import torch
        import torchaudio
        encoder = self.model.query_encoder
        audio_rows, text_rows = [], []
        with torch.inference_mode():
            for start in range(0, len(windows), 8):
                batch = torch.as_tensor(np.stack(windows[start:start + 8]).astype(np.float32))
                # The encoder takes 32 kHz and resamples to CLAP's 48 kHz itself.
                batch = torchaudio.functional.resample(batch, self.sample_rate, 32_000).to(self.device)
                audio_rows.append(encoder.get_query_embed(modality='audio', audio=batch, device=self.device).float())
            for start in range(0, len(labels), 16):
                text_rows.append(encoder.get_query_embed(modality='text', text=labels[start:start + 16], device=self.device).float())
            audio = torch.nn.functional.normalize(torch.cat(audio_rows).to(self.device), dim=-1)
            text = torch.nn.functional.normalize(torch.cat(text_rows).to(self.device), dim=-1)
            sims = (audio @ text.T).cpu().numpy()
        scale = 30.0
        clap = getattr(encoder, 'model', None)
        logit = getattr(clap, 'logit_scale_a', None)
        if logit is not None:
            try:
                scale = float(min(100.0, max(10.0, float(logit.exp()))))
            except (TypeError, ValueError, RuntimeError):
                pass
        return sims, scale

    def unload(self) -> None:
        self.model = None
        torch = sys.modules.get('torch')
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
