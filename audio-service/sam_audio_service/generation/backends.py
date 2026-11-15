"""Optional model backends for the generate pipeline. All lazy: nothing is imported or loaded until
first use, and every one can be absent (the pipeline then falls back and says so).

* ``BeatThisBackend``      beat_this (CPJKU, MIT code + weights)      beats + downbeats
* ``BasicPitchTranscriber`` Basic Pitch (Spotify, Apache-2.0, ONNX)    vocal melody / bass notes
* ``AmtComposer``          Anticipatory Music Transformer (Stanford CRFM, Apache-2.0 code + weights):
                           accompaniment infilling conditioned on vocal melody + chord controls,
                           restricted to the requested GM program
* ``MusicGenGenerator``    MusicGen-melody (Meta; code MIT, **weights CC-BY-NC-4.0**): text +
                           chroma-conditioned audio generation. Non-commercial: refused unless
                           ``AUDIO_GEN_ALLOW_NONCOMMERCIAL=true``.
"""
from __future__ import annotations

import logging
import tempfile
import threading
from pathlib import Path
from typing import Any

import numpy as np

from ..music_analysis import Note
from . import dsp
from .composer import PNote

log = logging.getLogger("sam_audio_service")

LICENSES = {
    "beat_this": {"code": "MIT", "weights": "MIT (CPJKU beat_this checkpoints)", "commercial_use": True},
    "basic_pitch": {"code": "Apache-2.0", "weights": "Apache-2.0 (bundled ONNX)", "commercial_use": True},
    "amt": {"code": "Apache-2.0", "weights": "Apache-2.0 (stanford-crfm/music-*-800k)", "commercial_use": True},
    "musicgen": {"code": "MIT (transformers / audiocraft)", "weights": "CC-BY-NC-4.0 (facebook/musicgen-melody)", "commercial_use": False},
    "stable_audio_open": {"code": "MIT (stable-audio-tools)", "weights": "Stability AI Community License (gated; revenue cap)", "commercial_use": "restricted"},
}


class BeatThisBackend:
    """beat_this. ``checkpoint`` is a beat_this short name / path / URL, or ``hf:<repo>`` for a
    safetensors export (PyTorchModelHubMixin layout: config.json + model.safetensors), which is
    converted once into a weights-only .ckpt next to the HF cache. The upstream JKU download host
    returned HTTP 500 on 2026-11-14, so the default is the MIT safetensors export."""

    name = "beat_this"

    def __init__(self, device: str = "cuda", checkpoint: str = "hf:safe-models/beat-this-final0"):
        self.device, self.checkpoint = device, checkpoint
        self._model = None
        self._lock = threading.Lock()

    def _resolve(self) -> str:
        if not self.checkpoint.startswith("hf:"):
            return self.checkpoint
        import json
        import torch
        from huggingface_hub import snapshot_download
        from safetensors.torch import load_file

        root = Path(snapshot_download(self.checkpoint[3:], allow_patterns=["config.json", "model.safetensors"]))
        out = root / "beat_this.ckpt"
        if not out.exists():
            sd = load_file(str(root / "model.safetensors"))
            torch.save({"hyper_parameters": json.loads((root / "config.json").read_text()), "state_dict": {"model." + k: v for k, v in sd.items()}}, out)
        return str(out)

    def track(self, mono: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
        with self._lock:
            if self._model is None:
                from beat_this.inference import Audio2Beats, load_model
                import beat_this.inference as bti

                path = self._resolve()
                orig = bti.BeatThis.load_state_dict

                def lenient(model, state_dict, strict=True, **kw):  # rotary freqs are rebuilt buffers, not in exports
                    return orig(model, state_dict, strict=False, **kw)

                bti.BeatThis.load_state_dict = lenient
                try:
                    self._model = Audio2Beats(checkpoint_path=path, device=self.device, dbn=False)
                finally:
                    bti.BeatThis.load_state_dict = orig
                del load_model
            beats, downbeats = self._model(np.asarray(mono, dtype=np.float32), sr)
        return np.asarray(beats, dtype=np.float64), np.asarray(downbeats, dtype=np.float64)


class BasicPitchTranscriber:
    name = "basic_pitch"

    def __init__(self, onset_threshold: float = 0.5, frame_threshold: float = 0.3, min_note_ms: float = 90.0):
        self.onset_threshold, self.frame_threshold, self.min_note_ms = onset_threshold, frame_threshold, min_note_ms
        self._model = None
        self._lock = threading.Lock()

    def transcribe(self, mono: np.ndarray, sr: int) -> list[Note]:
        from ..audio_io import encode_wav_float32

        from basic_pitch import ICASSP_2022_MODEL_PATH
        from basic_pitch.inference import Model, predict

        with self._lock:  # only model creation is serialised; ONNX sessions run concurrently
            if self._model is None:
                onnx = Path(str(ICASSP_2022_MODEL_PATH)).with_suffix(".onnx")
                self._model = Model(onnx if onnx.exists() else ICASSP_2022_MODEL_PATH)
        if True:
            with tempfile.TemporaryDirectory(prefix="gen-bp-") as tmp:
                wav = Path(tmp) / "x.wav"
                wav.write_bytes(encode_wav_float32(np.asarray(mono, dtype=np.float32)[None, :], sr))
                _, _, events = predict(str(wav), self._model, onset_threshold=self.onset_threshold, frame_threshold=self.frame_threshold,
                                       minimum_note_length=self.min_note_ms, melodia_trick=True)
        return [Note(float(s), float(e), int(p), int(np.clip(a * 127, 1, 127))) for s, e, p, a, *_ in events]


class AmtComposer:
    """Accompaniment infilling with the Anticipatory Music Transformer.

    Controls (anticipated): the vocal melody (as GM 52 voice) and block chords from the analysis
    (as GM 0 piano). The model then generates events in the window; sampling is restricted to the
    requested GM program so it writes *that* instrument, in time with the controls.
    """

    name = "amt"

    def __init__(self, model_id: str, device: str = "cuda"):
        self.model_id, self.device = model_id, device
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForCausalLM

            self._model = AutoModelForCausalLM.from_pretrained(self.model_id, torch_dtype=torch.float32).to(self.device).eval()
        return self._model

    def compose(self, analysis: dict[str, Any], program: int, low: int, high: int, windows: list[tuple[float, float]], seed: int = 0,
                top_p: float = 0.95, prompt_notes: list[PNote] | None = None, max_seconds: float = 90.0) -> list[PNote]:
        import torch
        from anticipation import ops, sample
        from anticipation.config import TIME_RESOLUTION
        from anticipation.vocab import CONTROL_OFFSET, DUR_OFFSET, NOTE_OFFSET, TIME_OFFSET

        def ev(start, end, pitch, instr):
            t = int(round(start * TIME_RESOLUTION))
            d = max(1, min(999, int(round((end - start) * TIME_RESOLUTION))))
            return [TIME_OFFSET + t, DUR_OFFSET + d, NOTE_OFFSET + instr * 128 + int(pitch)]

        out: list[PNote] = []
        with self._lock:
            model = self._load()
            original = sample.instr_logits

            def only_target(logits, full_history):
                mask = torch.full_like(logits[NOTE_OFFSET:NOTE_OFFSET + 129 * 128], -float("inf"))
                lo_i, hi_i = program * 128 + low, program * 128 + high + 1
                mask[lo_i:hi_i] = 0
                logits[NOTE_OFFSET:NOTE_OFFSET + 129 * 128] = logits[NOTE_OFFSET:NOTE_OFFSET + 129 * 128] + mask
                return logits

            sample.instr_logits = only_target
            try:
                for a, b in windows:
                    t0 = a
                    while t0 < b - 0.5:
                        t1 = min(b, t0 + max_seconds)
                        base = max(0.0, t0 - 0.0)
                        controls: list[int] = []
                        for n in analysis.get("melody_midi") or []:
                            if t0 <= n["start"] < t1:
                                controls += [CONTROL_OFFSET + x for x in ev(n["start"] - base, min(n["end"], t1) - base, n["pitch"], 52)]
                        for c in analysis.get("chords") or []:
                            if c.get("root_pc") is None or c["end"] <= t0 or c["start"] >= t1:
                                continue
                            s, e = max(c["start"], t0) - base, min(c["end"], t1) - base
                            root = 48 + c["root_pc"]
                            third = 3 if c.get("quality") == "min" else 4
                            for p in (root - 12, root, root + third, root + 7):
                                controls += [CONTROL_OFFSET + x for x in ev(s, e, p, 0)]
                        controls = ops.sort(controls) if controls else []
                        inputs: list[int] = []
                        start_gen = 0.0
                        if prompt_notes:
                            # Seed with the first bar of the rule-based part so style/register carry over.
                            seed_notes = [n for n in prompt_notes if t0 <= n.start < t0 + 2.0]
                            for n in seed_notes:
                                inputs += ev(n.start - base, n.end - base, n.pitch, program)
                            start_gen = 2.0 if seed_notes else 0.0
                        torch.manual_seed(seed)
                        events = sample.generate(model, start_gen, t1 - base, inputs=inputs, controls=controls, top_p=top_p)
                        events = ops.unpad(events) if hasattr(ops, "unpad") else events
                        for tt, dd, nn in zip(events[0::3], events[1::3], events[2::3]):
                            if nn >= CONTROL_OFFSET or nn < NOTE_OFFSET:
                                continue
                            instr, pitch = divmod(nn - NOTE_OFFSET, 128)
                            if instr != program:
                                continue
                            s = (tt - TIME_OFFSET) / TIME_RESOLUTION + base
                            e = s + (dd - DUR_OFFSET) / TIME_RESOLUTION
                            if t0 <= s < t1:
                                out.append(PNote(s, min(e, t1), int(pitch), 80))
                        t0 = t1
            finally:
                sample.instr_logits = original
        return sorted(out, key=lambda n: n.start)


class MusicGenGenerator:
    """MusicGen-melody: text prompt + chroma of a melody/condition signal -> audio (32 kHz mono)."""

    name = "musicgen"

    def __init__(self, model_id: str, device: str = "cuda"):
        self.model_id, self.device = model_id, device
        self._model = self._proc = None
        self._lock = threading.Lock()

    def generate(self, prompt: str, condition: np.ndarray, sr: int, seconds: float, seed: int = 0, guidance: float = 3.0) -> tuple[np.ndarray, int]:
        import torch
        from transformers import AutoProcessor, MusicgenMelodyForConditionalGeneration

        with self._lock:
            if self._model is None:
                self._proc = AutoProcessor.from_pretrained(self.model_id)
                self._model = MusicgenMelodyForConditionalGeneration.from_pretrained(self.model_id, torch_dtype=torch.float16).to(self.device).eval()
            out_sr = int(self._model.config.audio_encoder.sampling_rate)
            window, overlap = 29.0, 2.0  # MusicGen's 30 s limit; independent chunks are crossfaded over 2 s
            total = int(seconds * out_sr)
            y = np.zeros(total, np.float32)
            t = 0.0
            torch.manual_seed(seed)
            first = True
            while t < seconds - 0.1:
                seg_len = min(window, seconds - t)
                cond = dsp.resample(dsp.as_mono(condition[..., int(t * sr):int((t + seg_len) * sr)]), sr, out_sr)
                inputs = self._proc(audio=cond, sampling_rate=out_sr, text=[prompt], padding=True, return_tensors="pt").to(self.device)
                if "input_features" in inputs:
                    inputs["input_features"] = inputs["input_features"].to(torch.float16)
                tokens = int(seg_len * self._model.config.audio_encoder.frame_rate) + 2
                with torch.no_grad():
                    audio = self._model.generate(**inputs, do_sample=True, guidance_scale=guidance, max_new_tokens=tokens)
                seg = audio[0, 0].float().cpu().numpy()[: int(seg_len * out_sr)]
                s0 = int(t * out_sr)
                seg = seg[: total - s0]
                if not first:
                    xf = min(int(overlap * out_sr), len(seg))
                    ramp = np.linspace(0, 1, xf, dtype=np.float32)
                    y[s0:s0 + xf] = y[s0:s0 + xf] * (1 - ramp) + seg[:xf] * ramp
                    y[s0 + xf:s0 + len(seg)] = seg[xf:]
                else:
                    y[s0:s0 + len(seg)] = seg
                first = False
                if t + seg_len >= seconds - 0.1:
                    break
                t += seg_len - overlap
        return y.astype(np.float32), out_sr
