#!/usr/bin/env python3
"""Render authored dialogue with Apache-2.0 Kokoro weights on CPU only.

Install kokoro-onnx in a separate local environment. See its official README for
the model-files-v1.1 release's kokoro-v1.0.onnx and voices-v1.0.bin downloads.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import wave

import numpy as np
import onnxruntime as rt
from kokoro_onnx import Kokoro

LINES = [
    ('speaker_a', 'af_heart', 'en-us', 'Did you bring the camera? We should record the demo before lunch.'),
    ('speaker_b', 'bm_george', 'en-gb', "Yes, it is ready. Let's keep talking while our dog says hello."),
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--voices', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    options = rt.SessionOptions()
    options.intra_op_num_threads = 4
    session = rt.InferenceSession(str(a.model), sess_options=options, providers=['CPUExecutionProvider'])
    model = Kokoro.from_session(session, str(a.voices))
    renders = []
    for name, voice, lang, text in LINES:
        samples, rate = model.create(text, voice=voice, speed=1.15, lang=lang)
        with wave.open(str(a.output/f'{name}.wav'), 'wb') as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(rate)
            f.writeframes((np.clip(samples, -1, 1)*32767).astype('<i2').tobytes())
        renders.append(dict(speaker=name,voice=voice,text=text,lang=lang,speed=1.15,seconds=len(samples)/rate))
    metadata = {
        'model':'Kokoro-82M v1.0', 'license':'Apache-2.0',
        'model_url':'https://huggingface.co/hexgrad/Kokoro-82M',
        'runtime_url':'https://github.com/thewh1teagle/kokoro-onnx',
        'runtime_license':'MIT', 'runtime_version':importlib.metadata.version('kokoro-onnx'),
        'provider':session.get_providers(), 'renders':renders,
        'model_sha256':hashlib.sha256(a.model.read_bytes()).hexdigest(),
        'voices_sha256':hashlib.sha256(a.voices.read_bytes()).hexdigest(),
    }
    (a.output/'speech-provenance.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata,indent=2))


if __name__ == '__main__':
    main()
