# SoundCanvas audio inference service (Person 1)

Text-, span- and visually-prompted sound separation with Meta's SAM Audio, exposed as a
small HTTP service that Person 4's backend calls. See `docs/audio_integration_handoff.md`
at the repository root for the cross-team contract and open questions.

## Run locally (mock backend, no GPU)

```bash
cd audio-service
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m sam_audio_service        # http://localhost:8001/docs
.venv/bin/python -m pytest -q
```

Point Person 4's backend at it with `INFERENCE_ADAPTER=http SAM_AUDIO_SERVICE_URL=http://localhost:8001/separate`,
or run both with the compose overlay: `docker compose -f compose.yaml -f audio-service/compose.audio.yaml up --build`.

## Real model

```bash
uv pip install --python .venv/bin/python -e ".[sam]"   # Python >= 3.11, CUDA GPU recommended
hf auth login                                           # after access is approved on huggingface.co/facebook/sam-audio-large
AUDIO_BACKEND=sam AUDIO_MODEL_ID=facebook/sam-audio-large .venv/bin/python -m sam_audio_service
```

The mock backend (`AUDIO_BACKEND=mock`) is a band-pass filter, not separation; every response
produced by it carries `model.is_mock=true` and a warning.
