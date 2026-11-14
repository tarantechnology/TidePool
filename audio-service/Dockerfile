# SoundCanvas audio inference service (Person 1).
# Targets (audiosep is the final/default image):
#   audiosep: official text-only AudioSep, CUDA, cached public checkpoints.
#   mock : CPU-only image for local integration/CI (AUDIO_BACKEND=mock). No model weights.
#   sam  : CUDA image with SAM Audio installed from GitHub. Checkpoints are gated; mount a
#          Hugging Face cache or pass HF_TOKEN at runtime (never bake tokens into the image).
ARG BASE=python:3.12-slim

FROM ${BASE} AS mock
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
WORKDIR /srv
COPY pyproject.toml README.md ./
COPY sam_audio_service ./sam_audio_service
RUN pip install --no-cache-dir .
ENV AUDIO_BACKEND=mock AUDIO_PORT=8001 AUDIO_HOST=0.0.0.0
EXPOSE 8001
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8001/ready').status==200 else 1)"
CMD ["python", "-m", "sam_audio_service"]

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS sam
RUN apt-get update && apt-get install -y --no-install-recommends python3.11 python3.11-venv python3-pip ffmpeg git && rm -rf /var/lib/apt/lists/*
WORKDIR /srv
RUN python3.11 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
COPY pyproject.toml README.md ./
COPY sam_audio_service ./sam_audio_service
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir ".[sam]"
ENV AUDIO_BACKEND=sam AUDIO_MODEL_ID=facebook/sam-audio-large AUDIO_DEVICE=auto AUDIO_PORT=8001 AUDIO_HOST=0.0.0.0 HF_HOME=/models
VOLUME ["/models"]
EXPOSE 8001
CMD ["python", "-m", "sam_audio_service"]

# Default production separator. The official public checkpoints are downloaded
# lazily into HF_HOME and reused across container restarts.
FROM python:3.12-slim AS audiosep
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /srv
COPY pyproject.toml README.md ./
COPY sam_audio_service ./sam_audio_service
COPY patches/audiosep-checkpoint-compat.patch /tmp/audiosep-compat.patch
RUN git clone https://github.com/Audio-AGI/AudioSep.git /opt/audiosep \
 && git -C /opt/audiosep checkout 944583f18b84589dc965de3ad77525c945334252 \
 && git -C /opt/audiosep apply /tmp/audiosep-compat.patch \
 && pip install --no-cache-dir '.[audiosep]'
ENV AUDIO_BACKEND=audiosep AUDIO_AUDIOSEP_ROOT=/opt/audiosep AUDIO_MODEL_ID=audiosep-base-4M \
    AUDIO_DEVICE=cuda AUDIO_PORT=8001 AUDIO_HOST=0.0.0.0 HF_HOME=/models \
    AUDIO_CHUNK_BATCH=1 AUDIO_SUPPRESSION=conservative AUDIO_SUPPRESSION_STRENGTH=0.9 \
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
VOLUME ["/models"]
EXPOSE 8001
CMD ["python", "-m", "sam_audio_service"]
