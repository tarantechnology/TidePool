"""SoundCanvas audio inference service (workstream: Person 1, Audio AI Pipeline).

Owns SAM Audio model loading and inference, audio preprocessing for the model,
target/residual stem generation and the audio inference HTTP API. It does not own
uploads, storage, the application job queue, object detection, the editor or export.
"""

__version__ = "0.1.0"
CONTRACT_VERSION = "2025-11-draft-1"
