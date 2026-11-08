"""``python -m sam_audio_service`` runs the API with uvicorn (default port 8001)."""
import os

import uvicorn

if __name__ == "__main__":
    uvicorn.run("sam_audio_service.api:app", host=os.getenv("AUDIO_HOST", "0.0.0.0"), port=int(os.getenv("AUDIO_PORT", "8001")), workers=1)
