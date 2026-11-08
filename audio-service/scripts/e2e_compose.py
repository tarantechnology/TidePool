#!/usr/bin/env python
"""Cross-team Test 3/4 against the running compose stack (Person 4 backend + this service).

Bring the stack up first::

    docker compose -f compose.yaml -f audio-service/compose.audio.yaml up -d --build

then run ``python audio-service/scripts/e2e_compose.py [--backend http://127.0.0.1:8000] [--video clip.mp4]``.
Walks Person 4's documented API: create project -> upload -> visual metadata -> /process ->
poll -> fetch stem -> automation -> natural-language webhook -> MP4 export -> ffprobe. With
``AUDIO_BACKEND=mock`` in the stack this verifies plumbing and timing, not separation quality.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx


def synth_demo(path: Path, seconds: float = 6.0) -> Path:
    audio = ("aevalsrc=0.3*sin(2*PI*220*t)*(1+0.5*sin(2*PI*3*t))+0.3*sin(2*PI*330*t)*(1+0.5*sin(2*PI*2.3*t))"
             f"+0.3*sin(2*PI*(900+300*sin(2*PI*1.5*t))*t):s=48000:c=stereo:d={seconds}")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c=blue:s=320x240:r=24:d={seconds}", "-f", "lavfi", "-i", audio,
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path)], check=True)
    return path


def poll(client: httpx.Client, url: str, headers: dict, timeout: float) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        data = client.get(url, headers=headers).json()
        if data["status"] in ("completed", "failed"):
            return data
        time.sleep(0.25)
    raise SystemExit(f"timed out polling {url}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="http://127.0.0.1:8000")
    ap.add_argument("--video", type=Path, default=None, help="MP4 to upload; a synthetic two-voices+siren clip when omitted")
    ap.add_argument("--prompt", default="siren")
    ap.add_argument("--timeout", type=float, default=600.0)
    args = ap.parse_args(argv)
    with tempfile.TemporaryDirectory() as tmp, httpx.Client(timeout=120) as c:
        video = args.video or synth_demo(Path(tmp) / "demo.mp4")
        p = c.post(f"{args.backend}/api/projects").json()
        pid, h = p["project_id"], {"Authorization": "Bearer " + p["access_token"]}
        with video.open("rb") as f:
            r = c.post(f"{args.backend}/api/projects/{pid}/upload", headers=h, files={"file": (video.name, f, "video/mp4")})
        r.raise_for_status()
        source = r.json()["metadata"]
        print("upload:", source)
        c.put(f"{args.backend}/api/projects/{pid}/visual-metadata", headers=h, json={"object_id": "person_2", "label": "person", "time_range": {"start": 1.0, "end": 4.0}}).raise_for_status()
        job_id = c.post(f"{args.backend}/api/projects/{pid}/process", headers=h, json={"prompts": [args.prompt]}).json()["job_id"]
        job = poll(c, f"{args.backend}/api/jobs/{job_id}", h, args.timeout)
        print("job:", job["status"], job.get("error"), [(s["name"], round(s["duration"], 3)) for s in job.get("stems", [])])
        if job["status"] != "completed":
            return 1
        target = next(s for s in job["stems"] if s["name"] == args.prompt)
        audio = c.get(args.backend + target["audio_url"], headers=h)
        print("stem audio:", audio.status_code, len(audio.content), "bytes")
        c.patch(f"{args.backend}/api/projects/{pid}/stems/{target['id']}", headers=h, json={"automation": [{"time": 0, "value": 1}, {"time": 2, "value": 0.1}, {"time": 4, "value": 1}]}).raise_for_status()
        print("webhook:", c.post(f"{args.backend}/api/webhooks/local", json={"project_id": pid, "token": p["access_token"], "text": f"Lower the {args.prompt} volume to 30%"}).json()["status"])
        eid = c.post(f"{args.backend}/api/projects/{pid}/export", headers=h, json={"format": "mp4", "include_video": True}).json()["export_id"]
        export = poll(c, f"{args.backend}/api/exports/{eid}", h, args.timeout)
        print("export:", export["status"], export.get("error"))
        if export["status"] != "completed":
            return 1
        out = Path(tmp) / "export.mp4"
        out.write_bytes(c.get(args.backend + export["download_url"], headers=h).content)
        probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type,codec_name", "-of", "csv=p=0", str(out)], capture_output=True, text=True).stdout
        print("export probe:", probe.strip().replace("\n", " | "))
        ok = abs(float(probe.strip().splitlines()[-1]) - source["duration"]) < 0.1 and "video" in probe
        print("RESULT:", "PASS" if ok else "FAIL (duration or video stream mismatch)")
        return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
