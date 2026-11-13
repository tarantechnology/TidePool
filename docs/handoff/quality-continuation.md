# Handoff: finish the audio-separation quality work on a GPU host

Branch: `audio/quality-sweep` (pushed; main was `6e1f421` when it was cut). Everything below is
committed on that branch; nothing is pending on the Mac except gitignored local files
(`.env`, venvs, `benchmarks/out/`).

## Goal
Improve SAM Audio separation quality and, above all, remove the faint/muffled background that
is still audible inside an extracted stem. Measure first, ship winners as service + backend
defaults, no new editor controls. Deliver a Claude artifact (HTML page) explaining what was
done and measured, then merge the PR.

## What is already done (all tested: audio-service 325 tests, backend 114 tests)
- `audio-service/benchmarks/`: `run_benchmark.py` can sweep settings (`--sweep NAME`, named
  sweeps in `sweeps.py`: ode, chunks, refine, refine-floor, leakage, gate, candidates,
  quality-v1); flags for seed, chunk length/overlap, ODE step, candidates, refine options,
  model id; new 24 s `competing_voices` fixture; `SWEEP.md` tables with Δ vs baseline.
- `sam_audio_service/refine.py`: time-frequency mask refinement (`options.refine="mask"`,
  `refine_p`, `refine_n_fft`, `refine_floor`, `refine_blend`, `refine_threshold`,
  `refine_gate_db`). Default is `mask` for the real backend, `off` for mock (config.py).
- `sam_audio_service/candidates.py` + engine/backends: `options.candidates=K` returns K noise
  draws per chunk window; `candidate_select` = first | overlap | mean | auto (auto = mean for a
  single-window signal, overlap-consistent pick when chunked).
- Backend `app/guidance.py`: preview profile = mask refine (threshold 0); quality profile =
  candidates 2 auto + mask refine with threshold 0.5. `app/media.py`: quality retry no longer
  needs the text ranker (`quality_mode` = ranker | consistency). Env overrides:
  QUALITY_CANDIDATES, QUALITY_CANDIDATE_SELECT, QUALITY_REFINE*, PREVIEW_REFINE*, QUALITY_RERANKING.
- `scripts/runpod_setup.sh` (setup|start|stop|status|bench on a CUDA host),
  `scripts/run_local_real.sh` (real|mock|remote|stop on the Mac).
- Results so far: `audio-service/benchmarks/results/2026-11-13-quality-sweep/` (sam-audio-small,
  Apple M1 Max MPS). Headline, mean over 4 fixtures: baseline 11.35 dB SI-SDRi / −16.9 dB
  leakage; mask p=2 n_fft=2048: 14.61 / −20.4 (every fixture improved, free); + threshold 0.5:
  12.30 / −22.3; ODE steps: no effect; 20 s chunks collapse competing voices by 10 dB, K=2
  overlap selection repairs it (0.07 → 9.94 dB); mean ensemble K=3 gains 1.7-1.9 dB on
  single-window clips but hurts chunked ones (hence `auto`); 30 dB gate never triggers.

## Remaining steps
1. On the H100 VM (any Ubuntu/Debian with NVIDIA drivers): `HF_TOKEN=... bash scripts/runpod_setup.sh audio/quality-sweep`
   (installs, prefetches sam-audio-large on ≥40 GB GPUs). Work dir is `/workspace` if it
   exists, else `$HOME/soundcanvas-work` (override with `WS=`); the CUDA venv is `$WS/venv`.
   Confirm `python -m pytest -q` in `audio-service/` with that venv.
2. Sweeps with `bash scripts/runpod_setup.sh bench --sweep NAME --out /workspace/bench/NAME`
   (use `--speech-mode synthetic` on Linux; macOS `say` is unavailable):
   `gate` (interrupted on the Mac after 2 of 4 configs), `quality-v1` (final head-to-head),
   then `--model-id facebook/sam-audio-large` for `quality-v1`, and judge re-ranking
   (`AUDIO_LOAD_RANKERS=true AUDIO_JUDGE_ONLY=true AUDIO_LOAD_VISUAL_RANKER=false --candidates 2`).
   Gate: no fixture down > 0.5 dB, leakage not worse, recon ≤ 1e-6; `two_speakers` and
   `competing_voices` decide.
3. Adopt winners: service defaults in `config.py` (only if they win everywhere), backend profile
   values in `app/guidance.py`, launcher env in `scripts/*.sh`. Keep the H100 recommended
   config in `audio-service/README.md` (chunk 30 s, large, candidates).
4. Record: copy SWEEP.md/sweep.json into `benchmarks/results/<date>-quality-sweep-cuda/`, add
   rows to `benchmarks/results/README.md`, write `audio-service/docs/quality_tuning.md`
   (what each knob does, measured numbers, cost, recommended profiles).
5. Verify end to end: audio service on the GPU host, backend with `INFERENCE_ADAPTER=http`,
   `audio-service/scripts/e2e_compose.py --backend http://127.0.0.1:8000 --video <clip> --prompt
   "woman speaking"`; check `model.refine` / `model.candidates` in the job report; run root
   `pytest -q tests` and frontend `npm test`.
6. Open the PR from `audio/quality-sweep` (gh), merge when CI is green, then publish the
   artifact (Artifact tool, HTML): what changed, before/after tables, cost per setting, how to
   use the quality profile, and the H100 deployment recipe.

## Working agreements (from the owner)
- Open the PR early and push as work lands. Use Opus agents for coding, Sonnet for testing,
  in isolated worktrees; remove worktrees afterwards.
- Be mindful of disk and memory; delete what is not needed.
- Do not ask permission for reversible steps; report test output faithfully.
- Never print the Hugging Face token; access to facebook/sam-audio-small, -base, -large and
  -judge is approved on the owner's account (TaranTech).
