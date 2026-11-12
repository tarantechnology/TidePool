# Guided separation benchmark

`guided.py` compares text-only, an explicit positive anchor, two judge-ranked candidates,
20-second windows at batch one, and independent-channel separation. Defaults remain
10-second windows, 2-second overlap, batch two, one candidate, SAM Audio Large.
Automatic span prediction remains disabled. The independent-channel mode is an experiment;
it can change localization or assign different sources to left and right channels.

## Reproduce without labels or training

Install this service plus `pyarrow`, `soundfile`, and `httpx`. Supply the existing
LibriSpeech dummy validation parquet used by `real_audio.py`:

```sh
python -m benchmarks.prepare_guided --speech-parquet /path/to/librispeech-dummy.parquet \
  --out /path/to/guided-corpus
python -m benchmarks.guided --manifest /path/to/guided-corpus/manifest.json \
  --url http://127.0.0.1:18001/separate --out /path/to/guided-results
```

The preparation script retrieves existing recordings from the
[librosa example collection](https://librosa.org/doc/main/recordings.html), two speech
recordings from [LibriSpeech](https://www.openslr.org/12/), and the dog/rain categories of
[ESC-10](https://github.com/karolpiczak/ESC-50/blob/master/LICENSE). It preserves the
individual attribution/license texts in the manifest and sidecar files. It never invents
licenses or requires human annotations. The clean source is the reference. The manifest
records source hashes, fixed seeds, source placement, category and tuning/held-out split.
Do not remove the provenance when sharing derived audio.

The initial corpus has ten 12-second mixtures: one tuning and one held-out example for
speech/music, speech/environmental noise, intermittent effects, competing voices, and
stereo. Short recordings are repeated deterministically. Tuning speech overlaps may use
the same speaker; held-out competing voices use two male narrators. Recordings are disjoint
between splits, but reused within a split. This is a small smoke benchmark, not enough
coverage to establish general performance. Expand the manifest with more licensed sources
before promotion. No manually drawn labels, model fitting or new paid service is required.

Metrics include per-channel target and retained-background SI-SDR improvement, maximum
reconstruction error over both channels, error RMS near window boundaries, request latency,
and service-reported peak allocated GPU bytes. Returned WAVs and references are retained
for listening review. Missing/nonfinite scores and failed requests cannot pass the gate.
GPU allocated bytes are distinct from driver-reserved memory; monitor both during rollout.

Every held-out category must have no more than a 0.5 dB median regression relative to the
matched text baseline, for both target and background. The script never auto-promotes a
profile, even when the numerical gate passes. Review audible leakage, missing background,
window artifacts, peak memory and runtime. Judge scores are preferences, not confidence.
The earlier tiny benchmark's 6.88 versus 6.89 dB averages remain a baseline, not evidence
that ranking solved separation.

## GPU staging and rollback

Before stopping a shared service, inspect its backend SQLite database for queued/processing
jobs, inspect GPU use, and save the exact launcher/configuration. Use an isolated code copy
and project storage. Avoid loading two Large models together on the A40. Test the staged
service, record actual options (including sequential OOM fallback), then restart the prior
launcher and verify `/ready` and model identity. Do not overwrite project media or export
credentials into test reports. Keep the automatic span-predictor experiment separate:
loading extra weights changes memory requirements, so a failed load must roll back before
further work. Do not enable it from the editor until it both fits and improves intermittent
sound cases on held-out recordings.

Visual execution is measured separately using existing video/mask assets. Successful
mask decoding, frame alignment, preview/apply/undo and export do not establish whether the
visible object generated the selected sound. Visual guidance stays experimental.

## Recorded run

[The A40 validation report](results/guided-a40/VALIDATION.md) includes 50 successful
requests, category scores, memory/timing, visual execution and the separate automatic-span
fit check. Longer windows and independent-channel stereo failed the numerical regression
gate on this small set and remain unpromoted. The optional span predictor fit with rankers
disabled, but the two-example result does not isolate its quality effect; it stays off.
