# A40 guided separation validation — 2026-11-12

SAM Audio Large, float32, one A40. The original production launcher was saved before staging. Test project media and reports were isolated from existing projects. No model training or manual labels were used.

## Automated checks

- Frontend: TypeScript, lint, build, 433 passing tests; 8 skipped.
- Backend/audio: 298 passing tests, 2 opt-in browser skips, 1 pre-existing expected failure.
- Browser workflow passed separately against local fake inference and the real staged A40 API: upload, time guidance, timed mute preview, synchronized comparison, apply, replacement/quality retry, apply/undo, remote timed mute, and MP4 audio/video export.
- An existing pedestrian video/mask asset passed real visual execution, preview/apply/undo and authorization checks. Visual accuracy remains unverified.

## Benchmark results

Ten 12-second mixtures; one tuning and one held-out example per category, five settings each (50 successful requests). This corpus is too small for general quality claims. No listening review has been completed. All references, exact attribution, hashes, seed and options are in [report.json](report.json). WAVs are retained on the Pod under `/workspace/sc-guided-results` for listening.

| Profile | Median request time, all cases | Peak allocated GPU memory | Numerical held-out gate |
| --- | ---: | ---: | --- |
| text | 8.44 s | 24.47 GiB | baseline |
| anchor | 7.14 s | 23.09 GiB | passes this small set |
| judge | 15.43 s | 27.22 GiB | passes this small set |
| long | 10.20 s | 25.26 GiB | fails |
| stereo | 8.75 s | 24.47 GiB | fails |

Request timings include occasional contention with staged browser/visual tests; treat them as observed request latency, not an isolated throughput comparison.

Held-out target SI-SDR improvement relative to each constructed mixture (dB):

| Category | Text | Anchor | Judge (2) | 20 s / batch 1 | Per-channel |
| --- | ---: | ---: | ---: | ---: | ---: |
| competing_voices | -13.97 | -14.28 | 12.26 | -40.85 | -13.97 |
| intermittent | 12.06 | 13.48 | 12.41 | 13.04 | 12.06 |
| speech_music | 11.42 | 12.11 | 11.53 | 11.88 | 11.42 |
| speech_noise | 9.28 | 10.41 | 9.76 | 10.65 | 9.28 |
| stereo | 10.12 | 10.45 | 10.19 | 10.10 | 8.63 |

Maximum target + residual reconstruction error across all channels: 5.96e-08.

The longer-window candidate regressed 26.88 dB on the held-out competing-voice example; independent-channel separation regressed 1.50 dB in target quality on the stereo example. These settings are not promoted. The same-speaker tuning mixtures remain difficult. The large judge gain on one held-out voice mixture is not a calibrated confidence score or evidence of reliable speaker identification.

The default remains 10 s / 2 s overlap / batch two / one candidate. Quality retry is an explicit, reviewable two-candidate attempt. Visual guidance stays experimental. Automatic-span results and restoration status are recorded separately.

## Automatic span experiment

The pretrained span predictor loaded successfully with rankers disabled. Two intermittent
examples completed at batch one, using at most 22.97 GiB of allocated GPU memory during
inference. Target SI-SDR improvements were 2.02 dB (tuning) and 14.02 dB (held-out). See
[span-report.json](span-report.json). Because batch size and loaded auxiliaries differ from
the main benchmark, this is a memory-fit and execution check, not an isolated measurement
of the predictor's quality gain. It remains disabled by default and is not an editor option.

## Restoration

The previous production launcher was restored unchanged and `/ready` returned HTTP 200
with SAM Audio Large, the text judge loaded, and the span predictor disabled. See
[restoration.json](restoration.json). Production application code was not overwritten;
implementation was exercised in an isolated staging copy. Temporary local test servers
and the staging tunnel were stopped. The user's existing SSH tunnel was preserved.
