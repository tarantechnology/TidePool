# Repeated-event timing smoke test — A40

Run `replacement_timing.py` against the previously curated `shared-rendered` editor
fixture manifest with seed 20261114. This four-case test makes three **identical copies**
of a 250 ms high-energy excerpt from each known target stem and mixes them into its
background. Exact repetition favors example matching. It is not a substitute for varied
natural events or human-reviewed timing, and the source split names do not make this a
standalone held-out evaluation of the new thresholds. The JSON records file hashes,
queries, timings and individual results.

At 150 ms tolerance, mixture-energy peak picking matched 12/12 inserted samples with
84 extra detections; spectral-onset candidates matched 12/12 with 119 extra; matching
one example against candidates matched 12/12 with 10 extra (0 dog, 2 sneeze, 1 fire,
7 clock). FlexSED text-conditioned regions covered 6/12 (dog and sneeze) and covered
none of the short fire or clock excerpts at the fixed 0.5 threshold. Counts are not
comparable to a whole-video event detector's precision, and model scores are not
calibrated confidence. The FlexSED check and event-example matcher remain experimental;
users must review placements.

On the user's 48.7-second cooking clip, a separate FlexSED query for knife chopping
produced high-scoring regions around 3.9–4.0, 4.24–4.36 and several other times;
YAMNet's existing chopping category returned no regions. These are unverified model
outputs, not ground truth. One isolated model run took 17.59 s to load and 3.09 s to
infer with peak allocated GPU memory 661,337,088 bytes. A later offline subprocess run
reported 7.69 s total and 453,180,928 bytes. Model load and shared-Pod contention vary.
No real-video listening scores, separation quality improvement, or commercial-use
clearance beyond the model's published license are claimed.
