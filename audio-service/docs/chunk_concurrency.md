# Chunk windows: why they are batched, not threaded

Status: resolved. This replaces the "Known regression (investigation needed)" note PR #18
left in `audio-service/README.md`.

## What PR #18 did

PR #18 added `AUDIO_CHUNK_WORKERS` and used it two ways:

* `sam_audio_service/engine.py` ran the overlapping chunk windows of one request on a
  `ThreadPoolExecutor`, calling `backend.separate()` from up to `AUDIO_CHUNK_WORKERS`
  threads at a time and merging results in timeline order;
* `backends/sam_audio.py` wrapped each of those calls in its own CUDA stream
  (`_separate_on_stream`), on the theory that the default stream would otherwise serialise
  the GPU work.

Both threads hit **one shared `SAMAudio` module and one shared `SAMAudioProcessor`**: the
backend loads them once and keeps them in `self._model` / `self._processor`.

## The failure

On a RunPod A40 with `facebook/sam-audio-large`, float32, `AUDIO_CHUNK_SECONDS=10`,
`AUDIO_CHUNK_OVERLAP_SECONDS=2`, `AUDIO_CHUNK_WORKERS=2`, span predictor and rankers
disabled, a real video failed with:

```text
INFERENCE_FAILURE: RuntimeError: Given groups=1, weight of size [16, 32, 3],
expected input[2, 1, 480002] to have 32 channels, but got 1 channels instead
```

## Root cause

A single mechanism explains the error, and every number in the message. It is a data race
in the DAC-VAE **decoder**, not a shape bug in our input pipeline.

### 1. Where a `Conv1d` with weight `[16, 32, 3]` lives

There are exactly two such convolutions in the loaded model, both in the audio codec's
decoder:

```
SAMAudio.audio_codec.decoder.model[4].block[6].block[1]
SAMAudio.audio_codec.decoder.model[4].block[7].block[1]
```

They are `NormConv1d` layers (`dacvae/nn/layers.py:54`) inside `ResidualUnit`
(`dacvae/model/dacvae.py:25-84`) inside the last `DecoderBlock`, constructed at
`dacvae/model/dacvae.py:211-212`:

```python
ResidualUnit(output_dim // downsampling_factor, kernel=3, act="ELU", compress=2,
             causal=True, pad_mode="auto", norm="none", true_skip=True),
```

With the SAM Audio checkpoint config the last decoder block has `output_dim = 96` and
`downsampling_factor = 3`, so `dim = 32`, `hidden = dim // compress = 16`, and the inner
convolution's weight is `[out=16, in=32, k=3]`. **It expects 32 channels.**

For contrast, the codec *encoder* starts at `NormConv1d(1, 64, kernel_size=7)`
(`dacvae/model/dacvae.py:131`) and grows 64 → 128 → 256 → 512 → 1024. A raw `[B, 1, T]`
waveform cannot reach a 32-channel layer through the encoder at all; the encoder's first
conv is the one that *wants* 1 channel.

### 2. The one path that feeds those convolutions

They are reached only from `Decoder.watermark()` (`dacvae/model/dacvae.py:454-479`), which
runs on every decode: `Decoder.forward` ends with `return self.watermark(x, message)`
(`:448-451`), `blending` defaults to `"linear"` (`:426`) and `alpha = wm_channels / d_wm_out =
32 / 128 = 0.25`, so the `alpha == 0.0` early return never fires.

`watermark()` starts from the decoder's finished **1-channel waveform** `x` and lifts it to
32 channels:

```python
h = self.wm_model.encoder_block(x)          # :457  -> [B, 32, T], via NormConv1d(1, 32, 7)
upsampler = map(lambda x: x.upsample_group(), self.model[1:])
for layer in upsampler:
    h = layer(h)                            # :459-461  -> ResidualUnit(32) -> NormConv1d(32, 16, 3)
...
return self.wm_model.encoder_block.forward_no_conv(x) + self.alpha * h   # :479
```

### 3. The race: in-place module surgery during `forward`

Line 479 calls `WatermarkEncoderBlock.forward_no_conv`
(`dacvae/model/dacvae.py:326-332`):

```python
def forward_no_conv(self, x):
    _conv = self.pre[-1]
    try:
        self.pre[-1] = nn.Identity()
        return self.pre(x)
    finally:
        self.pre[-1] = _conv
```

It **mutates the shared module in place** for the duration of its own forward pass,
swapping the `NormConv1d(1, 32, 7)` at `pre[-1]` for an `nn.Identity()`.

Line 457 of the *same* method needs that convolution to be the real one. So with two
threads inside `model.separate(...)` → `audio_codec.decode(...)` on one shared model:

* thread A is between `self.pre[-1] = nn.Identity()` and the `finally`;
* thread B executes line 457, `encoder_block(x)` becomes a no-op on the channel axis and
  returns the **1-channel** waveform instead of a 32-channel feature map;
* thread B's next layer is the `ResidualUnit(32)`, i.e. `NormConv1d(32, 16, 3)`, which
  raises exactly the reported error.

This was reproduced directly against the real `dacvae` decoder built with the SAM Audio
config (one thread holding `pre[-1] == Identity`, another calling `encoder_block`):

```text
encoder_block(x96)                              -> (1, 32, 40)
encoder_block(x96) WHILE pre[-1] == Identity    -> (1,  1, 40)
RuntimeError: Given groups=1, weight of size [16, 32, 3],
expected input[1, 1, 42] to have 32 channels, but got 1 channels instead
```

### 4. Every number in the production message

| Reported | Why |
| --- | --- |
| `weight of size [16, 32, 3]` | `NormConv1d(32, 16, 3)` in the last decoder block's `ResidualUnit(32)` (`dacvae.py:211`). |
| `got 1 channels` | `pre[-1]` was `nn.Identity()` in the other thread, so the 1-channel waveform passed through unchanged. |
| batch `2` | Not two threads merging. `SAMAudio.separate` decodes target and residual together: `self.audio_codec.decode(generated_features.reshape(2 * B, C, T))` (`sam_audio/model/model.py:305-307`). One window (`B = 1`, `reranking_candidates = 1`) decodes as batch 2. |
| `480002` samples | The window was `AUDIO_CHUNK_SECONDS=10` at 48 kHz = 480 000 samples, exactly 250 codec frames, so the decoded waveform is 480 000 samples. `NormConv1d.pad` with `kernel_size=3`, `stride=1`, `causal=True` prepends `padding_total = 2` (`dacvae/nn/layers.py:89-108`), and the error prints the padded tensor: 480 000 + 2. |

So the failing tensor is the first full 10 s window's decoded waveform, in a thread whose
watermark convolution had been swapped out from under it by the other worker.

### 5. Secondary hazards in the same design

Serialising fixes these too; they are listed because they rule out "just retry" as a fix.

* **Permanent model corruption.** If two threads enter `forward_no_conv` at once, the
  second can capture `_conv = nn.Identity()` and its `finally` then installs `Identity`
  *permanently*. The model is silently broken for the rest of the process, with no
  exception — every later separation returns wrong audio.
* **Process-global cuDNN flag flipped mid-forward.** `sam_audio/model/codec.py:66` and
  `:87` wrap the codec in `torch.backends.cudnn.flags(enabled=False)`, which is global and
  not thread-local. One thread leaving the block re-enables cuDNN while another is still
  inside the codec, defeating the authors' deliberate disable.
* **`module.weight` rewritten on every forward.** `dacvae/nn/layers.py:9-14` deliberately
  pins the deprecated `torch.nn.utils.weight_norm` instead of
  `torch.nn.utils.parametrizations`; its forward pre-hook does
  `setattr(module, "weight", compute_weight(module))` on ~90 convolutions per pass.
  Concurrent threads overwrite each other's `weight` attribute between the hook and the
  kernel launch. We could not demonstrate a wrong result from this one, so it is reported
  as a genuine race of undemonstrated impact — unlike the `forward_no_conv` race above,
  which is reproduced.
* **Global RNG.** `noise = torch.randn_like(audio_features)`
  (`sam_audio/model/model.py:287`) draws from the default generator with no `Generator`
  argument, so concurrent calls interleave draws and a seed no longer reproduces a result.
* **Buffer reassigned during forward.** `self.freqs = self.freqs.to(device=t.device)`
  (`sam_audio/model/transformer.py:246`).

Upstream documents no thread-safety guarantee anywhere: a search for
`thread|lock|mutex|reentran|race` across `facebookresearch/sam-audio` and
`facebookresearch/dacvae` returns only incidental matches such as `EncoderBlock`.

### 6. Why the CUDA streams did not help, and could not

`_separate_on_stream` gave each thread its own CUDA stream. A stream only reorders *device*
work; the race above is entirely in Python, on CPU, in module attributes
(`self.pre[-1] = nn.Identity()`). Separate streams cannot serialise that. They also add a
hazard of their own: a tensor allocated on stream A and read on stream B needs explicit
event synchronisation, which nothing here performs.

The earlier synthetic 26-second concurrency test passed because the mock backend has no
shared `torch.nn.Module` — it never exercised the failure.

## Why batching is the correct GPU-side parallelism

Threads were the wrong tool for the goal. "Two windows at once" on one GPU does not need
two Python threads; it needs **one call with a batch dimension of two**, which is what the
upstream API already offers (verified in `sam_audio/processor.py` and
`sam_audio/model/model.py`):

* `SAMAudioProcessor.__call__(descriptions, audios, anchors=None, masked_videos=None)`
  takes `k` items. `batch_audio` (`processor.py:23-36`) averages each `(C, T)` tensor to
  mono with `wav.mean(0)` and `pad_sequence`-pads the batch to the longest item, returning
  `[k, 1, T_max]` plus per-item `wav_sizes`.
* `Batch.process_anchors` (`processor.py:78-124`) builds **per-item** anchor id rows and a
  per-item `anchor_alignment`, so each window keeps its own `+` / `-` spans.
* `model.separate(batch, noise=None, ode_opt=..., reranking_candidates=1,
  predict_spans=False)` runs one `odeint` solve and one `audio_codec.decode` for the whole
  batch, and returns `result.target` / `result.residual` as **lists of `k` 1-D tensors**
  (`model.py:344-350`). `unbatch` narrows each to `ceil(T_i / 1920) * 1920`, so every
  output is trimmed to **its own** input length, not the batch's padded length.

Batching therefore:

* enters the model exactly once per group of windows — no shared-state race is even
  expressible;
* keeps the GPU busier than threads could: the kernels are larger instead of interleaved,
  and the model weights are read once per batch rather than once per window;
* stays deterministic: one seeded draw per call, no interleaved RNG.

Visual windows are the exception and stay sequential, one window per call: `masked_videos`
is per item upstream and the frames must line up with exactly the audio they came from.
Anchored and unanchored windows are also not mixed in one call — an empty per-item anchor
list conditions identically to `anchors=None`, but `anchors is None` doubles as the
batch-wide "predict spans" gate in `SAMAudio.separate`, so mixing would silently drop span
prediction for windows that would otherwise get it.

## Implementation

* `backends/base.py` gains
  `separate_many(mixtures, prompts, options) -> list[BackendResult]`, with a default
  implementation that loops over `separate`. The mock and any other backend keep working
  unchanged.
* `backends/sam_audio.py` implements it for real: one processor call with `k` audios, `k`
  descriptions and per-item anchors, one `model.separate`, each output trimmed to its own
  input length, and CUDA OOM mapped to `InsufficientGpuMemory` after freeing the batch.
  The per-thread CUDA stream wrapper is gone.
* A `threading.Lock` inside `SamAudioBackend` serialises **load and inference**, so the
  shared module can never be entered concurrently regardless of what the engine, the HTTP
  layer or a future caller does above it. This is defence in depth: the engine no longer
  uses threads at all.
* `engine.py` collects up to `AUDIO_CHUNK_BATCH` windows, calls `separate_many` once per
  group, and merges in timeline order with the existing crossfade. The
  `ThreadPoolExecutor` is gone.

## Memory

A batch of `k` windows costs roughly `k ×` the activation memory of one window: upstream
runs the whole batch through a single `odeint` solve and a single `decode`, with no
internal splitting, so peak memory scales as `O(B · reranking_candidates · T)`. Weights
are loaded once either way, so only activations multiply.

`AUDIO_CHUNK_BATCH=2` with `AUDIO_CHUNK_SECONDS=10` therefore costs about what
`AUDIO_CHUNK_SECONDS=20` with batch 1 costs, and is a strictly better deal (two
independent windows instead of one long one). Reranking multiplies on top: the effective
batch is `AUDIO_CHUNK_BATCH × AUDIO_RERANKING_CANDIDATES`.

If a batched call runs out of GPU memory the engine retries the **whole signal** at batch
1 — not just the failing group, so the crossfade never mixes results from two different
batch sizes — and the response carries the warning
`GPU memory was insufficient for parallel chunks; processing retried sequentially`. A
batch-1 OOM is a real failure and is surfaced as `INSUFFICIENT_GPU_MEMORY`.

## Operator guidance

`AUDIO_CHUNK_BATCH` (1–8, default 1) replaces `AUDIO_CHUNK_WORKERS`.

`AUDIO_CHUNK_WORKERS` is still accepted as a **deprecated alias**: it now sets the batch
size and logs a warning at startup, so existing compose files (`compose.cloud.yaml` sets
`AUDIO_CHUNK_WORKERS=2`) keep working and mean the safe thing. `AUDIO_CHUNK_BATCH` wins if
both are set. `/capabilities` reports both under `defaults`, with `chunk_workers` mirroring
`chunk_batch`.

* Start at `AUDIO_CHUNK_BATCH=1`. Raise it only if the GPU has headroom: check
  `cuda_peak_allocated_bytes` in `/health` after a representative request, and keep peak
  under roughly 80 % of the card.
* On the A40 (48 GB) with `sam-audio-large`, float32, `AUDIO_CHUNK_SECONDS=10`, span
  predictor and rankers off, `AUDIO_CHUNK_BATCH=2` is the configuration this change was
  written for. That number is not measured on our hardware — raise it one step at a time
  and watch for `INSUFFICIENT_GPU_MEMORY`.
* Seeded requests may be batched: the seed is set immediately before the single
  `model.separate` call, so a given request reproduces exactly. The noise draw does depend
  on the batch size, so a seeded result reproduces for a *fixed* `AUDIO_CHUNK_BATCH`;
  the response says so in a warning. Pin the batch size if you are comparing runs.
* Visually prompted requests are unaffected: they still run as one aligned interval and
  ignore `AUDIO_CHUNK_BATCH`.
* `AUDIO_CHUNK_BATCH` bounds windows *within* one request. Concurrent HTTP requests are
  still serialised by the engine lock, so the setting does not multiply GPU usage across
  requests.
