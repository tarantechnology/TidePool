# SAM feasibility and compatibility

Sources were inspected on November 6, 2026. No SAM inference ran locally.

| Capability | YOLO26 + ByteTrack (implemented) | Official SAM 3 (investigated) |
| --- | --- | --- |
| Detection | Automatic fixed COCO classes | Prompted open-vocabulary concepts |
| Segmentation | Optional `yolo26n-seg.pt`, actual native instance masks tested | Prompted masks with text/geometry |
| Tracking | Box association, IDs may change | Video segmentation with interactive refinement |
| Local runtime | CPU measured; see VALIDATION.md | Unmeasured; not installed/run |
| Hardware/install | Small weights; pip package works on this M2 Mac | Official setup specifies Python ≥3.12, PyTorch ≥2.7 and CUDA GPU ≥12.6 |
| Access | YOLO weights downloaded | Hugging Face approval/authentication required |
| Editor suitability | Working box selection; optional per-frame masks | Potentially useful for precise/refined selections; needs GPU experiments |

SAM 3's official repository also contains SAM 3.1 updates. The comparison above concerns the requested SAM 3 baseline; it is a feasibility review, **not an accuracy or speed benchmark against SAM 3**. Keep the verified YOLO pipeline. Source: [official README at f1647e8d](https://github.com/facebookresearch/sam3/blob/f1647e8d0c1d280991f4d1b7d974c2704537fe66/README.md).

Local blockers: `sam3`, `sam_audio` and their checkpoints are absent, CUDA is unavailable (M2/MPS is not CUDA), and anonymous checkpoint/config HEAD requests returned HTTP 401 for `facebook/sam3` and `facebook/sam-audio-small-tv`. No inference service URL or team-selected SAM Audio revision is configured for this CV work. No credentials were requested, read or committed.

## Actual SAM Audio input

The [official SAM Audio README](https://github.com/facebookresearch/sam-audio/blob/bb4c6999d2677c7402360e426afc01ddfad6dce0/README.md) documents visual prompting through `masked_videos=processor.mask_videos([frames], [mask])`, alongside audio and a description. A bbox JSON object is not a direct visual input. Visual-oriented `*-tv` model variants are available, but the team must choose its checkpoint. Object visibility does not establish that it made a sound.

The [official processor implementation](https://github.com/facebookresearch/sam-audio/blob/bb4c6999d2677c7402360e426afc01ddfad6dce0/sam_audio/processor.py) was inspected directly. Tensor video is NCHW with three channels. Its tensor path maps frames uniformly onto audio features, so we provide every native CFR frame over the exact audio interval. Audio tensor input must already be 48kHz, shaped channels × samples; processing averages channels. `mask_videos` retains pixels where its mask equals **zero**, so our foreground=1 masks must be inverted and expanded to N×1×H×W. This polarity conversion is implemented and tested. RGB frames and masks come from original media, not annotated previews.

The adapter is source-compatible preparation, not a claim that the full official processor/model runs in this environment. Automated default tests use an injected mock processor. An additional local test executed the actual upstream `mask_videos` method extracted from the inspected source, with real tensors, without loading a checkpoint. Download that exact source to enable the same optional test:

```bash
# From vision/; downloaded research material remains ignored.
mkdir -p .research
curl --fail --location \
  https://raw.githubusercontent.com/facebookresearch/sam-audio/bb4c6999d2677c7402360e426afc01ddfad6dce0/sam_audio/processor.py \
  --output .research/sam-audio-sam_audio_processor.py
```

## Next joint experiment

Person 1 supplies the agreed model revision, access and GPU service. Person 1/4 agree original-video resolution and full-timeline stem alignment through the existing WAV+JSON boundary. Then process a short clip containing genuine overlapping sounds, choose an audible visible source, compare text-only and mask-guided stems, listen for target retention and leakage, and measure runtime. The pedestrian test's added audio is silence, so it cannot validate sound isolation. No siren, voice association or separation quality result is claimed.
