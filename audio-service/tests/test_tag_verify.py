import numpy as np
from sam_audio_service.tagging import HeardSound, verify

NAMES = ["Speech", "Music", "Fire", "Crackle", "Door", "Rain", "Squeal", "Vehicle"]


def h(prompt, group, tier):
    return HeardSound(prompt, group, tier, 0.5, [0])


def test_drops_what_the_tagger_does_not_hear_and_keeps_the_rest():
    probs = np.zeros((3, len(NAMES)), dtype=np.float32)
    probs[:, 0] = 0.9; probs[:, 1] = 0.8          # speech and music clip
    probs[1, NAMES.index("Rain")] = 0.4
    probs[2, NAMES.index("Vehicle")] = 0.3        # the screech reads as machinery
    heard = [h("people talking", "speech", "voices"), h("electronic music", "music", "music"), h("fire crackling", "fire", "background"),
             h("door slamming", "door", "hits"), h("heavy rain", "rain", "background"), h("high-pitched screech", "screech", "hits"),
             h("something new", "unmapped", "other")]
    kept = [x.prompt for x in verify(heard, probs, NAMES)]
    assert kept == ["people talking", "electronic music", "heavy rain", "high-pitched screech", "something new"]


def test_no_tagger_output_keeps_everything():
    heard = [h("fire crackling", "fire", "background")]
    assert verify(heard, np.zeros((0, 3)), ["a", "b", "c"]) == heard
