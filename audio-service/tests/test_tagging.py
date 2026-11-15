import numpy as np

from conftest import synth
from sam_audio_service.tagging import VOCABULARY, pick, windows_of


def test_windows_pad_and_count():
    w = windows_of(np.ones(25, dtype=np.float32), 1, seconds=10)
    assert len(w) == 3 and all(len(x) == 10 for x in w) and w[2][5:].sum() == 0


def test_pick_dominant_per_window_one_per_group():
    idx = {p: k for k, (p, _, _) in enumerate(VOCABULARY)}
    sims = np.zeros((3, len(VOCABULARY)))
    sims[:, idx['heavy rain']] = 0.4
    sims[:, idx['light rain dripping']] = 0.39
    sims[1, idx['thunder rumbling']] = 0.6
    sims[2, idx['orchestral music']] = 0.6
    heard = pick(sims, 30)
    prompts = [h.prompt for h in heard]
    assert prompts[:2] == ['orchestral music', 'thunder rumbling']  # music is kept first
    assert 'heavy rain' in prompts and 'light rain dripping' not in prompts
    assert heard[1].windows == [1]


def test_pick_ignores_silent_windows():
    sims = np.zeros((2, len(VOCABULARY)))
    sims[1, 0] = 1.0
    assert all(1 not in h.windows for h in pick(sims, 30, active=np.array([True, False])))
    assert pick(sims, 30, active=np.array([False, False])) == []


def test_tag_endpoint_mock(client, tmp_path):
    path = synth(tmp_path / 'clip.wav', seconds=12.0)
    with path.open('rb') as f:
        r = client.post('/tag', files={'audio': ('clip.wav', f, 'audio/wav')}, data={'max_sounds': '4'})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body['is_mock'] is True and body['windows'] == 2
    assert 1 <= len(body['sounds']) <= 4
    assert {'prompt', 'group', 'tier', 'score', 'windows'} <= set(body['sounds'][0])
    assert client.get('/capabilities').json()['prompting_modes']['tagging'] is True


def test_crowded_clip_keeps_speech_and_music_and_drops_weak_hits():
    idx = {p: k for k, (p, _, _) in enumerate(VOCABULARY)}
    logits = np.full((5, len(VOCABULARY)), -4.0)
    logits[:, idx['heavy rain']] = 4.0                 # dominates every window
    logits[2, idx['high-pitched screech']] = 4.4       # a clear hit
    logits[3, idx['gunshots']] = 3.2                   # a weak, wrong hit
    logits[:, idx['people talking']] = 0.5             # present but crowded out
    logits[:, idx['orchestral music']] = 0.2
    heard = pick(logits, 1.0)
    prompts = [h.prompt for h in heard]
    assert prompts[:2] == ['people talking', 'orchestral music']
    assert 'heavy rain' in prompts and 'high-pitched screech' in prompts and 'gunshots' not in prompts
