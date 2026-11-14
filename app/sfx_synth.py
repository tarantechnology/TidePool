"""Procedural recipes for the TidePool comedy / meme / editor SFX pack.

Every sound here is synthesized from oscillators and noise, so the pack is ours (CC0-1.0)
and carries no third-party licence risk. They recreate the *vibe* of familiar meme sounds
(a deep distorted boom, a record scratch, a sad trombone) without sampling any recording.

`synthesize(id)` is deterministic: each recipe gets a generator seeded from its id.
"""
import zlib

import numpy as np

from .sfx_dsp import (SR, adsr, smooth, bandpass, bump, concat, const, decay, finish, formant, gauss_band, glide,
                      harmonic, highpass, lowpass, midi, n_of, noise, pan_curve, partials, place, points,
                      reverb, saw, silence, sine, smooth_noise, softclip, square, stft_filter, times,
                      to_stereo, tremolo, triangle)

RECIPES = {}


def sound(sid, title, category, tags=(), aliases=(), hit=None, lufs=-16.0):
    """Register a recipe. `hit` is the moment (seconds) that should land on an anchor when the
    sound is placed *before* something; None means the sound's end."""
    def wrap(fn):
        RECIPES[sid] = {'id': sid, 'title': title, 'category': category, 'tags': list(tags),
                        'aliases': list(aliases), 'hit': hit, 'lufs': lufs, 'fn': fn}
        return fn
    return wrap


def synthesize(sid):
    """Finished float32 samples [frames, channels] at 48 kHz."""
    recipe = RECIPES[sid]
    rng = np.random.default_rng(zlib.crc32(sid.encode()))
    return finish(recipe['fn'](rng), recipe['lufs'])


# ================================================================ shared building blocks

VOWELS = {
    'a': [(800, 90, 1.0), (1200, 100, 0.55), (2500, 140, 0.22)],
    'o': [(500, 70, 1.0), (900, 80, 0.5), (2500, 140, 0.12)],
    'u': [(340, 60, 1.0), (800, 80, 0.25), (2400, 140, 0.08)],
    'uh': [(640, 80, 1.0), (1190, 90, 0.5), (2390, 140, 0.2)],
    'e': [(450, 70, 1.0), (1900, 110, 0.45), (2600, 140, 0.3)],
    'i': [(300, 60, 1.0), (2300, 120, 0.4), (3000, 150, 0.3)],
    'r': [(450, 80, 1.0), (1100, 100, 0.4), (1600, 120, 0.35)],
}


def glottal(f0, rng, jitter=0.012, breath=0.04, top=4800):
    """Voice source: harmonics falling at ~-8 dB/octave with jitter and breath noise."""
    n = len(f0)
    f = f0 * (1 + smooth_noise(n, rng, 18, jitter))
    src = harmonic(f, lambda k: 1 / k ** 1.35, kmax=60, ceiling=top)
    return src + breath * highpass(noise(n, rng), 1200) * np.max(np.abs(src))


def vowel(src, name, scale=1.0):
    peaks = [(c * scale, bw * scale, g) for c, bw, g in VOWELS[name]]
    return formant(src, peaks)


def vowel_glide(src, a, b, scale=1.0, power=1.0):
    """Formants moving from vowel a to vowel b across the clip."""
    total = len(src) / SR
    pa, pb = VOWELS[a], VOWELS[b]

    def mask(f, t):
        x = np.clip(t / total, 0, 1) ** power
        out = np.full(np.broadcast(f, t).shape, 0.02)
        for (ca, wa, ga), (cb, wb, gb) in zip(pa, pb):
            c = (ca + (cb - ca) * x) * scale
            w = (wa + (wb - wa) * x) * scale
            g = ga + (gb - ga) * x
            out = out + g / (1 + ((f - c) / w) ** 2)
        return out
    return stft_filter(src, mask)


def voice(seconds, f0_start, f0_end, rng, vowel_name='a', scale=1.0, env=None):
    n = n_of(seconds)
    f0 = glide(f0_start, f0_end, n)
    out = vowel(glottal(f0, rng), vowel_name, scale)
    return out * (env if env is not None else adsr(n, 0.03, 0.1, 0.8, 0.12))


def laugher(seconds, rng, f0, rate, vowel_name='a', scale=1.0, decline=0.25):
    """One person laughing: 'ha' syllables (aspiration + voiced vowel) at `rate` per second."""
    n = n_of(seconds)
    t = times(n)
    count = max(2, int(seconds * rate))
    env = np.zeros(n)
    asp = np.zeros(n)
    centers = np.cumsum(rng.uniform(0.8, 1.2, count) / rate)
    for i, c in enumerate(centers):
        if c > seconds - 0.05:
            break
        g = (1 - decline * i / count) * rng.uniform(0.7, 1.0)
        env += g * bump(n, c, 0.5 / rate)
        asp += g * bump(n, c - 0.35 / rate, 0.12 / rate)
    pitch = f0 * (1 + 0.12 * np.sin(2 * np.pi * rate * t)) * glide(1.0, 0.85, n)
    voiced = vowel(glottal(pitch, rng, jitter=0.02, breath=0.08), vowel_name, scale)
    breath = vowel(noise(n, rng), vowel_name, scale) * 0.25
    return voiced * env + breath * np.max(np.abs(voiced)) * asp / max(np.max(np.abs(breath)), 1e-9)


def crowd(seconds, rng, voices, make, spread=0.5, room=0.6, wet=0.25):
    out = silence(seconds, 2)
    for _ in range(voices):
        part = make(rng)
        out = place(out, to_stereo(part, rng.uniform(-spread, spread)), rng.uniform(0, 0.35), rng.uniform(0.5, 1.0))
    return reverb(out, rng, room, wet, damp=5000)


def clap(rng, center=None):
    n = n_of(0.03)
    burst = noise(n, rng) * decay(n, rng.uniform(0.003, 0.007), 0.0005)
    return bandpass(burst, center or rng.uniform(700, 1800), 6000)


def brass(freq, n, rng, bright=2500.0, attack=0.04):
    """Brass-ish tone: saw through a filter that opens on the attack."""
    tone = saw(freq * (1 + smooth_noise(n, rng, 5, 0.002)), kmax=40)

    def mask(f, t):
        cutoff = 300 + bright * np.clip(t / attack, 0, 1)
        return 1 / np.sqrt(1 + (f / cutoff) ** 4)
    return stft_filter(tone, mask)


def tom(f_start, f_end, length, rng):
    n = n_of(length)
    body = sine(glide(f_start, f_end, n)) * decay(n, length / 3)
    stick = bandpass(noise(n_of(0.01), rng), 1500, 6000) * decay(n_of(0.01), 0.002)
    return place(body, stick * 0.3, 0)


def snare(rng, length=0.25, tone=190):
    n = n_of(length)
    rattle = bandpass(noise(n, rng), 1500, 9000) * decay(n, 0.06)
    body = sine(glide(tone * 1.3, tone, n)) * decay(n, 0.04)
    return 0.8 * rattle + 0.6 * body


def crash(rng, length=2.0):
    n = n_of(length)
    hiss = highpass(noise(n, rng), 4000) * decay(n, length / 4, 0.001)
    shimmer = partials(n, 3200, rng.uniform(1, 3, 24), rng.uniform(0.02, 0.06, 24), rng.uniform(0.3, 1.0, 24), rng)
    return hiss * 0.5 + shimmer


def kick(rng, length=0.4):
    n = n_of(length)
    return sine(glide(160, 45, n, power=0.3)) * decay(n, 0.12) + 0.1 * bandpass(noise(n, rng), 2000, 6000) * decay(n, 0.004)


def bell(freq, length, rng, bright=1.0):
    n = n_of(length)
    return partials(n, freq, [1, 2.0, 2.76, 5.4, 8.93], [1, 0.35, 0.5 * bright, 0.2 * bright, 0.1 * bright],
                    [length / 2.5, length / 3.5, length / 4.5, length / 7, length / 10], rng)


def record_buffer(rng, seconds=3.0):
    """A stand-in 'record' to scratch: chord, pulsing drums and a vocal-ish formant."""
    n = n_of(seconds)
    t = times(n)
    chord = sum(saw(const(midi(m), n), kmax=30) for m in (45, 52, 57, 61)) * 0.25
    beat = sum(bump(n, b, 0.03) for b in np.arange(0.0, seconds, 0.25))
    drums = lowpass(noise(n, rng), 3000) * beat
    vox = vowel(glottal(const(220, n) * (1 + 0.03 * np.sin(2 * np.pi * 5 * t)), rng), 'a') * 0.4
    return lowpass(chord, 4000) + 0.6 * drums + vox


def scrub(buffer, velocity, start=1.0):
    """Read `buffer` with a time-varying playback velocity (1 = normal, negative = backwards)."""
    pos = start * SR + np.cumsum(velocity)
    pos = np.clip(pos, 0, len(buffer) - 2)
    i = pos.astype(int)
    frac = pos - i
    return buffer[i] * (1 - frac) + buffer[i + 1] * frac


# ================================================================ impacts and booms

@sound('vine_boom', 'Deep boom (vine-boom style)', 'impact', ['boom', 'meme', 'bass', 'dramatic', 'reveal'],
       ['vine boom', 'vine thud', 'meme boom', 'bass boom', 'dramatic boom', 'boom meme'], lufs=-14)
def vine_boom(rng):
    n = n_of(1.6)
    body = sine(glide(120, 42, n, power=0.25)) * decay(n, 0.42, 0.003) * points(n, [(0, 1), (1.2, 1), (1.6, 0)])
    body = softclip(body * 1.0 + 0.4 * sine(glide(240, 84, n, power=0.25)) * decay(n, 0.2), 3.0)
    click = bandpass(noise(n_of(0.012), rng), 800, 5000) * decay(n_of(0.012), 0.003, 0.0005)
    out = place(body, click * 0.35, 0)
    return reverb(lowpass(out, 3500), rng, 1.4, 0.18, damp=2500)


@sound('boom_impact', 'Cinematic boom impact', 'impact', ['boom', 'impact', 'trailer', 'hit', 'cinematic'],
       ['boom', 'impact', 'big boom', 'cinematic hit', 'trailer hit', 'trailer boom'], lufs=-14)
def boom_impact(rng):
    n = n_of(3.0)
    sub = sine(glide(75, 32, n, power=0.4)) * decay(n, 0.8)
    crack = lowpass(noise(n, rng), 900) * decay(n, 0.12, 0.001)
    out = softclip(sub + 0.6 * crack, 1.8)
    return reverb(out, rng, 2.5, 0.3, damp=3000)


@sound('sub_drop', 'Sub drop', 'impact', ['sub', 'drop', 'bass', 'transition'], ['sub drop', 'bass sweep', 'low drop'], lufs=-15)
def sub_drop(rng):
    n = n_of(2.2)
    return softclip(sine(glide(95, 28, n, power=0.6)) * decay(n, 0.9, 0.01), 1.5)


@sound('bass_drop', 'Bass drop', 'impact', ['bass', 'drop', 'edm', 'beat drop'], ['bass drop', 'drop the bass', 'beat drop', 'edm drop'], lufs=-14)
def bass_drop(rng):
    n = n_of(2.0)
    f = np.concatenate([glide(180, 42, n_of(0.5), power=0.5), const(42, n - n_of(0.5))])
    tone = softclip(sine(f) + 0.3 * sine(2 * f), 2.5)
    return tone * points(n, [(0, 0), (0.01, 1), (1.4, 0.8), (2.0, 0)])


@sound('punch', 'Punch hit', 'impact', ['punch', 'fight', 'hit', 'cartoon'], ['punch', 'hit', 'fight punch', 'smack', 'whack'], lufs=-14)
def punch(rng):
    n = n_of(0.35)
    thump = sine(glide(160, 55, n, power=0.4)) * decay(n, 0.07)
    snap = bandpass(noise(n, rng), 500, 3500) * decay(n, 0.025, 0.0005)
    return reverb(softclip(thump + 0.7 * snap, 2.0), rng, 0.3, 0.12)


@sound('slap', 'Slap', 'impact', ['slap', 'hit', 'face'], ['slap', 'face slap', 'smack'], lufs=-15)
def slap(rng):
    n = n_of(0.25)
    burst = highpass(noise(n, rng), 1000) * decay(n, 0.018, 0.0003)
    return reverb(burst + 0.3 * sine(const(220, n)) * decay(n, 0.02), rng, 0.25, 0.15)


@sound('bonk', 'Cartoon bonk', 'impact', ['bonk', 'head', 'cartoon', 'hit', 'meme'], ['bonk', 'head bonk', 'go to jail bonk', 'conk'], lufs=-15)
def bonk(rng):
    n = n_of(0.5)
    f = glide(820, 560, n, power=0.3)
    wood = sine(f) * decay(n, 0.09) + 0.5 * sine(f * 2.3) * decay(n, 0.04) + 0.3 * sine(f * 0.5) * decay(n, 0.12)
    knock = bandpass(noise(n_of(0.01), rng), 1000, 4000) * decay(n_of(0.01), 0.002)
    return reverb(place(wood, knock * 0.4, 0), rng, 0.4, 0.12)


@sound('thud', 'Body thud', 'impact', ['thud', 'fall', 'drop', 'land'], ['thud', 'body fall', 'fall down', 'drop'], lufs=-15)
def thud(rng):
    n = n_of(0.6)
    return sine(glide(95, 48, n)) * decay(n, 0.12) + 0.5 * lowpass(noise(n, rng), 600) * decay(n, 0.05)


@sound('metal_clang', 'Metal clang', 'impact', ['metal', 'clang', 'pan', 'hit'], ['clang', 'metal hit', 'frying pan', 'pan hit', 'anvil'], lufs=-16)
def metal_clang(rng):
    n = n_of(1.8)
    ring = partials(n, 420, [1, 2.32, 4.25, 6.63, 9.38, 3.1], [1, .7, .5, .35, .2, .4], [1.0, .8, .6, .4, .25, .5], rng, 0.01)
    hit = bandpass(noise(n, rng), 1500, 8000) * decay(n, 0.01)
    return reverb(ring + 0.5 * hit, rng, 0.8, 0.15)


@sound('glass_shatter', 'Glass shatter', 'impact', ['glass', 'break', 'shatter', 'window'], ['glass break', 'glass shatter', 'window smash', 'breaking glass'], lufs=-16)
def glass_shatter(rng):
    total = 1.8
    out = silence(total, 2)
    n = n_of(0.05)
    out = place(out, to_stereo(highpass(noise(n, rng), 2500) * decay(n, 0.012, 0.0003)), 0)
    for _ in range(70):
        at = float(min(total - 0.3, rng.exponential(0.22)))
        m = n_of(0.25)
        shard = sine(const(rng.uniform(2500, 9500), m)) * decay(m, rng.uniform(0.02, 0.09), 0.0005)
        shard += 0.4 * highpass(noise(m, rng), 4000) * decay(m, 0.006, 0.0003)
        out = place(out, to_stereo(shard, rng.uniform(-0.8, 0.8)), at, rng.uniform(0.15, 0.6))
    return reverb(out, rng, 0.6, 0.15)


@sound('explosion', 'Explosion', 'impact', ['explosion', 'blast', 'bomb', 'kaboom'], ['explosion', 'kaboom', 'blast', 'bomb', 'blow up'], lufs=-14)
def explosion(rng):
    n = n_of(3.2)
    rumble = lowpass(noise(n, rng), 400) * decay(n, 0.9, 0.004)
    blast = lowpass(noise(n, rng), 2500) * decay(n, 0.15, 0.001)
    crackle = np.zeros(n)
    for at in rng.uniform(0.05, 1.5, 120):
        i = int(at * SR)
        crackle[i:i + 60] += rng.uniform(-1, 1) * np.exp(-np.arange(60) / 10)[:max(0, min(60, n - i))]
    sub = sine(glide(60, 30, n)) * decay(n, 0.5)
    out = softclip(2.0 * rumble + blast + 0.5 * sub + 0.4 * highpass(crackle, 1500), 1.6)
    return reverb(out, rng, 2.2, 0.3, damp=2500)


@sound('heartbeat', 'Heartbeat', 'impact', ['heart', 'heartbeat', 'tension', 'pulse'], ['heartbeat', 'heart beat', 'heart pounding', 'pulse'], lufs=-17)
def heartbeat(rng):
    out = silence(2.8)
    for beat in (0.0, 0.9, 1.8):
        for at, f, g in ((0.0, 62, 1.0), (0.28, 52, 0.7)):
            n = n_of(0.18)
            out = place(out, sine(glide(f * 1.4, f, n)) * decay(n, 0.05, 0.01), beat + at, g)
    return lowpass(out, 300)


# ================================================================ comedy hits

@sound('rimshot', 'Rimshot (ba-dum-tss)', 'comedy', ['rimshot', 'joke', 'drum', 'punchline', 'comedy'],
       ['rimshot', 'ba dum tss', 'badum tss', 'ba dum tiss', 'joke drum', 'drum sting', 'badumtss', 'ba-dum-tss'], hit=0.38)
def rimshot(rng):
    out = silence(2.0)
    out = place(out, tom(210, 165, 0.25, rng), 0.0, 0.8)
    out = place(out, tom(150, 112, 0.3, rng), 0.17, 0.9)
    out = place(out, kick(rng), 0.38, 0.6)
    out = place(out, snare(rng), 0.38, 0.5)
    out = place(out, crash(rng, 1.6), 0.38, 0.7)
    return reverb(out, rng, 0.6, 0.12)


@sound('sad_trombone', 'Sad trombone (wah wah wah)', 'comedy', ['fail', 'sad', 'trombone', 'loser', 'comedy'],
       ['sad trombone', 'wah wah wah', 'wah wah', 'womp womp', 'fail horn', 'sad horn', 'price is wrong', 'you lose'], lufs=-15)
def sad_trombone(rng):
    notes = [(58, 0.42), (57, 0.42), (56, 0.42), (55, 1.5)]
    total = sum(d for _, d in notes)
    n = n_of(total)
    f = np.zeros(n)
    i = 0
    starts = []
    for note, d in notes:
        m = n_of(d)
        f[i:i + m] = midi(note)
        starts.append(i / SR)
        i += m
    f = smooth(f, 0.05)  # slide between notes like a trombone
    t = times(n)
    vib = 1 + 0.018 * np.sin(2 * np.pi * 5.5 * t) * np.clip((t - starts[-1] - 0.2) / 0.4, 0, 1)
    tone = harmonic(f * vib, lambda k: [0.6, 0.9, 1.0, 0.8, 0.6, 0.45, 0.3, 0.2, 0.15, 0.1][k - 1] if k <= 10 else 0.05 / k)

    def wah(fr, tt):
        tt0 = tt[:, 0]
        c = np.full(len(tt0), 400.0)
        for s, (_, d) in zip(starts, notes):
            local = (tt0 - s) / d
            inside = (local >= 0) & (local < 1)
            c += 1800 * np.sin(np.pi * np.minimum(np.clip(local, 0, 1) * 1.6, 1)) * inside
        return 1 / np.sqrt(1 + (fr / c[:, None]) ** 4)
    out = stft_filter(tone, wah)
    env = np.zeros(n)
    for s, (_, d) in zip(starts, notes):
        env += bump(n, s + d / 2, d / 2 + 0.03) ** 0.4 * (0.75 if d > 1 else 1.0)
    return reverb(out * env, rng, 0.8, 0.15)


@sound('slide_whistle_up', 'Slide whistle up', 'comedy', ['whistle', 'slide', 'up', 'cartoon', 'rise'],
       ['slide whistle', 'slide whistle up', 'whistle up', 'rising whistle', 'cartoon whistle'])
def slide_whistle_up(rng):
    n = n_of(0.9)
    t = times(n)
    f = glide(550, 2100, n, power=0.8) * (1 + 0.012 * np.sin(2 * np.pi * 6 * t))
    tone = sine(f) + 0.08 * sine(2 * f)
    breath = bandpass(noise(n, rng), 1500, 5000) * 0.04
    return (tone + breath) * adsr(n, 0.03, 0.1, 0.9, 0.08)


@sound('slide_whistle_down', 'Slide whistle down', 'comedy', ['whistle', 'slide', 'down', 'cartoon', 'fall'],
       ['slide whistle down', 'whistle down', 'falling whistle', 'descending whistle'])
def slide_whistle_down(rng):
    n = n_of(1.0)
    t = times(n)
    f = glide(2100, 480, n, power=1.2) * (1 + 0.012 * np.sin(2 * np.pi * 6 * t))
    return (sine(f) + 0.08 * sine(2 * f) + bandpass(noise(n, rng), 1500, 5000) * 0.04) * adsr(n, 0.03, 0.1, 0.9, 0.1)


@sound('falling_whistle', 'Falling bomb whistle', 'comedy', ['falling', 'whistle', 'bomb', 'drop', 'cartoon'],
       ['falling whistle', 'bomb whistle', 'falling bomb', 'incoming'], hit=None)
def falling_whistle(rng):
    n = n_of(2.2)
    t = times(n)
    f = glide(2600, 700, n, power=1.4) * (1 + 0.006 * np.sin(2 * np.pi * 7 * t))
    return sine(f) * points(n, [(0, 0), (0.4, 0.6), (2.0, 1.0), (2.2, 0)])


@sound('boing', 'Spring boing', 'comedy', ['boing', 'spring', 'bounce', 'cartoon', 'jump'],
       ['boing', 'spring', 'boing boing', 'bounce', 'jaw harp', 'doing'])
def boing(rng):
    n = n_of(1.0)
    f0 = glide(140, 165, n)
    src = harmonic(f0, lambda k: 1 / k ** 0.8, kmax=40)

    def wobble(f, t):
        c = 900 + 650 * np.sin(2 * np.pi * 11 * t) * np.exp(-t / 0.45)
        return 0.05 + 1 / (1 + ((f - c) / 120) ** 2)
    return stft_filter(src, wobble) * decay(n, 0.35, 0.003)


@sound('cartoon_run', 'Cartoon run (scramble)', 'comedy', ['run', 'cartoon', 'feet', 'escape', 'scramble'],
       ['cartoon run', 'running feet', 'scramble', 'run away', 'scurry'])
def cartoon_run(rng):
    out = silence(1.6)
    at = 0.0
    k = 0
    while at < 1.15:
        m = n_of(0.05)
        f = 900 if k % 2 else 1250
        tap = sine(const(f, m)) * decay(m, 0.012, 0.0005) + 0.3 * bandpass(noise(m, rng), 2000, 6000) * decay(m, 0.003)
        out = place(out, tap, at, 0.8)
        at += 1 / 15
        k += 1
    m = n_of(0.35)
    zip_ = sine(glide(500, 3200, m, power=0.7)) * adsr(m, 0.01, 0.1, 0.7, 0.1)
    return place(out, zip_ * 0.7, 1.2)


@sound('cartoon_zip', 'Zip away', 'comedy', ['zip', 'zoom', 'exit', 'cartoon', 'fast'], ['zip', 'zip away', 'zoom off', 'zoom', 'dash'])
def cartoon_zip(rng):
    n = n_of(0.35)
    tone = sine(glide(400, 3600, n, power=0.6)) * adsr(n, 0.01, 0.05, 0.7, 0.12)
    air = bandpass(noise(n, rng), 1500, 7000) * bump(n, 0.2, 0.15) * 0.3
    return pan_curve(tone + air, np.linspace(-0.7, 0.8, n))


@sound('airhorn', 'Air horn', 'comedy', ['airhorn', 'air horn', 'mlg', 'hype', 'celebration', 'meme', 'party'],
       ['airhorn', 'air horn', 'mlg airhorn', 'mlg horn', 'party horn', 'hype horn', 'stadium horn'], lufs=-14)
def airhorn(rng):
    out = silence(1.7)
    for at, length in ((0.0, 0.17), (0.24, 0.17), (0.48, 1.0)):
        n = n_of(length)
        scoop = points(n, [(0, 0.96), (0.03, 1.0), (length, 1.0)])
        tone = sum(saw(const(f, n) * scoop, kmax=40) for f in (466.2, 469.0, 554.4, 698.5))
        tone = softclip(bandpass(tone, 300, 5000) * 1.5, 2.5)
        out = place(out, tone * adsr(n, 0.01, 0.03, 0.9, 0.04 if length < 0.5 else 0.12), at)
    return out


@sound('pop', 'Pop', 'comedy', ['pop', 'bubble', 'cork', 'appear'], ['pop', 'bubble pop', 'cork pop', 'plop', 'blip'], lufs=-17)
def pop(rng):
    n = n_of(0.12)
    return sine(glide(350, 1600, n, power=0.3)) * decay(n, 0.025, 0.0005)


@sound('clown_horn', 'Clown horn (honk honk)', 'comedy', ['honk', 'clown', 'horn', 'bike horn', 'circus'],
       ['honk', 'honk honk', 'clown horn', 'bike horn', 'bulb horn', 'goose honk'])
def clown_horn(rng):
    out = silence(0.8)
    for at in (0.0, 0.36):
        n = n_of(0.26)
        f = glide(400, 370, n)
        tone = formant(saw(f, kmax=40), [(1300, 300, 1.0), (2500, 400, 0.5)], 0.08)
        out = place(out, softclip(tone * adsr(n, 0.01, 0.05, 0.9, 0.05) * 3, 2.0), at)
    return out


@sound('whoopee_cushion', 'Whoopee cushion', 'comedy', ['fart', 'whoopee', 'cushion', 'prank', 'toot'],
       ['whoopee cushion', 'fart', 'toot', 'raspberry', 'prank cushion'])
def whoopee_cushion(rng):
    n = n_of(1.1)
    f = glide(150, 85, n) * (1 + smooth_noise(n, rng, 14, 0.18))
    buzz = harmonic(f, lambda k: 1 / k ** 0.7, kmax=50)
    flutter = 0.65 + 0.35 * np.sin(2 * np.pi * 32 * times(n) + smooth_noise(n, rng, 10, 2.0))
    out = lowpass(buzz * flutter, 1800) * points(n, [(0, 0), (0.03, 1), (0.8, 0.7), (1.1, 0)])
    return out + 0.06 * lowpass(noise(n, rng), 1500) * points(n, [(0, 0), (0.05, 1), (1.1, 0)])


@sound('squeak_toy', 'Squeaky toy', 'comedy', ['squeak', 'toy', 'rubber', 'dog toy', 'cute'], ['squeak', 'squeaky toy', 'rubber duck', 'dog toy', 'squeeze'])
def squeak_toy(rng):
    out = silence(0.7)
    for at, peak in ((0.0, 2300), (0.33, 2050)):
        n = n_of(0.22)
        f = points(n, [(0, 1300), (0.07, peak), (0.22, peak * 0.8)])
        tone = sine(f) + 0.25 * sine(2 * f)
        out = place(out, tone * adsr(n, 0.01, 0.05, 0.8, 0.05) + 0.05 * highpass(noise(n, rng), 3000), at)
    return out


@sound('tada', 'Ta-da fanfare', 'comedy', ['tada', 'fanfare', 'reveal', 'magic', 'success'],
       ['tada', 'ta da', 'ta-da', 'fanfare', 'voila', 'presto', 'reveal fanfare'], hit=0.18)
def tada(rng):
    out = silence(1.9, 2)
    chord = (60, 64, 67, 72)
    for at, length in ((0.0, 0.14), (0.18, 1.45)):
        n = n_of(length)
        t = times(n)
        tone = sum(brass(midi(m) * (1 + 0.004 * np.sin(2 * np.pi * 5.5 * t) * (t > 0.3)), n, rng, 3000) for m in chord)
        env = adsr(n, 0.015, 0.08, 0.8, 0.25 if length > 1 else 0.04)
        out = place(out, to_stereo(tone * env / 3), at)
    out = place(out, to_stereo(crash(rng, 1.5) * 0.4), 0.18)
    return reverb(out, rng, 1.2, 0.2)


@sound('victory_fanfare', 'Victory fanfare', 'comedy', ['victory', 'win', 'fanfare', 'success', 'trumpet'],
       ['victory', 'victory fanfare', 'win', 'winner', 'we did it', 'trumpet fanfare', 'triumph'])
def victory_fanfare(rng):
    out = silence(2.6, 2)
    seq = [(67, 0.0, 0.14), (67, 0.16, 0.14), (67, 0.32, 0.14), (72, 0.5, 1.6)]
    for m, at, length in seq:
        n = n_of(length)
        tone = brass(midi(m), n, rng, 3500) + 0.6 * brass(midi(m - 12), n, rng, 2000)
        if length > 1:
            tone = tone + 0.7 * brass(midi(m + 4), n, rng, 3500) + 0.7 * brass(midi(m + 7), n, rng, 3500)
        out = place(out, to_stereo(tone * adsr(n, 0.015, 0.05, 0.85, 0.3 if length > 1 else 0.03) * 0.5), at)
    return reverb(out, rng, 1.4, 0.22)


@sound('drumroll', 'Drumroll with crash', 'comedy', ['drumroll', 'suspense', 'reveal', 'drum', 'announcement'],
       ['drumroll', 'drum roll', 'drumroll please', 'snare roll'], hit=2.5)
def drumroll(rng):
    out = silence(4.2)
    at = 0.0
    while at < 2.5:
        g = 0.25 + 0.75 * (at / 2.5) ** 1.5
        out = place(out, snare(rng, 0.12, 200), at + rng.uniform(-0.003, 0.003), g * rng.uniform(0.8, 1.0))
        at += 1 / 24
    out = place(out, kick(rng), 2.5, 0.9)
    out = place(out, snare(rng), 2.5, 0.7)
    out = place(out, crash(rng, 1.6), 2.5, 0.9)
    return reverb(out, rng, 0.8, 0.12)


@sound('drum_fill', 'Ba-dum (no cymbal)', 'comedy', ['drum', 'joke', 'badum', 'tom'], ['ba dum', 'badum', 'drum fill', 'tom fill'])
def drum_fill(rng):
    out = silence(0.8)
    out = place(out, tom(200, 160, 0.25, rng), 0.0)
    return place(out, tom(140, 105, 0.35, rng), 0.17)


@sound('cowbell', 'Cowbell', 'comedy', ['cowbell', 'percussion', 'more cowbell'], ['cowbell', 'more cowbell', 'bell hit'])
def cowbell(rng):
    n = n_of(0.45)
    tone = square(const(545, n), kmax=20) + square(const(815, n), kmax=14)
    return bandpass(tone, 400, 3500) * decay(n, 0.09, 0.001)


# ================================================================ voices and crowds (synthetic)

@sound('laugh_track', 'Laugh track (studio audience)', 'crowd', ['laugh', 'audience', 'sitcom', 'funny', 'crowd'],
       ['laugh track', 'laughter', 'audience laugh', 'sitcom laugh', 'crowd laughing', 'laugh', 'canned laughter', 'studio laugh', 'people laughing', 'audience laughing'], lufs=-17)
def laugh_track(rng):
    def person(r):
        female = r.random() < 0.5
        f0 = r.uniform(210, 290) if female else r.uniform(105, 160)
        return laugher(r.uniform(2.0, 3.2), r, f0, r.uniform(4.2, 6.0), 'a' if r.random() < 0.7 else 'e', 1.15 if female else 1.0)
    out = crowd(3.6, rng, 16, person, 0.7, 0.9, 0.3)
    return out * points(len(out), [(0, 0), (0.25, 1), (2.6, 0.8), (len(out) / SR, 0)])[:, None]


@sound('chuckle', 'Small audience chuckle', 'crowd', ['chuckle', 'laugh', 'audience', 'giggle'],
       ['chuckle', 'giggle', 'small laugh', 'light laughter', 'snicker'], lufs=-18)
def chuckle(rng):
    def person(r):
        female = r.random() < 0.5
        f0 = r.uniform(200, 270) if female else r.uniform(110, 150)
        return laugher(r.uniform(0.9, 1.6), r, f0, r.uniform(5, 6.5), 'uh', 1.1 if female else 1.0, 0.5) * 0.8
    out = crowd(2.0, rng, 6, person, 0.6, 0.6, 0.25)
    return out * points(len(out), [(0, 0), (0.1, 1), (1.3, 0.6), (len(out) / SR, 0)])[:, None]


@sound('evil_laugh', 'Evil laugh (mwahaha)', 'crowd', ['evil', 'laugh', 'villain', 'mwahaha'], ['evil laugh', 'mwahaha', 'villain laugh', 'muahaha'], lufs=-16)
def evil_laugh(rng):
    lead = voice(0.45, 95, 120, rng, 'uh', env=adsr(n_of(0.45), 0.05, 0.1, 0.9, 0.08))
    ha = laugher(1.9, rng, 118, 4.0, 'a', 0.95, 0.4)
    return reverb(concat(lead, ha), rng, 1.8, 0.35, damp=3500)


@sound('applause', 'Applause', 'crowd', ['applause', 'clapping', 'audience', 'crowd', 'ovation'],
       ['applause', 'clapping', 'claps', 'round of applause', 'audience applause', 'ovation', 'clap'], lufs=-17)
def applause(rng):
    total = 4.0
    out = silence(total, 2)
    env = lambda t: min(1.0, t / 0.4) * (1.0 if t < 2.6 else max(0.0, 1 - (t - 2.6) / 1.4))
    for clapper in range(28):
        pan = rng.uniform(-0.9, 0.9)
        rate = rng.uniform(3.5, 6.0)
        at = rng.uniform(0, 0.4)
        center = rng.uniform(800, 2200)
        while at < total - 0.05:
            g = env(at) * rng.uniform(0.5, 1.0)
            if g > 0.01:
                out = place(out, to_stereo(clap(rng, center), pan), at, g)
            at += rng.uniform(0.85, 1.15) / rate
    return reverb(out, rng, 0.9, 0.25)


@sound('cheer', 'Crowd cheer', 'crowd', ['cheer', 'crowd', 'yay', 'celebration', 'hooray'],
       ['cheer', 'cheering', 'crowd cheer', 'yay', 'hooray', 'woo', 'crowd goes wild'], lufs=-16)
def cheer(rng):
    def person(r):
        female = r.random() < 0.5
        f0 = r.uniform(230, 330) if female else r.uniform(130, 200)
        d = r.uniform(1.4, 2.4)
        n = n_of(d)
        src = glottal(glide(f0 * 0.85, f0 * 1.15, n) * (1 + 0.03 * np.sin(2 * np.pi * r.uniform(4, 7) * times(n))), r, 0.03, 0.15)
        return vowel_glide(src, 'e', 'a', 1.15 if female else 1.0) * adsr(n, 0.08, 0.2, 0.8, 0.5)
    voices = crowd(3.2, rng, 14, person, 0.8, 1.0, 0.3)
    claps = applause(rng)
    return place(voices, claps[:len(voices)] * 0.5, 0.1)


@sound('crowd_aww', 'Crowd "aww"', 'crowd', ['aww', 'cute', 'crowd', 'sympathy', 'adorable'], ['aww', 'awww', 'crowd aww', 'so cute', 'audience aww'], lufs=-17)
def crowd_aww(rng):
    def person(r):
        female = r.random() < 0.55
        f0 = r.uniform(230, 300) if female else r.uniform(120, 165)
        d = r.uniform(1.2, 1.7)
        n = n_of(d)
        src = glottal(points(n, [(0, f0), (0.25, f0 * 1.08), (d, f0 * 0.72)]), r, 0.015, 0.12)
        return vowel_glide(src, 'a', 'u', 1.12 if female else 1.0, 0.8) * adsr(n, 0.12, 0.2, 0.8, 0.45)
    return crowd(2.0, rng, 14, person, 0.7, 0.9, 0.3)


@sound('crowd_ooh', 'Crowd "ooh"', 'crowd', ['ooh', 'crowd', 'impressed', 'wow'], ['ooh', 'oooh', 'crowd ooh', 'audience ooh', 'ohh'], lufs=-17)
def crowd_ooh(rng):
    def person(r):
        female = r.random() < 0.5
        f0 = r.uniform(220, 290) if female else r.uniform(115, 160)
        d = r.uniform(1.0, 1.5)
        n = n_of(d)
        src = glottal(points(n, [(0, f0 * 0.9), (d * 0.45, f0 * 1.15), (d, f0 * 0.85)]), r, 0.015, 0.1)
        return vowel(src, 'u', 1.1 if female else 1.0) * adsr(n, 0.1, 0.2, 0.8, 0.4)
    return crowd(1.8, rng, 14, person, 0.7, 0.9, 0.3)


@sound('crowd_boo', 'Crowd boo', 'crowd', ['boo', 'crowd', 'disapprove', 'heckle'], ['boo', 'booing', 'crowd boo', 'audience boo', 'boo hiss'], lufs=-17)
def crowd_boo(rng):
    def person(r):
        female = r.random() < 0.4
        f0 = r.uniform(170, 230) if female else r.uniform(95, 140)
        d = r.uniform(1.3, 2.2)
        n = n_of(d)
        src = glottal(f0 * (1 + 0.02 * np.sin(2 * np.pi * r.uniform(3, 5) * times(n))) * glide(1.0, 0.92, n), r, 0.02, 0.12)
        return vowel(src, 'u', 1.1 if female else 1.0) * adsr(n, 0.15, 0.2, 0.85, 0.5)
    return crowd(2.6, rng, 16, person, 0.8, 0.9, 0.3)


@sound('wind_gust', 'Tumbleweed wind', 'ambience', ['wind', 'tumbleweed', 'empty', 'awkward', 'desert', 'silence'],
       ['wind', 'tumbleweed', 'desert wind', 'empty room', 'gust'], lufs=-22)
def wind_gust(rng):
    n = n_of(4.0)
    centre = lambda f, t: gauss_band(f, 500 * 2 ** (0.8 * np.sin(2 * np.pi * 0.2 * t)), 0.35)
    left, right = stft_filter(noise(n, rng), centre), stft_filter(noise(n, rng), centre)
    env = points(n, [(0, 0), (1.2, 1), (2.6, 0.7), (4.0, 0)])
    return np.stack([left * env, right * env], axis=1)


@sound('crowd_gasp', 'Crowd gasp', 'crowd', ['gasp', 'shock', 'crowd', 'surprise'], ['crowd gasp', 'audience gasp', 'everyone gasps'], lufs=-18)
def crowd_gasp(rng):
    return crowd(1.0, rng, 10, lambda r: gasp(r), 0.8, 0.7, 0.25)


@sound('gasp', 'Gasp', 'voice', ['gasp', 'shock', 'surprise', 'inhale', 'omg'], ['gasp', 'shocked gasp', 'surprised', 'inhale', 'oh my'], lufs=-18)
def gasp(rng):
    n = n_of(0.45)
    air = vowel(noise(n, rng), 'a' if rng.random() < 0.6 else 'o', rng.uniform(1.0, 1.15))
    return air * points(n, [(0, 0), (0.3, 1), (0.36, 0.8), (0.45, 0)])


@sound('bruh', 'Bruh (synthetic voice)', 'voice', ['bruh', 'meme', 'voice', 'disappointed', 'really'],
       ['bruh', 'bro', 'bruh moment', 'bruh sound'], lufs=-15)
def bruh(rng):
    n = n_of(0.5)
    f0 = points(n, [(0, 118), (0.12, 112), (0.5, 82)])
    src = glottal(f0, rng, 0.015, 0.05)
    out = vowel_glide(src, 'r', 'uh', 1.0, 0.35)
    env = points(n, [(0, 0), (0.015, 0.7), (0.06, 0.8), (0.12, 1), (0.4, 0.8), (0.5, 0)])
    burst = lowpass(noise(n_of(0.015), rng), 600) * decay(n_of(0.015), 0.004, 0.0005)
    breath = vowel(noise(n_of(0.12), rng), 'uh') * points(n_of(0.12), [(0, 0.5), (0.12, 0)])
    out = place(out * env, burst * 0.6, 0)
    out = place(out, breath * 0.08 * np.max(np.abs(out)) / max(np.max(np.abs(breath)), 1e-9), 0.46)
    return reverb(out, rng, 0.3, 0.08, stereo=False)


@sound('huh', 'Huh? (confused voice)', 'voice', ['huh', 'confused', 'what', 'question', 'voice'], ['huh', 'huh?', 'what?', 'confused', 'say what'], lufs=-16)
def huh(rng):
    n = n_of(0.42)
    f0 = points(n, [(0, 120), (0.1, 125), (0.42, 210)])
    out = vowel(glottal(f0, rng, 0.01, 0.06), 'uh') * points(n, [(0, 0), (0.06, 0.8), (0.3, 1), (0.42, 0)])
    asp = vowel(noise(n_of(0.07), rng), 'uh') * points(n_of(0.07), [(0, 0), (0.05, 1), (0.07, 0.3)])
    return place(out, asp * 0.2 * np.max(np.abs(out)) / max(np.max(np.abs(asp)), 1e-9), 0)


@sound('oof', 'Oof hit (synthetic voice)', 'voice', ['oof', 'hurt', 'ouch', 'hit', 'game', 'meme'],
       ['oof', 'ouch', 'ow', 'oof sound', 'hurt', 'damage', 'player hurt'], lufs=-15)
def oof(rng):
    n = n_of(0.28)
    voiced = vowel(glottal(points(n, [(0, 175), (0.05, 165), (0.28, 115)]), rng, 0.02, 0.08), 'u') * points(n, [(0, 0), (0.02, 1), (0.18, 0.7), (0.26, 0)])
    m = n_of(0.12)
    f = bandpass(noise(m, rng), 1200, 7000) * points(m, [(0, 0), (0.03, 1), (0.12, 0)])
    out = place(voiced, f * 0.1 * np.max(np.abs(voiced)) / max(np.max(np.abs(f)), 1e-9), 0.2)
    thump = sine(glide(120, 60, n_of(0.12))) * decay(n_of(0.12), 0.04)
    return place(out, thump * 0.5 * np.max(np.abs(out)), 0)


# ================================================================ game and UI

@sound('ding', 'Ding (bell)', 'ui', ['ding', 'bell', 'correct', 'idea', 'notification'], ['ding', 'bell ding', 'ding ding', 'idea', 'lightbulb', 'bell'], lufs=-17)
def ding(rng):
    return reverb(bell(1318.5, 1.8, rng), rng, 1.0, 0.15)


@sound('correct_chime', 'Correct answer chime', 'ui', ['correct', 'right', 'success', 'chime', 'yes'],
       ['correct', 'correct answer', 'right answer', 'success chime', 'chime', 'yes', 'ding ding'], lufs=-17)
def correct_chime(rng):
    out = silence(1.4)
    out = place(out, bell(1046.5, 0.9, rng), 0)
    out = place(out, bell(1568.0, 1.1, rng), 0.13)
    return reverb(out, rng, 0.8, 0.15)


@sound('wrong_buzzer', 'Wrong answer buzzer', 'ui', ['wrong', 'buzzer', 'fail', 'incorrect', 'game show', 'no'],
       ['wrong', 'wrong buzzer', 'wrong answer', 'buzzer', 'incorrect', 'eh eh', 'game show buzzer', 'nope'], lufs=-15)
def wrong_buzzer(rng):
    out = silence(1.0)
    for at, length in ((0.0, 0.32), (0.42, 0.55)):
        n = n_of(length)
        tone = square(const(116, n), kmax=40) + square(const(123, n), kmax=40)
        out = place(out, softclip(lowpass(tone, 2500) * adsr(n, 0.005, 0.02, 0.9, 0.03), 1.5), at)
    return out


@sound('error_beep', 'Error beep', 'ui', ['error', 'beep', 'denied', 'computer'], ['error', 'error beep', 'denied', 'access denied', 'computer error'], lufs=-17)
def error_beep(rng):
    out = silence(0.5)
    for at in (0.0, 0.2):
        n = n_of(0.14)
        out = place(out, triangle(const(330, n)) * adsr(n, 0.004, 0.02, 0.9, 0.02), at)
    return out


@sound('cash_register', 'Cash register (ka-ching)', 'ui', ['money', 'cash', 'register', 'ka-ching', 'sale', 'profit'],
       ['cash register', 'ka-ching', 'kaching', 'cha ching', 'money', 'cha-ching', 'sale', 'till'], hit=0.12)
def cash_register(rng):
    out = silence(1.6)
    n = n_of(0.08)
    clank = partials(n, 1800, [1, 1.6, 2.3], [1, .7, .5], [0.03, 0.02, 0.015], rng) + bandpass(noise(n, rng), 2000, 6000) * decay(n, 0.005)
    out = place(out, clank, 0.0, 0.6)
    m = n_of(0.18)
    out = place(out, lowpass(noise(m, rng), 900) * bump(m, 0.09, 0.09), 0.04, 0.3)
    n2 = n_of(1.4)
    ching = partials(n2, 2637, [1, 2.1, 3.3, 4.7], [1, .5, .3, .15], [0.6, 0.4, 0.25, 0.15], rng) * tremolo(n2, 9, 0.2)
    out = place(out, ching, 0.12)
    return reverb(out, rng, 0.7, 0.15)


@sound('coin', 'Coin pickup', 'ui', ['coin', 'game', 'pickup', 'retro', 'collect'], ['coin', 'coin pickup', 'collect coin', 'retro coin', 'point'], lufs=-17)
def coin(rng):
    a = square(const(987.8, n_of(0.07)), kmax=16) * 0.8
    n = n_of(0.4)
    b = square(const(1318.5, n), kmax=12) * decay(n, 0.12, 0.001)
    return concat(a, b)


@sound('level_up', 'Level up', 'ui', ['level up', 'game', 'achievement', 'retro', 'success'], ['level up', 'achievement', 'power level', 'upgrade'], lufs=-17)
def level_up(rng):
    out = silence(0.9)
    for i, m in enumerate((72, 76, 79, 84, 88, 91)):
        length = 0.08 if i < 5 else 0.4
        n = n_of(length)
        out = place(out, square(const(midi(m), n), kmax=12) * adsr(n, 0.003, 0.02, 0.8, 0.04 if i < 5 else 0.3), i * 0.07)
    return out


@sound('game_over', 'Game over', 'ui', ['game over', 'lose', 'retro', 'fail', 'death'], ['game over', 'you died', 'you lose', 'player died', 'lose'], lufs=-17)
def game_over(rng):
    out = silence(2.0)
    for i, m in enumerate((67, 66, 65, 64)):
        length = 0.3 if i < 3 else 0.9
        n = n_of(length)
        f = const(midi(m), n) * (1 + (0.01 * np.sin(2 * np.pi * 6 * times(n)) if i == 3 else 0))
        out = place(out, square(f, kmax=10) * adsr(n, 0.005, 0.05, 0.7, 0.1 if i < 3 else 0.6), i * 0.32)
    return out


@sound('power_up', 'Power up', 'ui', ['power up', 'boost', 'game', 'retro'], ['power up', 'powerup', 'boost', 'mushroom', 'buff'], lufs=-17)
def power_up(rng):
    n = n_of(0.7)
    steps = np.floor(times(n) * 30) / 30
    f = 300 * 2 ** (steps * 3.2) * (1 + 0.5 * (np.floor(times(n) * 60) % 2))
    return triangle(f, kmax=16) * adsr(n, 0.005, 0.05, 0.9, 0.08)


@sound('jump', 'Retro jump', 'ui', ['jump', 'game', 'retro', 'hop'], ['jump', 'hop', 'boing jump', 'game jump'], lufs=-17)
def jump(rng):
    n = n_of(0.22)
    return square(glide(280, 980, n, power=0.5), kmax=12) * adsr(n, 0.003, 0.03, 0.8, 0.06)


@sound('laser', 'Laser zap', 'ui', ['laser', 'zap', 'pew', 'shoot', 'sci-fi'], ['laser', 'pew', 'pew pew', 'zap', 'blaster', 'ray gun'], lufs=-17)
def laser(rng):
    n = n_of(0.3)
    f = glide(2400, 180, n, power=0.6)
    return sine(f + 120 * np.sin(phase_safe(f * 0.25))) * decay(n, 0.1, 0.001)


def phase_safe(freq):
    return 2 * np.pi * np.cumsum(freq) / SR


@sound('notification', 'Notification ping', 'ui', ['notification', 'ping', 'message', 'alert', 'phone'],
       ['notification', 'ping', 'message', 'text message', 'alert', 'new message'], lufs=-18)
def notification(rng):
    out = silence(0.6)
    out = place(out, sine(const(1760, n_of(0.4))) * decay(n_of(0.4), 0.09), 0)
    return place(out, sine(const(2637, n_of(0.45))) * decay(n_of(0.45), 0.12), 0.09)


@sound('mouse_click', 'Mouse click', 'ui', ['click', 'mouse', 'button', 'select'], ['click', 'mouse click', 'button click', 'tap'], lufs=-20)
def mouse_click(rng):
    out = silence(0.12)
    for at, f in ((0.0, 4500), (0.06, 3200)):
        n = n_of(0.01)
        out = place(out, bandpass(noise(n, rng), f * 0.6, f * 1.6) * decay(n, 0.0015, 0.0002), at)
    return out


@sound('typing', 'Keyboard typing', 'ui', ['typing', 'keyboard', 'hacker', 'computer', 'writing'],
       ['typing', 'keyboard', 'keyboard typing', 'hacking', 'hacker typing', 'clacking'], lufs=-19)
def typing(rng):
    total = 2.6
    out = silence(total)
    at = 0.0
    while at < total - 0.1:
        n = n_of(0.03)
        space = rng.random() < 0.12
        f = rng.uniform(1800, 4500) if not space else 900
        key = bandpass(noise(n, rng), f * 0.7, f * 1.5) * decay(n, 0.003 if not space else 0.007, 0.0002)
        key += 0.4 * sine(const(rng.uniform(140, 220), n)) * decay(n, 0.006)
        out = place(out, key, at, rng.uniform(0.5, 1.0))
        at += max(0.055, rng.exponential(0.11))
    return out


@sound('camera_shutter', 'Camera shutter', 'ui', ['camera', 'shutter', 'photo', 'snapshot', 'picture'],
       ['camera', 'camera shutter', 'photo', 'take a picture', 'snapshot', 'shutter click'], lufs=-18)
def camera_shutter(rng):
    out = silence(0.35)
    for at, g in ((0.0, 1.0), (0.085, 0.7)):
        n = n_of(0.03)
        out = place(out, bandpass(noise(n, rng), 2500, 9000) * decay(n, 0.003, 0.0002) + 0.5 * sine(const(3900, n)) * decay(n, 0.004), at, g)
    m = n_of(0.07)
    return place(out, bandpass(noise(m, rng), 400, 3000) * bump(m, 0.035, 0.035) * 0.25, 0.01)


@sound('clock_ticking', 'Clock ticking', 'ui', ['clock', 'tick', 'time', 'waiting', 'countdown'], ['clock', 'ticking', 'tick tock', 'clock ticking', 'waiting'], lufs=-19)
def clock_ticking(rng):
    out = silence(4.0)
    for i in range(8):
        n = n_of(0.02)
        f = 2600 if i % 2 == 0 else 2100
        out = place(out, bandpass(noise(n, rng), f * 0.7, f * 1.4) * decay(n, 0.002, 0.0002) + 0.5 * sine(const(f, n)) * decay(n, 0.003), i * 0.5)
    return out


@sound('countdown_beeps', 'Countdown beeps (3-2-1)', 'ui', ['countdown', 'beep', 'race', 'start', 'timer'],
       ['countdown', 'countdown beeps', '3 2 1', 'race start', 'timer beeps'], hit=3.0)
def countdown_beeps(rng):
    out = silence(3.8)
    for i in range(3):
        n = n_of(0.14)
        out = place(out, sine(const(880, n)) * adsr(n, 0.004, 0.02, 0.9, 0.03), i)
    n = n_of(0.6)
    return place(out, sine(const(1760, n)) * adsr(n, 0.004, 0.05, 0.9, 0.1), 3.0)


@sound('censor_bleep', 'Censor bleep (1 kHz)', 'ui', ['bleep', 'censor', 'beep', 'swear'], ['bleep', 'censor', 'censor beep', 'beep', 'swear beep'], lufs=-18)
def censor_bleep(rng):
    n = n_of(0.6)
    return sine(const(1000, n)) * adsr(n, 0.005, 0.01, 1.0, 0.005)


@sound('phone_vibrate', 'Phone vibrate', 'ui', ['phone', 'vibrate', 'buzz', 'call'], ['vibrate', 'phone buzz', 'phone vibrate', 'buzzing phone'], lufs=-19)
def phone_vibrate(rng):
    out = silence(1.9)
    for at in (0.0, 0.65, 1.3):
        n = n_of(0.42)
        out = place(out, lowpass(square(const(170 * (1 + 0.02 * rng.uniform()), n), kmax=30), 900) * adsr(n, 0.02, 0.02, 1, 0.03), at)
    return out


@sound('alarm_clock', 'Alarm clock ring', 'ui', ['alarm', 'clock', 'wake up', 'ring', 'bell'], ['alarm', 'alarm clock', 'wake up', 'ringing', 'school bell'], lufs=-17)
def alarm_clock(rng):
    n = n_of(2.2)
    ring = partials(n, 2050, [1, 1.5, 2.4], [1, .5, .3], [5, 5, 5], rng)
    strike = 0.5 + 0.5 * np.sign(np.sin(2 * np.pi * 22 * times(n)))
    return ring * lowpass(strike, 200, 1) * adsr(n, 0.01, 0.05, 1, 0.15)


@sound('doorbell', 'Doorbell (ding-dong)', 'ui', ['doorbell', 'ding dong', 'door', 'visitor'], ['doorbell', 'ding dong', 'door bell', 'someone at the door'], lufs=-17)
def doorbell(rng):
    out = silence(2.2)
    out = place(out, bell(659.3, 1.4, rng, 0.6), 0)
    return reverb(place(out, bell(523.3, 1.6, rng, 0.6), 0.55), rng, 0.9, 0.18)


@sound('door_knock', 'Door knock', 'ui', ['knock', 'door', 'knocking', 'visitor'], ['knock', 'knock knock', 'door knock', 'knocking'], lufs=-17)
def door_knock(rng):
    out = silence(0.8)
    for at in (0.0, 0.2, 0.4):
        n = n_of(0.12)
        hit = lowpass(noise(n, rng), 1300) * decay(n, 0.01, 0.0005) + 0.6 * (sine(const(185, n)) + 0.6 * sine(const(270, n))) * decay(n, 0.03)
        out = place(out, hit, at)
    return reverb(out, rng, 0.4, 0.12)


@sound('tv_static', 'TV static burst', 'ui', ['static', 'tv', 'glitch', 'noise', 'signal lost'], ['static', 'tv static', 'white noise', 'signal lost', 'interference'], lufs=-20)
def tv_static(rng):
    n = n_of(1.2)
    hiss = bandpass(noise(n, rng), 300, 9000)
    crackle = (rng.random(n) < 0.002) * rng.uniform(-3, 3, n)
    return (hiss + crackle) * adsr(n, 0.01, 0.05, 1, 0.08)


@sound('glitch', 'Digital glitch', 'ui', ['glitch', 'error', 'digital', 'stutter', 'malfunction'], ['glitch', 'digital glitch', 'malfunction', 'stutter', 'bug'], lufs=-18)
def glitch(rng):
    out = np.zeros(0)
    total = 0.0
    while total < 0.8:
        d = rng.uniform(0.015, 0.07)
        n = n_of(d)
        kind = rng.integers(0, 3)
        if kind == 0:
            seg = square(const(rng.uniform(80, 1800), n), kmax=20)
        elif kind == 1:
            seg = bandpass(noise(n, rng), 500, 8000)
        else:
            seg = np.zeros(n)
        out = np.concatenate([out, seg * rng.uniform(0.3, 1.0)])
        total += d
    return out


@sound('teleport', 'Teleport zap', 'ui', ['teleport', 'zap', 'sci-fi', 'magic', 'warp'], ['teleport', 'warp', 'beam me up', 'zap', 'vanish'], lufs=-18)
def teleport(rng):
    n = n_of(0.9)
    f = glide(200, 2400, n, power=0.7)
    tone = sine(f + 300 * np.sin(phase_safe(const(38, n))))
    return pan_curve(tone * adsr(n, 0.05, 0.1, 0.8, 0.3), 0.6 * np.sin(2 * np.pi * 3 * times(n)))


@sound('magic_sparkle', 'Magic sparkle', 'ui', ['magic', 'sparkle', 'twinkle', 'fairy', 'shine'], ['magic', 'sparkle', 'twinkle', 'fairy dust', 'shimmer', 'magic wand'], lufs=-19)
def magic_sparkle(rng):
    out = silence(1.8, 2)
    for i in range(30):
        at = i * 0.035 + rng.uniform(0, 0.02)
        f = 1800 * 2 ** (i / 30 * 1.6) * rng.uniform(0.95, 1.05)
        n = n_of(0.5)
        ping = sine(const(f, n)) * decay(n, rng.uniform(0.05, 0.15), 0.0005)
        out = place(out, to_stereo(ping, rng.uniform(-0.8, 0.8)), at, rng.uniform(0.3, 0.8))
    return reverb(out, rng, 1.2, 0.3, damp=9000)


# ================================================================ transitions

def whoosh_shape(rng, length, lo, hi, width=0.6, pan=(-0.7, 0.7)):
    n = n_of(length)
    src = noise(n, rng)
    peak_t = length * 0.55

    def mask(f, t):
        x = np.clip(t / length, 0, 1)
        c = lo + (hi - lo) * np.sin(np.pi * x) ** 1.5
        return gauss_band(f, c, width)
    out = stft_filter(src, mask) * bump(n, peak_t, length * 0.5) ** 1.3
    return pan_curve(out, np.linspace(pan[0], pan[1], n))


@sound('whoosh', 'Whoosh', 'transition', ['whoosh', 'swoosh', 'transition', 'swipe', 'fast'],
       ['whoosh', 'woosh', 'swoosh', 'swish', 'transition whoosh', 'swipe'], hit=0.35)
def whoosh(rng):
    return whoosh_shape(rng, 0.65, 400, 2600)


@sound('whoosh_fast', 'Fast whoosh', 'transition', ['whoosh', 'fast', 'swipe', 'zoom'], ['fast whoosh', 'quick whoosh', 'swish', 'whip'], hit=0.17)
def whoosh_fast(rng):
    return whoosh_shape(rng, 0.32, 700, 4500, 0.5, (-0.9, 0.9))


@sound('whoosh_deep', 'Deep whoosh', 'transition', ['whoosh', 'deep', 'heavy', 'cinematic', 'pass by'], ['deep whoosh', 'heavy whoosh', 'big whoosh', 'pass by'], hit=0.6)
def whoosh_deep(rng):
    out = whoosh_shape(rng, 1.1, 120, 900, 0.7)
    n = len(out)
    return out + to_stereo(sine(glide(70, 45, n)) * bump(n, 0.6, 0.5) * 0.3)


@sound('zoom_in', 'Dramatic zoom', 'transition', ['zoom', 'dramatic', 'zoom in', 'focus', 'meme'],
       ['dramatic zoom', 'zoom in', 'zoom', 'camera zoom', 'dramatic zoom sound'], hit=0.55)
def zoom_in(rng):
    n = n_of(0.75)
    tone = sine(glide(220, 1800, n, power=1.6)) * points(n, [(0, 0), (0.55, 1), (0.6, 0.3), (0.75, 0)])
    air = whoosh_shape(rng, 0.75, 500, 4000, 0.5, (0, 0))
    out = air + to_stereo(tone * 0.35)
    m = n_of(0.5)
    return place(out, to_stereo(sine(glide(110, 45, m)) * decay(m, 0.12) * 0.6), 0.55)


@sound('riser', 'Tension riser', 'transition', ['riser', 'build up', 'tension', 'transition', 'buildup'],
       ['riser', 'build up', 'buildup', 'rise', 'tension riser', 'uplifter'], lufs=-17)
def riser(rng):
    n = n_of(4.0)
    t = times(n)
    air = stft_filter(noise(n, rng), lambda f, tt: 1 / np.sqrt(1 + ((200 + 7800 * np.clip(tt / 4.0, 0, 1) ** 2) / np.maximum(f, 1)) ** 4))
    pitch = 2 ** (np.clip(t / 4.0, 0, 1) * 1.0)
    tone = sum(saw(const(midi(m), n) * pitch, kmax=24) for m in (45, 52, 57)) / 3
    trem = tremolo(n, glide(4, 18, n), 0.5)
    env = np.clip(t / 4.0, 0, 1) ** 2
    out = (0.6 * air / max(np.max(np.abs(air)), 1e-9) + 0.5 * lowpass(tone, 3000) * trem) * env
    return to_stereo(out)


@sound('riser_short', 'Short riser', 'transition', ['riser', 'short', 'build up', 'transition'], ['short riser', 'quick riser', 'mini build up'], lufs=-17)
def riser_short(rng):
    n = n_of(1.8)
    t = times(n)
    air = stft_filter(noise(n, rng), lambda f, tt: gauss_band(f, 400 * 2 ** (4 * np.clip(tt / 1.8, 0, 1)), 0.6))
    return to_stereo(air * np.clip(t / 1.8, 0, 1) ** 2)


@sound('reverse_cymbal', 'Reverse cymbal', 'transition', ['reverse', 'cymbal', 'swell', 'transition'], ['reverse cymbal', 'reversed crash', 'swell', 'suck in'], lufs=-17)
def reverse_cymbal(rng):
    return crash(rng, 2.0)[::-1].copy()


@sound('cymbal_crash', 'Cymbal crash', 'transition', ['cymbal', 'crash', 'hit', 'drum'], ['cymbal', 'crash', 'cymbal crash', 'crash cymbal'], lufs=-17)
def cymbal_crash(rng):
    return reverb(crash(rng, 2.4), rng, 0.8, 0.15)


@sound('record_scratch', 'Record scratch', 'transition', ['record', 'scratch', 'stop', 'freeze frame', 'dj', 'meme'],
       ['record scratch', 'vinyl scratch', 'scratch', 'freeze frame', 'needle scratch', 'yep thats me', 'dj scratch'], lufs=-15)
def record_scratch(rng):
    buffer = record_buffer(rng)
    v = np.concatenate([const(1.0, n_of(0.06)), points(n_of(0.5), [(0, 1.0), (0.04, -2.6), (0.12, -2.0), (0.16, 3.2), (0.24, 2.4), (0.3, -1.8), (0.42, -0.8), (0.5, 0.0)])])
    v = lowpass(v, 60, 1)
    out = scrub(buffer, v, 1.0)
    friction = np.abs(v) ** 0.6
    hiss = bandpass(noise(len(v), rng), 1200, 7000) * np.abs(np.gradient(v)) * 800
    out = highpass(out * friction + 0.25 * hiss * np.max(np.abs(out)), 150)
    return concat(out, silence(0.15))


@sound('record_stop', 'Tape stop (power down)', 'transition', ['tape stop', 'power down', 'slow down', 'record stop'],
       ['tape stop', 'record stop', 'power down', 'slow down', 'turntable stop', 'wind down'], lufs=-16)
def record_stop(rng):
    buffer = record_buffer(rng, 4.0)
    n = n_of(1.1)
    v = np.concatenate([const(1.0, n_of(0.15)), glide(1.0, 0.0, n - n_of(0.15), 'lin', power=0.7)])
    return scrub(buffer, v, 0.5) * np.concatenate([const(1, n_of(0.15)), points(n - n_of(0.15), [(0, 1), (0.95, 0.4), (0.95 + 1e-3, 0)])])


@sound('record_rewind', 'Rewind', 'transition', ['rewind', 'tape', 'back', 'flashback', 'reverse'],
       ['rewind', 'tape rewind', 'record rewind', 'go back', 'flashback', 'rewind sound'], lufs=-17)
def record_rewind(rng):
    buffer = np.tile(record_buffer(rng), 6)
    n = n_of(1.4)
    v = -glide(1.0, 9.0, n, power=0.6)
    out = scrub(buffer, v, len(buffer) / SR - 0.1)
    whirr = bandpass(noise(n, rng), 2000, 6000) * 0.05 * np.linspace(0.3, 1, n)
    return highpass(out * points(n, [(0, 0.3), (0.2, 1), (1.3, 1), (1.4, 0)]), 200) + whirr


# ================================================================ drama and suspense

@sound('dramatic_sting', 'Dramatic sting (dun dun dunnn)', 'drama', ['dramatic', 'sting', 'dun dun dun', 'reveal', 'shock', 'plot twist'],
       ['dramatic sting', 'dun dun dun', 'dun dun dunnn', 'dramatic chipmunk', 'plot twist', 'dramatic music', 'shock sting'], hit=0.6)
def dramatic_sting(rng):
    out = silence(3.2, 2)
    for m, at, length in ((50, 0.0, 0.24), (50, 0.3, 0.24), (46, 0.6, 2.2)):
        n = n_of(length)
        f = const(midi(m), n)
        tone = brass(f, n, rng, 1500, 0.03) + 0.8 * brass(f / 2, n, rng, 900, 0.03) + 0.6 * brass(f * 1.5, n, rng, 1500, 0.03)
        if length > 1:
            tone = tone * (1 + 0.15 * np.sin(2 * np.pi * 6 * times(n)))
        out = place(out, to_stereo(tone * adsr(n, 0.01, 0.06, 0.8, 0.8 if length > 1 else 0.04) * 0.4), at)
        tim = sine(glide(110, 70, n_of(0.6))) * decay(n_of(0.6), 0.25)
        out = place(out, to_stereo(tim * 0.5), at)
    return reverb(out, rng, 2.0, 0.3, damp=4000)


@sound('braam', 'Cinematic braam', 'drama', ['braam', 'trailer', 'horn', 'epic', 'cinematic'], ['braam', 'inception horn', 'trailer horn', 'epic horn', 'bwaaam'], lufs=-15)
def braam(rng):
    n = n_of(3.2)
    tone = sum(saw(const(midi(m), n) * (1 + 0.003 * rng.uniform(-1, 1)), kmax=40) for m in (33, 40, 45, 52)) / 4

    def mask(f, t):
        cutoff = 150 + 2400 * np.exp(-((t - 0.35) / 0.5) ** 2) + 300 * (t < 0.35)
        return 1 / np.sqrt(1 + (f / cutoff) ** 4)
    out = softclip(stft_filter(tone, mask) * 2.5, 2.0) * adsr(n, 0.04, 0.3, 0.6, 1.8)
    return reverb(out, rng, 2.5, 0.3, damp=3000)


@sound('suspense_drone', 'Suspense drone', 'drama', ['suspense', 'drone', 'tension', 'horror', 'ominous', 'awkward'],
       ['suspense', 'suspense drone', 'tension drone', 'ominous', 'horror drone', 'creepy', 'uneasy'], lufs=-20)
def suspense_drone(rng):
    n = n_of(6.0)
    t = times(n)
    cluster = sum(saw(const(midi(m), n) * (1 + 0.004 * np.sin(2 * np.pi * rng.uniform(0.1, 0.3) * t)), kmax=20) for m in (33, 34, 40, 45.5))
    pad = lowpass(cluster, 500) * (1 + 0.3 * np.sin(2 * np.pi * 0.25 * t))
    wind = stft_filter(noise(n, rng), lambda f, tt: gauss_band(f, 700 * 2 ** (0.5 * np.sin(2 * np.pi * 0.15 * tt)), 0.4)) * 0.3
    out = (pad / max(np.max(np.abs(pad)), 1e-9) + wind / max(np.max(np.abs(wind)), 1e-9) * 0.3)
    out = out * points(n, [(0, 0), (1.5, 1), (4.5, 1), (6.0, 0)])
    return reverb(out, rng, 2.0, 0.3, damp=2500)


@sound('suspense_sting', 'Horror strings sting', 'drama', ['horror', 'strings', 'scream', 'jump scare', 'sting', 'scary'],
       ['horror sting', 'jump scare', 'scary strings', 'screeching violins', 'shower scene', 'scary'], lufs=-16)
def suspense_sting(rng):
    n = n_of(1.6)
    t = times(n)
    strings = sum(saw(const(f, n) * (1 + 0.01 * np.sin(2 * np.pi * rng.uniform(5, 7) * t)), kmax=12)
                  for f in rng.uniform(1450, 1750, 6))
    bowed = bandpass(strings, 900, 6000) * (0.75 + 0.25 * np.sin(2 * np.pi * 13 * t))
    return reverb(bowed * adsr(n, 0.005, 0.15, 0.6, 0.9), rng, 1.5, 0.3)


@sound('gong', 'Gong', 'drama', ['gong', 'asian', 'announcement', 'hit', 'meditation'], ['gong', 'gong hit', 'tam tam', 'big gong'], lufs=-17)
def gong(rng):
    n = n_of(5.0)
    ratios = [1, 1.48, 2.1, 2.7, 3.3, 4.2, 5.4, 6.9]
    body = partials(n, 98, ratios, [1, .8, .7, .55, .45, .35, .25, .2], [3.5, 3.0, 2.5, 2.0, 1.6, 1.2, 1.0, 0.8], rng, 0.01)
    swell = highpass(noise(n, rng), 1500) * points(n, [(0, 0), (0.4, 0.15), (2.5, 0.05), (5, 0)])
    return reverb(body + swell * 0.5, rng, 2.0, 0.2)


@sound('crickets', 'Crickets (awkward silence)', 'ambience', ['crickets', 'awkward', 'silence', 'night', 'nobody laughed'],
       ['crickets', 'cricket', 'awkward silence', 'chirping crickets', 'nobody laughed'], lufs=-22)
def crickets(rng):
    total = 4.5
    out = silence(total, 2)
    for freq, period, pan in ((4400, 0.62, -0.45), (4750, 0.81, 0.5)):
        at = rng.uniform(0, 0.3)
        while at < total - 0.2:
            for p in range(rng.integers(3, 5)):
                n = n_of(0.016)
                pulse = sine(const(freq * rng.uniform(0.995, 1.005), n)) * np.hanning(n)
                out = place(out, to_stereo(pulse, pan), at + p * 0.024, rng.uniform(0.6, 1.0))
            at += period * rng.uniform(0.92, 1.08)
    night = to_stereo(lowpass(noise(n_of(total), rng), 1500) * 0.01)
    out = place(out, night, 0)
    n = len(out)
    return out * points(n, [(0, 0), (0.3, 1), (total - 0.6, 1), (n / SR, 0)])[:, None]
