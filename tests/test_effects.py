import subprocess
import numpy as np, soundfile as sf, pytest
from app.stem_fx import apply_effect

@pytest.mark.parametrize('effect', ['deepen', 'impact', 'disorient', 'muffle'])
def test_effect_only_changes_the_range(tmp_path, effect):
    src = tmp_path / 'a.wav'
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i', 'sine=frequency=440:duration=3', '-ar', '48000', str(src)], check=True)
    out = apply_effect(src, effect, 1.0, 2.0)
    a, sr = sf.read(src, always_2d=True); b, _ = sf.read(out, always_2d=True)
    assert len(a) == len(b)
    outside = np.r_[0:int(0.9 * sr), int(2.1 * sr):len(a)]
    assert np.allclose(b[outside, 0], a[outside, 0], atol=1e-4)
    inside = slice(int(1.2 * sr), int(1.8 * sr))
    assert not np.allclose(b[inside, 0], a[inside, 0], atol=1e-3)
    assert np.abs(b).max() <= 1.0

def test_rejects_range_outside_stem(tmp_path):
    src = tmp_path / 'a.wav'
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i', 'sine=duration=1', str(src)], check=True)
    with pytest.raises(ValueError): apply_effect(src, 'deepen', 0.5, 5)
