import numpy as np
import pytest

from sam_audio_service.suppression import suppress_target


def test_no_target_or_zero_strength_leaves_original_samples_intact():
    mixture = np.random.default_rng(7).normal(0, .1, (2, 2000)).astype(np.float32)
    assert np.array_equal(suppress_target(mixture, np.zeros_like(mixture)), mixture)
    assert np.array_equal(suppress_target(mixture, mixture, strength=0), mixture)


def test_weak_speech_leakage_does_not_change_quiet_target_passage():
    sr = 48000
    time = np.arange(sr)/sr
    speech = .1*np.sin(2*np.pi*300*time)
    barking = .3*np.sin(2*np.pi*2400*time)*(time > .5)
    mixture = (speech+barking)[None].astype(np.float32)
    estimate = (barking+.01*speech)[None].astype(np.float32)
    out = suppress_target(mixture, estimate)
    # The model erroneously includes speech throughout, but the quiet-target
    # interval must pass through unchanged away from the activity transition.
    np.testing.assert_array_equal(out[:, 1000:20000], mixture[:, 1000:20000])
    # Reduction must still do real work once the target becomes loud.
    segment = slice(30000, 44000)
    projection = np.dot(out[0, segment], barking[segment])/np.dot(barking[segment], barking[segment])
    assert .08 < projection < .14


@pytest.mark.parametrize('options', [{'strength':1.1}, {'mask_threshold':1},
    {'activity_low_db':-20, 'activity_high_db':-30}, {'strength':float('nan')}])
def test_invalid_options_are_rejected(options):
    with pytest.raises(ValueError):
        suppress_target(np.ones((1, 100)), np.ones((1, 100)), **options)


def test_timeline_mismatch_is_rejected():
    with pytest.raises(ValueError, match='identical'):
        suppress_target(np.ones((1, 100)), np.ones((1, 99)))
