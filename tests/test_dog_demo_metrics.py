import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('dog_demo', Path(__file__).parents[1]/'scripts/run_audiosep_dog_demo.py')
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


def sources():
    rng = np.random.default_rng(42)
    return {name:rng.normal(size=1000) for name in ['speaker_a','speaker_b','barking']}


def test_projection_distinguishes_bark_reduction_from_speech_damage():
    refs = sources()
    mixture = sum(refs.values())
    residual = refs['speaker_a'] + .5*refs['speaker_b'] + .1*refs['barking']
    result = demo.reference_metrics(residual, mixture, refs)
    assert result['barking_reduction_db_projection'] == pytest.approx(20)
    assert result['speaker_a_gain_db'] == pytest.approx(0, abs=1e-10)
    assert result['speaker_b_gain_db'] == pytest.approx(-6.0205999)


def test_unedited_mixture_does_not_claim_improvement():
    refs = sources()
    mixture = sum(refs.values())
    result = demo.reference_metrics(mixture, mixture, refs)
    assert result['barking_reduction_db_projection'] == pytest.approx(0, abs=1e-10)
    assert result['speech_si_sdri_db'] == pytest.approx(0)


def test_metric_rejects_length_drift_and_degenerate_references():
    refs = sources()
    mixture = sum(refs.values())
    with pytest.raises(ValueError, match='sample counts'):
        demo.reference_metrics(mixture[:-1], mixture, refs)
    refs['speaker_b'] = refs['speaker_a']
    with pytest.raises(ValueError, match='independent'):
        demo.reference_metrics(mixture, mixture, refs)
