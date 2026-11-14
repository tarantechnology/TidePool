

def test_refine_defaults_to_mask_for_the_real_backend_and_off_for_mock(monkeypatch):
    from sam_audio_service.config import Settings
    monkeypatch.delenv("AUDIO_REFINE", raising=False)
    assert Settings(backend="mock").refine == "off"
    assert Settings(backend="sam").refine == "mask"
    monkeypatch.setenv("AUDIO_REFINE", "off")
    assert Settings(backend="sam").refine == "off"
    monkeypatch.setenv("AUDIO_CANDIDATE_SELECT", "auto")
    assert Settings(backend="mock").candidate_select == "auto"


def test_pass_safeguards_default_on_for_the_real_backend_only(monkeypatch):
    from sam_audio_service.config import Settings
    monkeypatch.delenv("AUDIO_PASS_MAX_RATIO_DB", raising=False); monkeypatch.delenv("AUDIO_PASS_MIN_SIMILARITY", raising=False)
    sam, mock = Settings(backend="sam"), Settings(backend="mock")
    assert (sam.pass_max_ratio_db, sam.pass_min_similarity) == (-3.0, 0.6)
    assert (mock.pass_max_ratio_db, mock.pass_min_similarity) == (20.0, 0.0)
    monkeypatch.setenv("AUDIO_PASS_MAX_RATIO_DB", "-6"); monkeypatch.setenv("AUDIO_PASS_MIN_SIMILARITY", "0.8")
    env = Settings(backend="mock")
    assert (env.pass_max_ratio_db, env.pass_min_similarity) == (-6.0, 0.8)
