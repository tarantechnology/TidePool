

def test_refine_defaults_to_mask_for_the_real_backend_and_off_for_mock(monkeypatch):
    from sam_audio_service.config import Settings
    monkeypatch.delenv("AUDIO_REFINE", raising=False)
    assert Settings(backend="mock").refine == "off"
    assert Settings(backend="sam").refine == "mask"
    monkeypatch.setenv("AUDIO_REFINE", "off")
    assert Settings(backend="sam").refine == "off"
    monkeypatch.setenv("AUDIO_CANDIDATE_SELECT", "auto")
    assert Settings(backend="mock").candidate_select == "auto"
