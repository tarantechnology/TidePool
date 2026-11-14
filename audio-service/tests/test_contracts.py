

def test_model_aliases_map_unreliable_wordings_to_caption_vocabulary():
    from sam_audio_service.contracts import PromptSpec, normalise_description
    assert normalise_description("Narrator speaking!") == "speech"
    assert normalise_description("isolate the voiceover") == "speech"
    assert normalise_description("man speaking") == "man speaking"      # gendered wording stays
    assert normalise_description("dog barking") == "dog barking"
    spec = PromptSpec.model_validate({"text": "Narrator speaking"})
    assert spec.text == "speech" and spec.name == "Narrator speaking"   # the stem keeps the caller's words
