"""Stem names must keep the caller's wording: Person 3's editor matches stems by the prompt it sent."""
import json

from sam_audio_service.contracts import SeparationMetadata


def test_plain_string_prompt_keeps_callers_wording_as_stem_name():
    specs = SeparationMetadata(prompts=["Isolate the Dog Barking!", " siren "]).resolved_prompts(8)
    assert [s.display_name for s in specs] == ["Isolate the Dog Barking!", "siren"]
    assert [s.text for s in specs] == ["dog barking", "siren"]  # what the model sees


def test_prompt_object_name_still_wins():
    specs = SeparationMetadata(prompts=[{"text": "Police Siren", "name": "Siren track"}]).resolved_prompts(8)
    assert specs[0].display_name == "Siren track" and specs[0].text == "police siren"


def test_stem_name_round_trips_through_the_api(client, wav_path):
    r = client.post("/separate", files={"audio": ("a.wav", wav_path.read_bytes(), "audio/wav")}, data={"metadata": json.dumps({"prompts": ["Isolate the dog barking!"]})})
    assert r.status_code == 200, r.text
    stems = r.json()["stems"]
    assert stems[0]["name"] == "Isolate the dog barking!" and stems[0]["prompt"] == "dog barking"
    assert stems[-1]["name"] == "residual"
