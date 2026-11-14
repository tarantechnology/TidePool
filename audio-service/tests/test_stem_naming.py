"""Stem names must keep the caller's wording: Person 3's editor matches stems by the prompt it sent.

The rule lives on ``PromptSpec`` itself, so the string form (``["Isolate the Dog!"]``) and the
object form (``[{"text": "Isolate the Dog!"}]``) name their stem identically.
"""
import json

import pytest

from sam_audio_service.contracts import PromptSpec, SeparationMetadata


def test_plain_string_prompt_keeps_callers_wording_as_stem_name():
    specs = SeparationMetadata(prompts=["Isolate the Dog Barking!", " siren "]).resolved_prompts(8)
    assert [s.display_name for s in specs] == ["Isolate the Dog Barking!", "siren"]
    assert [s.text for s in specs] == ["dog barking", "siren"]  # what the model sees


@pytest.mark.parametrize("wording", ["Isolate the Dog Barking!", "police  siren\n", "a" * 300])
def test_string_and_object_forms_name_the_stem_identically(wording):
    """The naming rule is a ``PromptSpec`` default, not something the string branch adds."""
    from_string, from_object = SeparationMetadata(prompts=[wording, {"text": wording}]).resolved_prompts(8)
    assert from_string.name == from_object.name
    assert from_string.text == from_object.text
    assert from_string.display_name == from_object.display_name


def test_default_name_collapses_whitespace_but_keeps_case_and_punctuation():
    assert PromptSpec(text="police  siren\n").name == "police siren"
    assert PromptSpec(text="  Isolate the Dog Barking!  ").name == "Isolate the Dog Barking!"


def test_long_prompt_round_trips_untruncated():
    """``name`` allows as much as ``text`` (500), so no stem name is silently cut short."""
    wording = "Isolate the " + "dog barking " * 24  # 300 characters
    assert len(wording) == 300
    spec = PromptSpec(text=wording)
    assert spec.name == wording.strip()
    assert spec.display_name == wording.strip()


def test_punctuation_only_prompt_stays_unnamed_and_falls_back_to_the_visual_label():
    spec = PromptSpec(text="?", visual_prompt={"label": "dog"})
    assert spec.name is None and spec.text == ""
    assert spec.display_name == "dog"


def test_prompt_object_name_still_wins():
    specs = SeparationMetadata(prompts=[{"text": "Police Siren", "name": "Siren track"}]).resolved_prompts(8)
    assert specs[0].display_name == "Siren track" and specs[0].text == "police siren"


def test_stem_name_round_trips_through_the_api(client, wav_path):
    r = client.post("/separate", files={"audio": ("a.wav", wav_path.read_bytes(), "audio/wav")}, data={"metadata": json.dumps({"prompts": ["Isolate the dog barking!"]})})
    assert r.status_code == 200, r.text
    stems = r.json()["stems"]
    assert stems[0]["name"] == "Isolate the dog barking!" and stems[0]["prompt"] == "dog barking"
    # The residual is named by ``options.residual_name`` and is untouched by the prompt rule.
    assert stems[-1]["name"] == "residual"
