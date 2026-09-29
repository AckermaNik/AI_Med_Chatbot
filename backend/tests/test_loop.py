"""Unit tests for orchestration rules in the manual tool-calling loop."""

from app.llm.loop import (
    _defer_until_next_round,
    _merge_matched_symptoms,
    normalize_reply,
)


class _Call:
    def __init__(self, name):
        self.name = name


def test_diagnosis_waits_for_a_same_batch_symptom_search():
    assert _defer_until_next_round("diagnose", {"search_symptoms", "diagnose"})


def test_unrelated_calls_are_not_deferred():
    assert not _defer_until_next_round("search_symptoms", {"search_symptoms", "diagnose"})
    assert not _defer_until_next_round("recommend_specialty", {"recommend_specialty"})


def test_diagnosis_can_run_in_a_later_tool_round():
    assert not _defer_until_next_round("diagnose", {"diagnose"})


def test_disease_info_waits_for_same_batch_diagnosis():
    assert _defer_until_next_round("get_disease_info", {"diagnose", "get_disease_info"})


def test_specialty_waits_for_its_same_batch_inputs():
    assert _defer_until_next_round("recommend_specialty", {"search_symptoms", "recommend_specialty"})
    assert _defer_until_next_round("recommend_specialty", {"diagnose", "recommend_specialty"})


def test_normalize_reply_enforces_plain_text_and_final_disclaimer():
    reply = normalize_reply(
        "You may have **common cold**, and vomiting. Please note that I am not a "
        "substitute for a real medical assessment. See a GP."
    )
    assert "**" not in reply
    assert ", and" not in reply


def test_empty_follow_up_search_keeps_previous_symptoms():
    active = ["headache"]
    assert not _merge_matched_symptoms(
        active,
        [_Call("search_symptoms")],
        [{"matched": [], "unmatched": [{"phrase": "no"}]}],
    )
    assert active == ["headache"]


def test_search_with_a_match_can_continue_to_diagnosis():
    active = ["headache"]
    assert _merge_matched_symptoms(
        active,
        [_Call("search_symptoms")],
        [{"matched": [{"slug": "headache"}]}],
    )
    assert active == ["headache"]


def test_new_symptoms_are_added_to_previous_positive_symptoms():
    active = ["headache"]
    assert _merge_matched_symptoms(
        active,
        [_Call("search_symptoms")],
        [{"matched": [{"slug": "high-fever"}]}],
    )
    assert active == ["headache", "high-fever"]
