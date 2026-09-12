"""Unit tests for orchestration rules in the manual tool-calling loop."""

from app.llm.loop import _defer_until_next_round


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
