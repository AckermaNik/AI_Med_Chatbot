"""Unit tests for the ranking engine. No database, no LLM, no network.

The point of M2 is that all of this is verifiable while nothing is stochastic.
"""

from __future__ import annotations

import math

import pytest

from app.config import ScoringConfig
from app.engine.scoring import (
    DiseaseProfile,
    SymptomInfo,
    discriminating_symptom,
    rank,
    score_one,
    similarity,
)

# --------------------------------------------------------------------------- #
# fixtures mirroring the worked example in docs/PLAN.md section 5
# --------------------------------------------------------------------------- #

ITCHING, RASH, NODULES, PATCHES = 1, 2, 3, 4
FATIGUE, FEVER, HEADACHE = 5, 6, 7

SYMPTOMS = {
    ITCHING: SymptomInfo(ITCHING, "itching", "itching", idf=1.0, severity=1),
    RASH: SymptomInfo(RASH, "skin-rash", "skin rash", idf=1.5, severity=3),
    NODULES: SymptomInfo(NODULES, "nodal-skin-eruptions", "nodal skin eruptions", idf=4.0, severity=4),
    PATCHES: SymptomInfo(PATCHES, "discoloured-patches", "discoloured patches", idf=4.5, severity=6),
    FATIGUE: SymptomInfo(FATIGUE, "fatigue", "fatigue", idf=0.6, severity=4),
    FEVER: SymptomInfo(FEVER, "high-fever", "high fever", idf=0.8, severity=7),
    HEADACHE: SymptomInfo(HEADACHE, "headache", "headache", idf=0.5, severity=3),
}

FUNGAL = DiseaseProfile(
    id=1, slug="fungal-infection", name="fungal infection",
    support={ITCHING: 1.0, RASH: 1.0, NODULES: 0.9, PATCHES: 0.8},
)
CHICKENPOX = DiseaseProfile(
    id=2, slug="chickenpox", name="chickenpox",
    support={ITCHING: 0.9, RASH: 1.0, FATIGUE: 0.9, FEVER: 1.0, HEADACHE: 0.8},
)


def cfg(**overrides) -> ScoringConfig:
    base = {"mu": 0.0, "mmr_lambda": 1.0, "top_k": 5, "mmr_pool": 20}
    return ScoringConfig(**{**base, **overrides})


# --------------------------------------------------------------------------- #
# the documented worked example
# --------------------------------------------------------------------------- #


def test_worked_example_matches_documented_numbers():
    """docs/PLAN.md section 5 claims 0.51 and 0.75. Verify, don't trust."""
    reported = {ITCHING, RASH}
    norm_u = math.sqrt(1.0 + 1.5)

    fungal = score_one(reported, FUNGAL, SYMPTOMS, norm_u, mu=0.0)
    chicken = score_one(reported, CHICKENPOX, SYMPTOMS, norm_u, mu=0.0)

    assert fungal == pytest.approx(0.51, abs=0.01)
    assert chicken == pytest.approx(0.75, abs=0.01)


def test_denominator_penalises_unreported_distinctive_symptoms():
    """Chickenpox beats fungal infection precisely because fungal infection is
    missing its two rarest symptoms. That is the D(d) term working, not a bug."""
    reported = {ITCHING, RASH}
    norm_u = math.sqrt(1.0 + 1.5)

    assert score_one(reported, CHICKENPOX, SYMPTOMS, norm_u, 0.0) > score_one(
        reported, FUNGAL, SYMPTOMS, norm_u, 0.0
    )


def test_one_extra_symptom_flips_the_ranking():
    """Confirming the discriminating symptom reverses the leader decisively."""
    reported = {ITCHING, RASH, NODULES}
    norm_u = math.sqrt(1.0 + 1.5 + 4.0)

    fungal = score_one(reported, FUNGAL, SYMPTOMS, norm_u, 0.0)
    chicken = score_one(reported, CHICKENPOX, SYMPTOMS, norm_u, 0.0)

    assert fungal == pytest.approx(0.77, abs=0.01)
    assert chicken == pytest.approx(0.46, abs=0.01)
    assert fungal > chicken


# --------------------------------------------------------------------------- #
# the mu prior
# --------------------------------------------------------------------------- #


def test_thin_disease_scores_perfectly_without_mu():
    """The failure mode mu exists to prevent: one symptom, one disease, score 1.00."""
    thin = DiseaseProfile(id=9, slug="thin", name="thin", support={ITCHING: 1.0})
    score = score_one({ITCHING}, thin, SYMPTOMS, math.sqrt(1.0), mu=0.0)
    assert score == pytest.approx(1.0)


def test_mu_damps_thin_diseases_far_more_than_well_documented_ones():
    thin = DiseaseProfile(id=9, slug="thin", name="thin", support={ITCHING: 1.0})

    thin_before = score_one({ITCHING, RASH}, thin, SYMPTOMS, math.sqrt(2.5), mu=0.0)
    thin_after = score_one({ITCHING, RASH}, thin, SYMPTOMS, math.sqrt(2.5), mu=2.0)

    rich_before = score_one({ITCHING, RASH}, FUNGAL, SYMPTOMS, math.sqrt(2.5), mu=0.0)
    rich_after = score_one({ITCHING, RASH}, FUNGAL, SYMPTOMS, math.sqrt(2.5), mu=2.0)

    thin_drop = (thin_before - thin_after) / thin_before
    rich_drop = (rich_before - rich_after) / rich_before

    assert thin_drop > 3 * rich_drop


# --------------------------------------------------------------------------- #
# invariants
# --------------------------------------------------------------------------- #


def test_score_never_exceeds_one():
    perfect = DiseaseProfile(
        id=8, slug="p", name="p", support={ITCHING: 1.0, RASH: 1.0}
    )
    score = score_one({ITCHING, RASH}, perfect, SYMPTOMS, math.sqrt(2.5), mu=0.0)
    assert score <= 1.0 + 1e-9


def test_no_overlap_scores_zero():
    assert score_one({FEVER}, FUNGAL, SYMPTOMS, math.sqrt(0.8), mu=0.0) == 0.0


def test_unknown_symptom_ids_are_ignored_not_crashed():
    result = rank({ITCHING, 9999}, [FUNGAL, CHICKENPOX], SYMPTOMS, cfg())
    assert result.candidates


def test_empty_input_returns_nothing():
    assert rank(set(), [FUNGAL], SYMPTOMS, cfg()).candidates == []


def test_ranking_is_deterministic():
    a = rank({ITCHING, RASH}, [FUNGAL, CHICKENPOX], SYMPTOMS, cfg())
    b = rank({ITCHING, RASH}, [CHICKENPOX, FUNGAL], SYMPTOMS, cfg())
    assert [c.slug for c in a.candidates] == [c.slug for c in b.candidates]


# --------------------------------------------------------------------------- #
# explanation payload
# --------------------------------------------------------------------------- #


def test_matched_symptoms_sorted_by_contribution():
    result = rank({ITCHING, RASH, NODULES}, [FUNGAL], SYMPTOMS, cfg())
    contributions = [m.contribution for m in result.candidates[0].matched]
    assert contributions == sorted(contributions, reverse=True)
    assert result.candidates[0].matched[0].name == "nodal skin eruptions"


def test_missing_key_lists_highest_idf_unreported():
    result = rank({ITCHING, RASH}, [FUNGAL], SYMPTOMS, cfg())
    assert result.candidates[0].missing_key[0] == "discoloured patches"


def test_urgency_is_max_severity_of_matched_only():
    """Fungal infection's severity-6 symptom is not reported, so urgency is 3."""
    result = rank({ITCHING, RASH}, [FUNGAL], SYMPTOMS, cfg())
    assert result.candidates[0].urgency == 3


def test_urgency_ignores_unknown_severity():
    mystery = SymptomInfo(20, "mystery", "mystery", idf=3.0, severity=None)
    symptoms = {**SYMPTOMS, 20: mystery}
    profile = DiseaseProfile(id=10, slug="x", name="x", support={20: 1.0})
    result = rank({20}, [profile], symptoms, cfg())
    assert result.candidates[0].urgency is None


# --------------------------------------------------------------------------- #
# MMR
# --------------------------------------------------------------------------- #


def test_similarity_is_symmetric_and_self_is_one():
    assert similarity(FUNGAL, CHICKENPOX, SYMPTOMS) == pytest.approx(
        similarity(CHICKENPOX, FUNGAL, SYMPTOMS)
    )
    assert similarity(FUNGAL, FUNGAL, SYMPTOMS) == pytest.approx(1.0)


def test_mmr_lambda_one_leaves_order_untouched():
    profiles = [FUNGAL, CHICKENPOX]
    pure = rank({ITCHING, RASH}, profiles, SYMPTOMS, cfg(mmr_lambda=1.0))
    assert [c.slug for c in pure.candidates] == ["chickenpox", "fungal-infection"]


def test_mmr_never_changes_the_top_result():
    """S is empty on the first pick, so it is pure argmax either way. This is why
    top-1 accuracy is unaffected by MMR while top-3 can move."""
    profiles = [FUNGAL, CHICKENPOX]
    pure = rank({ITCHING, RASH}, profiles, SYMPTOMS, cfg(mmr_lambda=1.0))
    diverse = rank({ITCHING, RASH}, profiles, SYMPTOMS, cfg(mmr_lambda=0.5))
    assert pure.candidates[0].slug == diverse.candidates[0].slug


def test_mmr_demotes_a_near_duplicate():
    twin = DiseaseProfile(
        id=3, slug="chickenpox-twin", name="chickenpox twin",
        support=dict(CHICKENPOX.support),
    )
    distinct = DiseaseProfile(
        id=4, slug="distinct", name="distinct",
        support={ITCHING: 0.8, PATCHES: 0.7},
    )
    profiles = [CHICKENPOX, twin, distinct]

    pure = rank({ITCHING, RASH}, profiles, SYMPTOMS, cfg(mmr_lambda=1.0, top_k=2))
    diverse = rank({ITCHING, RASH}, profiles, SYMPTOMS, cfg(mmr_lambda=0.3, top_k=2))

    assert pure.candidates[1].slug == "chickenpox-twin"
    assert diverse.candidates[1].slug == "distinct"


# --------------------------------------------------------------------------- #
# discriminating symptom
# --------------------------------------------------------------------------- #


def test_picks_a_symptom_that_splits_the_candidates():
    result = rank({ITCHING, RASH}, [FUNGAL, CHICKENPOX], SYMPTOMS, cfg())
    assert result.discriminating_symptom in {
        "nodal-skin-eruptions", "discoloured-patches", "high-fever",
    }


def test_never_asks_about_a_symptom_all_candidates_share():
    """A question whose answer you already know is a wasted turn."""
    shared = DiseaseProfile(id=5, slug="a", name="a", support={ITCHING: 1.0, FATIGUE: 1.0})
    other = DiseaseProfile(id=6, slug="b", name="b", support={ITCHING: 1.0, FATIGUE: 1.0})
    result = rank({ITCHING}, [shared, other], SYMPTOMS, cfg())
    assert result.discriminating_symptom != "fatigue"


def test_prefers_rarer_symptom_when_split_is_equal():
    a = DiseaseProfile(id=5, slug="a", name="a", support={ITCHING: 1.0, PATCHES: 1.0})
    b = DiseaseProfile(id=6, slug="b", name="b", support={ITCHING: 1.0, HEADACHE: 1.0})
    result = rank({ITCHING}, [a, b], SYMPTOMS, cfg())
    assert result.discriminating_symptom == "discoloured-patches"


def test_no_question_when_only_one_candidate():
    result = rank({ITCHING, RASH}, [FUNGAL], SYMPTOMS, cfg())
    assert result.discriminating_symptom is None


def test_confidence_margin_gates_further_questions():
    clear = rank({ITCHING, RASH, NODULES, PATCHES}, [FUNGAL, CHICKENPOX], SYMPTOMS, cfg())
    assert clear.is_confident
