"""Integrity checks on the hand-authored data in data/curated/.

Pure file validation — no database. These catch the failure mode where a typo in a
CSV silently produces a fallback referral instead of the curated one, which the
application would never complain about.
"""

from __future__ import annotations

import csv

import pytest

from app.config import CURATED_DIR

VALID_LEVELS = {"emergency", "urgent"}


def read(name: str) -> list[dict]:
    with (CURATED_DIR / name).open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


@pytest.fixture(scope="module")
def specialty_slugs() -> set[str]:
    return {r["slug"] for r in read("specialties.csv")}


def test_specialty_slugs_are_unique():
    rows = read("specialties.csv")
    slugs = [r["slug"] for r in rows]
    assert len(slugs) == len(set(slugs))


def test_every_specialty_has_a_description():
    for r in read("specialties.csv"):
        assert r["description"].strip(), f"{r['slug']} has no description"


def test_general_practice_exists(specialty_slugs):
    """The fallback tier depends on it; without it every unmapped disease breaks."""
    assert "general-practice" in specialty_slugs


def test_disease_mappings_reference_real_specialties(specialty_slugs):
    for r in read("disease_specialty.csv"):
        assert r["specialty_slug"] in specialty_slugs, (
            f"{r['disease_slug']} maps to unknown specialty {r['specialty_slug']}"
        )


def test_no_disease_mapped_twice_to_same_specialty():
    seen = set()
    for r in read("disease_specialty.csv"):
        key = (r["disease_slug"], r["specialty_slug"])
        assert key not in seen, f"duplicate mapping {key}"
        seen.add(key)


def test_body_system_routes_reference_real_specialties(specialty_slugs):
    for r in read("body_system_specialty.csv"):
        assert r["specialty_slug"] in specialty_slugs, (
            f"{r['body_system']} routes to unknown specialty {r['specialty_slug']}"
        )


def test_every_symptom_has_a_body_system():
    rows = read("body_systems.csv")
    missing = [r["symptom_name"] for r in rows if not r["body_system"].strip()]
    assert not missing, f"{len(missing)} symptoms untagged: {missing[:5]}"


def test_body_systems_are_all_routable():
    """A tag with no route produces a symptom that can never reach a specialty."""
    routes = {r["body_system"] for r in read("body_system_specialty.csv")}
    used = {r["body_system"] for r in read("body_systems.csv") if r["body_system"]}
    assert used <= routes, f"unroutable body systems: {sorted(used - routes)}"


def test_red_flag_levels_are_valid():
    for r in read("red_flags.csv"):
        assert r["level"] in VALID_LEVELS, f"{r['name']} has level {r['level']!r}"


def test_red_flags_have_requirements_and_messages():
    for r in read("red_flags.csv"):
        assert r["require_all"].strip(), f"{r['name']} requires no symptoms"
        assert len(r["message"]) > 30, f"{r['name']} message is too terse to act on"


def test_emergency_flags_tell_the_user_to_seek_care_now():
    """An emergency alert that does not say what to do is not an alert."""
    for r in read("red_flags.csv"):
        if r["level"] != "emergency":
            continue
        assert any(
            phrase in r["message"].lower()
            for phrase in ("emergency", "immediately", "now")
        ), f"{r['name']} does not convey urgency"
