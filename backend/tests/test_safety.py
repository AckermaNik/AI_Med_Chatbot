"""Regression tests for deterministic raw-text safety phrases."""

import re

from app.engine.safety import TEXT_SYMPTOM_PATTERNS


def test_spitting_blood_is_detected_as_hemoptysis():
    patterns = TEXT_SYMPTOM_PATTERNS["hemoptysis"]
    assert any(re.search(pattern, "i also spit blood") for pattern in patterns)
