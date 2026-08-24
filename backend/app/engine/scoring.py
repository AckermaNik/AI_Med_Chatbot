"""It takes a list of symptoms a user typed in, compares it against a library of diseases, 
and ranks them purely using math—no AI guessing, just hard numbers.

Everything here is a pure function over plain data structures, which is why it can
be unit-tested exhaustively without Postgres running. The repository layer is
responsible for turning database rows into the inputs below.

    u (User Symptoms):      The list of symptoms the user reported having (e.g., "itching", "skin rash").
    S(d) (Disease Symptoms):    The list of symptoms a specific disease is known to cause.
    IDF (Inverse Document Frequency):       The rarity score of a symptom. A common symptom like fatigue has a low IDF (e.g., 0.5)
    Support:        A probability score of how strongly a disease is linked to a symptom.
    
    GOAL: calculate a final score (between 0 and 1) for every possible disease.

        score(d) = num(d) / (norm_u · √(D(d) + μ))          ∈ [0, 1]

        WHERE: 
        num(d)   = Σ_{s ∈ u ∩ S(d)} idf(s) · support(d, s) -> match on a rare symptom jumps THE score massively.
        norm_u   = √( Σ_{s ∈ u} idf(s) ) -> Penalizes if user reported a bunch of symptoms the disease doesn't explain.
        D(d)     = Σ_{s ∈ S(d)} idf(s) · support(d, s) -> Penalizes if the disease causes highly specific symptoms that the user didn't mention.
        μ (mu)   = Penalty that stops thinly-documented diseases scoring 1.00 off a single common symptom with user.
        mmr (Maximal Marginal Relevance)  = Aacts as a diversity filter. After scoring the diseases and choosing the 20 most fitted ones, it penalizes candidates that are too similar to the ones already chosen.
    
    NUMERATOR -> The Match at the symptoms the user and the disease share.
    DENOMINATOR -> The Reality Check. If we only looked at the match, a disease with 500 symptoms (like the common cold) would match everything. 
    
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from app.config import ScoringConfig


@dataclass(frozen=True)
class SymptomInfo:
    """Everything ranking needs to know about one symptom."""

    id: int
    slug: str
    name: str
    idf: float
    severity: int | None = None
    body_system: str | None = None


@dataclass(frozen=True)
class DiseaseProfile:
    """One disease and its symptom support distribution."""

    id: int
    slug: str
    name: str
    support: dict[int, float]  # symptom_id -> P(symptom | disease)
    low_evidence: bool = False


@dataclass(frozen=True)
class MatchedSymptom:
    slug: str
    name: str
    idf: float
    support: float

    @property
    def contribution(self) -> float:
        """How much this symptom added to the numerator. Drives the UI's 'because…'."""
        return self.idf * self.support


@dataclass
class Candidate:
    disease_id: int
    slug: str
    name: str
    score: float
    matched: list[MatchedSymptom] = field(default_factory=list)
    missing_key: list[str] = field(default_factory=list)
    urgency: int | None = None
    low_evidence: bool = False


@dataclass
class Ranking:
    candidates: list[Candidate]
    discriminating_symptom: str | None = None
    unmatched_reported: list[str] = field(default_factory=list)

    @property
    def is_confident(self) -> bool:
        """True when the leader is clear enough to stop asking questions."""
        if not self.candidates:
            return False
        if len(self.candidates) == 1:
            return True
        return (self.candidates[0].score - self.candidates[1].score) >= self._margin

    _margin: float = 0.15


# --------------------------------------------------------------------------- #
# core scoring
# --------------------------------------------------------------------------- #


def _evidence_mass(profile: DiseaseProfile, symptoms: dict[int, SymptomInfo]) -> float:
    """D(d) — total IDF-weighted evidence this disease is expected to present."""
    return sum(
        symptoms[sid].idf * support
        for sid, support in profile.support.items()
        if sid in symptoms
    )


def score_one(
    reported: set[int],
    profile: DiseaseProfile,
    symptoms: dict[int, SymptomInfo],
    norm_u: float,
    mu: float,
) -> float:
    numerator = sum(
        symptoms[sid].idf * profile.support[sid]
        for sid in reported & profile.support.keys()
        if sid in symptoms
    )
    if numerator <= 0 or norm_u <= 0:
        return 0.0
    denominator = norm_u * math.sqrt(_evidence_mass(profile, symptoms) + mu)
    return numerator / denominator if denominator > 0 else 0.0


def rank(
    reported: set[int],
    profiles: list[DiseaseProfile],
    symptoms: dict[int, SymptomInfo],
    config: ScoringConfig,
) -> Ranking:
    """Score, pool, MMR re-rank, then pick the next question."""
    known = {sid for sid in reported if sid in symptoms}
    if not known:
        return Ranking(candidates=[])

    norm_u = math.sqrt(sum(symptoms[sid].idf for sid in known))

    scored: list[tuple[float, DiseaseProfile]] = []
    for profile in profiles:
        s = score_one(known, profile, symptoms, norm_u, config.mu)
        if s > 0:
            scored.append((s, profile))

    scored.sort(key=lambda pair: (-pair[0], pair[1].name))
    pool = scored[: config.mmr_pool]

    selected = _mmr(pool, symptoms, config)

    candidates = [
        _explain(score, profile, known, symptoms) for score, profile in selected
    ]

    ranking = Ranking(
        candidates=candidates,
        unmatched_reported=sorted(
            symptoms[sid].slug
            for sid in known
            if not any(sid in p.support for _, p in selected)
        ),
    )
    ranking._margin = config.confident_margin
    ranking.discriminating_symptom = discriminating_symptom(
        candidates, selected, known, symptoms
    )
    return ranking


def _explain(
    score: float,
    profile: DiseaseProfile,
    reported: set[int],
    symptoms: dict[int, SymptomInfo],
) -> Candidate:
    matched = [
        MatchedSymptom(
            slug=symptoms[sid].slug,
            name=symptoms[sid].name,
            idf=symptoms[sid].idf,
            support=profile.support[sid],
        )
        for sid in reported & profile.support.keys()
        if sid in symptoms
    ]
    matched.sort(key=lambda m: -m.contribution)

    unreported = [
        sid for sid in profile.support if sid not in reported and sid in symptoms
    ]
    unreported.sort(key=lambda sid: -symptoms[sid].idf)

    severities = [
        symptoms[m_sid].severity
        for m_sid in reported & profile.support.keys()
        if m_sid in symptoms and symptoms[m_sid].severity is not None
    ]

    return Candidate(
        disease_id=profile.id,
        slug=profile.slug,
        name=profile.name,
        score=round(score, 4),
        matched=matched,
        missing_key=[symptoms[sid].name for sid in unreported[:3]],
        urgency=max(severities) if severities else None,
        low_evidence=profile.low_evidence,
    )


# --------------------------------------------------------------------------- #
# MMR
# --------------------------------------------------------------------------- #


def _vector_norm(profile: DiseaseProfile, symptoms: dict[int, SymptomInfo]) -> float:
    """L2 norm of the disease's (idf · support) vector."""
    return math.sqrt(
        sum(
            (symptoms[sid].idf * support) ** 2
            for sid, support in profile.support.items()
            if sid in symptoms
        )
    )


def similarity(
    a: DiseaseProfile, b: DiseaseProfile, symptoms: dict[int, SymptomInfo]
) -> float:
    """True IDF-weighted cosine between two diseases, in [0, 1].

    Deliberately normalised differently from score_one(): that function compares a
    presence-weighted user vector against a support-weighted disease vector, which
    is a useful hybrid but not a strict cosine. MMR needs a genuine similarity —
    in particular similarity(d, d) must be exactly 1.0, or a perfect duplicate
    receives too small a redundancy penalty.
    """
    shared = a.support.keys() & b.support.keys()
    if not shared:
        return 0.0
    num = sum(
        (symptoms[sid].idf * a.support[sid]) * (symptoms[sid].idf * b.support[sid])
        for sid in shared
        if sid in symptoms
    )
    denom = _vector_norm(a, symptoms) * _vector_norm(b, symptoms)
    return num / denom if denom > 0 else 0.0


def _mmr(
    pool: list[tuple[float, DiseaseProfile]],
    symptoms: dict[int, SymptomInfo],
    config: ScoringConfig,
) -> list[tuple[float, DiseaseProfile]]:
    """Maximal Marginal Relevance: penalise candidates resembling those already picked.

    lambda = 1.0 disables this entirely (pure relevance), which is what the accuracy
    evals use — MMR deliberately trades top-3 accuracy for variety.
    """
    if config.mmr_lambda >= 1.0 or len(pool) <= 1:
        return pool[: config.top_k]

    remaining = list(pool)
    selected: list[tuple[float, DiseaseProfile]] = [remaining.pop(0)]

    while remaining and len(selected) < config.top_k:
        best_index, best_value = 0, -math.inf
        for i, (score, profile) in enumerate(remaining):
            redundancy = max(
                similarity(profile, chosen, symptoms) for _, chosen in selected
            )
            value = config.mmr_lambda * score - (1 - config.mmr_lambda) * redundancy
            if value > best_value:
                best_index, best_value = i, value
        selected.append(remaining.pop(best_index))

    return selected


# --------------------------------------------------------------------------- #
# next question
# --------------------------------------------------------------------------- #


def discriminating_symptom(
    candidates: list[Candidate],
    selected: list[tuple[float, DiseaseProfile]],
    reported: set[int],
    symptoms: dict[int, SymptomInfo],
) -> str | None:
    """The unreported symptom whose answer would most change the ranking.

    A good question is one whose answer you cannot predict. A symptom every
    candidate shares tells you nothing, and neither does one no candidate has —
    so prefer the symptom splitting the candidates closest to 50/50 by score mass,
    weighted by rarity.

    Computed over the MMR-SELECTED set, not the wider pool: asking about a disease
    that never reaches the screen cannot change what the user sees.
    """
    total = sum(score for score, _ in selected)
    if total <= 0 or len(selected) < 2:
        return None

    best_slug, best_gain = None, 0.0
    seen: set[int] = set()
    for _, profile in selected:
        for sid in profile.support:
            if sid in reported or sid in seen or sid not in symptoms:
                continue
            seen.add(sid)

            yes_mass = sum(
                score * p.support.get(sid, 0.0) for score, p in selected
            )
            balance = 1 - abs(2 * yes_mass / total - 1)
            gain = balance * symptoms[sid].idf

            if gain > best_gain:
                best_slug, best_gain = symptoms[sid].slug, gain

    return best_slug
