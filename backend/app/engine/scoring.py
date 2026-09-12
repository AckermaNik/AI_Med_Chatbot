"""It takes a list of symptoms a user typed in, compares it against a library of diseases, 
and ranks them purely using math—no AI guessing, just hard numbers.

Everything here is a pure function over plain data structures, which is why it can
be unit-tested exhaustively without Postgres running. The repository layer is
responsible for turning database rows into the inputs below.

    u (User Symptoms):    The list of symptoms the user reported having (e.g., "itching", "skin rash").
    S(d) (Disease Symptoms):    The list of symptoms a specific disease is known to cause.
    IDF (Inverse Document Frequency):    The rarity score of a symptom. A common symptom like fatigue has a low IDF (e.g., 0.5)
    Support:    A probability score of how strongly a disease is linked to a symptom.
    
    GOAL: calculate a final score (between 0 and 1) for every possible disease.

        score(d) = num(d) / (norm_u · √(D(d) + μ))  ∈ [0, 1] like cosine-similarity

        WHERE: 
        num(d)   = Σ_{s ∈ u ∩ S(d)} idf(s) · support(d, s) -> match on a rare symptom jumps THE score massively.
        norm_u   = √( Σ_{s ∈ u} idf(s) ) -> Penalizes if user reported a bunch of symptoms the disease doesn't explain.
        D(d)     = Σ_{s ∈ S(d)} idf(s) · support(d, s) -> Penalizes if the disease causes highly specific symptoms that the user didn't mention.
        μ (mu)   = Penalty that stops thinly-documented diseases scoring 1.00 off a single common symptom with user.
        
        mmr (Maximal Marginal Relevance)  = Acts as a diversity filter. After scoring the diseases and choosing the 20 most fitted ones, it penalizes candidates that are too similar to the ones already chosen.
    
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

    id: int                       # The unique database ID for the symptom
    slug: str                     # The URL-friendly string (e.g., "stomach-pain")
    name: str                     # The human-readable name (e.g., "Stomach Pain")
    idf: float                    # Inverse Document Frequency (higher = rarer symptom)
    severity: int | None = None   # How dangerous it is (usually 1-10) for UI urgency flags
    body_system: str | None = None # The anatomical category (e.g., "digestive")


@dataclass(frozen=True)
class DiseaseProfile:
    """One disease and its symptom support distribution."""

    id: int                       # The unique database ID for the disease
    slug: str                     # The URL-friendly string (e.g., "common-cold")
    name: str                     # The human-readable name (e.g., "Common Cold")
    support: dict[int, float]     # symptom_id -> Probability the disease causes it (0.0 to 1.0)
    low_evidence: bool = False    # True if the database barely has any data on this disease


@dataclass(frozen=True)
class MatchedSymptom:
    slug: str                     # The URL-friendly string of the symptom
    name: str                     # The human-readable name of the symptom
    idf: float                    # The rarity score of this specific symptom
    support: float                # How strongly the disease is linked to this symptom

    @property
    def contribution(self) -> float:
        """How much this symptom added to the numerator. Drives the UI's 'because…'."""
        return self.idf * self.support


@dataclass
class Candidate:
    disease_id: int               # The database ID of the matched disease
    slug: str                     # The URL-friendly string of the disease
    name: str                     # The human-readable name of the disease
    score: float                  # The final calculated match score (0.0 to 1.0)
    matched: list[MatchedSymptom] = field(default_factory=list)  # The specific symptoms the user and disease share
    missing_key: list[str] = field(default_factory=list)         # Top rare symptoms the disease normally causes, but the user didn't have
    urgency: int | None = None    # The maximum severity score out of the matched symptoms
    low_evidence: bool = False    # Flag to warn the UI if this is a poorly documented disease


@dataclass
class Ranking:
    candidates: list[Candidate]   # The final, sorted, MMR-filtered list of diseases to show the user
    discriminating_symptom: str | None = None  # The slug of the single best symptom to ask the user about NEXT
    unmatched_reported: list[str] = field(default_factory=list) # Symptoms the user typed that don't match ANY of the top candidates

    @property
    def is_confident(self) -> bool:
        """True when the llm is clear enough to stop asking questions."""
        if not self.candidates:
            return False
        if len(self.candidates) == 1:
            return True
        return (self.candidates[0].score - self.candidates[1].score) >= self._margin

    _margin: float = 0.15         # !! The score gap needed between #1 and #2 to trigger is_confident


# --------------------------------------------------------------------------- #
# core scoring
# --------------------------------------------------------------------------- #


def _evidence_mass(profile: DiseaseProfile, symptoms: dict[int, SymptomInfo]) -> float:
    """It takes a specific disease profile and the global dictionary of all symptoms.
    It computes D(d) from the formula: the total expected evidence for this disease by summing up 
    the (IDF * support) of every single symptom this disease is known to cause.
    It returns a float representing the total "weight" of this disease's expected symptom profile.
    """
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
    """It takes the set of user symptoms, a single disease profile, the global symptom dictionary, 
    the user's penalty score (), and a baseline penalty (mu).
    It computes the core cosine-like similarity score. It calculates the numerator (the overlapping match) 
    and divides it by the denominator (the reality check penalty combining norm_u and the disease's evidence mass).
    It returns a final float score between 0.0 and 1.0 for this specific disease.
    """
    numerator = sum(
        symptoms[sid].idf * profile.support[sid]
        for sid in reported & profile.support.keys()
        if sid in symptoms
    )
    if numerator <= 0 or norm_u <= 0:
        return 0.0
    denominator = norm_u * math.sqrt(_evidence_mass(profile, symptoms) + mu)
    return numerator / denominator if denominator > 0 else 0.0

#################################################
#
# THE BRAIN OF THE SCORING COMPUTATION !!!!!!!
#
##################################################


def rank(
    reported: set[int], # user reported symptoms
    profiles: list[DiseaseProfile], # diseases that match
    symptoms: dict[int, SymptomInfo], # all the symptoms
    config: ScoringConfig,
) -> Ranking:
    """It takes the user's reported symptom IDs, every matched disease profile, the global symptom dictionary, 
    and the engine's configuration settings.
    It computes the entire ranking pipeline: calculates the user's norm penalty, scores every disease, 
    sorts them, grabs the top pool, applies MMR to filter out redundant diseases, packages them into UI Candidates, 
    and calculates the next best question to ask.
    It returns a fully populated Ranking object containing the top candidates and the next question.
    """
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
    pool = scored[: config.mmr_pool] # 20 samples

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
    
    # returns the slug of the chosen symptom for a follow up quaestion or None if no good question exists.
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
    """It takes the raw float score, a disease profile, the user's symptoms, and the symptom dictionary.
    It computes the metadata needed for the UI. It figures out exactly which symptoms matched, sorts them 
    by their mathematical contribution, finds the top expected symptoms the user was missing, and finds the highest severity score.
    It returns a rich Candidate dataclass object ready to be sent to the front-end.
    """
    matched = [ # sympotoms both the disease and the user has
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

    unreported = [ # symptoms the disease has but not the user
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
    """It takes a disease profile and the symptom dictionary.
    It computes the strict L2 norm (Euclidean length) of the disease's symptom vector by squaring 
    the (idf * support) of every symptom and taking the square root of the sum.
    It returns a float representing the geometric length of the disease, used purely for MMR similarity.
    """
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
    """It takes 2 different disease profiles and the symptom dictionary.
    It computes the true, mathematical cosine similarity between the two diseases by comparing 
    their overlapping symptoms and normalizing it against their vector lengths.
    It returns a float between 0.0 (totally unrelated) and 1.0 (identical profiles), used to penalize redundancy.
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
    """It takes the top pool of scored diseases, the symptom dictionary, and the config.
    It computes Maximal Marginal Relevance. It iteratively picks the next best disease from the pool, but artificially 
    lowers a disease's score if it is too mathematically similar (using similarity()) to the diseases already picked.
    It returns a smaller, filtered list of 5 (score, DiseaseProfile) tuples that prioritize a diverse set of diagnoses.
    """
    if config.mmr_lambda >= 1.0 or len(pool) <= 1:
        return pool[: config.top_k]

    remaining = list(pool) # brand new, flat copy of (score, DiseaseProfile)
    selected: list[tuple[float, DiseaseProfile]] = [remaining.pop(0)]

    while remaining and len(selected) < config.top_k:
        best_index, best_value = 0, -math.inf
        for i, (score, profile) in enumerate(remaining):
            redundancy = max(
                similarity(profile, chosen, symptoms) for _, chosen in selected
            )
            
            # literal mathematical formula for Maximal Marginal Relevance (MMR)
            # f.i: value = (0.7 * score) - (0.3 * redundancy) -> Give 70% of your attention to how accurate the disease is but subtract a 30% penalty if it is too similar to something we already showed the user.
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
    """It takes the list of UI candidates, the selected top disease tuples, the user's reported symptoms, and the symptom dict.
    It computes the single best follow-up question by iterating over all unreported symptoms from the top diseases 
    and finding the one that most evenly splits the probability mass of the top diseases, weighted by the symptom's rarity.
    It returns the slug (string) of the chosen symptom, or None if no good question exists.
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
            
            # Tries to find Next Perfect Symptom to question (Exactly 50% of diseases have it) -> P = yes_mass / total = 0.5
            balance = 1 - abs(2 * yes_mass / total - 1)
            
            # if it has to choose between two questions that both split the board 50/50, it will prefer to ask about a rare symptom rather than a generic one!
            gain = balance * symptoms[sid].idf 

            if gain > best_gain:
                best_slug, best_gain = symptoms[sid].slug, gain

    return best_slug