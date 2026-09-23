"""Typed settings, loaded from environment / .env.

This file is what we call a Configuration Manager.
It acts as the "control center" for the entire app. 
Instead of hard-coding values (like API keys, file paths, or AI math thresholds)
deep inside hundreds of different files, we put them all here. 
If we need to tweak how the AI behaves later, 
we only have to change it in one place (see docs/PLAN.md section 5).
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CURATED_DIR = DATA_DIR / "curated"


class ScoringConfig(BaseSettings):
    """Ranking hyperparameters. Every default is a starting point, not an answer."""

    model_config = SettingsConfigDict(env_prefix="SCORING_", extra="ignore")

    mu: float = Field(2.0, description="Prior mass μ; damps thinly-documented diseases.")
    mmr_lambda: float = Field(0.7, description="1.0 = pure relevance, MMR disabled.")
    mmr_pool: int = Field(20, description="Candidates considered before re-ranking.")
    top_k: int = Field(5, description="Candidates returned to the caller.")
    min_score: float = Field(
        0.15, description="Below this, no confident match -> body-system fallback."
    )
    confident_margin: float = Field(
        0.15, description="top1 - top2 gap that stops follow-up questions."
    )
    max_followups: int = Field(3, description="Hard cap on questions before committing.")


class MatchConfig(BaseSettings):
    """Symptom matcher thresholds (docs/PLAN.md section 3)."""

    model_config = SettingsConfigDict(env_prefix="MATCH_", extra="ignore")

    trigram_threshold: float = 0.35 # Lower threshold catches plausible spelling errors; embedding remains the first semantic check.
    embedding_threshold: float = 0.60


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    database_url: str = "postgresql+asyncpg://medai:medai@localhost:5433/medai"

    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.5-flash-lite"
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384

    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    match: MatchConfig = Field(default_factory=MatchConfig)



# @lru_cache is a decorator. It means "run this once, memorize the result, and 
# forever return the memorized result to save time."

@lru_cache
def get_settings() -> Settings:
    return Settings()
