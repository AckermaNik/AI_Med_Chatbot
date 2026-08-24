"""SQLAlchemy ORM models — the single source of truth for the schema.

Alembic reads these to autogenerate migrations. Change a model here, run
`alembic revision --autogenerate`, read the generated file, then `alembic upgrade head`.

Schema rationale lives in docs/PLAN.md section 4.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

EMBEDDING_DIM = 384  # all-MiniLM-L6-v2

"""
    In Python, the standard convention is to name classes using CamelCase
    (like DiseaseSymptom or ChatSession).
    But in PostgreSQL, capital letters can cause massive headaches. 
    SQL databases heavily prefer snake_case 
    (lowercase words separated by underscores).
    
    WE HAVE 5 DATA TABLES !!!!
"""



class Base(DeclarativeBase):
    pass


# --------------------------------------------------------------------------- #
# reference data
# --------------------------------------------------------------------------- #

# ------------- Parent table --------------
class Specialty(Base):
    """A medical specialty. ~30 rows, entirely hand-authored."""

    __tablename__ = "specialty"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(80), unique=True)
    slug: Mapped[str] = mapped_column(String(80), unique=True)
    description: Mapped[str] = mapped_column(Text)

    # The label tells us the target class is DiseaseSpecialty.
    # back_populates="specialty" tells us to look for a variable named 
    # 'specialty' inside that DiseaseSpecialty class.
    
    # This automatic detection only works because there is exactly one ForeignKey pointing to the Disease table.
    
    diseases: Mapped[list[DiseaseSpecialty]] = relationship(back_populates="specialty")



# ------------- Parent table --------------
class Disease(Base):
    __tablename__ = "disease"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200), unique=True)
    slug: Mapped[str] = mapped_column(String(200), unique=True, index=True)

    # Present for the curated core 41 only; NULL across the tail.
    description: Mapped[str | None] = mapped_column(Text)

    source: Mapped[str] = mapped_column(String(20))  # itachi9604 | dhivyeshrk | both

    # Fewer than 3 recorded symptoms. These would otherwise score 1.00 off a single
    # common symptom; the mu prior damps them (docs/PLAN.md section 5).
    low_evidence: Mapped[bool] = mapped_column(Boolean, default=False)

    symptoms: Mapped[list[DiseaseSymptom]] = relationship(
        back_populates="disease", cascade="all, delete-orphan"
    )
    specialties: Mapped[list[DiseaseSpecialty]] = relationship(
        back_populates="disease", cascade="all, delete-orphan"
    )
    
    # The label tells us the target class is Precaution.
    # back_populates="disease" tells us to look for a variable named 
    # 'disease' inside that Precaution class.
    
    precautions: Mapped[list[Precaution]] = relationship(
        back_populates="disease", cascade="all, delete-orphan"
    )

    __table_args__ = (
        CheckConstraint(
            "source IN ('itachi9604', 'dhivyeshrk', 'both')", name="ck_disease_source"
        ),
    )



# ------------- Parent table --------------
class Symptom(Base):
    __tablename__ = "symptom"

    id: Mapped[int] = mapped_column(primary_key=True)
    canonical_name: Mapped[str] = mapped_column(String(200), unique=True)
    slug: Mapped[str] = mapped_column(String(200), unique=True, index=True)

    # 1..7, from dataset A only. NULL means UNKNOWN, never zero — urgency is
    # computed only over symptoms whose severity we actually have.
    severity_weight: Mapped[int | None] = mapped_column(SmallInteger)

    # One of 13 tags. Used solely for specialty fallback when no disease matches
    # confidently (docs/PLAN.md section 6).
    body_system: Mapped[str | None] = mapped_column(String(30), index=True)

    # ln(N_diseases / n_diseases_containing). Precomputed by etl/stats.py so
    # scoring is one indexed query rather than a Python scan.
    idf: Mapped[float] = mapped_column(Float, default=0.0)

    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM))

    aliases: Mapped[list[SymptomAlias]] = relationship(
        back_populates="symptom", cascade="all, delete-orphan"
    )

    __table_args__ = (
        CheckConstraint(
            "severity_weight IS NULL OR severity_weight BETWEEN 1 AND 7",
            name="ck_symptom_severity_range",
        ),
        # Trigram index for stage 2 of the matcher (typos, morphology).
        Index(
            "ix_symptom_name_trgm",
            "canonical_name",
            postgresql_using="gin",
            postgresql_ops={"canonical_name": "gin_trgm_ops"},
        ),
        # Stage 3 (paraphrase). Declared here so autogenerate does not try to drop
        # it on the next revision. Unnecessary at ~440 rows — an exact scan is
        # sub-millisecond — but harmless and correct at scale.
        Index(
            "ix_symptom_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )



# ------------- Child table --------------
class SymptomAlias(Base):
    """Alternate surface forms — dataset B phrasings plus hand-curated synonyms."""

    __tablename__ = "symptom_alias"

    id: Mapped[int] = mapped_column(primary_key=True)
    symptom_id: Mapped[int] = mapped_column(
        ForeignKey("symptom.id", ondelete="CASCADE"), index=True
    )
    alias: Mapped[str] = mapped_column(String(200))
    source: Mapped[str] = mapped_column(String(20))  # dataset_b | curated

    symptom: Mapped[Symptom] = relationship(back_populates="aliases")

    __table_args__ = (
        UniqueConstraint("symptom_id", "alias", name="uq_symptom_alias"),
        Index(
            "ix_symptom_alias_trgm",
            "alias",
            
            # GIN (Generalized Inverted Index)
            # (A GIN index is the same technology that powers search engines and
            # it maps pieces of text to the rows where they appear).
            # Combined with trigram (3-letter chunk) 
            # operations creates a hyper-fast, fuzzy-search map in the database, 
            # allowing us to instantly match misspelled user inputs (e.g., "diarea") 
            # to the correct alias ("diarrhea") without scanning the whole table.
            
            postgresql_using="gin",
            postgresql_ops={"alias": "gin_trgm_ops"},
        ),
    )


# --------------------------------------------------------------------------- #
# edges
# --------------------------------------------------------------------------- #



# ------------- Child table --------------

class DiseaseSymptom(Base):
    __tablename__ = "disease_symptom"

    disease_id: Mapped[int] = mapped_column(
        ForeignKey("disease.id", ondelete="CASCADE"), primary_key=True
    )
    symptom_id: Mapped[int] = mapped_column(
        ForeignKey("symptom.id", ondelete="CASCADE"), primary_key=True, index=True
    )

    # P(symptom | disease), from row frequency in the source data.
    support: Mapped[float] = mapped_column(Float)

    disease: Mapped[Disease] = relationship(back_populates="symptoms")
    symptom: Mapped[Symptom] = relationship()

    __table_args__ = (
        CheckConstraint("support > 0 AND support <= 1", name="ck_support_range"),
    )

# ------------- Child table --------------

class DiseaseSpecialty(Base):
    __tablename__ = "disease_specialty"

    disease_id: Mapped[int] = mapped_column(
        ForeignKey("disease.id", ondelete="CASCADE"), primary_key=True
    )
    specialty_id: Mapped[int] = mapped_column(
        ForeignKey("specialty.id"), primary_key=True
    )

    # No ordering column: where a disease maps to several specialties they are all
    # equally valid referrals. Callers sort by specialty name so output stays
    # deterministic (evals depend on it).

    # Provenance. Drives how confidently the UI phrases the referral, and tells us
    # which mappings a human actually verified.
    mapping_source: Mapped[str] = mapped_column(String(20))  # curated | fallback

    disease: Mapped[Disease] = relationship(back_populates="specialties")
    specialty: Mapped[Specialty] = relationship(back_populates="diseases")

    __table_args__ = (
        CheckConstraint(
            "mapping_source IN ('curated', 'fallback')", name="ck_mapping_source"
        ),
    )



# ------------- Child table --------------
class BodySystemSpecialty(Base):
    """Fallback routing: body system -> specialty. 13 hand-authored rows."""

    __tablename__ = "body_system_specialty"

    body_system: Mapped[str] = mapped_column(String(30), primary_key=True)
    specialty_id: Mapped[int] = mapped_column(ForeignKey("specialty.id"))

    #"I want BodySystemSpecialty to know about the Specialty, 
    # but I do NOT care if the Specialty knows about BodySystemSpecialty."
    specialty: Mapped[Specialty] = relationship()


# ------------- Child table --------------
class Precaution(Base):
    __tablename__ = "precaution"

    id: Mapped[int] = mapped_column(primary_key=True)
    disease_id: Mapped[int] = mapped_column(
        ForeignKey("disease.id", ondelete="CASCADE"), index=True
    )
    ordinal: Mapped[int] = mapped_column(SmallInteger)
    text: Mapped[str] = mapped_column(Text)
    
    # When you type my_precaution.disease,
    # SQLAlchemy secretly looks at the disease_id column,
    # quietly writes the SQL query for you, goes to the database 
    # and fetches the right Disease object.

    disease: Mapped[Disease] = relationship(back_populates="precautions")


# --------------------------------------------------------------------------- #
# safety
# --------------------------------------------------------------------------- #


# ------------- Parent table --------------
class RedFlag(Base):
    """Deterministic escalation, evaluated BEFORE the LLM ever runs."""

    __tablename__ = "red_flag"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    require_all: Mapped[list[int]] = mapped_column(ARRAY(Integer))  # symptom ids that must ALL be present
    message: Mapped[str] = mapped_column(Text)
    level: Mapped[str] = mapped_column(String(20))  # emergency | urgent

    __table_args__ = (
        CheckConstraint("level IN ('emergency', 'urgent')", name="ck_red_flag_level"),
    )


# --------------------------------------------------------------------------- #
# conversations
# --------------------------------------------------------------------------- #


# ------------- Parent table --------------
class ChatSession(Base):
    __tablename__ = "chat_session"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    messages: Mapped[list[ChatMessage]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )

# ------------- Child table --------------
class ChatMessage(Base):
    __tablename__ = "chat_message"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("chat_session.id", ondelete="CASCADE")
    )
    role: Mapped[str] = mapped_column(String(10))  # user | model | tool
    content: Mapped[str | None] = mapped_column(Text)

    # The tool trace rendered in the UI.
    tool_calls: Mapped[dict | None] = mapped_column(JSONB)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    session: Mapped[ChatSession] = relationship(back_populates="messages")

    __table_args__ = (
        Index("ix_chat_message_session", "session_id", "id"),
        CheckConstraint("role IN ('user', 'model', 'tool')", name="ck_message_role"),
    )
