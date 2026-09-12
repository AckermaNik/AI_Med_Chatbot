"""Matcher query regression tests that do not require a running database."""

from __future__ import annotations

import pytest

from app.engine.matching import _trigram


class _Result:
    def first(self):
        return None


class _Session:
    def __init__(self):
        self.statement = None

    async def execute(self, statement, params):
        self.statement = statement
        return _Result()


@pytest.mark.asyncio
async def test_trigram_query_uses_postgres_similarity_operator_once():
    """`text()` sends `%` literally; `%%` is an undefined PostgreSQL operator."""
    session = _Session()

    assert await _trigram(session, "throat hurts", 0.45) is None

    sql = session.statement.text
    assert "s.canonical_name % :q" in sql
    assert "a.alias % :q" in sql
    assert "%%" not in sql
