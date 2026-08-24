"""M0 smoke test: is the environment actually ready?

    python scripts/check_env.py

Checks imports, the database connection, and that both extensions are live.
Exits non-zero on failure so it can gate later milestones.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402

OK = "[ ok ]"
BAD = "[FAIL]"


def check_imports() -> bool:
    ok = True
    for module, label in [
        ("fastapi", "FastAPI"),
        ("sqlalchemy", "SQLAlchemy"),
        ("asyncpg", "asyncpg"),
        ("alembic", "Alembic"),
        ("pgvector", "pgvector (python client)"),
        ("google.genai", "google-genai"),
        ("fastembed", "fastembed"),
        ("pandas", "pandas"),
        ("rapidfuzz", "rapidfuzz"),
    ]:
        try:
            __import__(module)
            print(f"{OK} import {label}")
        except ImportError as exc:
            print(f"{BAD} import {label}: {exc}")
            ok = False
    return ok


async def check_database() -> bool:
    from sqlalchemy import text

    from app.db import engine

    try:
        async with engine.connect() as conn:
            version = (await conn.execute(text("SHOW server_version"))).scalar()
            print(f"{OK} connected to PostgreSQL {version}")

            rows = (
                await conn.execute(
                    text(
                        "SELECT extname FROM pg_extension "
                        "WHERE extname IN ('pg_trgm', 'vector')"
                    )
                )
            ).scalars().all()

            ok = True
            for ext in ("pg_trgm", "vector"):
                if ext in rows:
                    print(f"{OK} extension {ext}")
                else:
                    print(f"{BAD} extension {ext} missing")
                    ok = False

            # Prove the operators actually work, not just that the extension loaded.
            sim = (
                await conn.execute(
                    text("SELECT similarity('stomach', 'stomache')")
                )
            ).scalar()
            print(f"{OK} pg_trgm similarity('stomach','stomache') = {sim:.3f}")

            dist = (
                await conn.execute(
                    text("SELECT '[1,0,0]'::vector <=> '[0,1,0]'::vector")
                )
            ).scalar()
            print(f"{OK} pgvector cosine distance operator = {dist:.3f}")
            return ok
    except Exception as exc:  # noqa: BLE001 - this is a diagnostic script
        print(f"{BAD} database: {type(exc).__name__}: {exc}")
        print("       is docker running?  docker compose up -d")
        return False
    finally:
        await engine.dispose()


def check_config() -> bool:
    s = get_settings()
    print(f"{OK} database_url  = {s.database_url}")
    print(f"{OK} gemini_model  = {s.gemini_model}")
    print(f"{OK} scoring       = mu={s.scoring.mu} lambda={s.scoring.mmr_lambda} "
          f"top_k={s.scoring.top_k}")
    if not s.gemini_api_key:
        print("[warn] GEMINI_API_KEY is empty - fine until M4")
    else:
        print(f"{OK} gemini key    = set ({len(s.gemini_api_key)} chars)")
    return True


async def main() -> int:
    print("--- config ---")
    config_ok = check_config()
    print("\n--- imports ---")
    imports_ok = check_imports()
    print("\n--- database ---")
    db_ok = await check_database()

    print()
    if config_ok and imports_ok and db_ok:
        print("M0 complete - environment is ready.")
        return 0
    print("M0 incomplete - see failures above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
