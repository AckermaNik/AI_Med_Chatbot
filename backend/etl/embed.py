"""Generate symptom embeddings for stage 3 of the matcher.

    python etl/embed.py

Uses fastembed (ONNX runtime, ~22MB model, no PyTorch) to embed every canonical
symptom name into a 384-dim vector stored in `symptom.embedding`.

This is what lets 'tummy hurts' reach 'stomach pain' — a comparison trigram
matching cannot make, because the two share almost no characters.

The model downloads on first run and is cached locally afterwards.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from sqlalchemy import select, text, update

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.db import SessionLocal, engine  # noqa: E402
from app.models import Symptom  # noqa: E402

BATCH = 256


async def main() -> int:
    settings = get_settings()

    async with SessionLocal() as session:
        rows = (
            await session.execute(
                select(Symptom.id, Symptom.canonical_name).order_by(Symptom.id)
            )
        ).all()

        if not rows:
            print("no symptoms loaded - run etl/load.py first")
            return 1

        print(f"embedding {len(rows)} symptoms with {settings.embedding_model}")
        print("(first run downloads the model, ~22MB)")

        from fastembed import TextEmbedding

        model = TextEmbedding(model_name=settings.embedding_model)
        names = [name for _, name in rows]
        vectors = list(model.embed(names, batch_size=BATCH)) # It gathers all the generated vectors of 384-dim into one giant, ordered Python list

        dim = len(vectors[0])
        if dim != settings.embedding_dim:
            print(f"DIMENSION MISMATCH: model gives {dim}, schema expects "
                  f"{settings.embedding_dim}. Nothing written.")
            return 1

        for (symptom_id, _), vector in zip(rows, vectors): # It takes the first item from the rows list and locks it to the first item in the vectors list and so on...
            await session.execute(
                update(Symptom)
                .where(Symptom.id == symptom_id)
                .values(embedding=vector.tolist())
            )
        await session.commit()

        missing = (
            await session.execute(
                text("SELECT count(*) FROM symptom WHERE embedding IS NULL")
            )
        ).scalar_one()
        print(f"\n  embedded          {len(rows)} ({dim} dims)")
        print(f"  still NULL        {missing}")

        # Prove it works on the case trigram matching cannot solve.
        for probe in ("tummy hurts", "cant catch my breath", "stomache ake"):
            hit = (
                await session.execute(
                    text(
                        """
                        SELECT canonical_name,
                               round((1 - (embedding <=> CAST(:v AS vector)))::numeric, 3) AS cosine
                        FROM symptom
                        WHERE embedding IS NOT NULL
                        ORDER BY embedding <=> CAST(:v AS vector)
                        LIMIT 3
                        """
                    ),
                    {"v": str(list(model.embed([probe]))[0].tolist())},
                )
            ).all()
            best = ", ".join(f"{r.canonical_name} ({r.cosine})" for r in hit)
            print(f"  {probe!r:26} -> {best}")

    await engine.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
