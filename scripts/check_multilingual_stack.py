from __future__ import annotations

import asyncio
import os

from app.agent.semantic_router import LocalSemanticRouter
from app.config import get_settings
from app.rag.embedding.factory import create_embedder


async def main() -> None:
    settings = get_settings()

    print("Embedding model:", settings.embedding_model)
    print("Embedding dimension:", settings.embedding_dimension)
    print(
        "Semantic router:",
        os.getenv("SEMANTIC_ROUTER_MODEL", "(not configured)"),
    )

    embedder = create_embedder(settings)

    q = await embedder.embed_query("Login Realty kya karta hai?")
    d = await embedder.embed_documents(
        ["Login Realty provides commercial real-estate services."]
    )

    print("Query dims:", len(q))
    print("Document dims:", len(d[0]))

    router = LocalSemanticRouter()
    route = await router.classify(
        "Login Realty kya karta hai?",
        "Login Realty",
    )

    print("Route scope:", route.scope)
    print("Route source:", route.source)
    print("Route language:", route.language)
    print("Route score:", round(route.score, 4))


if __name__ == "__main__":
    asyncio.run(main())
