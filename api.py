"""FastAPI wrapper around the arXiv semantic search.

Loads the FAISS index and embedding model once at startup (in the lifespan
handler), then serves searches over HTTP. This replaces the interactive
command-line prompt in search.py with a JSON endpoint.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from src.arxiv_search import index, embedder

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load once at boot, not per-request.
    state["index"], state["metadata"] = index.load_index()
    state["model"] = embedder.load_model()
    yield
    state.clear()


app = FastAPI(title="arXiv Semantic Search", lifespan=lifespan)


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/search")
def search(req: SearchRequest):
    query_embedding = embedder.embed_texts(state["model"], [req.query])
    hits = index.search(state["index"], query_embedding, req.top_k)

    results = []
    for score, row_index in hits:
        row = state["metadata"].iloc[int(row_index)]
        results.append({
            "score": float(score),
            "title": row["title"],
            "abstract": row["abstract"],
        })
    return {"results": results}
