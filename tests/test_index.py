"""Tests for the FAISS index pipeline.

These test build / save / load / search using synthetic normalized vectors,
so they run fast and need no model download or network access. They verify the
core correctness property: a vector searched against itself ranks first with a
cosine score of 1.0.

Run with:  pytest
"""

import numpy as np
import pandas as pd

from src.arxiv_search import config, index as index_module


def _fake_data(n: int = 5):
    """Create n normalized random vectors plus matching metadata."""
    rng = np.random.default_rng(0)
    vecs = rng.random((n, config.EMBEDDING_DIM)).astype("float32")
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    papers = pd.DataFrame(
        {
            "title": [f"Paper {i}" for i in range(n)],
            "abstract": [f"Abstract number {i}." for i in range(n)],
        }
    )
    return vecs, papers


def test_build_index_counts_vectors():
    vecs, _ = _fake_data()
    idx = index_module.build_index(vecs)
    assert idx.ntotal == len(vecs)


def test_save_and_load_roundtrip(tmp_path, monkeypatch):
    # redirect storage paths to a temp dir so the test is self-contained
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "INDEX_PATH", tmp_path / "abstracts.index")
    monkeypatch.setattr(config, "METADATA_PATH", tmp_path / "abstracts.pkl")

    vecs, papers = _fake_data()
    idx = index_module.build_index(vecs)
    index_module.save_index(idx, papers)

    loaded_idx, loaded_meta = index_module.load_index()
    assert loaded_idx.ntotal == len(vecs)
    assert len(loaded_meta) == len(papers)


def test_search_ranks_self_first():
    """A vector searched against itself should rank first with score ~1.0."""
    vecs, papers = _fake_data()
    idx = index_module.build_index(vecs)

    query = vecs[2:3]  # exact copy of vector 2
    results = index_module.search(idx, query, top_k=3)

    top_score, top_row = results[0]
    assert top_row == 2
    assert abs(top_score - 1.0) < 1e-4
