"""Building and searching the FAISS index.

This module owns everything to do with the vector index: creating it from
embeddings, saving it and its paper metadata to disk, loading it back, and
running searches against it. Separating this from embedding and from the
command-line scripts keeps each part small and testable.
"""

import faiss
import pandas as pd

from . import config


def build_index(embeddings):
    """Create a FAISS index from a matrix of normalized embeddings.

    IndexFlatIP does an exact inner-product search. Because the embeddings are
    normalized (see embedder.py), inner product == cosine similarity, so the
    scores this index returns are cosine-similarity scores.
    """
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index


def save_index(index, metadata: pd.DataFrame) -> None:
    """Save the index and the paper metadata (titles, abstracts) to disk.

    The index only stores vectors, not the papers they came from, so we save
    the metadata alongside it. Row N in the metadata corresponds to vector N
    in the index.
    """
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(config.INDEX_PATH))
    metadata.to_pickle(config.METADATA_PATH)


def load_index():
    """Load a previously built index and its metadata from disk.

    Returns (index, metadata). Raises FileNotFoundError if the index hasn't
    been built yet — run build_index.py first.
    """
    if not config.INDEX_PATH.exists() or not config.METADATA_PATH.exists():
        raise FileNotFoundError(
            f"No index found at {config.INDEX_PATH}. Run build_index.py first."
        )
    index = faiss.read_index(str(config.INDEX_PATH))
    metadata = pd.read_pickle(config.METADATA_PATH)
    return index, metadata


def search(index, query_embedding, top_k: int = config.DEFAULT_TOP_K):
    """Return the top_k nearest papers to a query embedding.

    Yields (score, row_index) pairs, best match first. The row_index maps
    back into the metadata dataframe to recover the paper's title and abstract.
    """
    scores, indices = index.search(query_embedding, top_k)
    return list(zip(scores[0], indices[0]))
