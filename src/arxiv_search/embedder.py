"""Turning text into vectors.

This module wraps the sentence-transformer model. It has one job: given text,
return normalized embedding vectors. Normalization matters because it lets us
use inner-product search in FAISS as an exact equivalent of cosine similarity.
"""

from sentence_transformers import SentenceTransformer

from . import config


def load_model() -> SentenceTransformer:
    """Load the sentence-transformer model named in config."""
    return SentenceTransformer(config.MODEL_NAME)


def embed_texts(model: SentenceTransformer, texts: list[str], show_progress: bool = False):
    """Embed a list of texts into normalized vectors.

    normalize_embeddings=True scales each vector to unit length, so that the
    inner product between two vectors equals their cosine similarity. This is
    what makes FAISS's IndexFlatIP behave like cosine search.
    """
    return model.encode(
        texts,
        show_progress_bar=show_progress,
        normalize_embeddings=True,
    )
