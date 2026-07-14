"""Retrieval evaluation for the arXiv semantic search index.

Runs a small set of hand-labelled queries through the pre-built index and
reports precision@k for each. GROUND_TRUTH maps each query to the row indices
(positions in the metadata DataFrame) that were judged genuinely relevant.

Note: row indices are only stable as long as the index is built over the same
data in the same order. Rebuild with different/reordered data and these labels
must be regenerated.

Usage:
    python evaluate.py
    python evaluate.py --top-k 5
"""

import argparse

from src.arxiv_search import config, embedder, index as index_module


GROUND_TRUTH = {
    "convolutional neural network": [8869, 7962, 8142, 18045, 11893],
    "neural networks for image classification": [16146, 6133],
    "transformer attention mechanism": [],  # corpus lacks transformer papers -- honest low case
}


def precision_at_k(retrieved_ids, relevant_ids, k):
    """Fraction of the top-k retrieved papers that are actually relevant."""
    top_k = retrieved_ids[:k]
    hits = [pid for pid in top_k if pid in relevant_ids]
    return len(hits) / len(top_k) if top_k else 0


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval with precision@k.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=config.DEFAULT_TOP_K,
        help=f"Cutoff k for precision@k (default: {config.DEFAULT_TOP_K}).",
    )
    args = parser.parse_args()

    print("Loading index...")
    index, metadata = index_module.load_index()
    print(f"Loaded {index.ntotal} papers.")

    print("Loading model...")
    model = embedder.load_model()

    scores = []
    print(f"\nEvaluating {len(GROUND_TRUTH)} queries at k={args.top_k}:\n")
    for query, relevant_ids in GROUND_TRUTH.items():
        query_embedding = embedder.embed_texts(model, [query])
        results = index_module.search(index, query_embedding, top_k=args.top_k)
        retrieved_ids = [row_index for (score, row_index) in results]

        p = precision_at_k(retrieved_ids, relevant_ids, args.top_k)
        scores.append(p)
        print(f"  precision@{args.top_k} = {p:.3f}  |  {query}")

    mean_p = sum(scores) / len(scores) if scores else 0
    print(f"\nMean precision@{args.top_k} across {len(scores)} queries: {mean_p:.3f}")


if __name__ == "__main__":
    main()
