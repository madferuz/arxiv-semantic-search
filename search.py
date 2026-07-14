"""Interactive semantic search over the pre-built arXiv index.

Loads the index built by build_index.py and lets you search it from the
terminal. Because the index is already built, this starts in about a second —
no embedding of the dataset happens here.

Usage:
    python search.py           # returns the default number of results
    python search.py --top-k 10
"""

import argparse

from src.arxiv_search import config, embedder, index as index_module


def format_result(rank: int, score: float, row_index: int, paper) -> str:
    """Format a single search result for printing."""
    title = paper["title"].strip()
    abstract = paper["abstract"].strip().replace("\n", " ")
    return (
        f"\n{rank}. (row {row_index}, score {score:.3f}) {title}\n"
        f"   {abstract[:200]}..."
    )


def main():
    parser = argparse.ArgumentParser(description="Search arXiv ML papers by meaning.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=config.DEFAULT_TOP_K,
        help=f"Number of results per query (default: {config.DEFAULT_TOP_K}).",
    )
    args = parser.parse_args()

    print("Loading index...")
    index, metadata = index_module.load_index()
    print(f"Loaded {index.ntotal} papers.")

    print("Loading model...")
    model = embedder.load_model()

    while True:
        query = input("\nSearch (or 'quit'): ").strip()
        if query.lower() in {"quit", "exit", "q"}:
            break
        if not query:
            continue

        query_embedding = embedder.embed_texts(model, [query])
        results = index_module.search(index, query_embedding, top_k=args.top_k)

        print(f"\nTop {args.top_k} results for: {query}")
        for rank, (score, row_index) in enumerate(results, 1):
            print(format_result(rank, score, row_index, metadata.iloc[row_index]))


if __name__ == "__main__":
    main()
