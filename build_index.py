"""Build the FAISS index from arXiv abstracts, then save it to disk.

Run this once. It downloads the dataset, embeds the abstracts (the slow part),
and saves the index + metadata so that search.py can load them instantly.

Usage:
    python build_index.py
"""

from datasets import load_dataset

from src.arxiv_search import config, embedder, index as index_module


def main():
    print(f"Loading dataset '{config.DATASET_NAME}'...")
    ds = load_dataset(config.DATASET_NAME, split="train")
    df = ds.to_pandas()

    if config.DATASET_SIZE is not None:
        df = df.head(config.DATASET_SIZE)
    print(f"Using {len(df)} abstracts.")

    print(f"Loading model '{config.MODEL_NAME}'...")
    model = embedder.load_model()

    print("Embedding abstracts (this is the slow part)...")
    embeddings = embedder.embed_texts(
        model, df["abstract"].tolist(), show_progress=True
    )

    print("Building FAISS index...")
    index = index_module.build_index(embeddings)

    print("Saving index and metadata...")
    index_module.save_index(index, df)

    print(f"Done. Indexed {index.ntotal} papers.")
    print("You can now run: python search.py")


if __name__ == "__main__":
    main()
