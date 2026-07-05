from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd
import os

INDEX_FILE = "abstracts.index"
DATA_FILE = "abstracts.pkl"

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# If we've already built and saved the index, just load it.
if os.path.exists(INDEX_FILE) and os.path.exists(DATA_FILE):
    print("Loading saved index...")
    index = faiss.read_index(INDEX_FILE)
    sample = pd.read_pickle(DATA_FILE)
    print(f"Loaded index with {index.ntotal} vectors.")
else:
    # First run: build everything and save it.
    print("No saved index found. Building from scratch...")
    print("Loading dataset...")
    ds = load_dataset("CShorten/ML-ArXiv-Papers", split="train")
    df = ds.to_pandas()
    sample = df.head(20000)

    print("Embedding abstracts...")
    embeddings = model.encode(
        sample["abstract"].tolist(),
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    # Save both, so next launch skips all the embedding.
    faiss.write_index(index, INDEX_FILE)
    sample.to_pickle(DATA_FILE)
    print(f"Built and saved index with {index.ntotal} vectors.")

while True:
    query = input("\nSearch (or 'quit'): ")
    if query.lower() == "quit":
        break

    query_embedding = model.encode([query], normalize_embeddings=True)
    scores, indices = index.search(query_embedding, 5)

    print(f"\nTop 5 results for: {query}")
    for rank, (score, i) in enumerate(zip(scores[0], indices[0]), 1):
        print(f"\n{rank}. (score {score:.3f}) {sample.iloc[i]['title']}")
        print(f"   {sample.iloc[i]['abstract'][:200]}...")
