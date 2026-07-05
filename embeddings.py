from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import faiss

print("Loading dataset...")
ds = load_dataset("CShorten/ML-ArXiv-Papers", split="train")
df = ds.to_pandas()
sample = df.head(1000)

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding abstracts...")
# normalize_embeddings=True makes each vector unit length,
# so inner product == cosine similarity
embeddings = model.encode(
    sample["abstract"].tolist(),
    show_progress_bar=True,
    normalize_embeddings=True,
)
print("Embeddings shape:", embeddings.shape)

# Build the FAISS index.
# IndexFlatIP = inner product. With normalized vectors, this is cosine.
dimension = embeddings.shape[1]      # 384
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)                # store all abstract vectors in the index
print(f"Index built with {index.ntotal} vectors.")

while True:
    query = input("\nSearch (or 'quit'): ")
    if query.lower() == "quit":
        break

    # embed the query the SAME way (normalized) so scores are comparable
    query_embedding = model.encode([query], normalize_embeddings=True)

    # search returns (scores, indices) for the top 5 nearest vectors
    scores, indices = index.search(query_embedding, 5)

    print(f"\nTop 5 results for: {query}")
    for rank, (score, i) in enumerate(zip(scores[0], indices[0]), 1):
        print(f"\n{rank}. (score {score:.3f}) {sample.iloc[i]['title']}")
        print(f"   {sample.iloc[i]['abstract'][:200]}...")
