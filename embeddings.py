from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

print("Loading dataset...")
ds = load_dataset("CShorten/ML-ArXiv-Papers", split="train")
df = ds.to_pandas()
sample = df.head(1000)

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding abstracts...")
embeddings = model.encode(sample["abstract"].tolist(), show_progress_bar=True)
print("Embeddings shape:", embeddings.shape)

while True:
    query = input("\nSearch (or 'quit'): ")
    if query.lower() == "quit":
        break

    query_embedding = model.encode([query])
    sims = cosine_similarity(query_embedding, embeddings)[0]

    top5 = sims.argsort()[::-1][:5]

    print(f"\nTop 5 results for: {query}")
    for rank, i in enumerate(top5, 1):
        print(f"\n{rank}. (score {sims[i]:.3f}) {sample.iloc[i]['title']}")
        print(f"   {sample.iloc[i]['abstract'][:200]}...")
