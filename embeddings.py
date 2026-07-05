from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

print("Loading dataset...")
ds = load_dataset("CShorten/ML-ArXiv-Papers", split="train")
df = ds.to_pandas()
sample = df.head(10)

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding abstracts...")
embeddings = model.encode(sample["abstract"].tolist())
print("Embeddings shape:", embeddings.shape)

sims = cosine_similarity([embeddings[0]], embeddings)[0]
print("\nSimilarity of abstract 0 to each abstract:")
for i, score in enumerate(sims):
    print(f"  Abstract {i}: {score:.3f}")