# arXiv Semantic Search

A command-line semantic search engine over 20,000 machine learning research papers. Type a natural-language query and get back the most conceptually relevant arXiv abstracts — ranked by meaning, not keywords.

Unlike a keyword search, this finds papers even when they don't contain your exact words. Searching `"neural networks for image classification"` returns papers about convolutional networks, MNIST, and classification architectures — because the search matches on *meaning*.

## Example

```
Search (or 'quit'): neural networks for image classification

Top 5 results for: neural networks for image classification

1. (score 0.681) Provably efficient neural network representation for image classification
   The state-of-the-art approaches for image classification are based on neural
   networks. Mathematically, the task of classifying images is equivalent to...

2. (score 0.590) Enhanced Image Classification With a Fast-Learning Shallow Convolutional Neural Network
   We present a neural network architecture and training method designed to
   enable very rapid training and low implementation complexity...

3. (score 0.569) Training Neural Networks by Using Power Linear Units (PoLUs)
   In this paper, we introduce "Power Linear Unit" (PoLU) which increases the
   nonlinearity capacity of a neural network...
```

## How it works

The engine turns text into vectors and compares them geometrically:

1. **Embedding** — each paper's abstract is converted into a 384-dimensional vector using the `all-MiniLM-L6-v2` sentence-transformer. Semantically similar abstracts end up close together in this vector space.
2. **Indexing** — all vectors are stored in a [FAISS](https://github.com/facebookresearch/faiss) `IndexFlatIP` index for fast nearest-neighbor search. Because the vectors are normalized to unit length, inner-product search is mathematically equivalent to cosine similarity.
3. **Querying** — your search string is embedded the same way, and FAISS returns the top 5 nearest abstracts, ranked by similarity score.

### Persistence

Embedding 20,000 abstracts takes a few minutes, so the built index is saved to disk (`abstracts.index` + `abstracts.pkl`) on the first run. Every launch after that loads the index in about a second instead of re-embedding everything.

## Tech stack

- **[sentence-transformers](https://www.sbert.net/)** — embedding model (`all-MiniLM-L6-v2`)
- **[FAISS](https://github.com/facebookresearch/faiss)** — vector index and similarity search
- **[Hugging Face `datasets`](https://huggingface.co/datasets/CShorten/ML-ArXiv-Papers)** — source data (`CShorten/ML-ArXiv-Papers`)
- **pandas** — data handling
- **Python 3.12**

## Setup

```bash
# clone and enter the repo
git clone https://github.com/madferuz/arxiv-semantic-search.git
cd arxiv-semantic-search

# create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate

# install dependencies
pip install -r requirements.txt
```

## Usage

```bash
python embeddings.py
```

On the **first run**, the script downloads the dataset and embeds 20,000 abstracts (a few minutes), then saves the index. On **subsequent runs**, it loads the saved index instantly.

At the prompt, type any query and press Enter. Type `quit` to exit.

## Notes

- The generated index files (`abstracts.index`, `abstracts.pkl`) are git-ignored — they're rebuilt automatically by running the script.
- The dataset size is set by `df.head(20000)` in `embeddings.py`. Increasing it toward the full ~117K papers works the same way; just delete the saved index files first to force a rebuild.

## Roadmap

- [ ] Scale to the full ~117K-paper dataset
- [ ] Split into separate `build_index.py` and `search.py` commands
- [ ] Add configurable result count (`top-k`)
