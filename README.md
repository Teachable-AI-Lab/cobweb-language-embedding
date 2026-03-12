# cobweb-language-embedding

Cobweb-based language embedding tools for:

- semantic retrieval using an incremental Cobweb tree
- BERTopic-compatible topic clustering and hierarchy evaluation
- PCA+ICA embedding whitening utilities for retrieval pipelines

## Installation

### Base package

```bash
pip install .
```

### With optional dependencies

```bash
pip install ".[retrieval]"
pip install ".[topic-modeling]"
pip install ".[all]"
```

## Package Layout

- `src/cobweb_language_embedding/CobwebWrapper.py`: base wrapper over the Cobweb tree implementation
- `src/cobweb_language_embedding/retrieval.py`: retrieval-focused wrapper (`CobwebRetriever`)
- `src/cobweb_language_embedding/topic_modeling.py`: BERTopic-compatible cluster wrappers
- `src/cobweb_language_embedding/preprocess_embedding.py`: PCA+ICA whitening model
- `benchmarks/retrieval/benchmark.py`: retrieval benchmark CLI
- `benchmarks/topic_modeling/benchmark.py`: topic modeling benchmark CLI

## Quick Usage

```python
from sentence_transformers import SentenceTransformer
from cobweb_language_embedding import CobwebRetriever

sentences = [
    "cats are playful",
    "dogs enjoy walks",
    "machine learning for retrieval",
]

embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedder.encode(sentences, convert_to_numpy=True)

retriever = CobwebRetriever(corpus=sentences, corpus_embeddings=embeddings)
query_embedding = embedder.encode("pet animals", convert_to_numpy=True)

print(retriever.query(query_embedding, k=2))
```

## Benchmarks

### Retrieval

```bash
python benchmarks/retrieval/benchmark.py --dataset qqp --model_name all-roberta-large-v1 --subset_size 7500 --target_size 750 --top_k 3 --method all
```

### Topic Modeling

```bash
python -m benchmarks.topic_modeling.benchmark agnews --max-docs 1000 --top-n-words 15
```

## Notes

- For topic modeling benchmarks, install optional dependencies via `.[topic-modeling]`.
- Some dataset pipelines require additional corpora downloads (for example, NLTK Reuters).
- If Graphviz system binaries are missing, visualization features in `CobwebWrapper` may be unavailable.

## License

MIT (see `LICENSE`).
