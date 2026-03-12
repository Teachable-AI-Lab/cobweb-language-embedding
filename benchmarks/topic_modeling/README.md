# Topic Modeling Benchmarks

This folder contains BERTopic benchmarking utilities for `cobweb-language-embedding`.

## What is included

- `benchmark.py`: CLI runner for end-to-end benchmark execution.
- `bertopic_utils.py`: dataset wrapper and non-hierarchical topic metrics.
- `hierarchical_utils.py`: hierarchical evaluation runner and metrics.
- `ag_news.py`, `reuters_21578.py`, `twenty_newsgroups.py`, `stackexchange.py`: dataset loaders.

The runner evaluates three clustering backends under BERTopic:

- HDBSCAN
- KMeans
- Cobweb (`BERTopicCobwebWrapper` from `src/cobweb_language_embedding/topic_modeling.py`)

## Quick start

Run from repository root (`cobweb-language-embedding`):

```bash
python -m benchmarks.topic_modeling.benchmark agnews --max-docs 1000 --top-n-words 15
```

Enable hierarchical evaluation:

```bash
python -m benchmarks.topic_modeling.benchmark agnews --max-docs 1000 --top-n-words 15 --test-hierarchical --leaf-level-zero
```

Try another dataset:

```bash
python -m benchmarks.topic_modeling.benchmark reuters --max-docs 2000
```

## Supported datasets

- `agnews`
- `reuters`
- `20newsgroups`
- `stackexchange`

## Useful CLI options

- `--model-name`: sentence embedding model name (default `all-roberta-large-v1`)
- `--no-umap`: disable UMAP before clustering
- `--umap-n-neighbors`
- `--umap-n-components`
- `--num-clusters`
- `--runtime-log`: save runtime metrics JSON

## Notes

- Some datasets require extra packages/corpora:
  - AG News / StackExchange: `datasets`
  - Reuters: `nltk` corpus (`nltk.download('reuters')`)
- For best runtime, use a GPU-enabled environment for embedding generation.
