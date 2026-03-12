# Retrieval Benchmarks

Unified retrieval benchmark entrypoint for `cobweb-language-embedding`.

## What It Benchmarks

- `QQP` and `MS MARCO` retrieval subsets from Hugging Face datasets.
- Sentence embedding models from `sentence-transformers`.
- Retrieval methods:
  - `FAISS` (if installed)
  - `Torch Dot`
  - `Cobweb Basic`
  - `Cobweb Fast` (indexed prediction)
  - Optional preprocessing variants with PCA+ICA and/or UMAP.

## Run

From repository root (`cobweb-language-embedding`):

```bash
python benchmarks/retrieval/benchmark.py --dataset qqp --model_name all-roberta-large-v1 --subset_size 7500 --target_size 750 --top_k 3 --method all
```

MS MARCO example:

```bash
python benchmarks/retrieval/benchmark.py --dataset msmarco --model_name all-roberta-large-v1 --subset_size 10000 --target_size 1000 --top_k 10 --method scale
```

Enable UMAP + PCA/ICA preprocessing:

```bash
python benchmarks/retrieval/benchmark.py --dataset qqp --model_name all-roberta-large-v1 --use_umap --umap_n_components 256 --umap_n_neighbors 20 --use_whitening --whitening_method pca_ica
```

## Notes

- `--use_whitening` defaults to enabled.
- `--use_umap` defaults to disabled.
- If `faiss` is not available, the benchmark skips FAISS methods and continues.
