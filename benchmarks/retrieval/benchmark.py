"""Unified retrieval benchmark CLI for cobweb-language-embedding."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    faiss = None

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cobweb_language_embedding.retrieval import CobwebRetriever
from cobweb_language_embedding.preprocess_embedding import PCAICAWhiteningModel


def _normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return x / norms


def _recall_at_k(hit_positions: list[int], k: int) -> float:
    return float(np.mean([1.0 if p < k else 0.0 for p in hit_positions]))


def _mrr_at_k(hit_positions: list[int], k: int) -> float:
    vals = [1.0 / (p + 1) if p < k else 0.0 for p in hit_positions]
    return float(np.mean(vals))


def _ndcg_at_k(hit_positions: list[int], k: int) -> float:
    vals = [1.0 / np.log2(p + 2) if p < k else 0.0 for p in hit_positions]
    return float(np.mean(vals))


@dataclass
class RetrievalData:
    corpus: list[str]
    queries: list[str]
    targets: list[str]


def load_qqp(subset_size: int, split: str, target_size: int) -> RetrievalData:
    ds = load_dataset("glue", "qqp", split=split)
    positives = [x for x in ds if x["label"] == 1]
    negatives = [x["question2"] for x in ds if x["label"] == 0]

    rng = np.random.default_rng(42)
    rng.shuffle(positives)

    sampled = positives[: min(subset_size, len(positives))]
    queries = [x["question1"] for x in sampled[:target_size]]
    targets = [x["question2"] for x in sampled[:target_size]]
    corpus = [x["question2"] for x in sampled]

    missing = max(0, subset_size - len(corpus))
    if missing > 0 and negatives:
        add_idx = rng.choice(len(negatives), size=min(missing, len(negatives)), replace=False)
        corpus.extend([negatives[i] for i in add_idx])

    return RetrievalData(corpus=corpus, queries=queries, targets=targets)


def load_msmarco(subset_size: int, split: str, target_size: int) -> RetrievalData:
    ds = load_dataset("ms_marco", "v2.1", split=split)
    queries: list[str] = []
    targets: list[str] = []
    corpus: list[str] = []
    seen = set()

    for ex in ds:
        passages = ex.get("passages", {})
        texts = passages.get("passage_text", [])
        selected = passages.get("is_selected", [])

        pos = [texts[i] for i, s in enumerate(selected) if s == 1 and i < len(texts)]
        if not pos:
            continue

        q = ex.get("query", "")
        t = pos[0]
        if not q or not t:
            continue

        queries.append(q)
        targets.append(t)
        if t not in seen:
            seen.add(t)
            corpus.append(t)

        for p in texts:
            if p and p not in seen:
                seen.add(p)
                corpus.append(p)
            if len(corpus) >= subset_size:
                break

        if len(queries) >= target_size and len(corpus) >= subset_size:
            break

    return RetrievalData(corpus=corpus[:subset_size], queries=queries[:target_size], targets=targets[:target_size])


def fit_preprocessor(
    corpus_embs: np.ndarray,
    n_components: int,
    use_whitening: bool
):
    steps = []
    x_fit = corpus_embs

    if use_whitening:
        whitener = PCAICAWhiteningModel.fit(x_fit, pca_dim=min(n_components, x_fit.shape[1]))
        steps.append(whitener)

    return steps


def apply_preprocessor(embs: np.ndarray, steps) -> np.ndarray:
    x = embs
    for s in steps:
        x = s.transform(x)
    return x


def build_faiss_index(corpus_embs: np.ndarray):
    if faiss is None:
        return None
    dim = corpus_embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(corpus_embs.astype(np.float32))
    return index


def retrieve_faiss(index, query: np.ndarray, k: int) -> list[int]:
    scores, idx = index.search(query[np.newaxis, :].astype(np.float32), k)
    _ = scores
    return idx[0].tolist()


def retrieve_torch(corpus_tensor: torch.Tensor, query: np.ndarray, k: int) -> list[int]:
    q = torch.from_numpy(query).to(corpus_tensor.device)
    scores = torch.matmul(corpus_tensor, q)
    top = torch.topk(scores, k=min(k, corpus_tensor.shape[0]), largest=True).indices
    return top.cpu().tolist()


def evaluate(method: str, get_ranked_ids, corpus: list[str], queries: list[str], targets: list[str], top_k: int):
    hit_positions: list[int] = []
    for q, t in zip(queries, targets):
        ranked_ids = get_ranked_ids(q, top_k)
        ranked_texts = [corpus[i] for i in ranked_ids if 0 <= i < len(corpus)]
        try:
            p = ranked_texts.index(t)
        except ValueError:
            p = 10**9
        hit_positions.append(p)

    return {
        "method": method,
        "recall@k": _recall_at_k(hit_positions, top_k),
        "mrr@k": _mrr_at_k(hit_positions, top_k),
        "ndcg@k": _ndcg_at_k(hit_positions, top_k),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run unified retrieval benchmark")
    parser.add_argument("--config", type=str, default=None, help="Optional JSON config path")
    parser.add_argument("--dataset", type=str, default="qqp", choices=["qqp", "msmarco"])
    parser.add_argument("--model_name", type=str, default="all-roberta-large-v1")
    parser.add_argument("--subset_size", type=int, default=7500)
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--target_size", type=int, default=750)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--method", type=str, default="all", choices=["all", "scale", "cobweb", "cobweb_pca"])
    parser.add_argument("--target_dim", type=int, default=256)
    parser.add_argument("--include_cobweb_fast", action="store_true", default=True)
    parser.add_argument("--use_whitening", action="store_true", default=True)
    parser.add_argument("--whitening_method", type=str, default="pca_ica", choices=["pca_ica"])
    return parser.parse_args()


def methods_for(mode: str) -> list[str]:
    if mode == "all":
        return ["faiss", "torch", "faiss_pre", "torch_pre", "cobweb", "cobweb_pre"]
    if mode == "scale":
        return ["faiss", "cobweb_pre"]
    if mode == "cobweb":
        return ["cobweb", "cobweb_pre"]
    if mode == "cobweb_pca":
        return ["cobweb_pre"]
    return []


def load_data(dataset: str, subset_size: int, split: str, target_size: int) -> RetrievalData:
    if dataset == "qqp":
        return load_qqp(subset_size=subset_size, split=split, target_size=target_size)
    return load_msmarco(subset_size=subset_size, split=split, target_size=target_size)


def main() -> None:
    args = parse_args()
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        for k, v in cfg.items():
            if hasattr(args, k):
                setattr(args, k, v)

    print(f"Running {args.dataset.upper()} benchmark")
    print(f"Model: {args.model_name} | method: {args.method} | top_k: {args.top_k}")

    data = load_data(args.dataset, args.subset_size, args.split, args.target_size)
    model = SentenceTransformer(args.model_name)

    corpus_embs = model.encode(data.corpus, convert_to_numpy=True, show_progress_bar=True)
    query_embs = model.encode(data.queries, convert_to_numpy=True, show_progress_bar=True)
    corpus_embs = _normalize(corpus_embs)
    query_embs = _normalize(query_embs)

    dim = min(args.target_dim, corpus_embs.shape[1])
    steps = fit_preprocessor(
        corpus_embs=corpus_embs,
        n_components=dim,
        use_whitening=args.use_whitening,
    )
    corpus_pre = _normalize(apply_preprocessor(corpus_embs, steps)) if steps else corpus_embs
    query_pre = _normalize(apply_preprocessor(query_embs, steps)) if steps else query_embs

    enabled = methods_for(args.method)
    results = []
    query_map = {q: query_embs[i] for i, q in enumerate(data.queries)}
    query_pre_map = {q: query_pre[i] for i, q in enumerate(data.queries)}

    if "faiss" in enabled and faiss is not None:
        idx = build_faiss_index(corpus_embs)
        results.append(
            evaluate(
                "FAISS",
                lambda q, k: retrieve_faiss(idx, query_map[q], k),
                data.corpus,
                data.queries,
                data.targets,
                args.top_k,
            )
        )

    if "torch" in enabled:
        corpus_t = torch.from_numpy(corpus_embs).float()
        results.append(
            evaluate(
                "Torch Dot",
                lambda q, k: retrieve_torch(corpus_t, query_map[q], k),
                data.corpus,
                data.queries,
                data.targets,
                args.top_k,
            )
        )

    if "faiss_pre" in enabled and faiss is not None:
        idx_pre = build_faiss_index(corpus_pre)
        results.append(
            evaluate(
                "FAISS PCA + ICA",
                lambda q, k: retrieve_faiss(idx_pre, query_pre_map[q], k),
                data.corpus,
                data.queries,
                data.targets,
                args.top_k,
            )
        )

    if "torch_pre" in enabled:
        corpus_pre_t = torch.from_numpy(corpus_pre).float()
        results.append(
            evaluate(
                "Torch PCA + ICA",
                lambda q, k: retrieve_torch(corpus_pre_t, query_pre_map[q], k),
                data.corpus,
                data.queries,
                data.targets,
                args.top_k,
            )
        )

    if "cobweb" in enabled:
        cobweb = CobwebRetriever(corpus=data.corpus, corpus_embeddings=corpus_embs)
        results.append(
            evaluate(
                "Cobweb Basic",
                lambda q, k: cobweb.query(query_map[q], k=k, return_ids=True, use_indexed=False),
                data.corpus,
                data.queries,
                data.targets,
                args.top_k,
            )
        )
        if args.include_cobweb_fast:
            cobweb.build_prediction_index()
            results.append(
                evaluate(
                    "Cobweb Fast",
                    lambda q, k: cobweb.query(query_map[q], k=k, return_ids=True, use_indexed=True),
                    data.corpus,
                    data.queries,
                    data.targets,
                    args.top_k,
                )
            )

    if "cobweb_pre" in enabled:
        cobweb_pre = CobwebRetriever(corpus=data.corpus, corpus_embeddings=corpus_pre)
        results.append(
            evaluate(
                "Cobweb PCA + ICA",
                lambda q, k: cobweb_pre.query(query_pre_map[q], k=k, return_ids=True, use_indexed=False),
                data.corpus,
                data.queries,
                data.targets,
                args.top_k,
            )
        )
        if args.include_cobweb_fast:
            cobweb_pre.build_prediction_index()
            results.append(
                evaluate(
                    "Cobweb PCA + ICA Fast",
                    lambda q, k: cobweb_pre.query(query_pre_map[q], k=k, return_ids=True, use_indexed=True),
                    data.corpus,
                    data.queries,
                    data.targets,
                    args.top_k,
                )
            )

    for r in results:
        print(r)


if __name__ == "__main__":
    main()
