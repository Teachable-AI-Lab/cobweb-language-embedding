"""CLI entry point to run BERTopic benchmarks for cobweb-language-embedding."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch

# Ensure local `src` and project root imports work when run as a script.
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

from cobweb_language_embedding.topic_modeling import BERTopicCobwebWrapper

try:
    from .ag_news import AGNewsDataset
    from .bertopic_utils import BERTopicRunner
    from .hierarchical_utils import BERTopicHierarchicalRunner
    from .reuters_21578 import Reuters21578Dataset
    from .stackexchange import StackExchangeDataset
    from .twenty_newsgroups import TwentyNewsgroupsDataset
except ImportError:
    # Fallback for direct script execution: `python benchmarks/topic_modeling/benchmark.py`.
    from benchmarks.topic_modeling.ag_news import AGNewsDataset
    from benchmarks.topic_modeling.bertopic_utils import BERTopicRunner
    from benchmarks.topic_modeling.hierarchical_utils import BERTopicHierarchicalRunner
    from benchmarks.topic_modeling.reuters_21578 import Reuters21578Dataset
    from benchmarks.topic_modeling.stackexchange import StackExchangeDataset
    from benchmarks.topic_modeling.twenty_newsgroups import TwentyNewsgroupsDataset

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Dispatch datasets and run BERTopic benchmarks."""

    def __init__(
        self,
        dataset: str,
        max_docs: int | None,
        top_n_words: int,
        run_hierarchical: bool = False,
        leaf_level_zero: bool = True,
        reverse_levels: bool = False,
        model_name: str = "all-roberta-large-v1",
        no_umap: bool = False,
        umap_n_neighbors: int = 15,
        umap_n_components: int = 128,
        num_clusters: int = 50,
        device: str | None = None,
        runtime_log: str | None = None,
    ):
        self.dataset = dataset.lower()
        self.max_docs = max_docs
        self.top_n_words = top_n_words
        self.run_hierarchical = run_hierarchical
        self.leaf_level_zero = leaf_level_zero
        self.reverse_levels = reverse_levels
        self.model_name = model_name
        self.no_umap = no_umap
        self.umap_n_neighbors = umap_n_neighbors
        self.umap_n_components = umap_n_components
        self.num_clusters = num_clusters
        self.device = self._resolve_device(device)
        self.runtime_log = runtime_log

    def _resolve_device(self, requested: str | None) -> str:
        if requested in (None, "", "auto"):
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = requested
        try:
            torch_device = torch.device(device)
        except (RuntimeError, ValueError) as exc:
            logger.warning("Unable to honor device '%s': %s. Falling back to auto.", device, exc)
            torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Using device %s for sentence embeddings", torch_device)
        return str(torch_device)

    def _load_dataset(self):
        if self.dataset in {"20newsgroups", "newsgroups"}:
            return TwentyNewsgroupsDataset.load(max_docs=self.max_docs)
        if self.dataset in {"reuters", "reuters21578", "reuters-21578"}:
            return Reuters21578Dataset.load(max_docs=self.max_docs)
        if self.dataset in {"ag", "agnews", "ag-news"}:
            return AGNewsDataset.load(max_docs=self.max_docs)
        if self.dataset in {"stackexchange", "stackoverflow"}:
            return StackExchangeDataset.load(max_docs=self.max_docs)
        raise ValueError(f"Unsupported dataset '{self.dataset}'")

    def _build_models(self, embedding_model):
        umap_model = None if self.no_umap else UMAP(
            n_neighbors=self.umap_n_neighbors,
            n_components=self.umap_n_components,
            metric="cosine",
        )
        vectorizer_model = CountVectorizer(stop_words="english")
        ctfidf_model = ClassTfidfTransformer()

        return [
            BERTopic(
                embedding_model=embedding_model,
                umap_model=umap_model,
                hdbscan_model=HDBSCAN(min_cluster_size=self.num_clusters, metric="euclidean"),
                vectorizer_model=vectorizer_model,
                ctfidf_model=ctfidf_model,
            ),
            BERTopic(
                embedding_model=embedding_model,
                umap_model=umap_model,
                hdbscan_model=KMeans(n_clusters=self.num_clusters),
                vectorizer_model=vectorizer_model,
                ctfidf_model=ctfidf_model,
            ),
            BERTopic(
                embedding_model=embedding_model,
                umap_model=umap_model,
                hdbscan_model=BERTopicCobwebWrapper(cluster_level=5, min_cluster_size=5),
                vectorizer_model=vectorizer_model,
                ctfidf_model=ctfidf_model,
            ),
        ]

    def _ensure_embeddings(self, dataset, embedding_model):
        if dataset.embeddings is not None:
            logger.info("Reusing %d precomputed embeddings", dataset.size)
            return 0.0
        encode_start = time.perf_counter()
        dataset.embeddings = embedding_model.encode(dataset.documents, show_progress_bar=True, convert_to_numpy=True)
        elapsed = time.perf_counter() - encode_start
        logger.info("Encoded %d documents in %.2fs on %s", dataset.size, elapsed, self.device)
        return elapsed

    def _write_runtime_log(self, payload):
        if not self.runtime_log:
            return
        parent = Path(self.runtime_log).parent
        if str(parent):
            parent.mkdir(parents=True, exist_ok=True)
        with open(self.runtime_log, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def _build_runtime_summary(self, dataset, results, embedding_seconds, total_seconds):
        model_entries = []
        for idx, entry in enumerate(results):
            model = entry["model"]
            runtime_keys = {"model", "fit_seconds", "total_seconds", "shared_overhead_seconds"}
            metrics = {k: v for k, v in entry.items() if k not in runtime_keys}
            model_entries.append(
                {
                    "index": idx,
                    "clusterer": model.hdbscan_model.__class__.__name__,
                    "fit_seconds": entry.get("fit_seconds"),
                    "total_seconds": entry.get("total_seconds"),
                    "metrics": metrics,
                }
            )
        return {
            "dataset": self.dataset,
            "num_documents": dataset.size,
            "model_name": self.model_name,
            "device": self.device,
            "embedding_seconds": embedding_seconds,
            "total_seconds": total_seconds,
            "num_clusters": self.num_clusters,
            "models": model_entries,
        }

    def run(self):
        dataset = self._load_dataset()
        total_start = time.perf_counter()

        embedding_model = SentenceTransformer(self.model_name, device=self.device)
        embedding_seconds = self._ensure_embeddings(dataset, embedding_model)
        topic_models = self._build_models(embedding_model)

        runner = BERTopicRunner(topic_models)
        results = runner.run(dataset, top_n_words=self.top_n_words, measure_runtime=True, shared_overhead=embedding_seconds)

        for idx, entry in enumerate(results):
            model = entry["model"]
            cluster_name = model.hdbscan_model.__class__.__name__
            runtime_keys = {"model", "fit_seconds", "total_seconds", "shared_overhead_seconds"}
            metrics = {k: v for k, v in entry.items() if k not in runtime_keys}
            print(f"Model {idx} ({cluster_name}) metrics: {metrics}")

        if self.run_hierarchical:
            hierarchical_runner = BERTopicHierarchicalRunner(
                topic_models,
                leaf_level_zero=self.leaf_level_zero,
                reverse_levels=self.reverse_levels,
            )
            hierarchical_results = hierarchical_runner.run(dataset, top_n_words=self.top_n_words)
            for idx, metrics in enumerate(hierarchical_results):
                model = metrics.pop("model")
                print(f"Hierarchical Model {idx} ({model.hdbscan_model.__class__.__name__}) metrics: {metrics}")

        total_seconds = time.perf_counter() - total_start
        summary = self._build_runtime_summary(dataset, results, embedding_seconds, total_seconds)
        self._write_runtime_log(summary)
        return results


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BERTopic benchmarks")
    parser.add_argument("dataset", help="Dataset to run: 20newsgroups | reuters | ag_news | stackexchange")
    parser.add_argument("--max-docs", type=int, default=None, help="Optional limit on documents for quick runs")
    parser.add_argument("--top-n-words", type=int, default=15, help="Top-N words per topic for metrics")
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    parser.add_argument("--test-hierarchical", action="store_true", help="Whether to test hierarchical clustering models")
    parser.add_argument("--leaf-level-zero", action="store_true", help="Align hierarchy reporting so leaves map to Level 0")
    parser.add_argument("--reverse-levels", action="store_true", help="Report hierarchical metrics from parents down to leaves")
    parser.add_argument("--model-name", type=str, default="all-roberta-large-v1", help="SentenceTransformer model name")
    parser.add_argument("--no-umap", action="store_true", help="Disable UMAP dimensionality reduction")
    parser.add_argument("--umap-n-neighbors", type=int, default=20, help="UMAP n_neighbors")
    parser.add_argument("--umap-n-components", type=int, default=512, help="UMAP n_components")
    parser.add_argument("--num-clusters", type=int, default=20, help="Number of clusters for KMeans/HDBSCAN")
    parser.add_argument("--device", default="auto", help="Torch device (cuda, cpu, auto)")
    parser.add_argument("--runtime-log", default=None, help="Optional path to store runtime metrics JSON")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    runner = BenchmarkRunner(
        dataset=args.dataset,
        max_docs=args.max_docs,
        top_n_words=args.top_n_words,
        run_hierarchical=args.test_hierarchical,
        leaf_level_zero=args.leaf_level_zero,
        reverse_levels=args.reverse_levels,
        model_name=args.model_name,
        no_umap=args.no_umap,
        umap_n_neighbors=args.umap_n_neighbors,
        umap_n_components=args.umap_n_components,
        num_clusters=args.num_clusters,
        device=args.device,
        runtime_log=args.runtime_log,
    )
    runner.run()


if __name__ == "__main__":
    main(sys.argv[1:])
