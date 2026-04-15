"""
CLI entry point to run incremental CobwebTM benchmarks.

Populate ``topic_models`` with incremental CobwebTM instances that support
``partial_fit``. The runner streams batches from an IncrementalCobwebTMDataset and
records metrics over time via IncrementalCobwebTMRunner.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benchmarks.topic_modeling.incremental.reuters_rcv1 import Reuters21578IncrementalDataset
from benchmarks.topic_modeling.incremental.stackexchange import StackExchangeIncrementalDataset
from benchmarks.topic_modeling.incremental.tweetner7 import TweetNER7IncrementalDataset
from benchmarks.topic_modeling.incremental.spatiotemporal_news import SpatioTemporalNewsIncrementalDataset
from benchmarks.topic_modeling.cobwebtm_utils import (
	IncrementalCobwebTMDataset,
	IncrementalCobwebTMRunner,
)

from cobweb_language_embedding import PersistentCobwebTM

from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from bertopic.vectorizers import ClassTfidfTransformer
from umap import UMAP
from bertopic.dimensionality import BaseDimensionalityReduction

logger = logging.getLogger(__name__)


class IncrementalBenchmarkRunner:
	"""Dispatch datasets and run incremental CobwebTM benchmarks."""

	class RefitCountVectorizer(CountVectorizer):
		"""CountVectorizer that re-fits from scratch on all seen documents each batch."""

		def __init__(self, **kwargs):
			super().__init__(**kwargs)
			self._all_docs: List[str] = []

		def partial_fit(self, raw_documents):
			self._all_docs.extend(list(raw_documents))
			super().fit(self._all_docs)
			return self

		def update_bow(self, raw_documents):
			if self._all_docs:
				super().fit(self._all_docs)
			return self.transform(raw_documents)

	class FrozenUMAP(UMAP):
		"""UMAP wrapper that fits once and reuses the embedding space on later batches."""

		def __init__(self, **kwargs):
			super().__init__(**kwargs)
			self._fitted = False

		def fit(self, X, y=None):
			if self._fitted:
				return self
			self._fitted = True
			return super().fit(X, y)

		def partial_fit(self, X, y=None):
			if not self._fitted:
				self.fit(X, y)
			return self

		def fit_transform(self, X, y=None):
			if not self._fitted:
				self._fitted = True
				return super().fit_transform(X, y)
			return self.transform(X)

	def __init__(self, config: Dict[str, Any]):
		self.config = config
		self.dataset = config["dataset"].lower()
		self.batch_size = config.get("batch_size")
		self.first_batch_size = config.get("first_batch_size")
		self.max_docs = config.get("max_docs")
		self.top_n_words = config.get("top_n_words", 10)
		self.model_configs = config.get("models", [])

	def _load_dataset(self) -> IncrementalCobwebTMDataset:
		if self.dataset in {"reuters", "rcv1", "reuters-21578", "reuters_rcv1"}:
			return Reuters21578IncrementalDataset.load(batch_size=self.batch_size, first_batch_size=self.first_batch_size, max_docs=self.max_docs)
		if self.dataset in {"spatiotemporal-news", "spatiotemporal", "spatio"}:
			return SpatioTemporalNewsIncrementalDataset.load(batch_size=self.batch_size, first_batch_size=self.first_batch_size, max_docs=self.max_docs)
		if self.dataset in {"tweetner7", "tweetner", "tner"}:
			return TweetNER7IncrementalDataset.load(batch_size=self.batch_size, first_batch_size=self.first_batch_size, max_docs=self.max_docs)
		if self.dataset in {"stackexchange", "stack-exchange", "stack"}:
			return StackExchangeIncrementalDataset.load(batch_size=self.batch_size, first_batch_size=self.first_batch_size, max_docs=self.max_docs)
		raise ValueError(f"Unsupported dataset '{self.dataset}'")

	def _build_umap_model(self, umap_cfg: Dict[str, Any]):
		n_components = umap_cfg.get("n_components", 0)
		if n_components and n_components > 0:
			return self.FrozenUMAP(**umap_cfg)
		return BaseDimensionalityReduction()

	def _build_models(self) -> Tuple[List, List[str]]:
		embedding_model_name = self.config.get("embedding_model", "all-roberta-large-v1")
		embedding_model = SentenceTransformer(embedding_model_name)
		topic_models: List = []
		labels: List[str] = []

		if not self.model_configs:
			raise ValueError("No models defined in configuration")

		for model_cfg in self.model_configs:
			model_type = model_cfg.get("type", "").lower()
			label = model_cfg.get("label", model_type or "model")
			vectorizer_cfg = model_cfg.get("vectorizer", {"stop_words": "english"})
			umap_cfg = model_cfg.get("umap", {})
			ctfidf_cfg = model_cfg.get("ctfidf", {})
			bow_vectorizer = self.RefitCountVectorizer(**vectorizer_cfg)
			umap_model = self._build_umap_model(umap_cfg)
			ctfidf_model = ClassTfidfTransformer(**ctfidf_cfg)
			common_kwargs = dict(
				embedding_model=embedding_model,
				umap_model=umap_model,
				vectorizer_model=bow_vectorizer,
				ctfidf_model=ctfidf_model,
			)

			if model_type == "cobwebtm":
				clustered = PersistentCobwebTM(**model_cfg.get("clusterer", {}))
				topic_models.append(BERTopic(hdbscan_model=clustered, **common_kwargs))
			else:
				raise ValueError(f"Unsupported model type '{model_type}' in configuration. Only 'cobwebtm' is supported.")

			labels.append(label)

		return topic_models, labels

	def run(self):
		dataset = self._load_dataset()
		topic_models, labels = self._build_models()
		if not topic_models:
			raise ValueError("Add incremental CobwebTM instances to topic_models before running benchmarks")

		runner = IncrementalCobwebTMRunner(topic_models, top_n_words=self.top_n_words, labels=labels)
		results = runner.run(dataset)
		for idx, model_results in enumerate(results):
			logger.info("Model %d produced %d timesteps", idx, len(model_results))

		return results, labels

	@staticmethod
	def plot_metrics(results: List[List[dict]], labels: List[str], output_dir: Path, dataset: str, run_name: str | None = None):
		output_dir = output_dir / (run_name if run_name else dataset)
		plots_dir = output_dir / "plots"
		tables_dir = output_dir / "tables"
		plots_dir.mkdir(parents=True, exist_ok=True)
		tables_dir.mkdir(parents=True, exist_ok=True)
		metrics = {
			"coherence_c_v": "Topic coherence (c_v)",
			"topic_stability_ari": "Temporal stability (ARI)",
			"topic_centroid_drift": "Topic centroid drift",
		}

		for metric_key, title in metrics.items():
			plt.figure(figsize=(8, 5))
			table_path = tables_dir / f"{metric_key}.csv"
			with table_path.open("w", newline="") as handle:
				writer = csv.writer(handle)
				writer.writerow(["batch_index", "model", metric_key])
				for idx, model_results in enumerate(results):
					y_vals = [step.get(metric_key, float("nan")) for step in model_results]
					x_vals = list(range(1, len(y_vals) + 1))
					plt.plot(x_vals, y_vals, marker="o", label=labels[idx])
					for batch_idx, val in zip(x_vals, y_vals):
						writer.writerow([batch_idx, labels[idx], val])
			plt.xlabel("Batch index")
			plt.ylabel(title)
			plt.title(title)
			plt.legend()
			plt.grid(True, alpha=0.3)
			plt.tight_layout()
			out_path = plots_dir / f"{metric_key}.png"
			plt.savefig(out_path)
			plt.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run incremental CobwebTM benchmarks from a JSON config")
	parser.add_argument("--config", required=True, help="Path to JSON configuration file")
	parser.add_argument("--log-level", default=None, help="Optional logging level override (DEBUG, INFO, WARNING, ERROR)")
	return parser.parse_args(argv)


def load_config(path: Path) -> Dict[str, Any]:
	with path.open("r") as handle:
		config = json.load(handle)
	if "dataset" not in config:
		raise ValueError("Configuration missing required field 'dataset'")
	config.setdefault("batch_size", 512)
	config.setdefault("first_batch_size", None)
	config.setdefault("max_docs", None)
	config.setdefault("top_n_words", 10)
	config.setdefault("plot_dir", "outputs/incremental_plots")
	config.setdefault("run_name", None)
	config.setdefault("models", [])
	return config


def main(argv: list[str] | None = None):
	args = parse_args(argv)
	config_path = Path(args.config)
	config = load_config(config_path)
	log_level = args.log_level or config.get("log_level", "INFO")
	logging.basicConfig(level=log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
	logger.info("Starting incremental benchmark for dataset=%s", config.get("dataset"))
	runner = IncrementalBenchmarkRunner(config)
	try:
		results, labels = runner.run()
		plot_dir = Path(config.get("plot_dir", "outputs/incremental_plots"))
		run_name = config.get("run_name")
		IncrementalBenchmarkRunner.plot_metrics(results, labels, plot_dir, dataset=config.get("dataset", "unknown"), run_name=run_name)
		logger.info("Saved per-batch metric plots to %s", plot_dir / (run_name if run_name else config.get("dataset", "unknown")))
	except Exception as exc:
		logger.error("Benchmark failed: %s", exc)
		raise


if __name__ == "__main__":
	main(sys.argv[1:])
