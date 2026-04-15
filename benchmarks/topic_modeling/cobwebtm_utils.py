"""Utility classes for running and evaluating CobwebTM models."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from bertopic import BERTopic
from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import adjusted_rand_score

Batch = List[str]

logger = logging.getLogger(__name__)


class MetricComputationError(RuntimeError):
    """Raised when a requested metric cannot be computed."""


def _default_analyzer() -> callable:
    """Return a default analyzer that mirrors CountVectorizer tokenization."""
    vectorizer = CountVectorizer(stop_words="english", token_pattern=r"(?u)\b\w\w+\b")
    return vectorizer.build_analyzer()


@dataclass
class CobwebTMDataset:
    """Lightweight dataset wrapper for CobwebTM."""

    documents: List[str]
    tokens: List[List[str]]
    dictionary: Dictionary
    corpus: List[List[Tuple[int, int]]]
    embeddings: Optional[np.ndarray] = None

    @classmethod
    def from_texts(
        cls,
        documents: Iterable[str],
        embeddings: Optional[Sequence[Sequence[float]]] = None,
        analyzer: Optional[callable] = None,
    ) -> "CobwebTMDataset":
        docs = [doc.strip() for doc in documents if str(doc).strip()]
        if not docs:
            raise ValueError("No documents provided to CobwebTMDataset")

        analyzer_fn = analyzer or _default_analyzer()
        tokens = [analyzer_fn(doc) for doc in docs]
        dictionary = Dictionary(tokens)
        corpus = [dictionary.doc2bow(doc_tokens) for doc_tokens in tokens]

        embed_array = None
        if embeddings is not None:
            embed_array = np.asarray(embeddings)
            if embed_array.shape[0] != len(docs):
                raise ValueError("Embeddings and documents must have the same length")

        return cls(documents=docs, tokens=tokens, dictionary=dictionary, corpus=corpus, embeddings=embed_array)

    @classmethod
    def from_text_file(
        cls,
        path: str | Path,
        encoding: str = "utf-8",
        analyzer: Optional[callable] = None,
    ) -> "CobwebTMDataset":
        text_path = Path(path)
        with text_path.open("r", encoding=encoding) as handle:
            docs = [line.strip() for line in handle if line.strip()]
        return cls.from_texts(docs, analyzer=analyzer)

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        text_column: str = "text",
        encoding: str = "utf-8",
        analyzer: Optional[callable] = None,
    ) -> "CobwebTMDataset":
        df = pd.read_csv(path, encoding=encoding)
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in CSV {path}")
        docs = df[text_column].astype(str).tolist()
        embeddings = df["embedding"].tolist() if "embedding" in df.columns else None
        return cls.from_texts(docs, embeddings=embeddings, analyzer=analyzer)

    @property
    def size(self) -> int:
        return len(self.documents)


class CobwebTMRunner:
    """Run one or more CobwebTM model instances and compute standard metrics."""

    def __init__(self, topic_models: Sequence[BERTopic]):
        if not topic_models:
            raise ValueError("Provide at least one CobwebTM instance to CobwebTMRunner")
        self.topic_models = list(topic_models)

    def run(
        self,
        dataset: CobwebTMDataset,
        top_n_words: int = 10,
        measure_runtime: bool = False,
        shared_overhead: float = 0.0,
    ) -> List[Dict[str, float]]:
        results: List[Dict[str, float]] = []
        for model in self.topic_models:
            logger.info("Fitting CobwebTM model %s on %d docs", model.hdbscan_model.__class__.__name__, dataset.size)
            start = time.perf_counter() if measure_runtime else None
            model.fit(dataset.documents, embeddings=dataset.embeddings)
            fit_seconds = (time.perf_counter() - start) if measure_runtime else None
            metrics = self._compute_metrics(model, dataset, top_n_words=top_n_words)
            if measure_runtime and fit_seconds is not None:
                metrics["fit_seconds"] = fit_seconds
                metrics["shared_overhead_seconds"] = shared_overhead
                metrics["total_seconds"] = fit_seconds + shared_overhead
            results.append({"model": model, **metrics})
        return results

    def _compute_metrics(
        self,
        model: BERTopic,
        dataset: CobwebTMDataset,
        top_n_words: int,
    ) -> Dict[str, float]:
        topic_words = self._extract_topic_words(model, top_n=top_n_words)
        coherence_cv = self._coherence(topic_words, dataset, measure="c_v")
        coherence_npmi = self._coherence(topic_words, dataset, measure="c_npmi")
        diversity = self._topic_diversity(topic_words)
        inter_sim = self._inter_topic_similarity(model)

        return {
            "coherence_c_v": coherence_cv,
            "coherence_npmi": coherence_npmi,
            "topic_diversity": diversity,
            "inter_topic_similarity": inter_sim,
        }

    @staticmethod
    def _extract_topic_words(model: BERTopic, top_n: int) -> List[List[str]]:
        topics = model.get_topics()
        if not topics:
            return []

        topic_words: List[List[str]] = []
        for topic_id, words in topics.items():
            if topic_id == -1:
                continue
            topic_words.append([word for word, _ in words[:top_n]])
        return topic_words

    @staticmethod
    def _coherence(topic_words: List[List[str]], dataset: CobwebTMDataset, measure: str) -> float:
        if not topic_words:
            return float("nan")

        try:
            coherence_model = CoherenceModel(
                topics=topic_words,
                texts=dataset.tokens,
                corpus=dataset.corpus,
                dictionary=dataset.dictionary,
                coherence=measure,
            )
            return float(coherence_model.get_coherence())
        except ValueError:
            logger.warning("Unable to compute %s coherence; returning NaN", measure)
            return float("nan")

    @staticmethod
    def _topic_diversity(topic_words: List[List[str]]) -> float:
        if not topic_words:
            return float("nan")

        unique_terms = set()
        total = 0
        for words in topic_words:
            unique_terms.update(words)
            total += len(words)
        return float(len(unique_terms) / total) if total else float("nan")

    @staticmethod
    def _inter_topic_similarity(model: BERTopic) -> float:
        topics = model.get_topics()
        if not topics or len(topics) < 2:
            return float("nan")

        topic_order = list(topics.keys())
        indices = [i for i, tid in enumerate(topic_order) if tid != -1]

        def _to_dense(matrix):
            return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)

        embeddings = getattr(model, "topic_embeddings_", None)
        if embeddings is not None:
            matrix = np.asarray(embeddings)
        elif getattr(model, "c_tf_idf_", None) is not None:
            matrix = _to_dense(model.c_tf_idf_)
        else:
            return float("nan")

        if matrix.shape[0] < 2 or not indices:
            return float("nan")

        filtered = matrix[indices]
        sim = cosine_similarity(filtered)
        upper = sim[np.triu_indices_from(sim, k=1)]
        return float(np.mean(upper)) if upper.size else float("nan")


# ---------------------------------------------------------------------------
# Helpers for topic coherence normalization
# ---------------------------------------------------------------------------

def _flatten_topic_tokens(topic: object) -> List[str]:
    """Flatten arbitrarily nested topic representations into a flat list of strings."""
    tokens: List[str] = []
    stack = [topic]
    while stack:
        item = stack.pop()
        if item is None:
            continue
        if isinstance(item, str):
            tok = item.strip()
            if tok:
                tokens.append(tok)
            continue
        if isinstance(item, (list, tuple, np.ndarray)):
            iterable = item.tolist() if isinstance(item, np.ndarray) else item
            stack.extend(reversed(iterable))
            continue
        tok = str(item).strip()
        if tok:
            tokens.append(tok)
    return tokens


def _normalize_topics_for_coherence(raw_topics: List[object]) -> Tuple[List[List[str]], List[int], List[int]]:
    """Return cleaned topics, raw lengths, and cleaned lengths for logging/validation."""
    cleaned: List[List[str]] = []
    raw_lengths: List[int] = []
    clean_lengths: List[int] = []
    for topic in raw_topics:
        raw_len = len(topic) if hasattr(topic, "__len__") and not isinstance(topic, str) else 1
        raw_lengths.append(raw_len)
        tokens = _flatten_topic_tokens(topic)
        clean_lengths.append(len(tokens))
        if tokens:
            cleaned.append(tokens)
    return cleaned, raw_lengths, clean_lengths


# ---------------------------------------------------------------------------
# Incremental (online) dataset and runner
# ---------------------------------------------------------------------------

@dataclass
class IncrementalCobwebTMDataset:
    """Dataset wrapper for incremental CobwebTM experiments.

    Documents are kept in-memory but exposed as batches to support partial-fit
    workflows. Tokenization and dictionary/corpus generation mirror CobwebTMDataset
    so coherence metrics can be computed on seen data at each timestep.
    """

    documents: List[str]
    tokens: List[List[str]]
    dictionary: Dictionary
    corpus: List[List[Tuple[int, int]]]
    batch_size: int = 512
    first_batch_size: Optional[int] = None
    embeddings: Optional[np.ndarray] = None

    @classmethod
    def from_texts(
        cls,
        documents: Iterable[str],
        batch_size: int = 512,
        first_batch_size: Optional[int] = None,
        embeddings: Optional[Sequence[Sequence[float]]] = None,
        analyzer: Optional[callable] = None,
    ) -> "IncrementalCobwebTMDataset":
        docs = [doc.strip() for doc in documents if str(doc).strip()]
        if not docs:
            raise ValueError("No documents provided to IncrementalCobwebTMDataset")

        analyzer_fn = analyzer or _default_analyzer()
        tokens = [analyzer_fn(doc) for doc in docs]
        dictionary = Dictionary(tokens)
        corpus = [dictionary.doc2bow(doc_tokens) for doc_tokens in tokens]

        embed_array = None
        if embeddings is not None:
            embed_array = np.asarray(embeddings)
            if embed_array.shape[0] != len(docs):
                raise ValueError("Embeddings and documents must have the same length")

        return cls(
            documents=docs,
            tokens=tokens,
            dictionary=dictionary,
            corpus=corpus,
            batch_size=batch_size,
            first_batch_size=first_batch_size,
            embeddings=embed_array,
        )

    def iter_batches(self) -> Iterable[Batch]:
        fb = self.first_batch_size if self.first_batch_size is not None else self.batch_size
        start = 0
        yield self.documents[start : start + fb]
        start += fb
        while start < len(self.documents):
            yield self.documents[start : start + self.batch_size]
            start += self.batch_size

    def iter_token_batches(self) -> Iterable[List[List[str]]]:
        fb = self.first_batch_size if self.first_batch_size is not None else self.batch_size
        start = 0
        yield self.tokens[start : start + fb]
        start += fb
        while start < len(self.tokens):
            yield self.tokens[start : start + self.batch_size]
            start += self.batch_size

    def iter_corpus_batches(self) -> Iterable[List[List[Tuple[int, int]]]]:
        fb = self.first_batch_size if self.first_batch_size is not None else self.batch_size
        start = 0
        yield self.corpus[start : start + fb]
        start += fb
        while start < len(self.corpus):
            yield self.corpus[start : start + self.batch_size]
            start += self.batch_size

    @property
    def size(self) -> int:
        return len(self.documents)

    @property
    def num_batches(self) -> int:
        if self.size == 0:
            return 0
        fb = self.first_batch_size if self.first_batch_size is not None else self.batch_size
        remaining = max(0, self.size - fb)
        return 1 + int(np.ceil(remaining / self.batch_size)) if remaining > 0 else 1


class IncrementalCobwebTMRunner:
    """Evaluate incremental CobwebTM models over document batches using partial_fit.

    Each timestep fits the next batch, computes topic quality metrics, and
    temporal stability metrics against the previous timestep.
    """

    def __init__(self, topic_models: Sequence[object], top_n_words: int = 10, labels: Optional[Sequence[str]] = None):
        if not topic_models:
            raise ValueError("Provide at least one incremental CobwebTM instance")
        self.topic_models = list(topic_models)
        self.top_n_words = top_n_words
        self.labels: List[str] = []
        if labels is not None:
            if len(labels) != len(self.topic_models):
                raise ValueError("Length of labels must match number of topic models")
            self.labels = list(labels)
        else:
            self.labels = [model.__class__.__name__ for model in self.topic_models]

    def run(self, dataset: IncrementalCobwebTMDataset) -> List[List[Dict[str, float]]]:
        analyzer = self._get_model_analyzer()
        if analyzer is not None:
            self._retokenize_dataset(dataset, analyzer)

        all_results: List[List[Dict[str, float]]] = [[] for _ in self.topic_models]
        seen_docs: List[str] = []
        seen_tokens: List[List[str]] = []
        seen_corpus: List[List[Tuple[int, int]]] = []
        prev_labels: List[Optional[np.ndarray]] = [None] * len(self.topic_models)
        prev_topics: List[Optional[Dict[int, List[Tuple[str, float]]]]] = [None] * len(self.topic_models)
        prev_c_tf_idf: List[Optional[np.ndarray]] = [None] * len(self.topic_models)
        prev_embeddings: List[Optional[np.ndarray]] = [None] * len(self.topic_models)

        for step_idx, (batch_docs, batch_tokens, batch_corpus) in enumerate(
            zip(dataset.iter_batches(), dataset.iter_token_batches(), dataset.iter_corpus_batches()), start=1
        ):
            seen_start = len(seen_docs)
            seen_docs.extend(batch_docs)
            seen_tokens.extend(batch_tokens)
            seen_corpus.extend(batch_corpus)

            for idx, model in enumerate(self.topic_models):
                label = self.labels[idx] if idx < len(self.labels) else model.__class__.__name__
                logger.info("Partial fitting model %s on batch starting %d (%d docs)", label, seen_start, len(batch_docs))
                model.partial_fit(batch_docs)

                metrics = self._compute_step_metrics(
                    model, label, step_idx,
                    seen_docs, seen_tokens, seen_corpus,
                    prev_topics[idx], prev_c_tf_idf[idx], prev_embeddings[idx], prev_labels[idx],
                    batch_docs=batch_docs, batch_tokens=batch_tokens, batch_corpus=batch_corpus,
                )

                prev_topics[idx] = model.get_topics()
                prev_c_tf_idf[idx] = getattr(model, "c_tf_idf_", None)
                prev_embeddings[idx] = getattr(model, "topic_embeddings_", None)
                prev_labels[idx] = metrics.pop("labels_curr", None)

                all_results[idx].append(metrics)

        return all_results

    def _compute_step_metrics(
        self, model, model_label, step_idx,
        docs_all, tokens_all, corpus_all,
        prev_topics, prev_c_tf_idf, prev_embeddings, prev_labels,
        batch_docs=None, batch_tokens=None, batch_corpus=None,
    ) -> Dict[str, float]:
        current_topics = model.get_topics()
        topic_words = self._extract_topic_words(current_topics, top_n=self.top_n_words)
        coherence_cv = self._coherence(topic_words, tokens_all, corpus_all, measure="c_v")

        # Try to obtain current labels for temporal stability metrics
        labels_curr = None
        try:
            if hasattr(model, "transform") and callable(model.transform):
                out = model.transform(docs_all)
                if isinstance(out, (tuple, list)):
                    labels_curr = np.asarray(out[0]) if len(out) > 0 else None
                else:
                    labels_curr = np.asarray(out)
        except Exception:
            pass
        if labels_curr is None:
            labels_attr = getattr(model, "labels_", None)
            if labels_attr is None:
                base = getattr(model, "base_model", None)
                labels_attr = getattr(base, "labels_", None) if base is not None else None
            if labels_attr is not None:
                labels_curr = np.asarray(labels_attr)

        stability = self._temporal_metrics(
            current_topics, prev_topics,
            getattr(model, "c_tf_idf_", None), prev_c_tf_idf,
            getattr(model, "topic_embeddings_", None), prev_embeddings,
            labels_curr, prev_labels,
            model_name=model_label, step_idx=step_idx,
        )

        return {
            "coherence_c_v": coherence_cv,
            "labels_curr": labels_curr,
            **stability,
        }

    @staticmethod
    def _extract_topic_words(topics: Dict[int, List[Tuple[str, float]]], top_n: int) -> List[List[str]]:
        if not topics:
            return []
        topic_words: List[List[str]] = []
        for topic_id, words in topics.items():
            if topic_id == -1:
                continue
            topic_words.append([str(word) for word, _ in words[:top_n] if word is not None])
        return topic_words

    @staticmethod
    def _coherence(topic_words, tokens, corpus, measure):
        cleaned_topics, _, _ = _normalize_topics_for_coherence(topic_words)
        if len(cleaned_topics) < 2 or not tokens or not corpus:
            return 0.0
        try:
            dictionary = Dictionary(tokens)
            if len(dictionary) == 0:
                return 0.0
            topics_for_coherence = [[str(w) for w in t if w is not None] for t in cleaned_topics]
            topics_for_coherence = [t for t in topics_for_coherence if t]
            if len(topics_for_coherence) < 2:
                return 0.0
            coherence_model = CoherenceModel(
                topics=topics_for_coherence,
                texts=tokens,
                corpus=corpus,
                dictionary=dictionary,
                coherence=measure,
            )
            return float(coherence_model.get_coherence())
        except (ValueError, Exception) as exc:
            logger.warning("Unable to compute %s coherence; returning 0 (%s)", measure, exc)
            return 0.0

    def _temporal_metrics(self, current_topics, prev_topics, current_ctfidf, prev_ctfidf,
                          current_embeddings, prev_embeddings, labels_curr, labels_prev,
                          model_name, step_idx):
        if not current_topics or prev_topics is None or not prev_topics:
            return {
                "topic_stability_ari": None,
                "topic_centroid_drift": None,
            }

        ari = None
        if labels_prev is not None and labels_curr is not None:
            min_len = min(len(labels_prev), len(labels_curr))
            if min_len > 0:
                ari = adjusted_rand_score(labels_prev[:min_len], labels_curr[:min_len])

        centroid_drift = None
        try:
            match = self._match_topics(
                current_topics, prev_topics,
                current_embeddings, prev_embeddings,
                current_ctfidf, prev_ctfidf,
                model_name=model_name, step_idx=step_idx,
            )
            if match:
                centroid_drifts = [float(1.0 - sim) for _, _, sim in match
                                   if current_embeddings is not None and prev_embeddings is not None]
                if centroid_drifts:
                    centroid_drift = float(np.mean(centroid_drifts))
        except MetricComputationError:
            pass

        return {
            "topic_stability_ari": float(ari) if ari is not None else None,
            "topic_centroid_drift": centroid_drift,
        }

    @staticmethod
    def _match_topics(current_topics, prev_topics, current_embeddings, prev_embeddings,
                      current_ctfidf, prev_ctfidf, model_name, step_idx):
        if not current_topics or not prev_topics:
            raise MetricComputationError("Topics required for matching")
        curr_ids = [tid for tid in current_topics.keys() if tid != -1]
        prev_ids = [tid for tid in prev_topics.keys() if tid != -1]
        if not curr_ids or not prev_ids:
            raise MetricComputationError("Non-empty topic ids required")

        sim_mat = None
        if current_embeddings is not None and prev_embeddings is not None:
            X = np.nan_to_num(np.asarray(current_embeddings), nan=0.0, posinf=0.0, neginf=0.0)
            Y = np.nan_to_num(np.asarray(prev_embeddings), nan=0.0, posinf=0.0, neginf=0.0)
            if X.ndim == 1:
                X = X.reshape(1, -1)
            if Y.ndim == 1:
                Y = Y.reshape(1, -1)
            if X.shape[1] != Y.shape[1]:
                maxc = max(X.shape[1], Y.shape[1])
                Xp = np.zeros((X.shape[0], maxc), dtype=X.dtype)
                Yp = np.zeros((Y.shape[0], maxc), dtype=Y.dtype)
                Xp[:, :X.shape[1]] = X
                Yp[:, :Y.shape[1]] = Y
                X, Y = Xp, Yp
            sim_mat = cosine_similarity(X, Y)
        elif current_ctfidf is not None and prev_ctfidf is not None:
            def _dense(m):
                return m.toarray() if hasattr(m, "toarray") else np.asarray(m)
            sim_mat = cosine_similarity(
                np.nan_to_num(_dense(current_ctfidf)),
                np.nan_to_num(_dense(prev_ctfidf)),
            )
        else:
            raise MetricComputationError("Embeddings or c-TF-IDF required for matching")

        if sim_mat is not None:
            if len(curr_ids) > sim_mat.shape[0]:
                curr_ids = curr_ids[:sim_mat.shape[0]]
            if len(prev_ids) > sim_mat.shape[1]:
                prev_ids = prev_ids[:sim_mat.shape[1]]

        matches = []
        used_prev = set()
        for curr_idx, curr_id in enumerate(curr_ids):
            best_prev = None
            best_sim = -1.0
            for prev_idx, prev_id in enumerate(prev_ids):
                if prev_id in used_prev:
                    continue
                sim = sim_mat[curr_idx, prev_idx]
                if sim > best_sim:
                    best_sim = sim
                    best_prev = prev_id
            if best_prev is not None:
                used_prev.add(best_prev)
                matches.append((curr_id, best_prev, float(best_sim)))
        return matches

    def _get_model_analyzer(self):
        for model in self.topic_models:
            vectorizer = getattr(model, "vectorizer_model", None)
            build_analyzer = getattr(vectorizer, "build_analyzer", None) if vectorizer is not None else None
            if callable(build_analyzer):
                try:
                    return build_analyzer()
                except Exception:
                    pass
        return _default_analyzer()

    @staticmethod
    def _retokenize_dataset(dataset: IncrementalCobwebTMDataset, analyzer) -> None:
        tokens = [analyzer(doc) for doc in dataset.documents]
        dictionary = Dictionary(tokens)
        dataset.tokens = tokens
        dataset.dictionary = dictionary
        dataset.corpus = [dictionary.doc2bow(doc_tokens) for doc_tokens in tokens]
