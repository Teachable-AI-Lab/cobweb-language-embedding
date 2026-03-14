"""Hierarchical CobwebTM runner and utilities."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from bertopic import BERTopic

from cobweb_language_embedding.topic_modeling import CobwebTMHierarchical
from .cobwebtm_utils import CobwebTMDataset

logger = logging.getLogger(__name__)


def compute_npmi(doc_word: np.ndarray, topic_word: np.ndarray, n_list: List[int]) -> float:
    topic_size, word_size = np.shape(topic_word)
    doc_size = np.shape(doc_word)[0]
    if topic_size == 0 or doc_size == 0 or word_size == 0:
        return 0.0
    scores = []
    for n in n_list:
        top_idxs = [np.argpartition(topic_word[t, :], -min(n, word_size))[-min(n, word_size):] for t in range(topic_size)]
        level_sum = 0.0
        for idxs in top_idxs:
            if len(idxs) < 2:
                continue
            topic_sum = 0.0
            pairs = 0
            for i in range(len(idxs)):
                wi = idxs[i]
                fi = doc_word[:, wi] > 0
                p_i = float(fi.sum()) / doc_size
                for j in range(i + 1, len(idxs)):
                    wj = idxs[j]
                    fj = doc_word[:, wj] > 0
                    p_j = float(fj.sum()) / doc_size
                    p_ij = float((fi & fj).sum()) / doc_size
                    if p_ij > 0 and p_i > 0 and p_j > 0:
                        topic_sum += np.log(p_ij / (p_i * p_j)) / (-np.log(p_ij))
                        pairs += 1
            if pairs > 0:
                topic_sum *= 2.0 / (len(idxs) * (len(idxs) - 1))
                level_sum += topic_sum
        scores.append(level_sum / max(topic_size, 1))
    return float(np.mean(scores)) if scores else 0.0


def compute_topic_uniqueness(topic_word: np.ndarray, n: int) -> float:
    t, v = topic_word.shape if topic_word.ndim == 2 else (0, 0)
    if t == 0 or v == 0:
        return 0.0
    top_lists = [np.argpartition(topic_word[i], -min(n, v))[-min(n, v):] for i in range(t)]
    counts = np.zeros(v, dtype=np.int32)
    for lst in top_lists:
        counts[lst] += 1
    tu_total = 0.0
    for lst in top_lists:
        inv_sum = 0.0
        for w in lst:
            c = counts[w]
            inv_sum += (1.0 / c) if c > 0 else 0.0
        tu_total += inv_sum / max(len(lst), 1)
    return float(tu_total / t)


def evaluate_tu(topic_word: np.ndarray, n_list: List[int]) -> float:
    return float(np.mean([compute_topic_uniqueness(topic_word, n) for n in n_list]))


def compute_topic_diversity(topic_words: List[List[str]]) -> float:
    if not topic_words:
        return 0.0
    flat = sum(topic_words, [])
    if not flat:
        return 0.0
    return float(len(set(flat)) / len(flat))


def compute_topic_specialization(topic_word: np.ndarray, doc_word: np.ndarray) -> float:
    if topic_word.size == 0 or doc_word.size == 0:
        return 0.0
    corpus_vec = doc_word.sum(axis=0).astype(np.float64)
    cnorm = np.linalg.norm(corpus_vec)
    if cnorm == 0:
        return 0.0
    corpus_vec = corpus_vec / cnorm
    tw = topic_word.astype(np.float64)
    norms = np.linalg.norm(tw, axis=1)
    norms[norms == 0] = 1.0
    tw_norm = (tw.T / norms).T
    sims = tw_norm.dot(corpus_vec)
    return float(np.mean(1.0 - sims))


def compute_hierarchical_affinity(relations: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[float, float]:
    if not relations:
        return 0.0, 0.0
    child_scores = []
    non_child_scores = []
    for child_dist, parent_dist in relations:
        if child_dist.size == 0 or parent_dist.size == 0:
            continue
        c_norm = np.linalg.norm(child_dist, axis=1, keepdims=True)
        p_norm = np.linalg.norm(parent_dist, axis=1, keepdims=True)
        c_norm[c_norm == 0] = 1.0
        p_norm[p_norm == 0] = 1.0
        sim = (child_dist / c_norm) @ (parent_dist / p_norm).T
        max_idx = np.argmax(sim, axis=1)
        for i in range(sim.shape[0]):
            child_scores.append(float(sim[i, max_idx[i]]))
            others = np.delete(sim[i], max_idx[i])
            if others.size:
                non_child_scores.extend(list(others))
    child_aff = float(np.mean(child_scores)) if child_scores else 0.0
    non_child_aff = float(np.mean(non_child_scores)) if non_child_scores else 0.0
    return child_aff, non_child_aff


def compute_clnpmi(child_topic: np.ndarray, parent_topic: np.ndarray, doc_word: np.ndarray) -> float:
    """Compute CLNPMI between child and parent topic distributions."""
    if child_topic.size == 0 or parent_topic.size == 0 or doc_word.size == 0:
        return 0.0

    def _clnpmi_pair(c_vec, p_vec):
        doc_n = doc_word.shape[0]
        vals = []
        for N in [5, 10, 15]:
            k_c = min(N, c_vec.size)
            k_p = min(N, p_vec.size)
            idx_c_full = np.argpartition(c_vec, -k_c)[-k_c:]
            idx_p_full = np.argpartition(p_vec, -k_p)[-k_p:]
            set_c = set(int(x) for x in idx_c_full)
            set_p = set(int(x) for x in idx_p_full)
            idx_c = [w for w in set_c if w not in set_p]
            idx_p = [w for w in set_p if w not in set_c]
            if not idx_c or not idx_p:
                vals.append(0.0)
                continue
            sum_score = 0.0
            pairs = 0
            for wi in idx_c:
                fi = doc_word[:, wi] > 0
                p_i = float(fi.sum()) / doc_n
                for wj in idx_p:
                    fj = doc_word[:, wj] > 0
                    p_j = float(fj.sum()) / doc_n
                    p_ij = float((fi & fj).sum()) / doc_n
                    if p_ij == 1.0:
                        sum_score += 1.0
                        pairs += 1
                    elif p_ij > 0 and p_i > 0 and p_j > 0:
                        p_ij = p_ij + 1e-10
                        sum_score += np.log(p_ij / (p_i * p_j)) / (-np.log(p_ij))
                        pairs += 1
            vals.append(sum_score / pairs if pairs > 0 else 0.0)
        return float(np.mean(vals)) if vals else 0.0

    def _normalize_rows(mat):
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return mat / norms

    if child_topic.ndim == 1 and parent_topic.ndim == 1:
        return _clnpmi_pair(child_topic, parent_topic)
    if child_topic.ndim == 1 and parent_topic.ndim == 2:
        child_topic = child_topic[np.newaxis, :]
    if child_topic.ndim == 2 and parent_topic.ndim == 1:
        parent_topic = parent_topic[np.newaxis, :]

    C = child_topic
    P = parent_topic
    if C.shape[1] != P.shape[1]:
        V = min(C.shape[1], P.shape[1])
        C = C[:, :V]
        P = P[:, :V]

    Cn = _normalize_rows(C.astype(np.float64))
    Pn = _normalize_rows(P.astype(np.float64))
    sim = Cn @ Pn.T
    best = np.argmax(sim, axis=1)
    scores = [_clnpmi_pair(C[i], P[best[i]]) for i in range(C.shape[0])]
    return float(np.mean(scores)) if scores else 0.0


def compute_topic_pair_difference(words_a: List[str], words_b: List[str]) -> float:
    """Fraction of words that appear in only one of two topics."""
    if not words_a and not words_b:
        return 0.0
    from collections import Counter
    c = Counter()
    c.update(words_a)
    c.update(words_b)
    unique_only = sum(1 for v in c.values() if v == 1)
    denom = len(words_a) + len(words_b)
    return float(unique_only / denom) if denom > 0 else 0.0


def compute_group_td(groups: List[List[List[str]]]) -> float:
    """Average topic diversity across sibling groups."""
    if not groups:
        return 0.0
    from collections import Counter
    group_scores = []
    for topics in groups:
        flat = []
        for tw in topics:
            flat.extend(tw)
        if not flat:
            group_scores.append(0.0)
            continue
        cnt = Counter(flat)
        unique_once = sum(1 for v in cnt.values() if v == 1)
        group_scores.append(float(unique_once / len(flat)))
    return float(np.mean(group_scores)) if group_scores else 0.0


def _get_vocabulary(model: BERTopic) -> List[str]:
    try:
        return list(model.vectorizer_model.get_feature_names_out())
    except Exception:
        return list(model.vectorizer_model.get_feature_names())


def _preprocess_and_bow(model: BERTopic, docs: List[str]):
    clean_docs = model._preprocess_text(docs)
    bow_counts = model.vectorizer_model.transform(clean_docs)
    counts = bow_counts.toarray() if hasattr(bow_counts, "toarray") else np.asarray(bow_counts)
    binary = (counts > 0).astype(np.int32)
    vocab = _get_vocabulary(model)
    return binary, counts, vocab


def _build_adjacency(df: pd.DataFrame) -> Dict[int, List[int]]:
    adj: Dict[int, List[int]] = {}
    for _, row in df.iterrows():
        nid = int(row.Node_ID)
        children = [int(c) for c in (row.children_ids or [])]
        adj[nid] = children
    return adj


def _compute_leaf_sets(df: pd.DataFrame) -> Dict[int, set]:
    adj = _build_adjacency(df)
    leaves = {nid for nid, children in adj.items() if len(children) == 0}
    leaf_sets: Dict[int, set] = {leaf: {leaf} for leaf in leaves}
    unresolved = {nid for nid, ch in adj.items() if len(ch) > 0}
    while unresolved:
        progressed = set()
        for nid in list(unresolved):
            ch = adj.get(nid, [])
            if ch and all((c in leaf_sets) for c in ch):
                s = set()
                for c in ch:
                    s.update(leaf_sets[c])
                leaf_sets[nid] = s
                progressed.add(nid)
        if not progressed:
            break
        unresolved = unresolved.difference(progressed)
    return leaf_sets


def _build_leaf_doc_index_map(model: BERTopic, df: pd.DataFrame) -> Dict[int, List[int]]:
    adj = _build_adjacency(df)
    leaf_ids = [nid for nid, children in adj.items() if len(children) == 0]
    topic_assignments = getattr(model, "topics_", None)
    if topic_assignments is None:
        return {lid: [] for lid in leaf_ids}
    mapping: Dict[int, List[int]] = {lid: [] for lid in leaf_ids}
    for i, t in enumerate(topic_assignments):
        if t in mapping:
            mapping[int(t)].append(i)
    return mapping


def _aggregate_topic_distributions(
    df: pd.DataFrame,
    leaf_sets: Dict[int, set],
    leaf_to_docs: Dict[int, List[int]],
    doc_word_counts: np.ndarray,
) -> Dict[int, np.ndarray]:
    node_dist: Dict[int, np.ndarray] = {}
    for _, row in df.iterrows():
        nid = int(row.Node_ID)
        leaves = list(leaf_sets.get(nid, {nid}))
        doc_indices: List[int] = []
        for lid in leaves:
            doc_indices.extend(leaf_to_docs.get(lid, []))
        if doc_indices:
            dist = doc_word_counts[doc_indices].sum(axis=0)
        else:
            dist = np.zeros((doc_word_counts.shape[1],), dtype=np.float32)
        node_dist[nid] = dist.astype(np.float32)
    return node_dist


def _level_nodes(df: pd.DataFrame) -> Dict[int, List[int]]:
    by_level: Dict[int, List[int]] = {}
    for _, row in df.iterrows():
        lvl = int(row.Level)
        nid = int(row.Node_ID)
        by_level.setdefault(lvl, []).append(nid)
    return by_level


class CobwebTMHierarchicalRunner:
    """Run CobwebTM instances and compute hierarchical metrics."""

    def __init__(self, topic_models: Sequence[BERTopic], *, leaf_level_zero: bool = True, reverse_levels: bool = False):
        if not topic_models:
            raise ValueError("Provide at least one CobwebTM instance to CobwebTMHierarchicalRunner")
        self.topic_models = list(topic_models)
        self.leaf_level_zero = bool(leaf_level_zero)
        self.reverse_levels = bool(reverse_levels)

    def run(self, dataset: CobwebTMDataset, top_n_words: int = 15, max_level: Optional[int] = None):
        results = []
        for model in self.topic_models:
            topic_assignments = getattr(model, "topics_", None)
            needs_fit = topic_assignments is None or (isinstance(topic_assignments, list) and len(topic_assignments) != dataset.size)
            if needs_fit:
                model.fit(dataset.documents, embeddings=dataset.embeddings)

            cobweb_clusterer = getattr(model.hdbscan_model, "clusterer", None)
            wrapper = CobwebTMHierarchical(
                docs=dataset.documents,
                model=model,
                linkage_function=None,
                cobweb_clusterer=cobweb_clusterer,
                topk=top_n_words,
            )

            df = getattr(wrapper, "hierachical_topics", None)
            if df is None or not isinstance(df, pd.DataFrame) or df.empty:
                results.append({"model": model})
                continue

            df = df.copy()
            df["Level"] = df["Level"].astype(int)
            if not self.leaf_level_zero:
                max_lvl_df = int(df["Level"].max()) if len(df) else 0
                df["Level"] = max_lvl_df - df["Level"]

            doc_word_binary, doc_word_counts, _ = _preprocess_and_bow(model, dataset.documents)
            leaf_sets = _compute_leaf_sets(df)
            leaf_to_docs = _build_leaf_doc_index_map(model, df)
            node_dist = _aggregate_topic_distributions(df, leaf_sets, leaf_to_docs, doc_word_counts)
            by_level = _level_nodes(df)

            levels_sorted = sorted(by_level.keys())
            if max_level is not None:
                levels_sorted = [lvl for lvl in levels_sorted if lvl <= max_level]
            levels_for_scoring = list(reversed(levels_sorted)) if self.reverse_levels else list(levels_sorted)

            level_topic_mats: Dict[int, np.ndarray] = {}
            level_topic_words: Dict[int, List[List[str]]] = {}
            for lvl in levels_sorted:
                nodes = by_level.get(lvl, [])
                mats = np.stack([node_dist[nid] for nid in nodes], axis=0) if nodes else np.zeros((0, doc_word_counts.shape[1]))
                level_topic_mats[lvl] = mats
                level_topic_words[lvl] = []
                for nid in nodes:
                    dist = node_dist[nid]
                    k = min(top_n_words, dist.size)
                    idx = np.argpartition(dist, -k)[-k:] if k > 0 else np.array([], dtype=int)
                    idx = idx[np.argsort(-dist[idx])] if idx.size else idx
                    level_topic_words[lvl].append([str(i) for i in idx.tolist()])

            npmi_vals = []
            tu_vals = []
            td_vals = []
            spec_vals = []
            for lvl in levels_for_scoring:
                mats = level_topic_mats[lvl]
                words_list = level_topic_words[lvl]
                if mats.shape[0] == 0:
                    continue
                npmi_vals.append(compute_npmi(doc_word_binary, mats, [top_n_words]))
                tu_vals.append(evaluate_tu(mats, [top_n_words]))
                td_vals.append(compute_topic_diversity(words_list))
                spec_vals.append(compute_topic_specialization(mats, doc_word_binary))

            relations = []
            for i_lvl in range(len(levels_sorted) - 1):
                child_lvl = levels_sorted[i_lvl]
                parent_lvl = levels_sorted[i_lvl + 1]
                child_mat = level_topic_mats.get(child_lvl, np.zeros((0, doc_word_counts.shape[1])))
                parent_mat = level_topic_mats.get(parent_lvl, np.zeros((0, doc_word_counts.shape[1])))
                if child_mat.shape[0] and parent_mat.shape[0]:
                    relations.append((child_mat, parent_mat))
            child_aff, non_child_aff = compute_hierarchical_affinity(relations) if relations else (float("nan"), float("nan"))

            # CLNPMI across consecutive levels
            clnpmi_vals = [compute_clnpmi(c, p, doc_word_binary) for (c, p) in relations] if relations else []
            hier_clnpmi = float(np.mean(clnpmi_vals)) if clnpmi_vals else float("nan")

            # TraCo-inspired metrics: PC_TD, PnonC_TD, Sibling_TD, Sibling clNPMI
            adj = _build_adjacency(df)
            node_top_words: Dict[int, List[str]] = {}
            for lvl in levels_sorted:
                for idx_n, nid in enumerate(by_level.get(lvl, [])):
                    node_top_words[nid] = level_topic_words[lvl][idx_n] if idx_n < len(level_topic_words[lvl]) else []

            pc_td_levels = []
            pnonc_td_levels = []
            sibling_td_levels = []
            sibling_clnpmi_levels = []
            for i_lvl in range(len(levels_sorted) - 1):
                child_lvl = levels_sorted[i_lvl]
                parent_lvl = levels_sorted[i_lvl + 1]
                child_nodes = by_level.get(child_lvl, [])
                parent_nodes = by_level.get(parent_lvl, [])
                if not child_nodes or not parent_nodes:
                    continue
                child_set = set(child_nodes)
                pc_scores = []
                pnonc_scores = []
                sibling_group_scores = []
                sibling_group_clnpmi = []
                for pid in parent_nodes:
                    children = [c for c in adj.get(pid, []) if c in child_set]
                    if children:
                        p_words = node_top_words.get(pid, [])
                        for cid in children:
                            pc_scores.append(compute_topic_pair_difference(p_words, node_top_words.get(cid, [])))
                        group_words = [node_top_words.get(cid, []) for cid in children]
                        sibling_group_scores.append(compute_group_td([group_words]))
                        pair_vals = []
                        for ii in range(len(children)):
                            for jj in range(ii + 1, len(children)):
                                d_i = node_dist.get(children[ii], np.zeros(doc_word_counts.shape[1], dtype=np.float32))
                                d_j = node_dist.get(children[jj], np.zeros(doc_word_counts.shape[1], dtype=np.float32))
                                pair_vals.append(compute_clnpmi(d_i, d_j, doc_word_binary))
                        if pair_vals:
                            sibling_group_clnpmi.append(float(np.mean(pair_vals)))
                    non_children = [c for c in child_nodes if c not in children]
                    if non_children:
                        p_words = node_top_words.get(pid, [])
                        for cid in non_children:
                            pnonc_scores.append(compute_topic_pair_difference(p_words, node_top_words.get(cid, [])))
                if pc_scores:
                    pc_td_levels.append(float(np.mean(pc_scores)))
                if pnonc_scores:
                    pnonc_td_levels.append(float(np.mean(pnonc_scores)))
                if sibling_group_scores:
                    sibling_td_levels.append(float(np.mean(sibling_group_scores)))
                if sibling_group_clnpmi:
                    sibling_clnpmi_levels.append(float(np.mean(sibling_group_clnpmi)))

            metrics = {
                "hier_coherence_npmi": float(np.mean(npmi_vals)) if npmi_vals else float("nan"),
                "hier_topic_uniqueness": float(np.mean(tu_vals)) if tu_vals else float("nan"),
                "hier_topic_diversity": float(np.mean(td_vals)) if td_vals else float("nan"),
                "hier_topic_specialization": float(np.mean(spec_vals)) if spec_vals else float("nan"),
                "hier_affinity_child": float(child_aff),
                "hier_affinity_non_child": float(non_child_aff),
                "hier_coherence_clnpmi": hier_clnpmi,
                "hier_PC_TD": float(np.mean(pc_td_levels)) if pc_td_levels else float("nan"),
                "hier_PnonC_TD": float(np.mean(pnonc_td_levels)) if pnonc_td_levels else float("nan"),
                "hier_sibling_TD": float(np.mean(sibling_td_levels)) if sibling_td_levels else float("nan"),
                "hier_sibling_clnpmi": float(np.mean(sibling_clnpmi_levels)) if sibling_clnpmi_levels else float("nan"),
            }
            results.append({"model": model, **metrics})

        return results
