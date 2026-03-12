"""Minimal BERTopic integration built on top of CobwebWrapper.

This module intentionally keeps only the pieces needed for BERTopic benchmark
style usage: a cluster model with ``fit`` and ``predict``.
"""

import math
import re

import numpy as np
import pandas as pd
import torch
from bertopic._utils import select_topic_representation
from scipy.cluster import hierarchy as sch
from scipy.sparse import csr_matrix
from scipy.spatial.distance import squareform
from sklearn.metrics.pairwise import cosine_similarity

from .CPPCobweb import CPPCobwebTree
from .CobwebWrapper import CobwebWrapper


class BERTopicCobwebWrapper(CobwebWrapper):
	"""BERTopic-compatible clustering wrapper backed by Cobweb."""

	def __init__(self, cluster_level=4, min_cluster_size=5):
		super().__init__(corpus=None, corpus_embeddings=None, empty_wrapper=True)
		self.cluster_level = cluster_level
		self.min_cluster_size = min_cluster_size

		self._cluster_index_valid = False
		self._means = None
		self._vars = None
		self._counts = None
		self._valid_mask = None
		self._large_enough_mask = None
		self.transition_nodes = []
		self.embedding_shape = None
		self.labels_ = None

	def _invalidate_prediction_index(self):
		# Reuse the base hook to clear cluster-side cached tensors.
		self._cluster_index_valid = False
		self._means = None
		self._vars = None
		self._counts = None
		self._valid_mask = None
		self._large_enough_mask = None

	def _init_tree_for_embeddings(self, x):
		self.embedding_shape = x.shape[1:]
		prior_var = np.var(x, axis=0, ddof=0)
		self.tree = CPPCobwebTree(
			shape=self.embedding_shape,
			device=self.device,
			use_info=True,
			acuity_cutoff=False,
			use_kl=True,
			prior_var=prior_var,
			alpha=1e-8,
		)

	def _gather_clusters(self):
		if self._cluster_index_valid:
			return self.labels_

		if len(self.sentences) == 0:
			self.labels_ = torch.empty(0, device=self.device, dtype=torch.long)
			self._cluster_index_valid = True
			return self.labels_

		transition_nodes = self.tree.categorize_transitions(
			torch.ones(self.embedding_shape, device=self.device),
			transition_depth=self.cluster_level,
			top_k=1e9,
		)
		self.transition_nodes = transition_nodes

		if len(transition_nodes) == 0:
			self.labels_ = torch.full((len(self.sentences),), -1, device=self.device, dtype=torch.long)
			self._cluster_index_valid = True
			return self.labels_

		c = len(transition_nodes)
		d = transition_nodes[0].mean.numel()

		means = torch.empty((c, d), device=self.device)
		vars_ = torch.empty((c, d), device=self.device)
		counts = torch.empty((c,), device=self.device)
		valid = torch.zeros((c,), device=self.device, dtype=torch.bool)
		large_enough = torch.zeros((c,), device=self.device, dtype=torch.bool)

		for i, node in enumerate(transition_nodes):
			means[i] = node.mean
			counts[i] = node.count
			valid[i] = node.count > 0
			large_enough[i] = node.count >= self.min_cluster_size

			if node.count > 0:
				if self.tree.covar_from == 1:
					vars_[i] = node.sum_sq / node.count + self.tree.prior_var
				elif node.parent is not None and node.parent.count > 0:
					vars_[i] = node.parent.sum_sq / node.parent.count + self.tree.prior_var
				else:
					vars_[i] = node.sum_sq / max(node.count, 1.0) + self.tree.prior_var
			else:
				vars_[i].fill_(1.0)

		training_labels = torch.full((len(self.sentences),), -1, device=self.device, dtype=torch.long)
		for i, tnode in enumerate(transition_nodes):
			queue = [tnode]
			while queue:
				curr = queue.pop()
				sid_list = getattr(curr, "sentence_id", None) or []
				for sid in sid_list:
					if sid is not None and 0 <= sid < len(self.sentences):
						training_labels[sid] = i
				for child in getattr(curr, "children", []):
					queue.append(child)

		# Enforce minimum cluster size by mapping small clusters to -1.
		small_cluster_ids = torch.where(~large_enough)[0]
		if small_cluster_ids.numel() > 0:
			for cid in small_cluster_ids.tolist():
				training_labels[training_labels == cid] = -1

		self._means = means
		self._vars = vars_
		self._counts = counts
		self._valid_mask = valid
		self._large_enough_mask = large_enough
		self.labels_ = training_labels
		self._cluster_index_valid = True
		return self.labels_

	def fit(self, x):
		"""Fit Cobweb on BERTopic-provided embeddings and expose ``labels_``."""
		if torch.is_tensor(x):
			x_tensor = x.to(self.device)
			x_np = x_tensor.detach().cpu().numpy()
		else:
			x_np = np.asarray(x)
			x_tensor = torch.tensor(x_np, device=self.device)

		self.sentences = []
		self.sentence_to_node = {}
		self._invalidate_prediction_index()

		self._init_tree_for_embeddings(x_np)
		buffer_texts = [None] * len(x_tensor)
		self.add_sentences(buffer_texts, x_tensor)

		self.labels_ = self._gather_clusters().detach().cpu().numpy()
		return self

	def predict_clusters(self, x):
		"""Predict transition-node cluster IDs for a batch of embeddings."""
		if not self._cluster_index_valid:
			self._gather_clusters()

		if self._means is None or self._means.numel() == 0:
			n = x.shape[0] if getattr(x, "ndim", 1) > 1 else 1
			return torch.full((n,), -1, device=self.device, dtype=torch.long), torch.full((n,), float("-inf"), device=self.device)

		x_t = x.to(self.device) if torch.is_tensor(x) else torch.tensor(x, device=self.device)
		if x_t.ndim == 1:
			x_t = x_t.unsqueeze(0)

		x_exp = x_t[:, None, :]
		mu = self._means[None, :, :]
		var = self._vars[None, :, :]

		log_gauss = -0.5 * (
			torch.log(var)
			+ math.log(2.0 * math.pi)
			+ (x_exp - mu).pow(2) / var
		).sum(dim=-1)

		log_prior = torch.log(self._counts.clamp_min(1e-12)) - math.log(max(self.tree.root.count, 1e-12))
		log_prob = log_gauss + log_prior[None, :]
		log_prob[:, ~self._valid_mask] = -float("inf")

		best_node_idxs = log_prob.argmax(dim=1)
		best_scores = log_prob.max(dim=1).values

		# Convert predictions from small clusters to outlier label -1.
		is_small = ~self._large_enough_mask[best_node_idxs]
		best_node_idxs = torch.where(is_small, torch.full_like(best_node_idxs, -1), best_node_idxs)

		return best_node_idxs, best_scores

	def predict(self, x):
		"""BERTopic cluster-model predict API."""
		labels, _ = self.predict_clusters(x)
		return labels.detach().cpu().numpy()


def process_text(docs):
	"""Remove numbers and very short tokens before vectorization."""
	processed_docs = []
	for doc in docs:
		doc = re.sub(r"\d+", "", doc)
		doc = " ".join([word for word in doc.split() if len(word) > 2])
		processed_docs.append(doc)
	return processed_docs


class BERTopicHierarchicalWrapper:
	"""Build a hierarchy dataframe compatible with hierarchical runner metrics."""

	def __init__(self, docs, bertopic_model, linkage_function=None, cobweb_clusterer=None, topk=15):
		self.topk = topk
		if cobweb_clusterer is not None:
			self._from_cobweb_clusterer(docs, cobweb_clusterer, bertopic_model)
		else:
			self._from_bertopic_hierarchical(docs, bertopic_model, linkage_function)

	def _topk_words_from_row(self, row: csr_matrix, words):
		counts = row.toarray().ravel()
		if counts.size == 0:
			return []
		non_zero = np.flatnonzero(counts)
		if non_zero.size == 0:
			return []
		k = min(self.topk, non_zero.size)
		if non_zero.size <= self.topk:
			indices_sorted = non_zero[np.argsort(-counts[non_zero])]
		else:
			top_idx = np.argpartition(counts[non_zero], -k)[-k:]
			indices_sorted = non_zero[top_idx][np.argsort(-counts[non_zero][top_idx])]
		return [str(words[j]) for j in indices_sorted]

	def _from_bertopic_hierarchical(self, docs, bertopic_model, linkage_function):
		try:
			words = bertopic_model.vectorizer_model.get_feature_names_out()
		except Exception:
			words = bertopic_model.vectorizer_model.get_feature_names()

		documents = pd.DataFrame(
			{
				"Document": docs,
				"ID": range(len(docs)),
				"Topic": bertopic_model.topics_,
			}
		)
		documents_per_topic = documents.groupby(["Topic"], as_index=False).agg({"Document": " ".join})
		documents_per_topic = documents_per_topic.loc[documents_per_topic.Topic != -1, :]

		clean_documents = process_text(bertopic_model._preprocess_text(documents_per_topic.Document.values))
		bow = bertopic_model.vectorizer_model.transform(clean_documents)
		topic_ids = [int(t) for t in documents_per_topic.Topic.values]

		embeddings = select_topic_representation(
			bertopic_model.c_tf_idf_,
			bertopic_model.topic_embeddings_,
			use_ctfidf=False,
		)[0]
		try:
			outliers = int(getattr(bertopic_model, "_outliers", 0))
		except Exception:
			outliers = 0
		embeddings = embeddings[outliers:]

		sim = cosine_similarity(embeddings)
		np.fill_diagonal(sim, 1.0)
		dist_square = 1.0 - sim
		np.fill_diagonal(dist_square, 0.0)
		dist_square[dist_square < 0] = 0.0
		dist_condensed = squareform(dist_square, checks=False)

		if linkage_function is None:
			z = sch.linkage(dist_condensed, method="ward", optimal_ordering=True)
		else:
			z = linkage_function(dist_condensed)

		n = embeddings.shape[0]
		adjacency = {}
		for i in range(len(z)):
			parent_id = int(n + i)
			left_raw = int(z[i][0])
			right_raw = int(z[i][1])
			left_id = topic_ids[left_raw] if left_raw < n else int(left_raw)
			right_id = topic_ids[right_raw] if right_raw < n else int(right_raw)
			adjacency[parent_id] = [left_id, right_id]

		leaves = set(topic_ids)
		levels = {int(l): 0 for l in leaves}
		unresolved_parents = set(adjacency.keys())
		while unresolved_parents:
			resolved_now = set()
			for parent in unresolved_parents:
				c1, c2 = adjacency[parent]
				if c1 in levels and c2 in levels:
					levels[parent] = max(levels[c1], levels[c2]) + 1
					resolved_now.add(parent)
			if not resolved_now:
				break
			unresolved_parents = unresolved_parents.difference(resolved_now)

		topic_id_to_row = {int(t): i for i, t in enumerate(documents_per_topic.Topic.values)}

		node_to_leaves = {int(l): {int(l)} for l in leaves}
		unresolved = set(adjacency.keys())
		while unresolved:
			progressed = set()
			for parent in unresolved:
				c1, c2 = adjacency[parent]
				if c1 in node_to_leaves and c2 in node_to_leaves:
					node_to_leaves[parent] = node_to_leaves[c1].union(node_to_leaves[c2])
					progressed.add(parent)
			if not progressed:
				break
			unresolved = unresolved.difference(progressed)

		rows = []
		for leaf in sorted(leaves):
			leaf_id = int(leaf)
			keywords = []
			if leaf_id in topic_id_to_row:
				keywords = self._topk_words_from_row(bow[topic_id_to_row[leaf_id]], words)
			rows.append(
				{
					"Node_ID": leaf_id,
					"Level": levels.get(leaf_id, 0),
					"Keywords": keywords,
					"children_ids": [],
				}
			)

		for parent in sorted(adjacency.keys()):
			leaf_ids = node_to_leaves.get(parent, set())
			indices = [topic_id_to_row[lid] for lid in leaf_ids if lid in topic_id_to_row]
			if indices:
				grouped = csr_matrix(bow[indices].sum(axis=0))
				keywords = self._topk_words_from_row(grouped, words)
			else:
				keywords = []
			rows.append(
				{
					"Node_ID": int(parent),
					"Level": levels.get(int(parent), None),
					"Keywords": keywords,
					"children_ids": adjacency.get(int(parent), []),
				}
			)

		self.hierachical_topics = pd.DataFrame(rows).sort_values(["Level", "Node_ID"]).reset_index(drop=True)

	def _from_cobweb_clusterer(self, docs, cobweb_clusterer, bertopic_model):
		doc_map, children_map = cobweb_clusterer._create_node_doc_assignment()

		try:
			words = bertopic_model.vectorizer_model.get_feature_names_out()
		except Exception:
			words = bertopic_model.vectorizer_model.get_feature_names()

		clean_docs = process_text(bertopic_model._preprocess_text(docs))
		bow_docs = bertopic_model.vectorizer_model.transform(clean_docs)

		adjacency = {}
		node_ids = set()
		for node, children in children_map.items():
			nid = int(getattr(node, "id", hash(node)))
			adjacency[nid] = [int(getattr(c, "id", hash(c))) for c in children]
			node_ids.add(nid)
			for child in children:
				node_ids.add(int(getattr(child, "id", hash(child))))

		for node in doc_map.keys():
			nid = int(getattr(node, "id", hash(node)))
			node_ids.add(nid)
			adjacency.setdefault(nid, [])

		leaves = {nid for nid in node_ids if len(adjacency.get(nid, [])) == 0}
		levels = {nid: 0 for nid in leaves}
		unresolved_parents = {nid for nid in node_ids if len(adjacency.get(nid, [])) > 0}
		id_to_node = {int(getattr(node, "id", hash(node))): node for node in doc_map.keys()}

		while unresolved_parents:
			resolved_now = set()
			for parent in list(unresolved_parents):
				children = adjacency[parent]
				if all((child in levels) for child in children):
					levels[parent] = (max(levels[child] for child in children) + 1) if children else 0
					resolved_now.add(parent)
			if not resolved_now:
				break
			unresolved_parents = unresolved_parents.difference(resolved_now)

		rows = []
		for nid in sorted(node_ids, key=lambda x: (levels.get(x, 0), x)):
			node_obj = id_to_node.get(nid, None)
			doc_indices = doc_map.get(node_obj, []) if node_obj is not None else []
			if doc_indices:
				grouped = csr_matrix(bow_docs[doc_indices].sum(axis=0))
				keywords = self._topk_words_from_row(grouped, words)
			else:
				keywords = []
			rows.append(
				{
					"Node_ID": nid,
					"Level": levels.get(nid, 0),
					"Keywords": keywords,
					"children_ids": adjacency.get(nid, []),
				}
			)

		self.hierachical_topics = pd.DataFrame(rows).sort_values(["Level", "Node_ID"]).reset_index(drop=True)
