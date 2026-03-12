import random

import torch

from .CobwebWrapper import CobwebWrapper


class CobwebRetriever(CobwebWrapper):
    def __init__(self, corpus=None, corpus_embeddings=None, cobweb_wrapper = None):
        if corpus is None and corpus_embeddings is None and cobweb_wrapper is None:
            raise ValueError("Must provide either corpus/corpus_embeddings or an existing CobwebWrapper.")
        empty_wrapper = cobweb_wrapper is not None
        super().__init__(corpus, corpus_embeddings, empty_wrapper=empty_wrapper)
        if cobweb_wrapper is not None:
            self.tree = cobweb_wrapper.tree
        self._prediction_index_valid = False
        self._index_to_node = {}
        self._node_means = None
        self._node_vars = None
        self._leaf_to_path_indices = None
        self._path_matrix = None
        self.max_depth = 0

    def query(self, embedding, k=5, return_ids=False, use_indexed=False):
        """Query the Cobweb tree for top-k similar sentences to the embedding."""
        if use_indexed:
            return self.cobweb_predict_indexed(embedding, k, return_ids)
        else:
            return self.cobweb_predict(embedding, k, return_ids)

    def cobweb_predict(self, embedding, k=5, return_ids=False):
        """Predict top-k similar entries from the tree for an embedding."""
        tensor = embedding.to(self.device) if isinstance(embedding, torch.Tensor) else torch.tensor(embedding, device=self.device)
        leaves = self.tree.categorize(tensor, use_best=True, max_nodes=self.max_init_search, retrieve_k=k)

        results = []
        for leaf in leaves:
            sid_lst = getattr(leaf, "sentence_id", None)
            if not sid_lst:
                continue
            random.shuffle(sid_lst)
            for sid in sid_lst:
                if sid is None or sid >= len(self.sentences):
                    continue
                results.append(sid if return_ids else self.sentences[sid])
        return results
    
    def _invalidate_prediction_index(self):
        """Invalidate the prediction index when tree structure changes."""
        self._prediction_index_valid = False
        self._index_to_node.clear()
        self._node_means = None
        self._node_vars = None
        self._leaf_to_path_indices = None
        self._path_matrix = None

    def build_prediction_index(self):
        """Build an index of all nodes in the tree for faster prediction."""
        if self._prediction_index_valid:
            return
        print("Building prediction index...")

        if set(self.sentence_to_node.keys()) != set(range(len(self.sentences))):
            raise ValueError("sentence_to_node mapping is inconsistent with sentence indices.")

        self._index_to_node.clear()

        # Collect all nodes via BFS traversal.
        idx = 0
        leaf_count = 0
        queue = [(self.tree.root, tuple())]
        self._leaf_to_path_indices = [None] * len(self.sentences)
        while queue:
            node, path = queue[0]
            queue = queue[1:]

            self._index_to_node[idx] = node
            for child in getattr(node, "children", []):
                queue.append((child, path + (idx,)))

            if hasattr(node, "sentence_id") and node.sentence_id:
                for sid in node.sentence_id:
                    if sid < len(self.sentences):
                        self._leaf_to_path_indices[sid] = list(path) + [idx]
                    else:
                        print(f"[Warning] Node has invalid sentence ID {sid}, skipping.")
                self.max_depth = max(self.max_depth, len(path) + 1)
                leaf_count += 1

            idx += 1

        if leaf_count != len(self.sentences):
            print(f"[Warning] Leaf count mismatch: expected {len(self.sentences)}, found {leaf_count}.")

        for sid, path in enumerate(self._leaf_to_path_indices):
            if path is None:
                print(
                    f"[Warning] Leaf path index for sentence ID {sid} is None. "
                    "This may indicate missing sentences in the tree."
                )
                node = self.sentence_to_node.get(sid)
                if node is not None:
                    print(node.sentence_id, sid)

            node = self.sentence_to_node.get(sid)
            if node is not None and node not in self._index_to_node.values():
                print(f"[Warning] Node for sentence ID {sid} not found in indexed nodes.")

        # Build sparse path matrix for efficient path scoring.
        num_nodes = idx
        path_row_indices = []
        path_col_indices = []
        path_weights = []

        for leaf_idx, path in enumerate(self._leaf_to_path_indices):
            if path is None:
                continue
            path_length = len(path)
            for node_idx in path:
                path_row_indices.append(leaf_idx)
                path_col_indices.append(node_idx)
                path_weights.append(1.0 / path_length)

        if path_row_indices:
            path_indices = torch.stack(
                [
                    torch.tensor(path_row_indices, device=self.device),
                    torch.tensor(path_col_indices, device=self.device),
                ]
            )
            path_values = torch.tensor(path_weights, device=self.device, dtype=torch.float)
            self._path_matrix = torch.sparse_coo_tensor(
                path_indices,
                path_values,
                (len(self._leaf_to_path_indices), num_nodes),
                device=self.device,
            ).coalesce()
        else:
            self._path_matrix = None

        # Pre-allocate tensors for means and variances.
        self._node_means = torch.zeros(num_nodes, self.tree.shape[0], device=self.device, dtype=torch.float)
        self._node_vars = torch.zeros(num_nodes, self.tree.shape[0], device=self.device, dtype=torch.float)

        for node_idx, node in self._index_to_node.items():
            self._node_means[node_idx] = node.mean
            if hasattr(node, "sum_sq") and node.count > 0:
                self._node_vars[node_idx] = self.tree.compute_var(node.sum_sq, node.count)
            else:
                self._node_vars[node_idx] = self.tree.prior_var

        self._prediction_index_valid = True
        print(f"Prediction index built: {num_nodes} nodes indexed, {leaf_count} leaf paths cached")
        if self._path_matrix is not None:
            print(f"Path matrix shape: {self._path_matrix.shape}, nnz: {self._path_matrix._nnz()}")

    def cobweb_predict_indexed(self, embedding, k=5, return_ids=False):
        """Ultra-fast prediction for an embedding using sparse path scoring."""
        self.build_prediction_index()

        x = embedding.to(self.device) if isinstance(embedding, torch.Tensor) else torch.tensor(embedding, device=self.device)

        num_leaves = len(self._leaf_to_path_indices)
        if num_leaves == 0 or self._path_matrix is None:
            return []

        diff_sq = (x.unsqueeze(0) - self._node_means) ** 2
        node_log_probs = -0.5 * (
            torch.log(self._node_vars).sum(dim=1) + (diff_sq / self._node_vars).sum(dim=1)
        )

        leaf_scores = torch.sparse.mm(self._path_matrix, node_log_probs.unsqueeze(1)).squeeze(1)

        noise = torch.randn_like(leaf_scores) * 1e-6
        noisy_scores = leaf_scores + noise
        if k >= num_leaves:
            selected_leaf_indices = torch.argsort(noisy_scores, descending=True).tolist()
        else:
            _, topk_indices = torch.topk(noisy_scores, k, largest=True)
            selected_leaf_indices = topk_indices.tolist()

        results = []
        for leaf_idx in selected_leaf_indices:
            if leaf_idx < len(self.sentences):
                results.append(leaf_idx if return_ids else self.sentences[leaf_idx])
        return results

    def cobweb_rank_scores(self, embedding):
        """Differentiable: return raw scores for all leaves."""
        self.build_prediction_index()

        x = embedding.to(self.device) if isinstance(embedding, torch.Tensor) else torch.tensor(embedding, device=self.device)

        if len(self._leaf_to_path_indices) == 0 or self._path_matrix is None:
            return torch.empty(0, device=self.device)

        diff_sq = (x.unsqueeze(0) - self._node_means) ** 2
        node_log_probs = -0.5 * (
            torch.log(self._node_vars).sum(dim=1) + (diff_sq / self._node_vars).sum(dim=1)
        )
        return torch.sparse.mm(self._path_matrix, node_log_probs.unsqueeze(1)).squeeze(1)

    def get_node_path_stats(self, sentence_id):
        """Get means and variances for all nodes on the root-to-leaf path."""
        self.build_prediction_index()

        if sentence_id < 0 or sentence_id >= len(self._leaf_to_path_indices):
            return None, None
        path_indices = self._leaf_to_path_indices[sentence_id]
        if path_indices is None:
            return None, None

        path_indices_tensor = torch.tensor(path_indices, device=self.device)
        return self._node_means[path_indices_tensor], self._node_vars[path_indices_tensor]

    def get_prediction_index_info(self):
        """Get diagnostic information about the prediction index."""
        info = {
            "index_valid": self._prediction_index_valid,
            "total_nodes": len(self._index_to_node) if self._prediction_index_valid else 0,
            "leaf_paths_cached": len(self._leaf_to_path_indices) if self._prediction_index_valid else 0,
            "means_cached": self._node_means is not None,
            "vars_cached": self._node_vars is not None,
        }

        if self._prediction_index_valid and self._node_means is not None:
            info["means_shape"] = tuple(self._node_means.shape)
            info["vars_shape"] = tuple(self._node_vars.shape)
            info["device"] = str(self._node_means.device)

        return info

    def force_rebuild_index(self):
        """Force rebuild of the prediction index."""
        self._invalidate_prediction_index()
        self.build_prediction_index()