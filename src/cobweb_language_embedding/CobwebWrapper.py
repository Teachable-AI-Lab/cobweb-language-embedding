import torch
from tqdm import tqdm
import os
import hashlib
from graphviz import Digraph
from .CPPCobweb import CPPCobwebTree as CobwebTorchTree

class CobwebWrapper:
    def __init__(self, corpus=None, corpus_embeddings=None, empty_wrapper=False):
        """
        Initializes the CobwebWrapper with optional sentences and/or embeddings.
        """
        self.sentences = []
        self.sentence_to_node = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if corpus is None and corpus_embeddings is None and not empty_wrapper:
            raise ValueError("Provide at least one of corpus, corpus_embeddings, or set empty_wrapper=True.")
        elif not empty_wrapper:
            self.init_tree(corpus, corpus_embeddings)


    def init_tree(self, corpus, corpus_embeddings):
        if corpus_embeddings is not None:
            corpus_embeddings = torch.tensor(corpus_embeddings)
            embedding_shape = embedding_shape if embedding_shape is not None else corpus_embeddings.shape[1:]
        elif embedding_shape is None:
            raise ValueError("Provide either corpus_embeddings or embedding_shape.")

        self.tree = CobwebTorchTree(shape=embedding_shape, device=self.device, use_info=True, acuity_cutoff=False, use_kl=True, alpha=1e-8)

        if corpus is not None and len(corpus) > 0 and corpus_embeddings is not None:
            self.add_sentences(corpus, corpus_embeddings)

    def add_sentences(self, new_sentences, new_embeddings):
        """
        Adds new sentences and/or embeddings to the Cobweb tree.
        If a sentence is None, it is treated as an embedding-only entry.
        """
        new_embeddings = torch.tensor(new_embeddings)
        start_index = len(self.sentences)
        for i, (sent, emb) in tqdm(enumerate(zip(new_sentences, new_embeddings)),
                                   total=len(new_sentences),
                                   desc="Training CobwebTree"):
            self.sentences.append(sent)
            leaf = self.tree.ifit(torch.tensor(emb, device=self.device))
            if leaf.sentence_id is None:
                leaf.sentence_id = []
            leaf.sentence_id.append(start_index + i)
            self.sentence_to_node[start_index + i] = leaf           
        self._invalidate_prediction_index()

    def _invalidate_prediction_index(self):
        """No-op in base wrapper; retrieval subclass overrides this."""
        return



    def print_tree(self):
        """
        Recursively prints the tree structure.
        """
        def _print_node(node, depth=0):
            indent = "  " * depth
            label = f"Sentence ID: {getattr(node, 'sentence_id', 'N/A')}"
            print(f"{indent}- Node ID {node.id} {label}")
            sid = getattr(node, "sentence_id", None)
            if sid is not None and sid < len(self.sentences):
                sentence = self.sentences[sid]
                if sentence is not None:
                    print(f"{indent}    \"{sentence}\"")
                else:
                    print(f"{indent}    [Embedding only]")
            for child in getattr(node, "children", []):
                _print_node(child, depth + 1)

        print("\nCobweb Sentence Clustering Tree:")
        _print_node(self.tree.root)

    def __len__(self):
        """
        Returns the number of sentences in the Cobweb tree.
        """
        return len(self.sentences)

    def _visualize_grandparent_tree(self, tree_root, sentences, output_dir="grandparent_trees", num_leaves=6):

        os.makedirs(output_dir, exist_ok=True)

        def get_sentence_label(sid, max_len=250, wrap=40):
            if sid is not None and sid < len(sentences):
                sentence = sentences[sid]
                if sentence:
                    needs_ellipsis = len(sentence) > max_len
                    truncated = sentence[:max_len].rstrip()
                    if needs_ellipsis:
                        truncated += "..."
                    # Wrap at word boundaries every ~wrap characters
                    words = truncated.split()
                    lines = []
                    current_line = ""
                    for word in words:
                        if len(current_line) + len(word) + 1 > wrap:
                            lines.append(current_line)
                            current_line = word
                        else:
                            current_line += (" " if current_line else "") + word
                    if current_line:
                        lines.append(current_line)
                    return "\n".join(lines)
            return None


        def is_leaf_with_sentence(node):
            sid = getattr(node, "sentence_id", None)
            return get_sentence_label(sid) is not None

        def is_grandparent(node):
            # A grandparent is a node whose children have children (i.e., grandchildren exist)
            return any(
                child and getattr(child, "children", None)
                for child in getattr(node, "children", [])
            )

        def collect_grandparents(node):
            result = []
            if is_grandparent(node):
                # Only include this grandparent if it has leaf descendants with valid sentences
                valid_leaf_count = sum(
                    is_leaf_with_sentence(leaf)
                    for child in getattr(node, "children", [])
                    for leaf in getattr(child, "children", [])
                )
                if valid_leaf_count > 0:
                    result.append(node)
            for child in getattr(node, "children", []):
                result.extend(collect_grandparents(child))
            return result

        def get_filename_for_grandparent(node, index=0):
            sid = getattr(node, "sentence_id", None)
            if sid is not None and sid < len(sentences):
                sentence = sentences[sid]
                if sentence:
                    short_hash = hashlib.sha1(sentence.encode()).hexdigest()[:8]
                    return f"gp_{sid}_{short_hash}_{index}.png"
            return f"gp_node_{getattr(node, 'id', 'unknown')}_{index}.png"

        def process_subtree(grandparent_node):
            all_leaves = []
            parent_map = {}

            # First collect only parents/leaves with valid sentences
            for parent in getattr(grandparent_node, "children", []):
                valid_leaves = [leaf for leaf in getattr(parent, "children", []) if is_leaf_with_sentence(leaf)]
                if valid_leaves:
                    parent_map[parent] = valid_leaves
                    all_leaves.extend(valid_leaves)

            if not all_leaves:
                return  # No valid subtree to render

            # Split leaves into batches of 6
            leaf_batches = [all_leaves[i:i + num_leaves] for i in range(0, len(all_leaves), 6)]

            for batch_index, batch in enumerate(leaf_batches):
                dot = Digraph(comment="Grandparent Subtree", format='png')
                dot.attr(rankdir='TB')
                dot.attr('edge', color='lightblue')

                node_ids = {}
                local_counter = {"id": 0}

                def local_next_id():
                    local_counter["id"] += 1
                    return f"n{local_counter['id']}"

                # Grandparent node
                gp_node_id = local_next_id()
                node_ids[grandparent_node] = gp_node_id
                dot.node(gp_node_id, "", shape='circle', width='0.5', style='filled', color='lightblue')

                # Include only relevant parents and children
                for parent, leaves in parent_map.items():
                    # Only include this parent if it has leaves in current batch
                    filtered_leaves = [leaf for leaf in leaves if leaf in batch]
                    if not filtered_leaves:
                        continue

                    parent_id = local_next_id()
                    node_ids[parent] = parent_id
                    dot.node(parent_id, "", shape='circle', width='0.25', style='filled', color='#666666')
                    dot.edge(gp_node_id, parent_id)

                    for leaf in filtered_leaves:
                        sid = getattr(leaf, "sentence_id", None)
                        label = get_sentence_label(sid)
                        if not label:
                            continue  # already filtered, but double-check

                        leaf_id = local_next_id()
                        dot.node(leaf_id, label, shape='box', style='filled', color='lightgrey')
                        dot.edge(parent_id, leaf_id)

                filename = get_filename_for_grandparent(grandparent_node, batch_index)
                filepath = os.path.join(output_dir, filename)
                dot.render(filepath, cleanup=True)
                print(f"Saved: {filepath}")

        grandparents = collect_grandparents(tree_root)
        for gp in grandparents:
            process_subtree(gp)



    def visualize_subtrees(self, directory, num_leaves=6):
        self._visualize_grandparent_tree(self.tree.root, self.sentences, directory, num_leaves)
