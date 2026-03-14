"""
Incremental 20 Newsgroups dataset loader for CobwebTM experiments.

Loads the 20 Newsgroups corpus (scikit-learn) and wraps it as an
IncrementalCobwebTMDataset so it can be consumed batch-by-batch.
"""

from __future__ import annotations

import random
from typing import List, Optional, Sequence

from sklearn.datasets import fetch_20newsgroups

from ..cobwebtm_utils import IncrementalCobwebTMDataset


class TwentyNewsgroupsIncrementalDataset:
    """Load 20 Newsgroups into an IncrementalCobwebTMDataset."""

    @classmethod
    def load(
        cls,
        subset: str = "all",
        categories: Optional[Sequence[str]] = None,
        remove_headers: bool = True,
        batch_size: int = 512,
        first_batch_size: Optional[int] = None,
        max_docs: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 42,
        analyzer: Optional[callable] = None,
    ) -> IncrementalCobwebTMDataset:
        remove = ("headers", "footers", "quotes") if remove_headers else ()
        dataset = fetch_20newsgroups(
            subset=subset, categories=categories, remove=remove
        )
        docs: List[str] = list(dataset.data)

        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(docs)

        if max_docs is not None:
            docs = docs[:max_docs]

        return IncrementalCobwebTMDataset.from_texts(
            documents=docs,
            batch_size=batch_size,
            first_batch_size=first_batch_size,
            analyzer=analyzer,
        )
