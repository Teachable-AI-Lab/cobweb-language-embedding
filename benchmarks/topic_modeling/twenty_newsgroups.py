"""20 Newsgroups dataset loader compatible with CobwebTM experiments."""

from __future__ import annotations

from typing import List, Optional, Sequence

from sklearn.datasets import fetch_20newsgroups

from .cobwebtm_utils import CobwebTMDataset


class TwentyNewsgroupsDataset(CobwebTMDataset):
    """Load 20 Newsgroups into a CobwebTMDataset."""

    @classmethod
    def load(
        cls,
        subset: str = "test",
        categories: Optional[Sequence[str]] = None,
        remove_headers: bool = True,
        max_docs: Optional[int] = None,
        analyzer: Optional[callable] = None,
    ) -> "TwentyNewsgroupsDataset":
        remove = ("headers", "footers", "quotes") if remove_headers else ()
        dataset = fetch_20newsgroups(subset=subset, categories=categories, remove=remove)
        docs: List[str] = dataset.data
        if max_docs is not None:
            docs = docs[:max_docs]
        return cls.from_texts(docs, analyzer=analyzer)
