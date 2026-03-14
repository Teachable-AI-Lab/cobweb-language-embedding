"""StackExchange / StackOverflow dataset loader compatible with CobwebTM experiments."""

from __future__ import annotations

from typing import List, Optional

from .cobwebtm_utils import CobwebTMDataset


class StackExchangeDataset(CobwebTMDataset):
    """Load StackExchange posts into a CobwebTMDataset."""

    @classmethod
    def load(
        cls,
        split: str = "test",
        max_docs: Optional[int] = 5000,
        analyzer: Optional[callable] = None,
        no_body: bool = False,
    ) -> "StackExchangeDataset":
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError("StackExchange benchmark requires datasets: pip install datasets") from exc

        dataset = load_dataset("pacovaldez/stackoverflow-questions", split=split)

        docs: List[str] = []
        for row in dataset:
            title = row.get("Title", "") or row.get("title", "")
            body = row.get("Body", "") or row.get("body", "")
            docs.append(title.strip() if no_body else f"{title}\n{body}".strip())

        if max_docs is not None:
            docs = docs[:max_docs]

        return cls.from_texts(docs, analyzer=analyzer)
