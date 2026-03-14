"""
Time-sorted SpatioTemporal News Corpus incremental loader for CobwebTM.

Loads the ``Artur-B/SpatioTemporal-News-Corpus`` dataset from HuggingFace,
orders records by the ``date`` field and uses the ``article`` field as the
text for each document.
"""

from __future__ import annotations

from typing import List, Optional
import random

from ..cobwebtm_utils import IncrementalCobwebTMDataset


class SpatioTemporalNewsIncrementalDataset:
    """Load SpatioTemporal-News-Corpus in temporal order into an IncrementalCobwebTMDataset."""

    @classmethod
    def load(
        cls,
        split: str = "train",
        batch_size: int = 1,
        first_batch_size: Optional[int] = None,
        max_docs: Optional[int] = None,
        analyzer: Optional[callable] = None,
    ) -> IncrementalCobwebTMDataset:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "SpatioTemporal-News loader requires the 'datasets' package: pip install datasets"
            ) from exc

        ds = load_dataset("Artur-B/SpatioTemporal-News-Corpus", split=split)

        total = len(ds)
        desired_n = total if max_docs is None else min(max_docs, total)

        if desired_n < total:
            try:
                indices = random.sample(range(total), desired_n)
                ds = ds.select(indices)
            except Exception:
                all_recs = list(ds)
                ds = random.sample(all_recs, desired_n)

        if isinstance(ds, list):
            if any(isinstance(r, dict) and "date" in r for r in ds):
                ds.sort(key=lambda r: r.get("date"))
        else:
            if "date" in ds.column_names:
                ds = ds.sort("date")

        text_field = "article"
        col_names = getattr(ds, "column_names", None) or ([] if isinstance(ds, list) else [])
        if not col_names and isinstance(ds, list):
            if ds:
                col_names = list(ds[0].keys())

        if text_field not in col_names:
            for alt in ("text", "content", "body"):
                if alt in col_names:
                    text_field = alt
                    break

        docs: List[str] = [str(rec.get(text_field, "")) for rec in (ds if isinstance(ds, list) else ds)]

        return IncrementalCobwebTMDataset.from_texts(
            documents=docs,
            batch_size=batch_size,
            first_batch_size=first_batch_size,
            analyzer=analyzer,
        )


# Backwards-compatible alias
GDELTGKGIncrementalDataset = SpatioTemporalNewsIncrementalDataset
