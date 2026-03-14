"""
Time-sorted TweetNER7 incremental loader for CobwebTM experiments.

Uses the tner/tweetner7 dataset and orders tweets by timestamp if available,
otherwise by id/index. Tokens are joined into space-separated text for CobwebTM.
"""

from __future__ import annotations

from typing import List, Optional
import random

from ..cobwebtm_utils import IncrementalCobwebTMDataset


class TweetNER7IncrementalDataset:
	"""Load TweetNER7 in temporal order into an IncrementalCobwebTMDataset."""

	@classmethod
	def load(
		cls,
		split: str = "test_2021",
		batch_size: int = 1,
		first_batch_size: Optional[int] = None,
		max_docs: Optional[int] = None,
		analyzer: Optional[callable] = None,
	) -> IncrementalCobwebTMDataset:
		try:
			from datasets import load_dataset
		except ImportError as exc:
			raise ImportError("TweetNER7 loader requires the 'datasets' package: pip install datasets") from exc

		ds = load_dataset("tner/tweetner7", split=split)

		total = len(ds)
		desired_n = total if max_docs is None else min(max_docs, total)
		if desired_n < total:
			try:
				indices = random.sample(range(total), desired_n)
				ds = ds.select(indices)
			except Exception:
				all_recs = list(ds)
				ds = random.sample(all_recs, desired_n)

		date_field = None
		col_names = getattr(ds, "column_names", None) or ([] if isinstance(ds, list) else [])
		if not col_names and isinstance(ds, list) and ds:
			col_names = list(ds[0].keys())

		for field in ("timestamp", "created_at", "date", "id"):
			if field in col_names:
				date_field = field
				break

		if date_field is not None:
			if isinstance(ds, list):
				ds.sort(key=lambda r: r.get(date_field))
			else:
				ds = ds.sort(date_field)

		if "tokens" in col_names:
			docs: List[str] = [" ".join(tokens) for tokens in (ds if isinstance(ds, list) else ds["tokens"])]
		elif "text" in col_names:
			docs = [str(t) for t in (ds if isinstance(ds, list) else ds["text"])]
		else:
			raise ValueError("No tokens/text field found in TweetNER7 dataset")

		return IncrementalCobwebTMDataset.from_texts(
			documents=docs,
			batch_size=batch_size,
			first_batch_size=first_batch_size,
			analyzer=analyzer,
		)
