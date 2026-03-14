"""
Time-sorted StackOverflow/StackExchange incremental loader for CobwebTM experiments.

Attempts multiple Hugging Face datasets (e.g., ``pacovaldez/stackoverflow-questions``,
``c17hawke/stackoverflow-dataset``) and orders entries by creation timestamp when
available, otherwise by id/index.
"""

from __future__ import annotations

from typing import List, Optional
import random

from ..cobwebtm_utils import IncrementalCobwebTMDataset


class StackExchangeIncrementalDataset:
	"""Load StackExchange posts in temporal order into an IncrementalCobwebTMDataset."""

	@classmethod
	def load(
		cls,
		split: str = "test",
		batch_size: int = 512,
		first_batch_size: Optional[int] = None,
		max_docs: Optional[int] = None,
		analyzer: Optional[callable] = None,
	) -> IncrementalCobwebTMDataset:
		try:
			from datasets import load_dataset
		except ImportError as exc:
			raise ImportError("StackExchange loader requires the 'datasets' package: pip install datasets") from exc

		candidates = [
			{"path": "pacovaldez/stackoverflow-questions", "kwargs": {"split": split}},
			{"path": "c17hawke/stackoverflow-dataset", "kwargs": {"split": split}},
		]

		ds = None
		last_err = None
		for cand in candidates:
			try:
				ds = load_dataset(cand["path"], **cand["kwargs"])
				x = cand["path"]
				print(f"Using dataset {x}")
				break
			except Exception as exc:  # noqa: BLE001
				last_err = exc

		if ds is None:
			raise RuntimeError("No accessible StackOverflow/StackExchange dataset found on Hugging Face") from last_err

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

		for field in ("CreationDate", "creation_date", "creation", "timestamp", "date", "Id", "pid"):
			if field in col_names:
				date_field = field
				break

		if date_field is not None:
			if isinstance(ds, list):
				ds.sort(key=lambda r: r.get(date_field))
			else:
				ds = ds.sort(date_field)

		docs: List[str] = []
		for row in (ds if isinstance(ds, list) else ds):
			title = row.get("Title", "") if isinstance(row, dict) else ""
			if not title and isinstance(row, dict):
				title = row.get("title", "")
			body = row.get("Body", "") if isinstance(row, dict) else ""
			if not body and isinstance(row, dict):
				body = row.get("body", "")
			question = (title or "") + "\n" + (body or "")
			docs.append(question.strip())

		return IncrementalCobwebTMDataset.from_texts(
			documents=docs,
			batch_size=batch_size,
			first_batch_size=first_batch_size,
			analyzer=analyzer,
		)
