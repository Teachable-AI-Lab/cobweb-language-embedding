"""Embedding preprocessing utilities (PCA+ICA whitening)."""

from __future__ import annotations

import pickle

import numpy as np
from sklearn.decomposition import FastICA, PCA


class PCAICAWhiteningModel:
	"""PCA + ICA whitening transform for embedding vectors."""

	def __init__(
		self,
		mean: np.ndarray,
		pca_components: np.ndarray,
		ica_unmixing: np.ndarray,
		pca_explained_var: np.ndarray,
		eps: float = 1e-8,
	):
		self.mean = mean
		self.pca_components = pca_components
		self.pca_explained_var = pca_explained_var
		self.ica_unmixing = ica_unmixing
		self.eps = eps

	def __repr__(self) -> str:
		return (
			f"{self.__class__.__name__}(\n"
			f"  mean.shape={self.mean.shape},\n"
			f"  pca_components.shape={self.pca_components.shape},\n"
			f"  pca_explained_var.shape={self.pca_explained_var.shape},\n"
			f"  ica_unmixing.shape={self.ica_unmixing.shape},\n"
			f"  eps={self.eps}\n"
			f")"
		)

	def transform(self, x: np.ndarray, is_ica: bool = True) -> np.ndarray:
		"""Apply PCA + ICA whitening to a single embedding or a batch."""
		is_single = x.ndim == 1
		if is_single:
			x = x[np.newaxis, :]

		x_centered = x - self.mean
		x_pca = np.dot(x_centered, self.pca_components.T)
		x_pca /= np.sqrt(self.pca_explained_var + self.eps)

		if is_ica:
			x_ica = np.dot(x_pca, self.ica_unmixing.T)
			return x_ica[0] if is_single else x_ica
		return x_pca[0] if is_single else x_pca

	@classmethod
	def fit(
		cls,
		x: np.ndarray,
		pca_dim: int = 256,
		eps: float = 1e-8,
		ica_max_iter: int = 5000,
		ica_tol: float = 1e-3,
	) -> "PCAICAWhiteningModel":
		"""Fit PCA -> ICA whitening on embedding matrix x."""
		mean = x.mean(axis=0)
		x_centered = x - mean

		dim = max(1, min(pca_dim, x.shape[1], x.shape[0]))
		pca = PCA(n_components=dim)
		x_pca = pca.fit_transform(x_centered)
		components = pca.components_
		explained_var = pca.explained_variance_

		x_pca_normalized = x_pca / np.sqrt(explained_var + eps)
		ica = FastICA(
			n_components=components.shape[0],
			whiten="unit-variance",
			max_iter=ica_max_iter,
			tol=ica_tol,
			random_state=42,
		)
		ica.fit(x_pca_normalized)

		return cls(mean, components, ica.components_, explained_var, eps)

	def save(self, filepath: str) -> None:
		with open(filepath, "wb") as f:
			pickle.dump(
				{
					"mean": self.mean,
					"pca_components": self.pca_components,
					"pca_explained_var": self.pca_explained_var,
					"ica_unmixing": self.ica_unmixing,
					"eps": self.eps,
				},
				f,
			)

	@classmethod
	def load(cls, filepath: str) -> "PCAICAWhiteningModel":
		with open(filepath, "rb") as f:
			data = pickle.load(f)
		return cls(
			mean=data["mean"],
			pca_components=data["pca_components"],
			pca_explained_var=data["pca_explained_var"],
			ica_unmixing=data["ica_unmixing"],
			eps=data["eps"],
		)

