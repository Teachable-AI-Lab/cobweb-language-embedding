"""Public package interface for cobweb-language-embedding.

This module keeps imports lazy so ``import cobweb_language_embedding`` does not
eagerly import heavy optional dependencies.
"""

__all__ = [
    "CobwebWrapper",
    "CobwebRetriever",
    "PCAICAWhiteningModel",
    "CobwebTM",
    "CobwebTMHierarchical",
    "IncrementalCobwebTM",
    "PersistentCobwebTM",
]


def __getattr__(name):
    if name == "CobwebWrapper":
        from .CobwebWrapper import CobwebWrapper as _CobwebWrapper
        return _CobwebWrapper
    if name == "CobwebRetriever":
        from .retrieval import CobwebRetriever as _CobwebRetriever
        return _CobwebRetriever
    if name == "PCAICAWhiteningModel":
        from .preprocess_embedding import PCAICAWhiteningModel as _PCAICAWhiteningModel
        return _PCAICAWhiteningModel
    if name == "CobwebTM":
        from .topic_modeling import CobwebTM as _CobwebTM
        return _CobwebTM
    if name == "CobwebTMHierarchical":
        from .topic_modeling import CobwebTMHierarchical as _CobwebTMHierarchical
        return _CobwebTMHierarchical
    if name == "IncrementalCobwebTM":
        from .topic_modeling import IncrementalCobwebTM as _cls
        return _cls
    if name == "PersistentCobwebTM":
        from .topic_modeling import PersistentCobwebTM as _cls
        return _cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
