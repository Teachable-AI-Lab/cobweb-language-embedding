"""Public package interface for cobweb-language-embedding.

This module keeps imports lazy so ``import cobweb_language_embedding`` does not
eagerly import heavy optional dependencies.
"""

__all__ = [
    "CobwebWrapper",
    "CobwebRetriever",
    "PCAICAWhiteningModel",
    "BERTopicCobwebWrapper",
    "BERTopicHierarchicalWrapper",
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
    if name == "BERTopicCobwebWrapper":
        from .topic_modeling import BERTopicCobwebWrapper as _BERTopicCobwebWrapper
        return _BERTopicCobwebWrapper
    if name == "BERTopicHierarchicalWrapper":
        from .topic_modeling import BERTopicHierarchicalWrapper as _BERTopicHierarchicalWrapper
        return _BERTopicHierarchicalWrapper
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
