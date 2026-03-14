"""Incremental dataset loaders for streaming CobwebTM experiments."""

from .reuters_rcv1 import Reuters21578IncrementalDataset
from .stackexchange import StackExchangeIncrementalDataset
from .tweetner7 import TweetNER7IncrementalDataset
from .spatiotemporal_news import SpatioTemporalNewsIncrementalDataset, GDELTGKGIncrementalDataset
from .twenty_newsgroups import TwentyNewsgroupsIncrementalDataset

__all__ = [
    "Reuters21578IncrementalDataset",
    "StackExchangeIncrementalDataset",
    "TweetNER7IncrementalDataset",
    "SpatioTemporalNewsIncrementalDataset",
    "GDELTGKGIncrementalDataset",
    "TwentyNewsgroupsIncrementalDataset",
]
