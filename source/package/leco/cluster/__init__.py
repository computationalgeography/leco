"""Cluster language profiles into languages."""

from .all import classify_all
from .feed_forward import diversify
from .version import __version__

__all__ = ["__version__", "classify_all", "diversify"]
