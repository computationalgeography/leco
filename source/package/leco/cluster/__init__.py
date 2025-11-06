"""Cluster language profiles into languages."""

from .language_classification import language_classification, read_geoparquet, classify_all
from .version import __version__

__all__ = ["__version__", "language_classification", "read_geoparquet", "classify_all"]
