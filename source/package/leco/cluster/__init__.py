"""Cluster language profiles into languages."""

from .all import classify_all
from .feed_forward import diversify
from .per_step import diversify_stepwise

__all__ = ["classify_all", "diversify", "diversify_stepwise"]
