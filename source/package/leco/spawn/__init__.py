"""Code related to spawning multiple leco runs."""

import os

from .cluster import cluster
from .run import run

__all__ = ["cluster", "default_max_nr_workers", "run"]


def default_max_nr_workers() -> int:
    """Return default maximum number of workers (processes using a single CPU core) to use."""
    return (os.cpu_count() or 2) // 2
