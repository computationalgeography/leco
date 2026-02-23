"""Run the leco model of language evolution."""

from .initialization import initialize_barrier, initialize_population
from .interaction import interact, nearest_neighbors
from .movement import move
from .population_dynamics import population_dynamics
from .simulation import mutate_profile, simulate

__all__ = [
    "initialize_barrier",
    "initialize_population",
    "interact",
    "move",
    "mutate_profile",
    "nearest_neighbors",
    "population_dynamics",
    "simulate",
]
