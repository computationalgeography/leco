"""Run the leco model of language evolution."""

from .initialization import initialize_barrier, initialize_population
from .interaction import interact
from .simulation import mutate_profile, simulate
from .movement import move
from .population_dynamics import population_dynamics
from .version import __version__

__all__ = [
    "__version__",
    "initialize_barrier",
    "initialize_population",
    "interact",
    "move",
    "mutate_profile",
    "population_dynamics",
    "simulate",
]
