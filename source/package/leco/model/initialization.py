"""Functions for initialization of the agent population and space."""

import logging

import geopandas as gpd
import numpy as np
from shapely import (
    Polygon,
    box,
)

logger = logging.getLogger(__name__)


def initialize_barrier(
    space: list[float, float],
    bar_x: list[float, float],
    bar_y: list[float, float],
) -> Polygon:
    """Initialize a barrier in continuous space."""
    # Create the barrier as a shapely Polygon
    barrier = box(bar_x[0], bar_y[0], bar_x[1], bar_y[1])

    # Bounding box of the entire space
    bounds = box(0, 0, space[0], space[1])

    # Clip the barrier to fit inside the bounds
    barrier_clipped = barrier.intersection(bounds)

    # Check if clipping occurred
    if not barrier_clipped.equals(barrier):
        logger.warning("Warning: Spatial barrier was clipped to fit within the entire space.")

    return barrier_clipped


def initialize_coordinates(
    min_coordinates: float,
    max_coordinates: float,
    nr_agents: int,
    rng: np.random.default_rng,
) -> np.ndarray[float]:
    """Randomly initialize coordinates along one axis within the specified range."""
    return rng.uniform(low=min_coordinates, high=max_coordinates, size=nr_agents)


def initialize_positions(
    space: list[float],
    subset_area: dict[str, bool | list[float]],
    nr_agents: int,
    rng: np.random.default_rng,
) -> tuple[np.ndarray[float], np.ndarray[float]]:
    """Initialize coordinates for the number of start agents within the specified initialization area."""
    if subset_area["present"] is True:
        # If a initialization area is specified, use those coordinates as the range
        x = initialize_coordinates(subset_area["x_extent"][0], subset_area["x_extent"][1], nr_agents, rng)
        y = initialize_coordinates(subset_area["y_extent"][0], subset_area["y_extent"][1], nr_agents, rng)
        return x, y
    # Agents can be initialized across the entire space
    x = initialize_coordinates(0.0, space[0], nr_agents, rng)
    y = initialize_coordinates(0.0, space[1], nr_agents, rng)
    return x, y


def initialize_language_profile(
    nr_meanings: int,
    nr_forms: int,
    rng: np.random.default_rng,
) -> np.ndarray[int]:
    """Randomly initialize a language profile with size 'nr_meanings'."""
    # Each meaning is randomly assigned a form
    return rng.integers(0, nr_forms, nr_meanings)


def initialize_spatial_profile_assignments(
    x: np.ndarray,
    y: np.ndarray,
    nr_languages: int,
    rng: np.random.default_rng,
) -> np.ndarray:
    """Assign language-profile indices to agents so that profiles are spatially grouped.

    Voronoi partitioning: pick `nr_languages` seed agents and assign every agent to the nearest seed
    """
    nr_agents = len(x)

    coordinates = np.column_stack([x, y])

    # Randomly pick start agents (unique indices)
    seed_idx = rng.choice(nr_agents, size=nr_languages, replace=False)
    seed_coords = coordinates[seed_idx]

    # Squared Euclidean distance to each seed; shape: (nr_agents, k)
    diff = coordinates[:, None, :] - seed_coords[None, :, :]
    sq_dist = np.sum(diff * diff, axis=2)

    # Assign each agent to its nearest seed
    # If nr_languages > nr_agents, just keep assignments in [0, nr_agents-1]
    return np.argmin(sq_dist, axis=1).astype(int)


def initialize_population(
    nr_agents: int,
    space: list[float],
    subset_area: dict[str, bool | list[float]],
    nr_languages: int,
    nr_forms: int,
    nr_meanings: int,
    rng: np.random.default_rng,
) -> list[gpd.GeoDataFrame, int]:
    """Return a data frame containing for each agent the following properties."""
    """
    - id
    - language_profile
    - point position
    - parent id (set to None at initialization)
    """
    ids = list(range(1, nr_agents + 1))  # ids from 1 to number of agents

    # Assign positions
    x, y = initialize_positions(space, subset_area, nr_agents, rng)

    # Assign language profiles, each profile represented by a string of integers
    # For each agent:
    #     For each meaning:
    #         A random int [0, forms]

    # Create the start language profiles, number is equal to nr_languages
    start_profiles = [initialize_language_profile(nr_meanings, nr_forms, rng) for _ in range(nr_languages)]

    # Assign language profiles to agents
    # If there are multiple languages initialized, assign the language profiles spatially grouped.
    if nr_languages > 1:
        profile_assignments = initialize_spatial_profile_assignments(x, y, nr_languages, rng)
    else:
        profile_assignments = np.zeros(nr_agents, dtype=int)

    # Assign the start profiles to the agents
    language_profile = [start_profiles[assignment].copy() for assignment in profile_assignments]

    # Keep track of the maximum ID for agent births
    max_id = max(ids)

    # Create a geopandas dataframe with agent id, positions and language profile
    return (
        gpd.GeoDataFrame(
            {
                "id": ids,
                "language_profile": language_profile,
                "parent_id": [(-1) for _ in range(nr_agents)],
                "time_step": [(0) for _ in range(nr_agents)],
            },
            geometry=gpd.points_from_xy(x, y),
            crs="+proj=cart +units=km +type=crs",
        ),
        max_id,
    )
