"""Functions for initialization of the agent population and space."""

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import numpy.typing as npt
from shapely import (
    box,
)
from shapely.geometry.base import BaseGeometry

logger = logging.getLogger(__name__)


def initialize_barrier(
    space: list[float],
    bar_x: list[float],
    bar_y: list[float],
) -> BaseGeometry:
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
    rng: np.random.Generator,
) -> npt.NDArray[np.float64]:
    """Randomly initialize coordinates along one axis within the specified range."""
    return rng.uniform(low=min_coordinates, high=max_coordinates, size=nr_agents)


def initialize_positions(
    space: list[float],
    subset_area_present: bool,
    subset_x_extent: list[float],
    subset_y_extent: list[float],
    nr_agents: int,
    rng: np.random.Generator,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Initialize coordinates for the number of start agents within the specified initialization area."""
    if subset_area_present is True:
        # If a initialization area is specified, use those coordinates as the range
        x = initialize_coordinates(subset_x_extent[0], subset_x_extent[1], nr_agents, rng)
        y = initialize_coordinates(subset_y_extent[0], subset_y_extent[1], nr_agents, rng)
        return x, y
    # Agents can be initialized across the entire space
    x = initialize_coordinates(0.0, space[0], nr_agents, rng)
    y = initialize_coordinates(0.0, space[1], nr_agents, rng)
    return x, y


def initialize_language_profile(
    nr_meanings: int,
    nr_forms: int,
    rng: np.random.Generator,
) -> npt.NDArray[np.int64]:
    """Randomly initialize a language profile with size 'nr_meanings'."""
    # Each meaning is randomly assigned a form
    return rng.integers(0, nr_forms, nr_meanings)


def initialize_spatial_profile_assignments(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    nr_languages: int,
    rng: np.random.Generator,
) -> npt.NDArray[np.int64]:
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


def initialize_spatial_cross_distribution(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    x_size: float,
    y_size: float,
) -> npt.NDArray[np.int64]:
    """Assign language-profile indices to agents so that profiles are spatially grouped.

    In cross manner: four even-sized squares.
    """
    coordinates = np.column_stack([x, y])

    # Assign quadrant index (0-3) based on position relative to midpoint
    # Quadrant layout:
    #   0 | 1
    #   -----
    #   2 | 3
    return np.where(
        coordinates[:, 1] < (y_size / 2),  # bottom half
        np.where(coordinates[:, 0] < (x_size / 2), 2, 3),  # bottom-left=2, bottom-right=3
        np.where(coordinates[:, 0] < (x_size / 2), 0, 1),  # top-left=0,    top-right=1
    )


def initialize_population(
    nr_agents: int,
    space: list[float],
    subset_area_present: bool,
    subset_x_extent: list[float],
    subset_y_extent: list[float],
    nr_languages: int,
    nr_forms: int,
    nr_meanings: int,
    rng: np.random.Generator,
) -> tuple[gpd.GeoDataFrame, int]:
    """Return a data frame containing for each agent the following properties."""
    """
    - id
    - language_profile
    - point position
    - parent id (set to None at initialization)
    """
    ids = list(range(1, nr_agents + 1))  # ids from 1 to number of agents

    # Assign positions
    x, y = initialize_positions(space, subset_area_present, subset_x_extent, subset_y_extent, nr_agents, rng)

    # Assign language profiles, each profile represented by a string of integers
    # For each agent:
    #     For each meaning:
    #         A random int [0, forms]

    # Create the start language profiles, number is equal to nr_languages
    start_profiles = [initialize_language_profile(nr_meanings, nr_forms, rng) for _ in range(nr_languages)]

    # Assign language profiles to agents
    # If there are multiple languages initialized, assign the language profiles spatially grouped.
    if nr_languages == 1:
        profile_assignments = np.zeros(nr_agents, dtype=int)
    elif nr_languages == 4:
        # Use the spatial cross distribution
        profile_assignments = initialize_spatial_cross_distribution(x, y, space[0], space[1])
    else:
        profile_assignments = initialize_spatial_profile_assignments(x, y, nr_languages, rng)

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


def initialize_intermediate_start(file_path: Path, time_step: int) -> tuple[gpd.GeoDataFrame, int]:
    """Initialize population from an intermediate point."""
    populations = gpd.read_parquet(file_path)

    population = populations.loc[populations["time_step"] == time_step, :]
    if population.empty:
        raise ValueError(f"No population found for time_step {time_step}")
    max_id = max(population["id"])

    return population, max_id
