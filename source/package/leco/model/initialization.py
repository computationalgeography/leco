import numpy as np
import geopandas as gpd
from shapely import (
    Polygon,
    box,
)


def initialize_barrier(
    space: list[float, float],
    bar_x: list[float, float],
    bar_y: list[float, float],
) -> Polygon:
    """Initialize a barrier in continuous space"""

    # Create the barrier as a shapely Polygon
    barrier = box(bar_x[0], bar_y[0], bar_x[1], bar_y[1])

    # Bounding box of the entire space
    bounds = box(0, 0, space[0], space[1])

    # Clip the barrier to fit inside the bounds
    barrier_clipped = barrier.intersection(bounds)

    # Check if clipping occurred
    if not barrier_clipped.equals(barrier):
        print("Warning: Spatial barrier was clipped to fit within the entire space.")

    return barrier_clipped


def initialize_coordinates(
    min_coor: float,
    max_coor: float,
    nr_agents: int,
    rng: np.random.default_rng,
) -> np.ndarray[float]:
    """Randomly initialize coordinates along one axis within the specified range"""

    return rng.uniform(low=min_coor, high=max_coor, size=nr_agents)


def initialize_positions(
    space: list[float],
    subset_area: dict[bool, list[float], list[float]],
    nr_agents: int,
    rng: np.random.default_rng,
) -> tuple[np.ndarray[float], np.ndarray[float]]:
    """Initialize x and y coordinates for the number of start agents within the specified initialization area"""

    if subset_area["present"] is True:
        # If a initialization area is specified, use those coordinates as the range
        x = initialize_coordinates(
            subset_area["x_extent"][0], subset_area["x_extent"][1], nr_agents, rng
        )
        y = initialize_coordinates(
            subset_area["y_extent"][0], subset_area["y_extent"][1], nr_agents, rng
        )
        return x, y
    else:
        # Agents can be initialized across the entire space
        x = initialize_coordinates(0.0, space[0], nr_agents, rng)
        y = initialize_coordinates(0.0, space[1], nr_agents, rng)
        return x, y


def initialize_language_profile(
    nr_meanings: int,
    nr_forms: int,
    rng: np.random.default_rng,
) -> np.ndarray[int]:
    """Randomly initialize a language profile with size 'nr_meanings', where each meaning is randomly assigned a form"""
    return rng.integers(0, nr_forms, nr_meanings)


def initialize_population(
    nr_agents: int,
    space: list[float],
    subset_area: dict[bool, list[float], list[float]],
    nr_languages: int,
    nr_forms: int,
    nr_meanings: int,
    rng: np.random.default_rng,
) -> gpd.GeoDataFrame:
    """
    Returns a data frame containing for each agent the following properties:
    - id
    - language_profile
    - point position
    """

    ids = list(range(1, nr_agents + 1))  # ids from 1 to number of agents

    # Assign positions
    x, y = initialize_positions(space, subset_area, nr_agents, rng)

    # Assign language profiles, each profile represented by a string of integers
    # For each agent:
    #     For each meaning:
    #         A random int [0, forms]

    # Create the start language profiles, number is equal to nr_languages
    start_profiles = [
        initialize_language_profile(nr_meanings, nr_forms, rng)
        for _ in range(nr_languages)
    ]

    # Evenly distribute the start language profiles across the agents
    profile_assignments = np.array([i % nr_languages for i in range(nr_agents)])
    rng.shuffle(profile_assignments)  # Randomize the order

    # Assign the start profiles to the agents
    language_profile = [
        start_profiles[assignment].copy() for assignment in profile_assignments
    ]

    # Create a geopandas dataframe with agent id, positions and language profile
    population = gpd.GeoDataFrame(
        {
            "id": ids,
            "language_profile": language_profile,
        },
        geometry=gpd.points_from_xy(x, y),
    )

    return population
