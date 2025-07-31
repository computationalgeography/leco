import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import Polygon, LineString, box  # I could also just import shapely
from scipy.spatial import KDTree
import time
import os


def set_birth_rate(
    multiplier: float,
    growth_rate: float,
    birth_rate: float,
    death_rate: float,
    init_population: int,
    N: int,
) -> float:
    """Calculate the birth rate based on the multiplier and growth rate"""
    if multiplier > 1:
        # If the multiplier is not set to 1, apply logistic growth
        K = init_population * multiplier
        effective_growth_rate = growth_rate * (1 - N / K)
        birth_rate = death_rate + effective_growth_rate
        return birth_rate
    else:
        # Use the constant initalized birth rates
        return birth_rate


def population_dynamics(
    population: gpd.GeoDataFrame,
    death_rate: float,
    birth_rate: float,
    max_id: int,
    rng: np.random.default_rng,
) -> gpd.GeoDataFrame:
    """Update population based on death and birth rates"""

    N = len(population)

    # Determine for every agent whether it will die based on the deathrate
    death_masks = rng.choice([False, True], size=N, p=[1 - death_rate, death_rate])

    # Remove the agents that die while remaining consecutive row count number
    survivors = population[~death_masks].copy().reset_index(drop=True)

    # Determine for every agent whether it will reproduce based on the birthrate
    birth_masks = rng.choice(
        [False, True], size=len(survivors), p=[1 - birth_rate, birth_rate]
    )

    # Handle births: duplicate the agents that give birth
    if birth_masks.any():
        # Generate new unique IDs for offspring
        num_births = birth_masks.sum()
        new_ids = np.arange(max_id + 1, max_id + 1 + num_births)
        max_id = max(new_ids)

        # Get parent indices for births
        birth_indices = np.where(birth_masks)[0]

        # Create offspring by taking parent data and updating IDs
        offspring = population.iloc[birth_indices].copy()
        offspring["id"] = new_ids

        # Combine survivors with offspring
        new_population = pd.concat([survivors, offspring], ignore_index=True)
    else:
        new_population = survivors

    return new_population, max_id


def movement_direction(rng: np.random.default_rng, nr_agents: int) -> np.ndarray[float]:
    """Get a random movement direction"""
    angle = rng.uniform(
        0, 2 * np.pi, size=nr_agents
    )  # Get a random angle in radians between 0 and 2π for each agent

    return angle


def move_x(
    current_x: np.ndarray[float], speed: float, angle: float, x_max: int
) -> np.ndarray[float]:
    """Move in x direction"""
    dx = speed * np.cos(
        angle
    )  # Calculate the change in x position based on speed and angle
    new_pos = np.clip(
        current_x + dx, a_min=0.0, a_max=x_max
    )  # Values outside the interval are clipped to the interval edges

    return new_pos


def move_y(
    current_y: np.ndarray[float], speed: float, angle: float, y_max: int
) -> np.ndarray[float]:
    """Move in y direction"""
    dy = speed * np.sin(
        angle
    )  # Calculate the change in y position based on speed and angle
    new_pos = np.clip(
        current_y + dy, a_min=0.0, a_max=y_max
    )  # Values outside the interval are clipped to the interval edges

    return new_pos


def find_intersecting_movements(
    old_coor: gpd.GeoSeries.geometry,
    new_coor: gpd.GeoSeries.geometry,
    barrier: Polygon,
) -> np.ndarray[bool]:
    """Find which agent movements intersect with the barrier"""

    # Create movement lines from old to new positions
    movement_lines = gpd.GeoSeries(
        [LineString([p1, p2]) for p1, p2 in zip(old_coor, new_coor)],
        index=old_coor.index,
    )

    # Find agents already intersecting the barrier at their old position
    already_at_barrier = old_coor.intersects(barrier)

    # Find which movement lines intersect the barrier
    crosses_barrier = movement_lines.intersects(barrier)

    # Only keep those who weren't already at the barrier
    result = crosses_barrier & ~already_at_barrier

    return result.to_numpy()


def calculate_away_angle(
    x_coor: float,
    y_coor: float,
    barrier: Polygon,
    rng: np.random.default_rng,
) -> float:
    """Calculate movement direction that points away from rectangular barrier"""

    # Get barrier bounds for directional movement
    minx, miny, maxx, maxy = barrier.bounds

    # Determine which side of the rectangle the agent is closest to and point away from that side
    # Calculate distances to each side of the barrier
    dist_to_left = abs(x_coor - minx)
    dist_to_right = abs(x_coor - maxx)
    dist_to_bottom = abs(y_coor - miny)
    dist_to_top = abs(y_coor - maxy)

    # Stack distances to find the closest side
    dists = np.stack([dist_to_left, dist_to_right, dist_to_bottom, dist_to_top], axis=0)
    closest_side = np.argmin(dists, axis=0)  # 0 = left, 1 = right, 2 = bottom, 3 = top

    # Random variance for each agent
    angle_variance = np.pi / 3  # ±60 degrees
    variance = rng.uniform(-angle_variance, angle_variance, size=x_coor.shape)

    # Initialize angles
    angles = np.zeros_like(x_coor)

    # Set angles away from the closest side of the rectangle
    angles[closest_side == 0] = np.pi  # left → move left
    angles[closest_side == 1] = 0  # right → move right
    angles[closest_side == 2] = -np.pi / 2  # bottom → move down
    angles[closest_side == 3] = np.pi / 2  # top → move up

    # Add random variance
    angles += variance

    return angles


def move(
    position: gpd.GeoSeries.geometry,
    barrier: Polygon | None,
    bar_impediment: float,
    rng: np.random.default_rng,
    speed: float,
    x_max: int,
    y_max: int,
) -> gpd.GeoSeries.geometry:
    """Move agents across a continuous space"""

    angle = movement_direction(rng, len(position))  # Get random angle for all agents
    new_x = move_x(position.x.values, speed, angle, x_max)  # Move in x direction
    new_y = move_y(position.y.values, speed, angle, y_max)  # Move in y direction
    new_position = gpd.points_from_xy(new_x, new_y)

    # If there is no barrier or no impediment from the barrier, return the new positions
    if barrier is None or bar_impediment == 0:
        return new_position

    # Check if the barrier impediment falls within the range of [0,1]
    if bar_impediment < 0 or bar_impediment > 1:
        raise ValueError(
            f"The barrier impediment value {bar_impediment} must be between 0 and 1."
        )

    # Find agents which migration routes intersect with the barrier
    intersecting = find_intersecting_movements(position, new_position, barrier)

    # If none of the routes are intersecting with the barrier, return new positions
    if not np.any(intersecting):
        return new_position

    # Get the agent ids that intersect with the barrier
    intersecting_indices = np.where(intersecting)[0]

    # Generate impediment masks for the intersecting agents based on the barrier impediment
    impeded_prob = rng.random(len(intersecting_indices))
    impeded_agents = intersecting_indices[impeded_prob < bar_impediment]

    # If none of the agents is impeded by the barrier, return new positions
    if not np.any(impeded_agents):
        return new_position

    # Get the x and y coordinates from the current positions of the impeded agents
    impeded_x = position.x.iloc[impeded_agents].values
    impeded_y = position.y.iloc[impeded_agents].values

    # Calculate a new movement direction for each impeded agent away from the barrier
    away_angles = calculate_away_angle(impeded_x, impeded_y, barrier, rng)

    # Update the new positions of the impeded agents
    new_x[impeded_agents] = move_x(impeded_x, speed, away_angles, x_max)
    new_y[impeded_agents] = move_y(impeded_y, speed, away_angles, y_max)
    new_position = gpd.points_from_xy(new_x, new_y)

    return new_position


def mutate_profile(
    language_profiles: np.ndarray[int],
    rng: np.random.default_rng,
    nr_forms: int,
    mutation_rate: float,
) -> np.ndarray[int]:
    """Mutate language profile of agents"""

    # Get the current number of agents and the number of meanings
    nr_agents, nr_meanings = language_profiles.shape

    # Generate mutation masks for all agents at once
    mutation_prob = rng.random((nr_agents, nr_meanings))
    mutation_mask = mutation_prob < mutation_rate

    # Generate the new forms
    mutated_forms = rng.integers(0, nr_forms, size=(nr_agents, nr_meanings))

    # Mutate the forms if mask is true
    mutated_profiles = np.where(mutation_mask, mutated_forms, language_profiles)

    return list(mutated_profiles)


def nearest_neighbors(
    positions: np.ndarray[float], radius: float
) -> list[np.ndarray[int]]:
    """Find all neighbors within a radius"""

    # Build a K-Dimensional tree (spatial index)
    pos_kdt = KDTree(positions)

    # Find all neighbors within a radius around an agent
    neighbors = pos_kdt.query_ball_point(positions, r=radius)
    # Remove self and store in list format
    neighbors = [
        [n for n in neighbor_list if n != i]
        for i, neighbor_list in enumerate(neighbors)
    ]

    return neighbors


def interact(
    language_profiles: np.ndarray[int],
    neighbors_list: list[np.ndarray[int]],
    rng: np.random.default_rng,
    int_partner_prob: float,
    diffusion_rate: float,
) -> np.ndarray[int]:
    """Interaction between agents whereby linguistic diffusion occurs"""

    # Get the current number of agents and the number of meanings
    nr_agents, nr_meanings = language_profiles.shape
    # Create a copy of the language profiles
    new_profiles = language_profiles.copy()

    ## Pre-generate all random numbers at once
    # Calculate the maximum number of neighbors an agent has
    max_neighbors = max(len(nbs) for nbs in neighbors_list) if neighbors_list else 0
    if max_neighbors == 0:
        return list(new_profiles)

    # Generate interaction masks for all agents at once
    interaction_probs = rng.random(
        (nr_agents, max_neighbors)
    )  # The probability that an agent interact with each of its neighbors
    diffusion_probs = rng.random(
        (nr_agents, max_neighbors, nr_meanings)
    )  # The probability that a form diffuses from a neighbor to an agent, probabilities are taken for every meaning in the language profile

    # Loop through the agents
    for agent_idx, neighbors in enumerate(neighbors_list):
        if len(neighbors) == 0:
            continue

        # Convert neighbor list to an array to make use of the masks
        neighbors = np.array(neighbors)

        # Based on a probability, the agent interacts with 'int_partner_prob' proportion of their neighbors
        interaction_mask = (
            interaction_probs[agent_idx, : len(neighbors)] < int_partner_prob
        )
        # Select the interaction partners
        interacting_neighbors = neighbors[interaction_mask]

        if len(interacting_neighbors) == 0:
            continue

        # Shuffle the order of the interaction partners
        rng.shuffle(interacting_neighbors)

        # Interact with the interaction partners
        for i, nb_idx in enumerate(interacting_neighbors):
            # Select the forms from the language profile that will be diffused during the interaction with neighbor i based on the diffusion rate
            diffusion_mask = diffusion_probs[agent_idx, i, :] < diffusion_rate
            # Diffuse these forms from the neighbor's language profile to the active agent's profile
            new_profiles[agent_idx] = np.where(
                diffusion_mask, language_profiles[nb_idx], new_profiles[agent_idx]
            )

    # Return list format to add to geopanda's dataframe
    return list(new_profiles)


def initialize_coordinates(
    rng: np.random.default_rng,
    max_value: float,
    init_radius: float | bool,
    init_coord: float | bool,
    nr_agents: int,
) -> np.ndarray[float]:
    """Randomly initialize coordinates for all agents within the specified range"""

    if init_radius is not False:
        if init_coord is not False:
            # If a specific coordinate is given, use it for all agents
            if init_coord < 0 or init_coord > max_value:
                raise ValueError(
                    f"Initial coordinate {init_coord} must be between 0 and {max_value}."
                )
            mid_point = init_coord
        else:
            mid_point = rng.uniform(0.0, max_value, size=1)

        lowest = mid_point - init_radius
        if lowest < 0.0:
            lowest = 0.0
        highest = mid_point + init_radius
        if highest > max_value:
            highest = max_value
        return rng.uniform(low=lowest, high=highest, size=nr_agents)

    else:
        return rng.uniform(low=0.0, high=max_value, size=nr_agents)


def initialize_barrier(
    max_coor: float,
    bar_x: float,
    bar_y: float,
    bar_x_radius: float,
    bar_y_radius: float,
) -> Polygon:
    """Initialize a barrier in the continuous space"""

    barrier = Polygon(
        [
            (bar_x - bar_x_radius, bar_y - bar_y_radius),
            (bar_x + bar_x_radius, bar_y - bar_y_radius),
            (bar_x + bar_x_radius, bar_y + bar_y_radius),
            (bar_x - bar_x_radius, bar_y + bar_y_radius),
        ]
    )

    # Bounding box for valid space
    bounds = box(0, 0, max_coor, max_coor)

    # Clip the barrier to fit inside the bounds
    barrier_clipped = barrier.intersection(bounds)

    # Check if clipping occurred
    if not barrier_clipped.equals(barrier):
        print(
            f"Warning: Barrier at ({bar_x:.2f}, {bar_y:.2f}) was clipped to fit within "
            f"bounds [0, {max_coor}]"
        )

    return barrier_clipped


def initialize_language_profile(
    rng: np.random.default_rng, nr_meanings: int, nr_forms: int
) -> np.ndarray[int]:
    """Randomly initialize a language profile with length = nr_meaning for an agent"""
    return rng.integers(0, nr_forms, nr_meanings)


def initialize_population(
    nr_agents: int,
    x_max: int,
    y_max: int,
    init_radius: float | bool,
    init_x: float | bool,
    init_y: float | bool,
    nr_languages: int,
    nr_forms: int,
    nr_meanings: int,
    rng: np.random.default_rng,
) -> gpd.GeoDataFrame:
    """
    Returns a data frame containing for each agent these properties:
    - id
    - language_profile
    - language
    - point position
    """

    ids = list(range(1, nr_agents + 1))  # ids from 1 to number of agents

    # Initial spatial distribution of the agents
    x = initialize_coordinates(rng, x_max, init_radius, init_x, nr_agents)
    y = initialize_coordinates(rng, y_max, init_radius, init_y, nr_agents)

    # Initial language profile, represented by a string of integers
    # For each agent:
    #     For each meaning:
    #         A random int [0, forms)

    # Create the original language profiles, number is equal to nr_languages
    start_profiles = [
        initialize_language_profile(rng, nr_meanings, nr_forms)
        for _ in range(nr_languages)
    ]

    # Create evenly distributed assignments
    profile_assignments = np.array([i % nr_languages for i in range(nr_agents)])
    rng.shuffle(profile_assignments)  # Randomize the order

    # Assign the start profiles following the profile_assignments
    language_profile = [
        start_profiles[assignment].copy() for assignment in profile_assignments
    ]

    # Create a geopandas dataframe with agent id, point positions, language profile and language
    population = gpd.GeoDataFrame(
        {
            "id": ids,
            "language_profile": language_profile,
        },
        geometry=gpd.points_from_xy(x, y),
    )

    return population


def run_model(p: dict, output_run: str):
    """Run the LECo model of language evolution"""

    # Initialize seed
    rng = np.random.default_rng(p["seed"])

    # Start to track model run time
    start_time = time.time()

    # Initialize a spatial barrier if specified
    barrier = None
    if p["barrier"]:
        barrier = initialize_barrier(
            p["x_max"],
            p["bar_x"],
            p["bar_y"],
            p["bar_x_radius"],
            p["bar_y_radius"],
        )

    # Initialize population of agents
    population = initialize_population(
        p["agents"],
        p["x_max"],
        p["y_max"],
        p["init_area_radius"],
        p["init_x"],
        p["init_y"],
        p["nr_start_languages"],
        p["forms"],
        p["meanings"],
        rng,
    )

    if output_run:
        # Write initialization dataframe to a .geoparquet file
        population.to_parquet(os.path.join(output_run, "output000.geoparquet"))

    # Keep track of the maximum ID for births
    max_id = population["id"].max()

    for step in range(1, p["steps"] + 1):
        print(step)
        population["timestep"] = step
        current_pop_size = len(population)

        # Determine the current birh rate based on constant rates (multiplier == 1) or logistic growth (multiplier != 1)
        current_birth_rate = set_birth_rate(
            p["multiplier"],
            p["growth_rate"],
            p["birth_rate"],
            p["death_rate"],
            p["agents"],
            current_pop_size,
        )
        # Apply birth and death rates to the population
        population, max_id = population_dynamics(
            population,
            p["death_rate"],
            current_birth_rate,
            max_id,
            rng,
        )

        # Move the agents within space
        population.geometry = move(
            population.geometry,
            barrier,
            p["bar_impediment"],
            rng,
            p["speed"],
            p["x_max"],
            p["y_max"],
        )

        # Mutate language profiles
        population["language_profile"] = mutate_profile(
            np.stack(population["language_profile"]),
            rng,
            p["forms"],
            p["mutation_rate"],
        )

        # Interact with nearby neighbors
        nbs = nearest_neighbors(
            population.get_coordinates().to_numpy(), p["int_radius"]
        )

        population["language_profile"] = interact(
            np.stack(population["language_profile"]),
            nbs,
            rng,
            p["int_partner_prob"],
            p["diffusion_rate"],
        )

        # Save output to geoparquet file
        if output_run:
            population.to_parquet(
                os.path.join(output_run, f"output{step:03d}.geoparquet")
            )

    print("--- %s seconds ---" % (time.time() - start_time))
    print("--- %s minutes ---" % ((time.time() - start_time) / 60))


# %%
