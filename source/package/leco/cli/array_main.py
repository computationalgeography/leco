import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import KDTree
import time
import os


def population_dynamics(
    population: gpd.GeoDataFrame,
    rng: np.random.default_rng,
    death_rate: float,
    birth_rate: float,
    max_id: int,
) -> gpd.GeoDataFrame:
    """Update population based on death and birth rates"""

    # Determine for every agent whether it will die based on the deathrate
    death_masks = rng.choice(
        [False, True], size=len(population), p=[1 - death_rate, death_rate]
    )

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


def move(
    position: gpd.GeoSeries.geometry,
    rng: np.random.default_rng,
    speed: float,
    x_max: int,
    y_max: int,
) -> gpd.GeoSeries.geometry:
    """Move agents across a continuous space"""
    angle = movement_direction(rng, len(position))  # Get random angle for all agents
    new_x = move_x(position.x.values, speed, angle, x_max)  # Move in x direction
    new_y = move_y(position.y.values, speed, angle, y_max)  # Move in y direction

    new_positions = gpd.points_from_xy(new_x, new_y)

    return new_positions


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
    rng: np.random.default_rng, max_value: float, nr_agents: int
) -> np.ndarray[float]:
    """Randomly initialize coordinates for all agents within the specified range"""
    return rng.uniform(low=0.0, high=max_value, size=nr_agents)


def initialize_language_profile(
    rng: np.random.default_rng, nr_meanings: int, nr_forms: int
) -> np.ndarray[int]:
    """Randomly initialize a language profile with length = nr_meaning for an agent"""
    return rng.integers(0, nr_forms, nr_meanings)


def initialize_board(
    nr_agents: int,
    x_max: int,
    y_max: int,
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
    x = initialize_coordinates(rng, x_max, nr_agents)
    y = initialize_coordinates(rng, y_max, nr_agents)

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

    # Initialize population of agents
    population = initialize_board(
        p["agents"],
        p["x_max"],
        p["y_max"],
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

        # Apply birth and death rates to the population
        population, max_id = population_dynamics(
            population, rng, p["death_rate"], p["birth_rate"], max_id
        )

        # Move the agents within space
        population.geometry = move(
            population.geometry, rng, p["speed"], p["x_max"], p["y_max"]
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
