import json
import jsonschema
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from sklearn.cluster import AgglomerativeClustering
import os
from datetime import datetime
import time
from .create_plots import point_plotje, language_number_plotje


def load_parameters_from_json(json_file: str) -> dict:
    """Load parameters from a JSON file and validate them against a schema"""

    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, json_file)
    with open(json_path, "r") as f:
        parameters = json.load(f)

    # Define the schema for validation
    schema = {
        "type": "object",
        "properties": {
            "agents": {"type": "integer", "minimum": 1},
            "steps": {"type": "integer", "minimum": 1},
            "seed": {"type": "integer"},
            "x_max": {"type": "number", "minimum": 0},
            "y_max": {"type": "number", "minimum": 0},
            "speed": {"type": "number", "minimum": 0},
            "meanings": {"type": "integer", "minimum": 1},
            "forms": {"type": "integer", "minimum": 1},
            "birth_rate": {"type": "number", "minimum": 0, "maximum": 1},
            "death_rate": {"type": "number", "minimum": 0, "maximum": 1},
            "mutation_rate": {"type": "number", "minimum": 0, "maximum": 1},
            "int_radius": {"type": "number", "minimum": 0},
            "int_partner_prob": {"type": "number", "minimum": 0, "maximum": 1},
            "diffusion_rate": {"type": "number", "minimum": 0, "maximum": 1},
            "similarity": {"type": "number", "minimum": 0, "maximum": 1},
            "output": {"type": "boolean"},
            "plot_step": {"type": "integer", "minimum": 1},
            "random_init": {"type": "boolean"},
        },
        "required": [
            "agents",
            "steps",
            "seed",
            "x_max",
            "y_max",
            "speed",
            "meanings",
            "forms",
            "birth_rate",
            "death_rate",
            "mutation_rate",
            "int_radius",
            "int_partner_prob",
            "diffusion_rate",
            "similarity",
            "output",
            "plot_step",
            "random_init",
        ],
    }

    # Validate the parameters against the schema
    try:
        jsonschema.validate(instance=parameters, schema=schema)
    except jsonschema.ValidationError as e:
        raise ValueError(f"Invalid parameter configuration: {e.message}")

    return parameters


def output_handling(dir_plots: str, p: dict) -> str:
    """Create output directory and store parameter values in a text file"""
    os.makedirs(
        dir_plots, exist_ok=True
    )  # Create the directory if it does not exist yet

    # Create a subdirectory for each run named after date and time
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir_run = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), dir_plots, f"results_{timestamp}"
    )
    os.makedirs(output_dir_run, exist_ok=True)

    # Write parameter values to text file and store in the ouput directory of the specific run
    with open(os.path.join(output_dir_run, "parameters.txt"), "w") as f:
        for key, value in p.items():
            f.write(f"{key} = {value}\n")

    return output_dir_run


def population_dynamics(
    population: pd.DataFrame,
    rng: np.random.default_rng,
    death_rate: float,
    birth_rate: float,
) -> pd.DataFrame:
    """Update population based on birth and death rates"""

    nothing_prob = 1 - death_rate - birth_rate
    probabilities = [nothing_prob, death_rate, birth_rate]

    # Compute the population dynamic action for every agent: nothing, die or reproduce
    outcomes = rng.choice(
        ["nothing", "death", "birth"],
        size=len(population),
        p=probabilities,
    )

    # Create masks for the death and birth events
    death_mask = outcomes == "death"
    birth_mask = outcomes == "birth"

    # Remove the agents that die while remaining consecutive row count number
    survivors = population[~death_mask].copy().reset_index(drop=True)

    # Handle births: duplicate the agents that give birth
    if birth_mask.any():
        # Generate new unique IDs for offspring
        max_id = population["id"].max()
        num_births = birth_mask.sum()
        new_ids = np.arange(max_id + 1, max_id + 1 + num_births)

        # Get parent indices for births
        birth_indices = np.where(birth_mask)[0]

        # Create offspring by taking parent data and updating IDs
        offspring = population.iloc[birth_indices].copy()
        offspring["id"] = new_ids

        # Combine survivors with offspring
        new_population = pd.concat([survivors, offspring], ignore_index=True)
    else:
        new_population = survivors

    return new_population


def movement_direction(rng: np.random.default_rng, nr_agents: int) -> np.ndarray:
    """Get a random movement direction"""
    angle = rng.uniform(
        0, 2 * np.pi, size=nr_agents
    )  # Get a random angle in radians between 0 and 2π for each agent

    return angle


def move_x(current_x: pd.Series, speed: float, angle: float, x_max: int) -> pd.Series:
    """Move in x direction"""
    dx = speed * np.cos(
        angle
    )  # Calculate the change in x position based on speed and angle
    new_pos = np.clip(
        current_x + dx, a_min=0.0, a_max=x_max
    )  # Values outside the interval are clipped to the interval edges

    return new_pos


def move_y(current_y: pd.Series, speed: float, angle: float, y_max: int) -> pd.Series:
    """Move in y direction"""
    dy = speed * np.sin(
        angle
    )  # Calculate the change in y position based on speed and angle
    new_pos = np.clip(
        current_y + dy, a_min=0.0, a_max=y_max
    )  # Values outside the interval are clipped to the interval edges

    return new_pos


def move(
    x_position: pd.Series,
    y_position: pd.Series,
    rng: np.random.default_rng,
    speed: float,
    x_max: int,
    y_max: int,
) -> tuple[pd.Series, pd.Series]:
    """Move agents across a continuous space"""
    angle = movement_direction(rng, len(x_position))  # Get random angles for each agent
    new_x = move_x(x_position, speed, angle, x_max)  # Move in x direction
    new_y = move_y(y_position, speed, angle, y_max)  # Move in y direction

    return new_x, new_y


def mutate_profile(
    profile: np.ndarray,
    rng: np.random.default_rng,
    nr_forms: int,
    mutation_rate: float,
) -> np.ndarray:
    """Mutate language profile of agents"""

    mutation_mask = rng.random(len(profile)) < mutation_rate

    mutated_forms = rng.integers(0, nr_forms, size=len(profile))

    return np.where(mutation_mask, mutated_forms, profile)


def precise_intersection_neighbors(
    row: gpd.GeoSeries,
    geometry_agents: gpd.GeoSeries,
    potential_matches: pd.Series,
    radius: float,
) -> np.ndarray:
    """Find the actual neighbors within the interaction radius using precise intersection for one agent at a time"""
    i = row.name  # DataFrame index
    point = row.geometry
    candidates = potential_matches[
        i
    ]  # Obtain potential neighbors for the current agent

    if not candidates:
        return np.empty(0, dtype=int)  # Return empty array if no candidates

    # Convert to numppy array and remove self
    candidates_array = np.array(candidates)
    valid_candidates = candidates_array[candidates_array != i]

    # Return empty array if no valid candidates after removing self
    if len(valid_candidates) == 0:
        return np.empty(0, dtype=int)

    # Precise distance calculation for all valid candidates
    candidate_geometries = geometry_agents.iloc[valid_candidates]
    distances = candidate_geometries.distance(point)

    # Filter by radius
    within_radius_mask = distances <= radius
    neighbors = valid_candidates[within_radius_mask]

    return neighbors


def get_neighbors(positions: np.ndarray, radius: float) -> pd.Series:
    """Get neighbors within interaction radius using GeoPandas spatial indexing"""

    # Create GeoDataFrame from point positions
    gdf = gpd.GeoDataFrame(geometry=[Point(x, y) for x, y in positions])

    # Create spatial index (R-tree based)
    spatial_index = gdf.sindex

    # Create the interaction radius polygons for all agents
    all_buffers = gdf.geometry.buffer(radius)

    # Use spatial index to find potential neighbors (bounding box intersection) from which the buffers intersect
    potential_matches = all_buffers.apply(
        lambda buffer: list(spatial_index.intersection(buffer.bounds))
    )

    # Filter to actual neighbors within radius (precise intersection)
    neighbor_results = gdf.apply(
        precise_intersection_neighbors,
        args=(gdf.geometry, potential_matches, radius),
        axis=1,
    )

    return neighbor_results


def conversating(
    agent: pd.Series,
    language_profiles: pd.Series,
    rng: np.random.default_rng,
    int_partner_prob: float,
    diffusion_rate: float,
) -> np.ndarray:
    """Interactions between the active agent and its neighbors whereby linguistic diffusion occurs"""

    # Check if the agent has any neighbors to interact with
    if len(agent["nbs"]) == 0:
        return agent["language_profile"]

    # Based on a probability, the agent interacts with its neighbors
    interaction_mask = rng.random(len(agent["nbs"])) < int_partner_prob
    # interacting_neighbors = shuffled_nbs[interaction_mask]
    interacting_neighbors = agent["nbs"][interaction_mask]

    # Check if the agent has any neighbors to interact with after probability selection
    if len(interacting_neighbors) == 0:
        return agent["language_profile"]

    # Shuffle the order of interaction partners
    shuffled_nbs = rng.permutation(interacting_neighbors)

    new_profile = agent["language_profile"]

    # Loop through the neighbors
    for nb in shuffled_nbs:
        # Select the features from the language profile that will be diffused based on the diffusion rate
        diffusion_mask = rng.random(len(new_profile)) < diffusion_rate
        # Diffuse these features from the neighbor's language profile to the active agent's profile
        new_profile = np.where(diffusion_mask, language_profiles[nb], new_profile)

    return new_profile


def interact(
    language_profiles: pd.Series,
    nbs: pd.Series,
    rng: np.random.default_rng,
    int_partner_prob: float,
    diffusion_rate: float,
) -> pd.Series:
    """Interaction between agents whereby linguistic diffusion occurs"""

    nbs.name = "nbs"
    df_interaction = pd.concat([language_profiles, nbs], axis=1)

    # Interactions between the agents and their neighbors
    new_profiles = df_interaction.apply(
        conversating,
        args=(language_profiles, rng, int_partner_prob, diffusion_rate),
        axis=1,
    )

    return new_profiles


def distance_matrix(language_profiles: pd.Series) -> np.ndarray:
    """Computes a distance matrix based on the Hamming distance between all agents"""

    # Stack the language profiles into a 2D numpy array
    profiles = np.stack(language_profiles.values)

    # Create 3D arrays for pairwise comparison
    # profiles_i: (n, 1, features) - adds a new dimension in the middle
    # profiles_j: (1, n, features) - adds a new dimension at the beginning
    profiles_i = profiles[:, np.newaxis, :]
    profiles_j = profiles[np.newaxis, :, :]

    # Comparison and computation of the mean differences
    dist_matrix = np.mean(profiles_i != profiles_j, axis=2)

    return dist_matrix


def language_classification(
    distance_matrix: np.ndarray, sim_threshold: float
) -> np.ndarray:
    """Group language profiles into languages based on similarity threshold following hierarchical clustering"""

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=1 - sim_threshold,  # Threshold for clustering
        metric="precomputed",
        linkage="average",
    )  # Average linkage calculates over the mean of the distances between all points in the clusters

    categories = clustering.fit_predict(distance_matrix)

    return categories


def initialize_coordinates(
    rng: np.random.default_rng, max_value: float, nr_agents: int
) -> np.ndarray:
    """Randomly initialize coordinates for all agents within the specified range"""
    return rng.uniform(low=0.0, high=max_value, size=nr_agents)


def initialize_language_profile(
    rng: np.random.default_rng, nr_meanings: int, nr_forms: int
) -> np.ndarray:
    """Randomly initialize a language profile with length = nr_meaning for an agent"""
    return rng.integers(0, nr_forms, nr_meanings)


def initialize_board(
    rng: np.random.default_rng,
    nr_agents: int,
    x_max: int,
    y_max: int,
    random_initialization: bool,
    nr_forms: int,
    nr_meanings: int,
) -> pd.DataFrame:
    """
    Returns a data frame containing for each agent these properties:
    - id
    - x
    - y
    - language_profile
    - language
    """

    ids = list(range(1, nr_agents + 1))  # ids from 1 to number of agents

    # Initial spatial distribution of the agents
    x = initialize_coordinates(rng, x_max, nr_agents)
    y = initialize_coordinates(rng, y_max, nr_agents)

    # Initial language profile, represented by a string of integers
    # For each agent:
    #     For each meaning:
    #         A random int [0, forms)

    language_profile = []

    if random_initialization:
        # All agents are initialized with random language profile
        language_profile = [
            initialize_language_profile(rng, nr_meanings, nr_forms)
            for _ in range(nr_agents)
        ]
    else:
        # All agents are initialized with same language profile
        profile = initialize_language_profile(rng, nr_forms, nr_meanings)
        language_profile = [profile.copy() for _ in range(nr_agents)]

    classification = np.zeros(
        nr_agents, dtype=int
    )  # Initialize language classification to zero for all

    # Create a pandas dataframe with all information
    population = pd.DataFrame(
        {
            "id": ids,
            "x": x,
            "y": y,
            "language_profile": language_profile,
            "language": classification,
        }
    )

    return population


def run_model():
    """Run the LECo model of language evolution"""

    # Request JSON parameter file
    configfile = input(
        "Please provide the relative path and file name with the parameter values (config.json for now): "
    )
    p = load_parameters_from_json(configfile)

    # Initialize seed
    rng = np.random.default_rng(p["seed"])

    # If the ouput parameter is set to true, the output of the run is saved in a directory within the directory provided by the user
    output_run = None
    if p["output"]:
        dir_plots = input(
            "Please provide the relative path and directory name to store output plots: "
        )
        output_run = output_handling(dir_plots, p)

    # Start to track model run time
    start_time = time.time()

    # Initialize population of agents
    population = initialize_board(
        rng,
        p["agents"],
        p["x_max"],
        p["y_max"],
        p["random_init"],
        p["forms"],
        p["meanings"],
    )

    # Group agents into different languages based on their language profiles
    dist = distance_matrix(population["language_profile"])
    population["language"] = language_classification(dist, p["similarity"])

    # Initialize variable that tracks the number of languages at each time step
    languagenumber = []

    for step in range(1, p["steps"] + 1):
        print(step)

        # Apply birth and death rates to the population
        population = population_dynamics(
            population, rng, p["death_rate"], p["birth_rate"]
        )

        # Move the agents within space
        population["x"], population["y"] = move(
            population["x"], population["y"], rng, p["speed"], p["x_max"], p["y_max"]
        )

        # Mutate language profiles
        population["language_profile"] = population["language_profile"].apply(
            mutate_profile, args=(rng, p["forms"], p["mutation_rate"])
        )

        # Interact with nearby neighbors
        nbs = get_neighbors(population[["x", "y"]].values, p["int_radius"])
        population["language_profile"] = interact(
            population["language_profile"],
            nbs,
            rng,
            p["int_partner_prob"],
            p["diffusion_rate"],
        )

        # Group agents into different languages based on their language profiles
        dist_matrix = distance_matrix(population["language_profile"])
        population["language"] = language_classification(dist_matrix, p["similarity"])

        languagenumber.append(
            len(np.unique(population["language"]))
        )  # Store the number of languages

        # Plot the positions of agents in space colored by language
        if p["output"]:
            if (step % p["plot_step"]) == 0:  # plot every plot_step years
                point_plotje(output_run, population, step, p["x_max"], p["y_max"])

    # Plot the number of languages over time
    if p["output"]:
        nr_agents = len(population)
        language_number_plotje(output_run, languagenumber, nr_agents, p["steps"])

    print("--- %s seconds ---" % (time.time() - start_time))
    print("--- %s minutes ---" % ((time.time() - start_time) / 60))


# run_model()
# %%
