import numpy as np
import geopandas as gpd
from shapely import (
    Polygon,
    get_coordinates,
    MultiPolygon,
    vectorized,
)
from scipy.spatial import KDTree


def nearest_neighbors(
    positions: np.ndarray[float], radius: float
) -> list[np.ndarray[int]]:
    """Find all neighboring agents within a radius"""

    # Build a K-Dimensional tree (spatial index)
    pos_kdt = KDTree(positions)

    # Find all neighbors within a radius around an agent
    neighbors = pos_kdt.query_ball_point(positions, r=radius)
    # Remove self and store in list format
    neighbors = [
        [n for n in neighbor_list if n != i]
        for i, neighbor_list in enumerate(neighbors)
    ]

    # Return a list of the neighbor ids
    return neighbors


def compute_interact_prob_neighbors(
    agent_id: int,
    positions: gpd.GeoDataFrame.geometry,
    interact_attributes: dict[float, float, float],
    neighbors: np.ndarray[int],
    barrier: Polygon | None,
    bar_impermeability: float,
) -> np.ndarray[float]:
    """Compute the interaction probability for neighbors of a single agent dependent on barrier presence"""

    # Get the position of the active agent
    agent_pos = positions[agent_id]

    if barrier is None:
        # Without barrier, all neighbors have an equal probability [int_partner_prob] to interact
        return np.repeat(interact_attributes["partner_prob"], len(neighbors))

    # Create a Polygon area of the interaction radius
    int_circle = agent_pos.buffer(interact_attributes["radius"])
    # Find the area that is intersected by the barrier
    impeded_area = int_circle.intersection(barrier)

    if impeded_area == 0:
        # If the barrier is not intersecting with the interaction radius, all neighbors have an equal probability of int_partner_prob
        return np.repeat(interact_attributes["partner_prob"], len(neighbors))

    # The neighbors on or behind the barrier have a lower probability to interact, which is proportional to the bar_impermeability
    barrier_prob = interact_attributes["partner_prob"] * (1.0 - bar_impermeability)

    if impeded_area.contains(agent_pos):
        # If the active agent is positioned on a barrier, interaction with all of its neighbors has a lower probability
        return np.repeat(barrier_prob, len(neighbors))

    # Select the area without barrier
    int_area_withoutbar = int_circle.difference(barrier)
    if int_area_withoutbar.is_empty:
        # Barrier completely covers the interaction area
        valid_int_area = None
    elif isinstance(int_area_withoutbar, MultiPolygon):
        # Barrier has split the interaction area in two, keep the area where the agent resides
        valid_int_area = next(
            (geom for geom in int_area_withoutbar.geoms if geom.contains(agent_pos)),
        )
    elif isinstance(int_area_withoutbar, Polygon):
        # Barrier has cut off a side of the area, keep the remaining area
        valid_int_area = int_area_withoutbar

    # Add for every neighbor the probability dependent on whether they are located in the valid_int_area or not
    nb_positions = get_coordinates(
        positions[neighbors]
    )  # Get the positions of the neighbors

    # Check whether the neighbors reside on the reachable area
    mask = vectorized.contains(valid_int_area, nb_positions[:, 0], nb_positions[:, 1])

    # If they reside on the reachable area, assign the high probability, if not the low probability
    int_probs = np.where(mask, interact_attributes["partner_prob"], barrier_prob)

    return int_probs


def interact(
    language_profiles: np.ndarray[int],
    interact_attributes: dict[float, float, float],
    positions: gpd.GeoDataFrame.geometry,
    barrier: Polygon | None,
    bar_impermeability: float,
    rng: np.random.default_rng,
) -> np.ndarray[int]:
    """Interaction between agents whereby linguistic diffusion occurs"""

    # Get neighbors for all agents within a radius of int_radius
    neighbors_list = nearest_neighbors(
        positions.get_coordinates().to_numpy(), interact_attributes["radius"]
    )

    # Get the current number of agents and the number of meanings
    nr_agents, nr_meanings = language_profiles.shape
    # Create a copy of the language profiles
    new_profiles = language_profiles.copy()

    ## Pre-generate all random numbers at once
    # Calculate the maximum number of neighbors an agent has
    max_neighbors = max(len(nbs) for nbs in neighbors_list) if neighbors_list else 0
    if max_neighbors == 0:
        return list(new_profiles)

    # Generate interaction probabilities for all agents at once
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

        # Compute the probabilities for an agent to interact with each of its neighbours, based on the presence of a barrier
        int_probs_nbs = compute_interact_prob_neighbors(
            agent_idx,
            positions,
            interact_attributes,
            neighbors,
            barrier,
            bar_impermeability,
        )
        # Convert neighbor list to an array to make use of the masks
        neighbors = np.array(neighbors)

        if len(neighbors) != len(int_probs_nbs):
            print(
                "Error! Number of neighbors is not equal to the number of neighbor probabilities!!"
            )

        # Based on the interaction probabilities, the agent interacts with 'int_partner_prob' proportion of their neighbors
        interaction_mask = (
            interaction_probs[agent_idx, : len(neighbors)] < int_probs_nbs
        )
        # Select the interaction partners
        interacting_neighbors = neighbors[interaction_mask]

        if len(interacting_neighbors) == 0:
            continue

        # Shuffle the order of the interaction partners
        rng.shuffle(interacting_neighbors)

        # Select the diffusion probabilities for the interacting neighbors
        agent_diffusion_probs = diffusion_probs[agent_idx, : len(neighbors), :]
        agent_diffusion_probs = agent_diffusion_probs[
            interaction_mask
        ]  # Only keep the probabilities for the interacting neighbors

        # Determine which meanings from neighbors' language profiles will be diffused based on the diffusion_rate
        diffusion_mask = agent_diffusion_probs < interact_attributes["diffusion_rate"]

        # Select the language profiles of the interacting neighbors
        neighbor_profiles = language_profiles[interacting_neighbors]

        # Adopt the forms from the neighbor's language profile to the active agent's profile
        for j in range(len(interacting_neighbors)):
            new_profiles[agent_idx] = np.where(
                diffusion_mask[j], neighbor_profiles[j], new_profiles[agent_idx]
            )

    # Return list format to add to geopanda's dataframe
    return list(new_profiles)
