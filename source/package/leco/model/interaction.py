"""Functions for agent interaction and linguistic diffusion."""

import geopandas as gpd
import logging
import numpy as np
from scipy.spatial import KDTree
from shapely import (
    MultiPolygon,
    Polygon,
    get_coordinates,
    vectorized,
)


def nearest_neighbors(
    positions: np.ndarray[float], radius: float
) -> list[np.ndarray[int]]:
    """Find all neighboring agents within a radius."""
    # Build a K-Dimensional tree (spatial index)
    positions_kdt = KDTree(positions)

    # Find all neighbors within a radius around an agent
    neighbors = positions_kdt.query_ball_point(positions, r=radius)
    # Remove self and store neighbors of every agent in list format
    return [
        [n for n in neighbor_list if n != i]
        for i, neighbor_list in enumerate(neighbors)
    ]


def compute_interact_prob_neighbors(
    agent_id: int,
    positions: gpd.GeoDataFrame.geometry,
    interact_attributes: dict[float, float, float],
    neighbors: np.ndarray[int],
    barrier: Polygon | None,
    impermeability: float,
) -> np.ndarray[float]:
    """Compute the interaction probability for neighbors of a single agent dependent on barrier presence."""
    # Get the position of the active agent
    agent_pos = positions[agent_id]

    if barrier is None:
        # Without barrier, all neighbors have an equal probability [partner_prob] to interact
        return np.repeat(interact_attributes["partner_prob"], len(neighbors))

    # Create a Polygon area of the interaction radius
    interaction_circle = agent_pos.buffer(interact_attributes["radius"])
    # Find the area that is intersected by the barrier
    impeded_area = interaction_circle.intersection(barrier)

    if impeded_area == 0:
        # No intersection of barrier and radius, all neighbors have equal probability of partner_probability
        return np.repeat(interact_attributes["partner_probability"], len(neighbors))

    # The neighbors on/behind the barrier have a lower probability to interact,
    # proportional to the impermeability
    barrier_probability = interact_attributes["partner_probability"] * (
        1.0 - impermeability
    )

    if impeded_area.contains(agent_pos):
        # If the active agent is positioned on a barrier,
        # interaction with all of its neighbors has a lower probability
        return np.repeat(barrier_probability, len(neighbors))

    # Select the area without barrier
    interaction_area_no_barrier = interaction_circle.difference(barrier)
    if interaction_area_no_barrier.is_empty:
        # Barrier completely covers the interaction area
        valid_interaction_area = None
    elif isinstance(interaction_area_no_barrier, MultiPolygon):
        # Barrier has split the interaction area in two, keep the area where the agent resides
        valid_interaction_area = next(
            (
                geom
                for geom in interaction_area_no_barrier.geoms
                if geom.contains(agent_pos)
            ),
        )
    elif isinstance(interaction_area_no_barrier, Polygon):
        # Barrier has cut off a side of the area, keep the remaining area
        valid_interaction_area = interaction_area_no_barrier

    # Add for every neighbor the probability dependent on whether they are located in the valid_int_area
    nb_positions = get_coordinates(
        positions[neighbors]
    )  # Get the positions of the neighbors

    # Check whether the neighbors reside on the reachable area
    mask = vectorized.contains(
        valid_interaction_area, nb_positions[:, 0], nb_positions[:, 1]
    )

    # If they reside on the reachable area, assign the high probability, if not the low probability
    return np.where(mask, interact_attributes["partner_prob"], barrier_probability)


def calculate_similarity_pairwise(
    profile_a: np.ndarray[int],
    profile_b: np.ndarray[int],
) -> float:
    """Calculate the similarity between two language profiles."""
    # Count the number of meanings with the same form
    matching_meanings = np.sum(profile_a == profile_b)
    # Similarity is the proportion of matching meanings
    return matching_meanings / len(profile_a)


def interact(
    language_profiles: np.ndarray[int],
    interact_attributes: dict[float, float, float, float],
    profile_length: int,
    positions: gpd.GeoDataFrame.geometry,
    barrier: Polygon | None,
    impermeability: float,
    rng: np.random.default_rng,
) -> list[np.ndarray[int], int]:
    """Interaction between agents whereby linguistic diffusion occurs."""
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
        # No neighbors anywhere, hence no diffusion occurred
        return list(new_profiles), 0

    # Generate interaction probabilities for all agents at once
    interaction_probabilities = rng.random(
        (nr_agents, max_neighbors),
    )  # The probability that an agent interact with each of its neighbors
    diffusion_probabilities = rng.random(
        (nr_agents, max_neighbors, nr_meanings),
    )  # The probability that a form diffuses from a neighbor to an agent,
    # probabilities are taken for every meaning in the language profile

    # Loop through the agents
    for agent_idx, neighbors in enumerate(neighbors_list):
        if len(neighbors) == 0:
            continue

        # Compute the probabilities for an agent to interact with each of its neighbors,
        # based on the presence of a barrier
        interaction_probabilities_nbs = compute_interact_prob_neighbors(
            agent_idx,
            positions,
            interact_attributes,
            neighbors,
            barrier,
            impermeability,
        )
        # Convert neighbor list to an array to make use of the masks
        neighbors_array = np.array(neighbors)

        if len(neighbors_array) != len(interaction_probabilities_nbs):
            logging.error(
                "Error! Number of neighbors is not equal to the number of neighbor probabilities!!"
            )

        # Based on the interaction probabilities, the agent interact with 'partner_prob' proportion
        # of their neighbors
        interaction_mask = (
            interaction_probabilities[agent_idx, : len(neighbors_array)]
            < interaction_probabilities_nbs
        )
        # Select the interaction partners
        interacting_neighbors = neighbors_array[interaction_mask]

        if len(interacting_neighbors) == 0:
            continue

        # Shuffle the order of the interaction partners
        rng.shuffle(interacting_neighbors)

        # Select the diffusion probabilities for the interacting neighbors
        agent_diffusion_probabilities = diffusion_probabilities[
            agent_idx, : len(neighbors), :
        ]
        agent_diffusion_probabilities = agent_diffusion_probabilities[
            interaction_mask
        ]  # Only keep the probabilities for the interacting neighbors

        # Select the language profiles of the interacting neighbors
        neighbor_profiles = language_profiles[interacting_neighbors]

        # Adopt the forms from the neighbor's language profile to the active agent's profile
        for j in range(len(interacting_neighbors)):
            if interact_attributes["similarity_preference"] != 0.0:
                # Calculate similarity between the agent and the neighbor
                similarity = calculate_similarity_pairwise(
                    language_profiles[agent_idx],
                    neighbor_profiles[j],
                )
                # Similarity factor is weighted by the similarity preference
                similarity_factor = (
                    similarity * interact_attributes["similarity_preference"]
                )
                # Neutral factor, diffusion probability is weighted by the inverse of the similarity preference
                neutral_factor = (
                    1 - interact_attributes["similarity_preference"]
                ) * interact_attributes["diffusion_rate"]
                # Adoption probability is the sum of both factors per feature
                adoption_probability = (
                    neutral_factor + similarity_factor
                ) / profile_length
            else:
                # Avoid calculating pairwise similarity between all agents
                # Instead use the diffusion rate per feature
                adoption_probability = (
                    interact_attributes["diffusion_rate"] / profile_length
                )

            # Determine which meanings from neighbors' language profiles will be diffused
            diffusion_mask = agent_diffusion_probabilities[j] < adoption_probability

            new_profiles[agent_idx] = np.where(
                diffusion_mask,
                neighbor_profiles[j],
                new_profiles[agent_idx],
            )

    # Count the number of diffusions that have occurred to track external change
    diffusions_count = np.sum(language_profiles != new_profiles)

    # Return list format to add to geopandas dataframe
    return list(new_profiles), diffusions_count
