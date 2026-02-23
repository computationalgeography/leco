"""Functions for agent interaction and linguistic diffusion."""

import logging

import geopandas as gpd
import numpy as np
from scipy.spatial import KDTree

logger = logging.getLogger(__name__)


def nearest_neighbors(positions: np.ndarray[float], radius: float) -> list[np.ndarray[int]]:
    """Find all neighboring agents within a radius."""
    # Build a K-Dimensional tree (spatial index)
    positions_kdt = KDTree(positions)

    # Find all neighbors within a radius around an agent
    neighbors = positions_kdt.query_ball_point(positions, r=radius)
    # Remove self and store neighbors of every agent in list format
    return [[n for n in neighbor_list if n != i] for i, neighbor_list in enumerate(neighbors)]


def select_interacting_partners(
    agent_profile: np.ndarray[int],
    neighbors: np.ndarray[int],
    neighbors_profiles: np.ndarray[int],
    partner_proportion: float,
    similarity_preference: float,
    rng: np.random.default_rng,
) -> np.ndarray[int]:
    """Select proportion / fixed number of neighbors an agent interacts with.

    Neighbor weights are calculated following under positive similarity preference s:
        weights = (1 - s) * uniform_weights + s * similarity_weights
    Or a negative similarity preference s:
        weights = (1 + s) * uniform_weights - s * dissimilarity_weights
    """
    number_neighbors = len(neighbors)
    # Select partner_proportion interaction partners, or the number of neighbors when this value is lower
    number_interaction_partners = min(number_neighbors, int(number_neighbors * partner_proportion))

    # Calculate all pairwise similarities between active agent and it's neighbors
    similarities = np.sum(neighbors_profiles == agent_profile, axis=1) / len(agent_profile)

    uniform_weights = np.ones(number_neighbors) / number_neighbors

    # Select weighting scheme based on similarity preference in [-1.0, 1.0]
    if 0.0 <= similarity_preference <= 1.0:
        # A positive preference normalizes the similarities
        similarity_sum = similarities.sum()
        # Avoid dividing by zero
        similarity_weights = similarities / similarity_sum if similarity_sum > 0 else uniform_weights
        weights = (1 - similarity_preference) * uniform_weights + similarity_preference * similarity_weights
    elif -1.0 <= similarity_preference < 0.0:
        # A negative preferences normalizes the dissimilarities
        dissimilarity_sum = (1 - similarities).sum()
        # Avoid dividing by zero
        dissimilarity_weights = (
            (1 - similarities) / dissimilarity_sum if dissimilarity_sum > 0 else uniform_weights
        )
        weights = (1 + similarity_preference) * uniform_weights + (
            -similarity_preference
        ) * dissimilarity_weights
    else:
        logger.error("Error: similarity preference should be within the range of [-1.0, 1.0]")

    # Avoid weights of zero, so rng.choice always select number_interaction_partners
    epsilon = 1e-8
    weights = weights + epsilon
    weights /= weights.sum()

    # Return the chosen interaction partners
    return rng.choice(
        number_neighbors,
        size=number_interaction_partners,
        p=weights,
        replace=False,
    )


def interact(
    language_profiles: np.ndarray[int],
    interact_attributes: dict[str, float],
    profile_length: int,
    positions: gpd.GeoDataFrame.geometry,
    rng: np.random.default_rng,
) -> list[np.ndarray[int], int]:
    """Interaction between agents whereby linguistic diffusion occurs."""
    # Get neighbors for all agents within a radius of int_radius
    neighbors_list = nearest_neighbors(positions.get_coordinates().to_numpy(), interact_attributes["radius"])

    # Create a copy of the language profiles
    new_profiles = language_profiles.copy()

    # Loop through the agents
    for agent_idx, neighbors in enumerate(neighbors_list):
        if len(neighbors) == 0:
            continue

        # Select the language profiles of the interacting neighbors
        neighbor_profiles = language_profiles[neighbors]

        # Select subset of the neighbors with which agent interacts
        interaction_partners = select_interacting_partners(
            language_profiles[agent_idx],
            neighbors,
            neighbor_profiles,
            interact_attributes["partner_proportion"],
            interact_attributes["similarity_preference"],
            rng,
        )

        if len(interaction_partners) == 0:
            ### This could happen when number of neighbors is zero and using partner_proportion
            ### might chance to fixed number
            continue

        # Shuffle the order of the interaction partners as the agent's profile is updated continuously
        rng.shuffle(interaction_partners)

        # The probability that a form diffuses from a neighbor to the agent,
        # probabilities are taken for every meaning in the language profile
        diffusion_probabilities = rng.random(
            (len(interaction_partners), profile_length),
        )

        for i, partner_idx in enumerate(interaction_partners):
            # Determine which features from the partner's language profile will be diffused
            # The partner corresponds to the ith array from diffusion_probabilities
            diffusion_mask = diffusion_probabilities[i] < interact_attributes["diffusion_rate"]

            new_profiles[agent_idx] = np.where(
                diffusion_mask,
                neighbor_profiles[partner_idx],
                new_profiles[agent_idx],
            )

    # Count the number of diffusions that have occurred to track external change
    diffusions_count = np.sum(language_profiles != new_profiles)

    # Return list format to add to geopandas dataframe
    return list(new_profiles), diffusions_count
