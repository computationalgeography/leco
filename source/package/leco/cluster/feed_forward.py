"""Feed-forward clustering of the language profiles into languages.

Classification based on evolutionary diversification processes.
"""

import heapq
import logging
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm

logger = logging.getLogger(__name__)


def read_population(directory: Path, step: int) -> pd.DataFrame:
    """Read in the population data at a certain time step."""
    file_name = f"output{step:03d}.geoparquet"

    return gpd.read_parquet(directory / file_name)


def agglomerative_classification(
    language_profiles: np.ndarray[int],
    distance_threshold: float,
    linkage: str,
) -> np.ndarray[int]:
    """Group profiles into clusters based on distance threshold using hierarchical clustering."""
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="hamming",
        linkage=linkage,
    )

    return clustering.fit_predict(language_profiles)


def initialize_languages(
    start_population: gpd.GeoDataFrame,
    distance_threshold: float,
    linkage: str,
) -> np.ndarray[int]:
    """Initialize languages for the first time_step based on distance threshold."""
    language_profiles = np.stack(start_population["language_profile"])
    if len(start_population) == 1:
        return np.array([0])

    return agglomerative_classification(language_profiles, distance_threshold, linkage)


def find_neighboring_clusters(current_step: gpd.GeoDataFrame, radius: int) -> dict:
    """Find all neighboring clusters within radius."""
    n_clusters = len(current_step["candidate_cluster"].unique())

    if n_clusters <= 1:
        # If there is only one cluster in the current step, there are no neighboring clusters
        return {}

    # Dissolve the geometries of individual agents into a single geometry per cluster
    clusters = current_step.dissolve(by="candidate_cluster", as_index=False).loc[
        :,
        ["candidate_cluster", "geometry", "previous_language"],
    ]

    # Create buffers around the clusters with a specified radius
    clusters["buffer"] = clusters.geometry.buffer(radius)
    # Store geometries of the clusters with buffers in the geometry column
    clusters = clusters.set_geometry("buffer")

    # Perform a spatial join to find all clusters that intersect with each other
    # For every cluster (left), you get all clusters (right) whose geometry intersects it
    neighbors = gpd.sjoin(
        clusters,
        clusters.set_geometry("geometry"),
        predicate="intersects",
        how="left",
    )

    # Remove self-intersections
    neighbors = neighbors[neighbors.candidate_cluster_left != neighbors.candidate_cluster_right]

    # Exclude neighbors that just split from the same previous language
    neighbors = neighbors[neighbors.previous_language_left != neighbors.previous_language_right]

    # Group by the left cluster and get the list of right clusters for each left cluster
    return neighbors.groupby("candidate_cluster_left")["candidate_cluster_right"].apply(list).to_dict()


def find_splitting_events(
    previous_step: gpd.GeoDataFrame,
    current_step: gpd.GeoDataFrame,
    distance_threshold: float,
    linkage: str,
) -> gpd.GeoDataFrame:
    """Determine splits in previous language clusters for a certain time step."""
    current_step["candidate_cluster"] = -1
    current_step["previous_language"] = -1
    # Keep track of the candidate cluster ids
    candidate_cluster_id = 0

    # Map for each agent id the language of the previous step
    previous_lang_by_id = previous_step.set_index("id")["language"]

    # Set the previous language column of the current step to the language of the previous step
    # for corresponding agents
    current_step["previous_language"] = current_step["id"].map(previous_lang_by_id).fillna(-1).astype(int)

    # For all newborns, set the previous language to the previous language of the parent
    newborn_mask = current_step["previous_language"] == -1
    if newborn_mask.any():
        current_step.loc[newborn_mask, "previous_language"] = (
            current_step.loc[newborn_mask, "parent_id"].map(previous_lang_by_id).fillna(-1).astype(int)
        )

    # Group by the previous language and store agent indices in dictionary mapping
    previous_language_clusters = current_step.groupby("previous_language", sort=False).groups

    # Loop through the previous language clusters
    for idx in previous_language_clusters.values():
        # Store the agent indices in the dataframe in an array
        idx_arr = np.asarray(list(idx))
        count = int(idx_arr.size)
        if count == 0:
            # If there are no agents in the cluster, continue to the next cluster
            continue
        if count == 1:
            # If there is only one agent in the cluster, assign the candidate cluster id
            current_step.loc[idx_arr, "candidate_cluster"] = candidate_cluster_id
            candidate_cluster_id += 1
            continue

        # When there are multiple agents in the cluster, perform agglomerative classification
        new_profiles = np.stack(current_step.loc[idx_arr, "language_profile"])
        clusters = agglomerative_classification(new_profiles, distance_threshold, linkage)

        # Obtain the number of clusters and the cluster assignment to each agent
        unique_labels, inverse = np.unique(clusters, return_inverse=True)
        # Get the appropriate number of candidate_cluster_ids
        assigned_ids = np.arange(candidate_cluster_id, candidate_cluster_id + len(unique_labels))
        # Assign the candidate cluster ids to the current_step dataframe
        current_step.loc[idx_arr, "candidate_cluster"] = assigned_ids[inverse]
        candidate_cluster_id += len(unique_labels)

    return current_step


def find_shifting_events(
    current_step: gpd.GeoDataFrame,
    all_neighbors: dict[int, list[int]],
    distance_threshold: float,
    linkage: str,
) -> gpd.GeoDataFrame:
    """Determine language shifts of candidate clusters due to linguistic diffusion."""
    cluster_ids = current_step["candidate_cluster"].unique()

    # No shifts have occurred when there is only one candidate cluster
    if len(cluster_ids) <= 1:
        return current_step

    # Precompute agent profiles per cluster
    cluster_profiles = (
        current_step.groupby("candidate_cluster")["language_profile"]
        .apply(
            lambda x: np.stack(x.values),
        )
        .to_dict()
    )

    # When linkage is single, a network approach is used to computationally efficiently identify shifts
    if linkage == "single":
        return find_shifts_network(
            current_step,
            cluster_ids,
            cluster_profiles,
            all_neighbors,
            distance_threshold,
        )

    # When linkage is complete or average, an agglomerative-based approach is used to identify shifts
    return find_shifts_distance_matrix(
        current_step,
        cluster_ids,
        cluster_profiles,
        all_neighbors,
        distance_threshold,
    )


def find_shifts_network(
    current_step: gpd.GeoDataFrame,
    cluster_ids: np.ndarray[int],
    cluster_profiles: dict[int, np.ndarray],
    all_neighbors: dict[int, list[int]],
    distance_threshold: float,
) -> gpd.GeoDataFrame:
    """Assign shifting events using a network approach when linkage is single.

    A single linkage means that the minimum distance between clusters A and B < distance threshold.
    """
    # Create a network graph with candidate clusters as nodes with edges when the minimum distance is met
    cluster_network = nx.Graph()
    cluster_network.add_nodes_from(cluster_ids)

    for cluster_id, neighbors in all_neighbors.items():
        if not neighbors:
            # Skip if a cluster has zero neighboring clusters
            continue
        profiles_i = cluster_profiles[cluster_id]
        for neighbor_cluster_id in neighbors:
            if cluster_network.has_edge(cluster_id, neighbor_cluster_id):
                continue
            # Calculate the minimum cluster distance between current cluster and neighboring cluster
            d = cdist(profiles_i, cluster_profiles[neighbor_cluster_id], metric="hamming").min()
            # When this minimum distance is lower than the threshold, add an edge
            if d <= distance_threshold:
                cluster_network.add_edge(cluster_id, neighbor_cluster_id)

    # Go through all connected components in the network
    for component in nx.connected_components(cluster_network):
        if len(component) <= 1:
            # Only one node in the component; nothing happens
            continue
        # When there are edges, all connected candidate clusters are assigned the same cluster id
        candidate_id = next(iter(component))
        current_step.loc[current_step["candidate_cluster"].isin(component), "candidate_cluster"] = (
            candidate_id
        )

    return current_step


def compute_neighbor_distances(
    all_neighbors: dict[int, list[int]],
    cluster_ids: np.ndarray[int],
    cluster_profiles: dict[int, np.ndarray],
    distance_threshold: float,
    distance_matrix: np.ndarray[float],
    is_neighbor: np.ndarray[bool],
    heap: list[tuple[float, int, int]],
) -> tuple[np.ndarray, np.ndarray, list]:
    """Compute the distances between all neighboring candidate clusters.

    Return a distance matrix, a neighbor track matrix and a heap.
    """
    # Create a node index for every candidate cluster
    node_index = {cid: i for i, cid in enumerate(cluster_ids)}
    # Compute initial distances between neighboring candidate clusters
    for cluster_id, neighbors in all_neighbors.items():
        if not neighbors:
            # Skip if a cluster has zero neighboring clusters
            continue
        i = node_index[cluster_id]
        profiles_i = cluster_profiles[cluster_id]

        for neighbor_cluster_id in neighbors:
            j = node_index[neighbor_cluster_id]
            if is_neighbor[i, j]:
                continue

            # Calculate the cluster distance between current cluster and neighboring cluster
            pairwise_distances = cdist(profiles_i, cluster_profiles[neighbor_cluster_id], metric="hamming")
            # Take the mean distance if linkage is average, and the maximum distance when linkage is complete
            d = pairwise_distances.mean() if linkage == "average" else pairwise_distances.max()

            # Update the distance matrix and the neighbor tracking matrix
            distance_matrix[i, j] = d
            distance_matrix[j, i] = d
            is_neighbor[i, j] = True
            is_neighbor[j, i] = True

            # Add distance and clusters onto the heap when below threshold
            if d <= distance_threshold:
                heapq.heappush(heap, (d, i, j))

    return distance_matrix, is_neighbor, heap


def assign_shift_cluster_ids(
    current_step: gpd.GeoDataFrame,
    active: set[int],
    super_nodes: dict[int, frozenset[int]],
) -> gpd.GeoDataFrame:
    """Update all the candidate clusters in the population dataframe after detecting shift events."""
    for i in active:
        members = super_nodes[i]
        if len(members) > 1:
            current_step.loc[current_step["candidate_cluster"].isin(members), "candidate_cluster"] = next(
                iter(members),
            )

    return current_step


def find_shifts_distance_matrix(
    current_step: gpd.GeoDataFrame,
    cluster_ids: np.ndarray[int],
    cluster_profiles: dict[int, np.ndarray],
    all_neighbors: dict[int, list[int]],
    distance_threshold: float,
) -> gpd.GeoDataFrame:
    """Assign shifting events using an agglomerative approach when linkage is complete or average.

    A complete linkage means that the maximum distance between clusters A and B < distance threshold.
    An average linkage means that the mean distance between clusters A and B < distance threshold.
    """
    number_of_clusters = len(cluster_ids)

    # Initialize distance matrix and neighbor tracking
    distance_matrix = np.full((number_of_clusters, number_of_clusters), fill_value=np.inf)
    np.fill_diagonal(distance_matrix, 0.0)
    is_neighbor = np.zeros((number_of_clusters, number_of_clusters), dtype=bool)

    # Priority queue for closest pairs
    heap = []

    # Initialize super nodes as initial candidate clusters, these can later on be merged
    super_nodes = {i: frozenset({cluster_ids[i]}) for i in range(number_of_clusters)}
    # Compute the sizes of each clusters (number of agents) to compute UPGMA, unweighted average
    sizes = {i: len(cluster_profiles[cluster_ids[i]]) for i in range(number_of_clusters)}
    # Keep track of the active candidate clusters
    active = set(range(number_of_clusters))

    # Compute initial distances between neighboring candidate clusters
    distance_matrix, is_neighbor, heap = compute_neighbor_distances(
        all_neighbors,
        cluster_ids,
        cluster_profiles,
        distance_threshold,
        distance_matrix,
        is_neighbor,
        heap,
    )

    # Agglomerative clustering
    while heap and len(active) > 1:
        # Pop and return smallest item from the heap
        d, i, j = heapq.heappop(heap)

        # Skip if either node was already merged or distance exceeds threshold
        if i not in active or j not in active or d > distance_threshold:
            continue

        # Verify distance is still current (may have been updated after a previous merge)
        if distance_matrix[i, j] != d:
            continue

        merged_size = sizes[i] + sizes[j]

        # Find neighbors of i and j that are still active
        # These are the only clusters that need distance updates
        neighbors_i = {k for k in active - {i, j} if is_neighbor[i, k]}
        neighbors_j = {k for k in active - {i, j} if is_neighbor[j, k]}
        all_known_neighbors = neighbors_i | neighbors_j
        only_in_one = neighbors_i.symmetric_difference(neighbors_j)

        # Update the distance with active neighbors
        for k in all_known_neighbors:
            d_ik = distance_matrix[i, k] if k in neighbors_i else np.inf
            d_jk = distance_matrix[j, k] if k in neighbors_j else np.inf

            # Only compute unknown distance when k neighbors one side but not the other
            # i.e. the merged cluster may now be close enough to k to warrant checking
            if k in only_in_one:
                known_d = d_ik if k in neighbors_i else d_jk
                # Only worth computing if the known side is already below threshold
                if known_d <= distance_threshold:
                    unknown_profiles = np.vstack(
                        [
                            cluster_profiles[cid]
                            for cid in (super_nodes[j] if k in neighbors_i else super_nodes[i])
                        ],
                    )
                    k_profiles = np.vstack([cluster_profiles[cid] for cid in super_nodes[k]])
                    pairwise = cdist(unknown_profiles, k_profiles, metric="hamming")
                    unknown_d = pairwise.mean() if linkage == "average" else pairwise.max()

                    d_ik = known_d if k in neighbors_i else unknown_d
                    d_jk = unknown_d if k in neighbors_i else known_d
                else:
                    # Known side already exceeds threshold, no need to check unknown side
                    continue

            # Update distance from merged node to k
            if linkage == "average":
                # unweighted average (UPGMA) with each agent contributing equally
                new_d = (sizes[i] * d_ik + sizes[j] * d_jk) / merged_size
            else:  # complete
                new_d = max(d_ik, d_jk)

            distance_matrix[i, k] = new_d
            distance_matrix[k, i] = new_d
            is_neighbor[i, k] = True
            is_neighbor[k, i] = True

            if new_d <= distance_threshold:
                heapq.heappush(heap, (new_d, i, k))

        # Merge j into i
        super_nodes[i] = super_nodes[i] | super_nodes[j]
        sizes[i] = merged_size
        active.remove(j)

    # Update current_step with merged clusters
    return assign_shift_cluster_ids(current_step, active, super_nodes)


def assign_languages(
    current_step: gpd.GeoDataFrame,
    max_language_id: int,
    shift_counter: int,
    merge_counter: int,
) -> tuple[gpd.GeoDataFrame, int, int, int]:
    """Assign language labels including splitting and shifting events."""
    current_step["language"] = -1
    # Count number of speakers per (candidate_cluster, previous_language) pair
    contribution = (
        current_step.groupby(["candidate_cluster", "previous_language"]).size().rename("count").reset_index()
    )

    # Assign per candidate cluster the previous language that contributes the most speakers
    candidate_to_language = (
        contribution.sort_values("count", ascending=False)
        .drop_duplicates("candidate_cluster")
        .set_index("candidate_cluster")["previous_language"]
    )

    # Add column in contribution with the dominant, most common previous language per cluster
    contribution["dominant_language"] = contribution["candidate_cluster"].map(candidate_to_language)
    # Select per candidate cluster the dominant language
    winning = contribution[contribution["dominant_language"] == contribution["previous_language"]]

    # Assign per previous language the most common candidate cluster
    # from the candidate clusters that have this previous language as most common
    language_to_cluster = (
        winning.sort_values("count", ascending=False)
        .drop_duplicates("previous_language")
        .set_index("previous_language")["candidate_cluster"]
    )

    # For every previous language map the winning cluster
    language_map = (
        language_to_cluster.reset_index()
        .rename(columns={"previous_language": "language", "candidate_cluster": "cluster"})
        .set_index("cluster")["language"]
        .to_dict()
    )

    # --- SHIFTS: among assigned clusters (language_to_cluster) ---
    assigned_clusters = set(language_to_cluster.values)

    assigned_contribution = contribution[contribution["candidate_cluster"].isin(assigned_clusters)]
    prev_languages_per_assigned_cluster = assigned_contribution.groupby("candidate_cluster")[
        "previous_language"
    ].apply(set)

    shift_counter = 0
    for cluster, prev_langs in prev_languages_per_assigned_cluster.items():
        if len(prev_langs) <= 1:
            continue
        current_lang_id = language_map[cluster]
        if any(lang == current_lang_id for lang in prev_langs):
            shift_counter += 1

    # All other clusters get new language IDs
    # These include splits and shifts
    all_clusters = set(current_step["candidate_cluster"].unique())
    unassigned = sorted(all_clusters - set(language_map.keys()))

    new_ids = range(max_language_id, max_language_id + len(unassigned))
    language_map.update(zip(unassigned, new_ids, strict=True))
    max_language_id += len(unassigned)

    # --- MERGES: among unassigned clusters ---
    unassigned_contribution = contribution[contribution["candidate_cluster"].isin(unassigned)]
    prev_languages_per_unassigned_cluster = unassigned_contribution.groupby("candidate_cluster")[
        "previous_language"
    ].apply(set)

    merge_counter = 0
    for cluster, prev_langs in prev_languages_per_unassigned_cluster.items():
        if len(prev_langs) <= 1:
            continue
        current_lang_id = language_map[cluster]
        if all(lang != current_lang_id for lang in prev_langs):
            merge_counter += 1

    # Assign language ids to dataframe
    current_step["language"] = current_step["candidate_cluster"].map(language_map).astype(int)

    return current_step, max_language_id, shift_counter, merge_counter


def diversify(
    directory: Path,
    distance_threshold: float,
    linkage: str,
    time_steps: int,
    radius: float,
    sensitivity: bool = False,
) -> None | int:
    """Cluster languages per time step based on clustering previous time step."""
    # Create a dataframe to store meta data on the language shifts and merges
    meta_data = pd.DataFrame(
        {
            "time_step": [0],
            "shifts": [0],
            "merges": [0],
        },
    )

    # Dataframe of the population at the initial time step
    population_current = read_population(directory, 0)
    # First time_step: initialize the start languages
    languages = initialize_languages(
        population_current,
        distance_threshold,
        linkage,
    )
    population_current["language"] = languages.astype(int)
    max_language_id = languages.max()

    #### THESE IF STATEMENTS ARE WEIRD RIGHT
    if "candidate_cluster" not in population_current.columns:
        # Initialize candidate column as integer type
        population_current["candidate_cluster"] = -1

    if "previous_language" not in population_current.columns:
        # Initialize previous language as integer type
        population_current["previous_language"] = -1

    # Initiate the total population data frame across all time steps
    population_total = [population_current]

    # Iterate over each time_step and check for splits
    for time_step in tqdm(range(1, time_steps + 1), desc="Processing time steps"):
        # Keep track of the number of language shifts
        shift_counter = 0
        merge_counter = 0

        population_previous = population_current
        population_current = read_population(directory, time_step)

        # Determine the candidate clusters formed after splitting
        population_current = find_splitting_events(
            population_previous,
            population_current,
            distance_threshold,
            linkage,
        )

        # Find all neighboring clusters within radius
        all_neighbors = find_neighboring_clusters(population_current, radius)

        # Merge clusters based on language similarity
        population_current = find_shifting_events(
            population_current,
            all_neighbors,
            distance_threshold,
            linkage,
        )

        population_current, max_language_id, shift_counter, merge_counter = assign_languages(
            population_current,
            max_language_id,
            shift_counter,
            merge_counter,
        )

        if np.any(population_current["language"] == -1):
            logger.error("Be careful! Language is not assigned")
        # Update population with new language assignments
        population_total.append(population_current)
        # Record meta data for this time step

        meta_data.loc[len(meta_data)] = [int(time_step), int(shift_counter), int(merge_counter)]

    # Save output to a single gpkg file
    pd.concat(population_total, ignore_index=True).to_file(
        directory / f"population_{linkage}.gpkg",
        driver="GPKG",
    )

    # Save meta data to csv file
    meta_data.to_csv(directory / "meta_data_cluster.csv", index=False)

    if sensitivity:
        # Compute number of unique languages at the last time step
        last_step = int(population_total["time_step"].max())
        languages_last = population_total.loc[population_total["time_step"] == last_step, "language"].unique()
        logger.info("Last time step: %s; number of languages: %s", last_step, len(languages_last))
        return len(languages_last)

    return None
