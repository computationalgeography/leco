"""Feed-forward clustering of the language profiles into languages.

Classification based on evolutionary diversification processes.
"""

import heapq
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import cast

import geopandas as gpd
import networkx as nx
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import KDTree
from tqdm import tqdm

logger = logging.getLogger(__name__)


def read_population(directory: Path, step: int) -> gpd.GeoDataFrame:
    """Read in the population data at a certain time step."""
    file_name = f"output{step:03d}.geoparquet"

    return gpd.read_parquet(directory / file_name)


def make_population_reader(
    directory: Path,
    chunk_size: int,
    time_steps: int,
) -> Callable[[int], gpd.GeoDataFrame]:
    """Return a read function with a built-in cache for reading population data from geoparquet files."""
    chunk: gpd.GeoDataFrame | None = None
    chunk_end: int | None = None

    def read(step: int) -> gpd.GeoDataFrame:
        nonlocal chunk, chunk_end  # Required to modify the outer variables

        if step == 0:
            return cast("gpd.GeoDataFrame", gpd.read_parquet(directory / "steps_0000.geoparquet"))

        new_chunk_end = ((step - 1) // chunk_size + 1) * chunk_size
        chunk_start = new_chunk_end - chunk_size + 1
        actual_chunk_end = min(new_chunk_end, time_steps)

        if new_chunk_end != chunk_end:
            file = directory / f"steps_{chunk_start:04d}_{actual_chunk_end:04d}.geoparquet"
            chunk = cast("gpd.GeoDataFrame", gpd.read_parquet(file))
            chunk_end = new_chunk_end

        assert chunk is not None  # Ensure chunk is not None
        result = chunk[chunk["time_step"] == step].reset_index(drop=True).copy()
        return cast("gpd.GeoDataFrame", result)

    return read


def agglomerative_classification(
    language_profiles: npt.NDArray[np.int64],
    distance_threshold: float,
    linkage: str,
) -> npt.NDArray[np.int64]:
    """Group profiles into clusters based on distance threshold using hierarchical clustering."""
    clustering = AgglomerativeClustering(
        n_clusters=None,  # type: ignore[arg-type]
        distance_threshold=distance_threshold,
        metric="hamming",
        linkage=linkage,
    )

    return clustering.fit_predict(language_profiles)


def initialize_languages(
    start_population: gpd.GeoDataFrame,
    distance_threshold: float,
    linkage: str,
) -> npt.NDArray[np.int_]:
    """Initialize languages for the first time_step based on distance threshold."""
    language_profiles = np.stack(start_population["language_profile"].to_list())

    if len(start_population) == 1:
        return np.array([0])

    language_classification = agglomerative_classification(language_profiles, distance_threshold, linkage)

    # Initialize the languages in consistent order when initial language number is four
    if len(np.unique(language_classification)) == 4:
        coordinates = np.stack([start_population.geometry.x, start_population.geometry.y], axis=1)
        x_mid = coordinates[:, 0].mean()
        y_mid = coordinates[:, 1].mean()
        # Assign quadrant index (0-3) based on position relative to midpoint
        # Quadrant layout:
        #   0 | 1
        #   -----
        #   2 | 3
        quadrants = np.where(
            coordinates[:, 1] >= y_mid,  # top half
            np.where(coordinates[:, 0] < x_mid, 0, 1),  # top-left = 0, top-right = 1
            np.where(coordinates[:, 0] < x_mid, 2, 3),  # bottom-left = 2, bottom-right = 3
        )
        new_labels = np.empty_like(language_classification)
        for cluster_id in np.unique(language_classification):
            mask = language_classification == cluster_id
            majority_quadrant = int(np.bincount(quadrants[mask]).argmax())
            new_labels[mask] = majority_quadrant
        return new_labels

    return language_classification


def find_neighboring_clusters(current_step: gpd.GeoDataFrame, radius: float) -> dict:
    """Find all neighboring clusters within radius."""
    n_clusters = len(current_step["candidate_language"].unique())

    if n_clusters <= 1:
        # If there is only one cluster in the current step, there are no neighboring clusters
        return {}

    # Dissolve the geometries of individual agents into a single geometry per cluster
    clusters = current_step.dissolve(by="candidate_language", as_index=False).loc[
        :,
        ["candidate_language", "geometry", "previous_language"],
    ]

    # Create buffers around the clusters with a specified radius
    clusters["buffer"] = clusters.geometry.buffer(radius)
    # Store geometries of the clusters with buffers in the geometry column
    clusters = clusters.set_geometry("buffer")

    # Perform a spatial join to find all clusters that intersect with each other
    # For every cluster (left), you get all clusters (right) whose geometry intersects it
    neighbors = cast(
        "pd.DataFrame",
        gpd.sjoin(
            clusters,
            clusters.set_geometry("geometry"),
            predicate="intersects",
            how="left",
        ),
    )

    # Remove self-intersections
    neighbors = neighbors[neighbors.candidate_language_left != neighbors.candidate_language_right]

    # Exclude neighbors that just split from the same previous language
    neighbors = neighbors[neighbors.previous_language_left != neighbors.previous_language_right]
    neighbors = cast("pd.DataFrame", neighbors)  # re-cast after boolean indexing

    # Drop any nan neighbor ids (unmatched left-join rows)
    neighbors = neighbors[neighbors["candidate_language_right"].notna()]

    # Group by the left cluster and get the list of right clusters for each left cluster
    return neighbors.groupby("candidate_language_left")["candidate_language_right"].apply(list).to_dict()


def find_splitting_events(
    previous_step: gpd.GeoDataFrame,
    current_step: gpd.GeoDataFrame,
    distance_threshold: float,
    linkage: str,
    divergence_counter: int,
) -> tuple[gpd.GeoDataFrame, int]:
    """Determine divergence in previous language clusters for a certain time step."""
    current_step["candidate_language"] = -1
    current_step["previous_language"] = -1
    # Keep track of the candidate cluster ids
    candidate_language_id = 0

    # Map for each agent id the language of the previous step
    previous_lang_by_id = cast("pd.Series", previous_step.set_index("id")["language"])

    # Set the previous language column of the current step to the language of the previous step
    # for corresponding agents
    current_step["previous_language"] = (
        current_step["id"].replace(previous_lang_by_id.to_dict()).fillna(-1).astype(int)
    )

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
            current_step.loc[idx_arr, "candidate_language"] = candidate_language_id
            candidate_language_id += 1
            continue

        # When there are multiple agents in the cluster, perform agglomerative classification
        new_profiles = np.stack(current_step.loc[idx_arr, "language_profile"])
        clusters = agglomerative_classification(new_profiles, distance_threshold, linkage)

        # Obtain the number of clusters and the cluster assignment to each agent
        unique_labels, inverse = np.unique(clusters, return_inverse=True)
        # Get the appropriate number of candidate_language_ids
        assigned_ids = np.arange(candidate_language_id, candidate_language_id + len(unique_labels))
        # Assign the candidate cluster ids to the current_step dataframe
        current_step.loc[idx_arr, "candidate_language"] = assigned_ids[inverse]
        candidate_language_id += len(unique_labels)

        # Update divergence_counter with the number of splitted candidate clusters
        if len(unique_labels) > 1:
            divergence_counter += len(unique_labels) - 1

    return current_step, divergence_counter


def find_merging_events(
    current_step: gpd.GeoDataFrame,
    all_neighbors: dict[int, list[int]],
    distance_threshold: float,
    linkage: str,
    convergence_counter: int,
) -> tuple[gpd.GeoDataFrame, int]:
    """Determine language convergence of candidate languages due to linguistic diffusion."""
    cluster_ids = np.unique(current_step["candidate_language"].to_numpy(dtype=np.int64))

    # No merges have occurred when there is only one candidate cluster
    if len(cluster_ids) <= 1:
        return current_step, convergence_counter

    # Precompute agent profiles per cluster
    cluster_profiles = (
        current_step.groupby("candidate_language")["language_profile"]
        .apply(
            lambda x: np.stack(x.values),
        )
        .to_dict()
    )

    # When linkage is single, a network approach is used to computationally efficiently identify merges
    if linkage == "single":
        return find_merges_network(
            current_step,
            cluster_ids,
            cluster_profiles,
            all_neighbors,
            distance_threshold,
            convergence_counter,
        )

    # When linkage is complete or average, an agglomerative-based approach is used to identify merges
    return find_merges_distance_matrix(
        current_step,
        cluster_ids,
        cluster_profiles,
        all_neighbors,
        distance_threshold,
        convergence_counter,
    )


def find_merges_network(
    current_step: gpd.GeoDataFrame,
    cluster_ids: npt.NDArray[np.int64],
    cluster_profiles: dict[int, npt.NDArray[np.int64]],
    all_neighbors: dict[int, list[int]],
    distance_threshold: float,
    convergence_counter: int,
) -> tuple[gpd.GeoDataFrame, int]:
    """Assign merging events using a network approach when linkage is single.

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
        current_step.loc[current_step["candidate_language"].isin(component), "candidate_language"] = (
            candidate_id
        )
        # Update convergence_counter if clusters have merged
        convergence_counter += 1

    return current_step, convergence_counter


def compute_neighbor_distances(
    all_neighbors: dict[int, list[int]],
    cluster_ids: npt.NDArray[np.int64],
    cluster_profiles: dict[int, np.ndarray],
    distance_threshold: float,
    distance_matrix: npt.NDArray[np.float64],
    is_neighbor: npt.NDArray[np.bool],
    heap: list[tuple[float, int, int]],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool], list]:
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


def assign_merge_cluster_ids(
    current_step: gpd.GeoDataFrame,
    active: set[int],
    super_nodes: dict[int, frozenset[int]],
    convergence_counter: int,
) -> tuple[gpd.GeoDataFrame, int]:
    """Update all the candidate clusters in the population dataframe after detecting merge events."""
    for i in active:
        members = super_nodes[i]
        if len(members) > 1:
            current_step.loc[current_step["candidate_language"].isin(list(members)), "candidate_language"] = (
                next(
                    iter(members),
                )
            )
            convergence_counter += 1

    return current_step, convergence_counter


def find_merges_distance_matrix(
    current_step: gpd.GeoDataFrame,
    cluster_ids: npt.NDArray[np.int64],
    cluster_profiles: dict[int, npt.NDArray[np.int64]],
    all_neighbors: dict[int, list[int]],
    distance_threshold: float,
    convergence_counter: int,
) -> tuple[gpd.GeoDataFrame, int]:
    """Assign merging events using an agglomerative approach when linkage is complete or average.

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
    return assign_merge_cluster_ids(current_step, active, super_nodes, convergence_counter)


def assign_languages(
    current_step: gpd.GeoDataFrame,
    max_language_id: int,
) -> tuple[gpd.GeoDataFrame, int]:
    """Assign language labels after divergence and convergence detection."""
    current_step["language"] = -1
    # Count number of speakers per (candidate_language, previous_language) pair
    contribution = (
        current_step.groupby(["candidate_language", "previous_language"])
        .size()
        .to_frame("count")
        .reset_index()
    )
    # Assign per candidate cluster the previous language that contributes the most speakers
    candidate_to_language = cast(
        "pd.Series",
        contribution.sort_values("count", ascending=False)
        .drop_duplicates("candidate_language")
        .set_index("candidate_language")["previous_language"],
    )
    # Add column in contribution with the dominant, most common previous language per cluster
    contribution["dominant_language"] = contribution["candidate_language"].map(
        candidate_to_language.to_dict(),  # type: ignore[arg-type]
    )

    # Select per candidate cluster the dominant language
    winning = cast(
        "pd.DataFrame",
        contribution[contribution["dominant_language"] == contribution["previous_language"]],
    )

    # Assign per previous language the most common candidate cluster
    # from the candidate clusters that have this previous language as most common
    language_to_cluster = (
        winning.sort_values("count", ascending=False)
        .drop_duplicates("previous_language")
        .set_index("previous_language")["candidate_language"]
    )

    # For every previous language map the winning cluster
    language_map: dict[int, int] = (
        language_to_cluster.reset_index()
        .rename(columns={"previous_language": "language", "candidate_language": "cluster"})
        .set_index("cluster")["language"]
        .to_dict()
    )

    # All other clusters get new language IDs
    all_clusters = set(current_step["candidate_language"].unique())
    unassigned = sorted(all_clusters - set(language_map.keys()))

    new_ids = range(max_language_id, max_language_id + len(unassigned))
    language_map.update(zip(unassigned, new_ids, strict=True))
    max_language_id += len(unassigned)

    # Assign language ids to dataframe
    current_step["language"] = current_step["candidate_language"].map(language_map).astype(int)  # type: ignore[arg-type]

    return current_step, max_language_id


def nncor_discrete(
    population: gpd.GeoDataFrame,
    k: int = 1,
) -> dict[str, float]:
    """Nearest-neighbour equality index for categorical marks.

    Returns unnormalised and normalised versions.
    """
    coords = np.column_stack(
        (
            population.geometry.x.to_numpy(dtype=np.float64),
            population.geometry.y.to_numpy(dtype=np.float64),
        ),
    )
    labels = population["language"].to_numpy(dtype=np.int_)
    tree = KDTree(coords)
    # k+1 because the point itself is included
    _dist, idx = tree.query(coords, k=k + 1)
    nearestneighbor_idx = idx[:, k]  # k-th nearest neighbour

    nearestneighbor_labels = labels[nearestneighbor_idx]

    # Unnormalised: P[M == M*]
    unnormalised = np.mean(labels == nearestneighbor_labels)

    # Null expectation: sum(p_c^2) = Simpson index
    props = np.unique(labels, return_counts=True)[1] / len(labels)
    expected = np.sum(props**2)

    # Normalised
    normalised = unnormalised / expected

    return {
        "unnormalised": float(unnormalised),
        "normalised": float(normalised),
        "simpson_baseline": float(expected),
    }


def diversify(
    directory: Path,
    distance_threshold: float,
    linkage: str,
    time_steps: int,
    radius: float,
    write_interval: int,
    intermediate_start: Path | None,
    intermediate_step: int | None,
) -> None:
    """Cluster languages per time step based on clustering previous time step."""
    # Dataframe of the population at the initial time step
    read_population = make_population_reader(directory, write_interval, time_steps)

    # First time_step: initialize the start languages
    if intermediate_start is not None and intermediate_step is not None:
        # Start at intermediate point after warm-up run
        population = gpd.read_file(intermediate_start)
        population_current = population[population["time_step"] == intermediate_step]
        languages = population_current["language"]
    else:
        # Initialize start languages
        population_current = read_population(0)
        languages = initialize_languages(
            population_current,
            distance_threshold,
            linkage,
        )
        population_current["language"] = languages.astype(int)

    population_current = cast("gpd.GeoDataFrame", population_current)
    max_language_id = cast("int", languages.max())

    # Calculate the spatial variation per language
    nncor = nncor_discrete(population_current)

    # Create a dataframe to store meta data on the language divergence and convergence
    meta_data = pd.DataFrame(
        {
            "time_step": [0],
            "divergences": [0],
            "convergences": [0],
            "language_number": [population_current["language"].nunique()],
            "speaker_numbers": [population_current.groupby("language")["id"].count().sort_index().to_numpy()],
            "nncor_unnormalized": nncor["unnormalised"],
            "nncor_normalized": nncor["normalised"],
            "simpson_baseline": nncor["simpson_baseline"],
        },
    )

    if "previous_language" not in population_current.columns:
        # Initialize previous language as integer type
        population_current["previous_language"] = -1

    # Initiate the total population data frame across all time steps
    population_total = [population_current]

    # Iterate over each time_step and check for splits
    for time_step in tqdm(range(1, time_steps + 1), desc="Processing time steps"):
        # Keep track of the language divergence and convergence
        divergence_counter = 0
        convergence_counter = 0

        population_previous = cast("gpd.GeoDataFrame", population_current)
        population_current = read_population(time_step)

        # Determine the candidate clusters formed after splitting
        population_current, divergence_counter = find_splitting_events(
            population_previous,
            population_current,
            distance_threshold,
            linkage,
            divergence_counter,
        )

        if np.any(population_current["candidate_language"] == -1):
            logger.error("Be careful! Candidate language not fully assigned after splitting")

        # Find all neighboring clusters within radius
        all_neighbors = find_neighboring_clusters(population_current, radius)

        # Merge clusters based on language similarity
        population_current, convergence_counter = find_merging_events(
            population_current,
            all_neighbors,
            distance_threshold,
            linkage,
            convergence_counter,
        )

        population_current, max_language_id = assign_languages(
            population_current,
            max_language_id,
        )

        if np.any(population_current["language"] == -1):
            logger.error("Be careful! Language is not assigned")

        # Remove candidate cluster column, as it is not needed anymore
        population_current = cast("gpd.GeoDataFrame", population_current.drop("candidate_language", axis=1))
        # Update population with new language assignments
        population_total.append(population_current)

        # Record meta data for this time step
        number_languages = population_current["language"].nunique()
        nncor = nncor_discrete(population_current)

        # 1D numpy array of speaker counts, ordered by language ID
        language_speakers = population_current.groupby("language")["id"].count().sort_index().to_numpy()

        meta_data.loc[len(meta_data)] = [
            int(time_step),
            int(divergence_counter),
            int(convergence_counter),
            number_languages,
            language_speakers,
            nncor["unnormalised"],
            nncor["normalised"],
            nncor["simpson_baseline"],
        ]

    # Save output to a single gpkg file
    pd.concat(population_total, ignore_index=True).to_file(
        directory / f"population_{linkage}.gpkg",
        driver="GPKG",
    )

    # Save meta data to csv with speaker numbers as string
    meta_data["speaker_numbers"] = meta_data["speaker_numbers"].apply(
        lambda x: np.array2string(x, separator=",", max_line_width=sys.maxsize, threshold=sys.maxsize).strip(
            "[]",
        ),
    )
    meta_data.to_csv(directory / "meta_data_cluster.csv", index=False)
