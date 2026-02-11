"""Feed-forward clustering of the language profiles into languages
following evolutionary diversification processes."""

from pathlib import Path
from tqdm import tqdm

import geopandas as gpd
import logging
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering


def read_geoparquet(
    directory: Path,
    file_pattern: str = "*.geoparquet",
) -> gpd.GeoDataFrame:
    """Read in multiple geoparquet files with time_steps in filenames."""
    # ~ (nitpicking): This approach is a bit brittle. If I copy a .geoparquet file in the directory with
    # unexpected content, it gets picked up here and will brake things.
    # Try to be as explicit as possible. Generalize only if needed. You know some things:
    # - The format of the file. No need for a file pattern.
    # - The time steps as well(?). If so, no need for globbing for files. Just open the files that should be
    #   there. Do you know the range (min / max time step)?

    # Find all matching files
    file_paths = list(directory.glob(file_pattern))

    dataframes = []

    for file in file_paths:
        gdf = gpd.read_parquet(file)

        # Extract time_step from filename
        filename = file.stem
        time_step = filename.removeprefix("output")
        gdf["time_step"] = int(time_step)

        dataframes.append(gdf)

    # ! (important): Code at the call-site of this function assumes dataframes are sorted by time step. This
    # would be a good place to assert this is indeed the case. You are implicitly assuming globbing returns paths in
    # order of time step. Is this assumption always true?
    # A comment about this postcondition would be good to add to the docstring. In general add information
    # about what is being returned. The type can be deduced from the type hint, but how is it organized?

    return pd.concat(dataframes, ignore_index=True)


def language_classification(
    language_profiles: np.ndarray[int],
    dist_threshold: float,
    linkage: str,
) -> np.ndarray[int]:
    """Group language profiles into languages based on distance threshold using hierarchical clustering."""
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=dist_threshold,  # Threshold for clustering
        metric="hamming",
        linkage=linkage,
    )  # Average linkage calculates over the mean of the distances between all points in the clusters

    return clustering.fit_predict(language_profiles)


def check_cluster_coherence(language_profiles: np.ndarray[int], dist_threshold) -> bool:
    """Check if maximum distance between language profiles in a cluster is within the distance threshold."""
    distances = pdist(language_profiles, metric="hamming")
    return np.all(distances <= dist_threshold)


def initialize_languages(
    start_population: gpd.GeoDataFrame,
    dist_threshold: float,
    linkage: str,
) -> np.ndarray[int]:
    """Initialize languages for the first time_step based on coherence."""
    language_profiles = np.stack(start_population["language_profile"])
    if len(start_population) == 1:
        return np.array([0])
    clusters = language_classification(language_profiles, dist_threshold, linkage)

    return clusters


def find_modal_profile(language_profiles: np.ndarray[int]) -> np.ndarray[int]:
    """Calculate the mode (most common value) for each categorical variable."""
    # For each column/feature, find the most common value
    return np.array(
        [np.bincount(language_profiles[:, i]).argmax() for i in range(language_profiles.shape[1])]
    )


def find_assigned_languages(new_step: gpd.GeoDataFrame) -> dict[int]:
    """Return list of currently assigned languages in the new step."""
    assigned_langs = (
        new_step.loc[new_step["language"].notna() & (new_step["language"] != -1), "language"]
        .astype(int)
        .unique()
        .tolist()
    )
    return assigned_langs


def get_neighboring_languages(
    new_step: gpd.GeoDataFrame,
    agent_idx_list: list[int],
    assigned_langs: list[int],
    radius: float,
) -> list[int]:
    """Find languages that are spoken within a certain radius around the language of interest."""

    # Create buffer around all agents speaking the language with a specified radius
    cluster_buffer = new_step.loc[agent_idx_list].geometry.unary_union.buffer(radius)

    # Only look at agents with already assigned languages
    assigned_mask = new_step["language"].isin(assigned_langs)
    candidates = new_step.loc[assigned_mask]

    if candidates.empty:
        return []

    intersecting = candidates[candidates.geometry.intersects(cluster_buffer)]

    return intersecting["language"].unique().tolist()


def meh():

    for idx in range(nr_cells):
        array3[idx] = array1[idx] + array2[idx]

    array1 = np.array()
    array2 = np.array()
    array3 = array1 + array2

    # Dataframe voor agents die dezelfde taal spreken. Attributes:
    # - intra_distance
    # - ...

    for language in old_languages:
        step1()

    for language in old_languages:
        step2()

    for language in old_languages:
        step3()

    ... = step1(...)
    for language in old_languages:
        ... step2(...)
    ... = step3(...)


def find_splitting_events(
    old_step: gpd.GeoDataFrame,
    new_step: gpd.GeoDataFrame,
    dist_threshold: float,
    linkage: str,
    similar: bool,
    max_language_id: int,
) -> tuple[dict, int]:
    """Determines the number of splits in previous language clusters for a certain time step"""
    # Clustering is based on the languages present in the previous time step
    old_languages = old_step["language"].unique()

    # Collect the new languages formed in this time step
    new_clusters = []

    # Loop through the languages of the previous time step
    # !: This is likely slow (but profile to proof it is a bottleneck). Can it be removed by calling functions
    # that operate on all agents at the same time? Think of turning each variable used in the for-loop's body
    # into an array / column. First create a test for this function, then create an alternative implementation
    # so you can verify that the new version behaves as the original. This also allows you to compare the
    # differences in performance (e.g. using Python's timeit package).
    for language in old_languages:
        # Get IDs of agents speaking this language in previous time step
        lang_old_agent_ids = old_step[old_step["language"] == language]["id"]
        # Get IDs of newborn agents that are born in the current time step
        # and whose parents spoke this language in previous time step
        lang_newborn_agent_ids = new_step[
            (~new_step["id"].isin(old_step["id"])) & (new_step["parent_id"].isin(lang_old_agent_ids))
        ]["id"]
        # Combine old and newborn agent IDs
        lang_agent_ids = pd.concat([lang_old_agent_ids, lang_newborn_agent_ids])

        # Generate mask of previous speakers in the new time step
        agent_mask = new_step["id"].isin(lang_agent_ids)

        if agent_mask.sum() == 0:
            # No agents remain from this language in the new time step: extinction
            continue
        if agent_mask.sum() == 1:
            # Only one agent remains, assign the language directly
            new_step.loc[agent_mask, "language"] = language
            continue

        # Extract language profiles of the previous speakers
        new_profiles = np.stack(new_step.loc[agent_mask, "language_profile"])
        # Check cluster coherence (max distance between any two profiles <= threshold)
        coherence = check_cluster_coherence(new_profiles, dist_threshold)

        if coherence is True:
            # All agents continue speaking the same language
            new_step.loc[agent_mask, "language"] = language
        else:
            # Cluster the current language
            # This could still be one cluster based on linkage
            clusters = language_classification(new_profiles, dist_threshold, linkage)
            # Get the indices in the dataframe in new_step that correspond to these agents
            agent_indices = new_step.index[agent_mask]
            # Find the number of new languages created and the counts of each language
            unique_labels, counts = np.unique(clusters, return_counts=True)

            if similar is True:
                # Find the cluster that is most similar to the original language to retain original ID
                # Calculate the modal profile from the speakers of the original language
                old_profiles = np.stack(old_step[old_step["id"].isin(lang_old_agent_ids)]["language_profile"])
                original_mode = find_modal_profile(old_profiles)

                cluster_modes = []
                for label in unique_labels:
                    # Calculate modal profiles for each new cluster
                    cluster_mask = clusters == label
                    modal = find_modal_profile(new_profiles[cluster_mask])
                    cluster_modes.append(modal)

                # Calculate Hamming distance to original profile mode
                distances = [np.sum(mode != original_mode) for mode in cluster_modes]
                # Returns first occurrence when there is a tie
                favorable_cluster = unique_labels[np.argmin(distances)]
            else:
                # Find the largest cluster to retain the original language ID
                favorable_cluster = unique_labels[np.argmax(counts)]

            for _i, label in enumerate(unique_labels):
                selected_idx = agent_indices[clusters == label]
                if label == favorable_cluster:
                    # Favorable cluster gets the original language ID assigned
                    new_step.loc[selected_idx, "language"] = int(language)
                else:
                    # Other clusters get temporary language IDs
                    max_language_id += 1
                    new_clusters.append([selected_idx.tolist(), int(max_language_id)])

    return new_clusters, max_language_id


def diversify(
    directory: Path,
    dist_threshold: float,
    linkage: str,
    radius: float,
    sensitivity: bool = False,
    similar: bool = True,
    merge: bool = False,
) -> int:
    """Cluster languages per time step based on clustering previous time step"""
    # Read the population data across all time_steps
    population = read_geoparquet(directory)
    # Initialize language column as -1 to keep track of unclassified
    population["language"] = -1
    # Keep track of the maximum language ID assigned
    max_language_id = 0

    # Create a dataframe to store data on the language shifts and merges
    meta_data = pd.DataFrame(
        {
            "time_step": [0],
            "shifts": [0],
            # "merges": [0], # For a later moment to potentially include merges
        }
    )

    # Iterate over each time_step and check for splits
    for time_step in tqdm(population["time_step"].unique()):
        # Keep track of the number of language shifts
        shift_counter = 0

        # !: The block within the if-statement can be moved to before the for-loop. More efficient and
        # clearer. Initialize the state of the model before simulating it forward through time.
        if time_step == 0:
            # First time_step: initialize the start languages
            clusters = initialize_languages(
                population[population["time_step"] == 0],
                dist_threshold,
                linkage,
            )
            population.loc[population["time_step"] == 0, "language"] = clusters.astype(int)
            max_language_id = clusters.max()
            continue

        # ~: Any previous step is an old step, but the previous step is a specific old step. Same for new vs
        # next. You can also think of a current vs next step. Current is what you have, next is what you are
        # busy with computing here. Previous would then be the one before current. You probably don't need to
        # use the work previous often in that case.

        # !: Your code relies on dataframes being ordered by time step. Otherwise the initial state of the
        # model would not be initialized. Still, you *search* for the data frames corresponding to the time
        # steps. Can't you just iterate forward over all data frames in population? Maybe population can be a
        # list of data frames?
        new_step = population[population["time_step"] == time_step]  # .copy()
        old_step = population[population["time_step"] == (time_step - 1)]

        # Determine the new clusters formed in this time step
        new_clusters, max_language_id = find_splitting_events(
            old_step, new_step, dist_threshold, linkage, similar, max_language_id
        )

        # Check if new clusters overlap in similarity with existing languages
        if merge is False:
            # Merge defines whether mixed languages can arise: a new language is formed out of two languages
            # if merge is set to false, only language shifts can take place to a previously assigned language
            assigned_langs = find_assigned_languages(new_step)

        # Loop through the new clusters that have not been assigned yet
        # !: Likely slow. See comment in find_assigned_languages().
        for agent_idx_list, label in new_clusters:
            if len(agent_idx_list) == 0:
                logging.debug(f"No agents in cluster {label} at time step {time_step}, skipping.")
                continue

            if merge is True:
                # New clusters can merge with newly assigned languages
                assigned_langs = find_assigned_languages(new_step)

            # Find neighboring languages within radius
            neighboring_languages = get_neighboring_languages(
                new_step,
                agent_idx_list,
                assigned_langs,
                radius,
            )

            # Get the language profiles of agents in the new cluster
            cluster_profiles = np.stack(new_step.loc[agent_idx_list, "language_profile"])

            # Check if any neighboring language clusters have a similarity below the distance threshold
            for neighbor_language in neighboring_languages:
                # Get the language profiles of agents speaking the neighboring language
                other_profiles = np.stack(
                    new_step[new_step["language"] == neighbor_language]["language_profile"]
                )

                # Combine all language profiles and check whether they form a coherent cluster
                combined_profiles = np.vstack([cluster_profiles, other_profiles])
                coherence = check_cluster_coherence(combined_profiles, dist_threshold)

                if coherence:
                    # Merge clusters by assigning the other language label
                    new_step.loc[agent_idx_list, "language"] = neighbor_language
                    logging.debug(
                        f"Merge cluster {label} to existing language {neighbor_language} at step {time_step}"
                    )
                    shift_counter += 1
                    break  # Exit after merging to avoid multiple merges

            # If agents in the new cluster have not been assigned a language yet, assign a new language ID
            language_values = new_step.loc[agent_idx_list, "language"]
            if (language_values.isna() | (language_values == -1)).all():
                logging.debug(f"Assigning new language {label} to cluster at time_step {time_step}")
                new_step.loc[agent_idx_list, "language"] = label

        # Update population with new language assignments
        population.loc[population["time_step"] == time_step, "language"] = new_step["language"].astype(int)

        # Record meta data for this time step
        meta_data.loc[len(meta_data)] = [time_step, shift_counter]

    # Save output to a single gpkg file
    population.to_file(directory / f"population_{linkage}.gpkg", driver="GPKG")

    # Save meta data to csv file
    meta_data.to_csv(directory / "meta_data_cluster.csv", index=False)

    if sensitivity:
        # Compute number of unique languages at the last time step
        last_step = int(population["time_step"].max())
        languages_last = population.loc[population["time_step"] == last_step, "language"].unique()
        logging.info(f"Last time step: {last_step}; number of languages: {len(languages_last)}")
        return len(languages_last)
