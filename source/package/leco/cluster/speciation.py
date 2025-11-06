"""Cluster language profiles into languages per timestep."""

### IT CREATES FEWER LANGUAGES THAN 3D CLUSTERING
### CHECK IF THIS IS BECAUSE MERGED LANGUAGES ARE MISSING OR BECAUSE OF GRADUAL CHANGE

from pathlib import Path

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
    """Read in multiple geoparquet files with timesteps in filenames."""
    # Find all matching files
    file_paths = list(directory.glob(file_pattern))

    dataframes = []

    for file in file_paths:
        gdf = gpd.read_parquet(file)

        # Extract timestep from filename
        filename = file.stem
        timestep = filename.removeprefix("output")
        gdf["timestep"] = int(timestep)

        dataframes.append(gdf)

    return pd.concat(dataframes, ignore_index=True)


def language_classification(
    language_profiles: np.ndarray[int],
    dist_threshold: float,
) -> np.ndarray[int]:
    """Group language profiles into languages based on distance threshold using hierarchical clustering."""
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=dist_threshold,  # Threshold for clustering
        metric="hamming",
        linkage="average",
    )  # Average linkage calculates over the mean of the distances between all points in the clusters

    return clustering.fit_predict(language_profiles)


def check_cluster_coherence(language_profiles: np.ndarray[int], dist_threshold) -> bool:
    """Check if maximum distance between language profiles in a cluster is within the distance threshold."""
    distances = pdist(language_profiles, metric="hamming")
    return np.all(distances <= dist_threshold)


def first_initialization(start_population: gpd.GeoDataFrame, dist_threshold: float) -> np.ndarray[int]:
    """Initialize languages for the first timestep based on coherence."""
    language_profiles = np.stack(start_population["language_profile"])
    clusters = language_classification(language_profiles, dist_threshold)

    return clusters


def split_cluster_agglomerative(language_profiles: np.ndarray, dist_threshold):
    """Split using agglomerative clustering with distance threshold."""

    clustering = AgglomerativeClustering(
        n_clusters=None, distance_threshold=dist_threshold, linkage="complete", metric="hamming"
    )
    labels = clustering.fit_predict(language_profiles)

    return labels


def speciate(directory: Path, dist_threshold: float, stepsize: int = 1) -> None:
    """Run the LECo model of language evolution."""
    # Read in the population data across all timesteps
    population = read_geoparquet(directory)
    # Initialize language column in integer type
    population["language"] = -1
    max_lang = 0

    for timestep in population["timestep"].unique():
        if timestep == 0:
            clusters = first_initialization(population[population["timestep"] == 0], dist_threshold)
            population.loc[population["timestep"] == 0, "language"] = clusters.astype(int)
            max_lang = clusters.max()
            continue

        new_step = population[population["timestep"] == timestep]
        old_step = population[population["timestep"] == (timestep - 1)]
        # max_parent_id = old_step["id"].max() if not old_step.empty else 0
        languages = old_step["language"].unique()
        for lang in languages:
            # Get IDs of agents speaking this language in previous timestep
            lang_old_agent_ids = old_step[old_step["language"] == lang]["id"]
            # Get IDs of newborn agents whose parents spoke this language in previous timestep
            lang_newborn_agent_ids = new_step[new_step["parent_id"].isin(lang_old_agent_ids)]["id"]
            ### AND THEY ONLY HAVE TO SEARCH FROM MAX_ID ONWARDS
            lang_agent_ids = pd.concat([lang_old_agent_ids, lang_newborn_agent_ids])

            # Generate mask of agents of interest in the new timestep
            agent_mask = new_step["id"].isin(lang_agent_ids)

            if agent_mask.sum() == 0:
                # No agents remain from this language in the new timestep
                continue
            if agent_mask.sum() == 1:
                # Only one agent remains
                new_step.loc[agent_mask, "language"] = lang
                continue

            # Extract language profiles of the agents of interest
            new_profiles = np.stack(new_step.loc[agent_mask, "language_profile"])
            # Check cluster coherence (max distance between any two profiles <= threshold)
            coherence = check_cluster_coherence(new_profiles, dist_threshold)

            if coherence:
                # All agents continue speaking the same language
                new_step.loc[agent_mask, "language"] = lang
            else:
                # clusters = KMeans(n_clusters=2, random_state=0).fit_predict(new_profiles)
                clusters = split_cluster_agglomerative(new_profiles, dist_threshold)
                # Get the indices in the dataframe in new_step that correspond to these agents
                agent_indices = new_step.index[agent_mask]  ## WHY DO I DO THIS

                # Find the number of new languages created and the counts of each language
                nr_new_languages, counts = np.unique(clusters, return_counts=True)
                # Find the largest cluster to retain the original language ID
                # largest_cluster = np.argmax(counts)

                logging.debug(
                    f"Timestep {timestep}, Lang {lang}: {agent_mask.sum()} agents split into {len(nr_new_languages)} clusters"
                )
                logging.debug(
                    f"  Current max_lang: {max_lang}, will create {len(nr_new_languages) - 1} new languages"
                )

                ### OR: assign old language ID to largest cluster. Check if smaller clusters are coherent with neighboring languages clusters.
                ### But then these clusters have to be final...

                for i, label in enumerate(nr_new_languages):
                    selected_idx = agent_indices[clusters == label]
                    logging.debug(selected_idx)
                    if i == 0:
                        # First cluster keeps original language
                        new_step.loc[selected_idx, "language"] = int(lang)
                    else:
                        # Subsequent clusters get new language IDs
                        ### or keep these seperated and only once you've been through all old languages
                        ### you try to cluster these again to the bigger ones / together
                        ### but is together valid? merging? --> maybe in a dialect continuum it is.
                        ### and then assign? maybe split these two steps
                        max_lang += 1  # Increment BEFORE assigning
                        new_step.loc[selected_idx, "language"] = int(max_lang)
                        logging.debug(max_lang)

        # Update population with new language assignments
        population.loc[population["timestep"] == timestep, "language"] = new_step["language"].astype(int)

    # Save output to a single gpkg file
    population.to_file(directory / "populationspeciationTest.gpkg", driver="GPKG")
