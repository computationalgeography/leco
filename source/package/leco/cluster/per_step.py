"""Clusters per step"""

from pathlib import Path
from tqdm import tqdm

import geopandas as gpd
import logging
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering


def read_geoparquet(
    directory: Path,
    file_pattern: str = "*.geoparquet",
) -> gpd.GeoDataFrame:
    """Read in multiple geoparquet files with time_steps in filenames."""
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


def diversify(
    directory: Path,
    dist_threshold: float,
    linkage: str,
    radius: float,
    sensitivity: bool = False,
    similar: bool = True,
    merge: bool = False,
) -> int:
    """ "Cluster languages per time step based on clustering previous time step"""
    # Read the population data across all time_steps
    population = read_geoparquet(directory)
    # Initialize language column as -1 to keep track of unclassified
    population["language"] = -1
    # Keep track of the maximum language ID assigned

    # Iterate over each time_step and check for splits
    for time_step in tqdm(population["time_step"].unique()):
        population_step = population[population["time_step"] == time_step]
        profiles = np.stack(population_step["language_profile"])
        clusters = language_classification(profiles, dist_threshold, linkage)
        # Update population with new language assignments
        population.loc[population["time_step"] == time_step, "language"] = clusters.astype(int)

    # Save output to a single gpkg file
    population.to_file(directory / f"population_step_{linkage}.gpkg", driver="GPKG")

    if sensitivity:
        # Compute number of unique languages at the last time step
        last_step = int(population["time_step"].max())
        languages_last = population.loc[population["time_step"] == last_step, "language"].unique()
        logging.info(f"Last time step: {last_step}; number of languages: {len(languages_last)}")
        return len(languages_last)
