"""Cluster language profiles into languages in 3d across time."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering


def read_geoparquet(
    directory: Path,
    file_pattern: str = "*.geoparquet",
) -> gpd.GeoDataFrame:
    """Read in multiple geoparquet files with time steps in filenames."""
    # Find all matching files
    file_paths = list(directory.glob(file_pattern))

    dataframes = []

    for file in file_paths:
        gdf = gpd.read_parquet(file)

        # Extract time step from filename
        filename = file.stem
        time_step = filename.removeprefix("output")
        gdf["time_step"] = int(time_step)

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
        linkage="complete",
    )  # Average linkage calculates over the mean of the distances between all points in the clusters

    return clustering.fit_predict(language_profiles)


def dynamic_clustering(population: gpd.GeoDataFrame, dist_threshold: float) -> gpd.GeoDataFrame:
    """Cluster language profiles into languages for each time step separately."""
    clustered_dfs = []

    for _time_step, step_data in population.groupby("time_step"):
        language_profiles = np.stack(step_data["language_profile"])
        step_data["language"] = language_classification(language_profiles, dist_threshold)

    return pd.concat(clustered_dfs, ignore_index=True)


def classify_all(directory: Path, dist_threshold: float) -> None:
    """Run the LECo model of language evolution."""
    # Read in the population data across all time steps
    population = read_geoparquet(directory)

    # Cluster the language profiles into languages based on the distance threshold
    population["language"] = language_classification(
        np.stack(population["language_profile"]),
        dist_threshold,
    )

    # Save output to a single gpkg file
    # Pass a CRS to get rid of the warning
    population.to_file(directory / "population.gpkg", driver="GPKG")
