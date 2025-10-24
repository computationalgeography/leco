"""Cluster language profiles into languages per timestep."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering


def read_geoparquet(
    directory_path: str,
    file_pattern: str = "*.geoparquet",
) -> gpd.GeoDataFrame:
    """Read in multiple geoparquet files with timesteps in filenames."""
    directory = Path(directory_path)

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


def analyze_cluster_transitions(population: gpd.GeoDataFrame) -> dict[tuple[int, int], int]:
    """Track how agents move between language clusters over time."""
    transitions = {}

    for _agent_id, agent_data in population.groupby("id"):
        # Get sequential pairs of timesteps
        timesteps = sorted(agent_data["timestep"].unique())
        for t1, t2 in iter.tools.pairwise(timesteps[:-1], timesteps[1:]):
            cluster1 = agent_data[agent_data["timestep"] == t1]["language"].iloc[0]
            cluster2 = agent_data[agent_data["timestep"] == t2]["language"].iloc[0]

            transition = (cluster1, cluster2)
            transitions[transition] = transitions.get(transition, 0) + 1

    return transitions


def run_classification(input_path: str, dist_threshold: float) -> None:
    """Run the LECo model of language evolution."""
    # Read in the population data across all timesteps
    population = read_geoparquet(input_path)

    for _timestep, stepdata in population.groupby("timestep"):
        language_profiles = np.stack(stepdata["language_profile"])
        stepdata["language"] = language_classification(language_profiles, dist_threshold)
        population.loc[stepdata.index, "language"] = stepdata["language"]

    transitions = analyze_cluster_transitions(population)
    print("Language cluster transitions between timesteps:")
    for (from_cluster, to_cluster), count in transitions.items():
        print(f"From {from_cluster} to {to_cluster}: {count} agents")

    # Save output to a single gpkg file
    population.to_file(Path(input_path) / "population.gpkg", driver="GPKG")
