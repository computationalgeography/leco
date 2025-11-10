"""Cluster language profiles into languages per time_step."""

from pathlib import Path

import geopandas as gpd
import numpy as np
from sklearn.cluster import AgglomerativeClustering


def read_specific_geoparquet(
    directory: Path,
    time_step: int,
    file_pattern: str = "*.geoparquet",
) -> gpd.GeoDataFrame:
    """Read in multiple geoparquet files with time_steps in filenames."""
    # Find all matching files
    file_path = directory / f"output{time_step}.geoparquet"

    gdf = gpd.read_parquet(file_path)

    gdf["time_step"] = int(time_step)

    return gdf


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


def analyze_cluster_transitions(population: gpd.GeoDataFrame) -> dict[tuple[int, int], int]:
    """Track how agents move between language clusters over time."""
    transitions = {}

    for _agent_id, agent_data in population.groupby("id"):
        # Get sequential pairs of time_steps
        time_steps = sorted(agent_data["time_step"].unique())
        for t1, t2 in iter.tools.pairwise(time_steps[:-1], time_steps[1:]):
            cluster1 = agent_data[agent_data["time_step"] == t1]["language"].iloc[0]
            cluster2 = agent_data[agent_data["time_step"] == t2]["language"].iloc[0]

            transition = (cluster1, cluster2)
            transitions[transition] = transitions.get(transition, 0) + 1

    return transitions


def classify_single(directory: Path, dist_threshold: float) -> None:
    """Run the LECo model of language evolution."""
    # Read in the population data across all time_steps
    time_step = 100
    population = read_specific_geoparquet(directory, time_step)

    clustering = language_classification(np.stack(population["language_profile"]), dist_threshold)
    print(np.unique(clustering).size)
    """for _time_step, stepdata in population.groupby("time_step"):
        language_profiles = np.stack(stepdata["language_profile"])
        stepdata["language"] = language_classification(language_profiles, dist_threshold)
        population.loc[stepdata.index, "language"] = stepdata["language"]

    transitions = analyze_cluster_transitions(population)
    print("Language cluster transitions between time_steps:")
    for (from_cluster, to_cluster), count in transitions.items():
        print(f"From {from_cluster} to {to_cluster}: {count} agents")

    # Save output to a single gpkg file
    population.to_file(directory / "population.gpkg", driver="GPKG")"""
