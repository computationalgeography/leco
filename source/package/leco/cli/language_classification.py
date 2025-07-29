import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.metrics import pairwise_distances
from pathlib import Path

# import re
import os
import matplotlib.pyplot as plt


def read_geoparquet(
    directory_path: str, file_pattern="*.geoparquet"
) -> gpd.GeoDataFrame:
    """Reads in multiple geoparquet files with timesteps in filenames"""
    directory = Path(directory_path)

    # Find all matching files
    file_paths = list(directory.glob(file_pattern))

    dataframes = []

    for file in file_paths:
        gdf = gpd.read_parquet(file)

        # Extract timestep from filename
        filename = file.stem
        # timestep = re.search(r"\d+", filename).group() # More flexible but higher computational cost
        timestep = filename.lstrip("output")
        gdf["timestep"] = int(timestep)

        dataframes.append(gdf)

    return pd.concat(dataframes, ignore_index=True)


def language_classification(
    language_profiles: np.ndarray[int],
    dist_threshold: float,
) -> np.ndarray[int]:
    """Group language profiles into languages based on similarity threshold following hierarchical clustering"""

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=dist_threshold,  # Threshold for clustering
        metric="hamming",
        linkage="average",
    )  # Average linkage calculates over the mean of the distances between all points in the clusters

    categories = clustering.fit_predict(language_profiles)

    return categories


def dbscan_classification(
    language_profiles: np.ndarray[int],
    distance_threshold: float,
) -> np.ndarray[int]:
    """Group language profiles into languages based on distance threshold following density-based clustering"""

    dist = pairwise_distances(language_profiles, metric="hamming")
    print(dist)
    dbscan = DBSCAN(eps=0.2, min_samples=1, metric="precomputed")
    labels = dbscan.fit_predict(dist)
    print(labels)
    """     clustering = DBSCAN(eps=distance_threshold, min_samples=2, metric="hamming")

    categories = clustering.fit(language_profiles)
    print(categories) """

    return labels


def thresholded_clustering(
    language_profiles: np.ndarray, threshold: float
) -> np.ndarray:
    n = language_profiles.shape[0]
    dist = pairwise_distances(language_profiles, metric="hamming")
    labels = -np.ones(n, dtype=int)
    cluster_id = 0

    for i in range(n):
        if labels[i] == -1:
            # Find all unassigned points within threshold to point i
            similar = (dist[i] <= threshold) & (labels == -1)
            labels[similar] = cluster_id
            cluster_id += 1

    return labels


def create_3d_fig(population: gpd.GeoDataFrame):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(
        population.geometry.x,
        population.geometry.y,
        population.timestep,
        c=population.language,
        s=1,
    )

    # Add lines to visualize the evolution of individual agents over time
    for pid, group in population.groupby("id"):
        # Sort by timestep to ensure correct line plotting
        group = group.sort_values("timestep")
        ax.plot(
            group.geometry.x,
            group.geometry.y,
            group.timestep,
            color="gray",  # or set color by some attribute
            linewidth=0.5,
            alpha=0.5,
        )

    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")
    ax.set_zlabel("Timestep")

    plt.show()


def run_classification(input_path: str, dist_threshold: float) -> None:
    """Run the LECo model of language evolution"""

    # Read in the population data across all timesteps
    population = read_geoparquet(input_path)

    # Cluster the language profiles into languages based on the distance threshold
    population["language"] = thresholded_clustering(
        np.stack(population["language_profile"]), dist_threshold
    )

    # Save output to a single gpkg file
    population.to_file(os.path.join(input_path, "population.gpkg"), driver="GPKG")

    create_3d_fig(population)
