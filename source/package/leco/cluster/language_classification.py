import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.cluster import AgglomerativeClustering
from pathlib import Path

# import re
import os


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


def run_classification(input_path: str, dist_threshold: float) -> None:
    """Run the LECo model of language evolution"""

    # Read in the population data across all timesteps
    population = read_geoparquet(input_path)

    # Cluster the language profiles into languages based on the distance threshold
    population["language"] = language_classification(
        np.stack(population["language_profile"]), dist_threshold
    )

    # Save output to a single gpkg file
    population.to_file(os.path.join(input_path, "population.gpkg"), driver="GPKG")
