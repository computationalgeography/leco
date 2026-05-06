"""Functions to create summary data of the leco model output for different parameter combinations."""

import argparse
import re
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

PARAMETER_NAMES = [
    "speed",
    "mutation_rate",
    "radius",
    "diffusion_rate",
    "similarity_preference",
]


def parse_path_metadata(filepath: Path) -> tuple[str, dict]:
    """Extract all parameter values and seed from a nested path.

    Path looks like:
    speed_0.5/mutation_rate_0.01/radius_30.0/diffusion_rate_0.1/similarity_preference_0.8/seed_42/population.gpkg

    It returns:
        seed: the seed value as string
        params: dict of all parameter values
    """
    filepath_str = str(filepath)
    params = {}

    for key in PARAMETER_NAMES:
        match = re.search(rf"{key}_(-?\d+(?:\.\d+)?)", filepath_str)
        if match:
            params[key] = match.group(1)
        else:
            raise ValueError(f"Could not parse '{key}' from {filepath}")

    seed_match = re.search(r"seed_(\d+)", filepath_str)
    if not seed_match:
        raise ValueError(f"Could not parse seed from {filepath}")

    return seed_match.group(1), params


def organize_by_params_and_seed(
    populations: list[gpd.GeoDataFrame],
    gpkg_filepaths: list[Path],
    metadata_files: list[pd.DataFrame],
) -> dict:
    """Organize GeoDataFrames by parameter combination and seed.

    Returns nested dictionary: {param_combo_tuple: {seed: (gdf, metadata)}}
    with param_combo_tuple = (speed, mutation_rate, radius, diffusion_rate, similarity_preference)
    """
    scenario_dict = defaultdict(dict)

    for gdf, filepath, metadata in zip(populations, gpkg_filepaths, metadata_files, strict=True):
        seed, params = parse_path_metadata(filepath)
        param_key = tuple(params[p] for p in PARAMETER_NAMES)
        scenario_dict[param_key][seed] = (gdf, metadata)

    return dict(scenario_dict)


def organize_meta_by_params_and_seed(
    csv_filepaths: list[Path],
    metadata_files: list[pd.DataFrame],
    metacluster_files: list[pd.DataFrame],
) -> dict:
    """Organize meta data by parameter combination and seed.

    Returns nested dictionary: {param_combo_tuple: {seed: (gdf, metadata)}}
    with param_combo_tuple = (speed, mutation_rate, radius, diffusion_rate, similarity_preference)
    """
    scenario_dict = defaultdict(dict)

    for filepath, metadata, metacluster in zip(csv_filepaths, metadata_files, metacluster_files, strict=True):
        seed, params = parse_path_metadata(filepath)
        param_key = tuple(params[p] for p in PARAMETER_NAMES)
        scenario_dict[param_key][seed] = (metadata, metacluster)

    return dict(scenario_dict)


def write_data(
    organized_data: dict,
    step_to_years: int,
) -> pd.DataFrame:
    """Calculate summary statistics across all parameter combinations and seeds."""
    # Get the first actual dataframe to extract time steps
    first_param_combo = next(iter(organized_data.values()))
    first_seed = next(iter(first_param_combo.values()))
    first_meta_data = first_seed[1]

    time_steps = sorted(first_meta_data["time_step"].unique())
    time_steps_years = (np.array(time_steps) * step_to_years).tolist()
    last_10_steps = time_steps[-10:]

    results = []

    for param_key, seeds_dict in organized_data.items():
        # Unpack parameter combination for labelling
        param_dict = dict(zip(PARAMETER_NAMES, param_key, strict=True))
        print(f"Processing params: {param_dict}")

        for seed, (gdf, metadata) in seeds_dict.items():
            print(f"  seed={seed}")

            number_languages = gdf.groupby("time_step")["language"].nunique().sort_index().tolist()

            language_speakers = (
                gdf.groupby(["time_step", "language"])["id"]
                .count()
                .reset_index(name="speaker_count")
                .pivot_table(
                    index="time_step",
                    columns="language",
                    values="speaker_count",
                    fill_value=0,
                )
                .sort_index()
            )

            external_changes = metadata["external_change"]
            internal_changes = metadata["internal_change"]

            # Compute speaker array only for last 10 time steps
            speakers_last_10 = {}
            for step in last_10_steps:
                speakers_at_step = language_speakers.loc[step]
                nonzero = speakers_at_step[speakers_at_step > 0]
                speakers_last_10[step] = sorted(nonzero.to_numpy(), reverse=True)

            for i, year in enumerate(time_steps_years):
                speakers_at_timestep = language_speakers.loc[time_steps[i]]
                nonzero_speakers = speakers_at_timestep[speakers_at_timestep > 0]

                row = {
                    "seed": seed,
                    **param_dict,
                    "year": year,
                    "language_count": number_languages[i],
                    "internal_change": internal_changes[i],
                    "external_change": external_changes[i],
                    "speaker_mean": np.mean(nonzero_speakers),
                    "speaker_min": np.min(nonzero_speakers),
                    "speaker_max": np.max(nonzero_speakers),
                }

                if time_steps[i] in last_10_steps:
                    row["speakers_per_language"] = np.array(speakers_last_10[time_steps[i]])
                else:
                    row["speakers_per_language"] = None

                results.append(row)

    return pd.DataFrame(results)


def write_meta_data(
    organized_data: dict,
    step_to_years: int,
    expected_seed_count: int = 5,
) -> pd.DataFrame:
    """Calculate summary statistics across all parameter combinations and seeds."""
    # Get the first actual dataframe to extract time steps
    first_param_combo = next(iter(organized_data.values()))
    first_seed = next(iter(first_param_combo.values()))
    first_meta_data = first_seed[0]

    time_steps = sorted(first_meta_data["time_step"].unique())
    time_steps_years = (np.array(time_steps) * step_to_years).tolist()

    results = []

    for param_key, seeds_dict in organized_data.items():
        # Unpack parameter combination for labelling
        param_dict = dict(zip(PARAMETER_NAMES, param_key, strict=True))

        # Check if every parameter combination has indeed the expected number of seeds
        if len(seeds_dict) != expected_seed_count:
            raise ValueError(
                f"Expected {expected_seed_count} seeds for {param_dict}, "
                f"but found {len(seeds_dict)}: {list(seeds_dict.keys())}",
            )

        for seed, (metadata, metacluster) in seeds_dict.items():
            external_changes = metadata["external_change"]
            internal_changes = metadata["internal_change"]
            number_languages = metacluster["language_number"]
            language_speakers = metacluster["speaker_numbers"]

            for i, year in enumerate(time_steps_years):
                """ if year < 1000:
                    continue """

                row = {
                    "seed": seed,
                    **param_dict,
                    "year": year,
                    "language_count": number_languages[i],
                    "internal_change": internal_changes[i],
                    "external_change": external_changes[i],
                    "speakers": language_speakers[i].tolist(),
                    "speaker_mean": float(language_speakers[i].mean()),
                    "speaker_min": int(language_speakers[i].min()),
                    "speaker_max": int(language_speakers[i].max()),
                }

                results.append(row)

    return pd.DataFrame(results)


def extract_files(gpkg_paths: list[Path]) -> tuple[list[gpd.GeoDataFrame], list[pd.DataFrame]]:
    """Extract gpkg files and the metadata files from the same run."""
    populations = []
    metadatas = []

    for gpkg_path in gpkg_paths:
        population = gpd.read_file(gpkg_path)
        population["language_profile"] = population["language_profile"].apply(
            lambda x: np.fromstring(str(x).strip("[]"), sep=" ", dtype=int),
        )
        populations.append(population)

        metadata_path = gpkg_path.parent / "meta_data.csv"
        if not metadata_path.exists():
            raise FileNotFoundError(f"No metadata file found in {gpkg_path.parent}")

        metadatas.append(pd.read_csv(metadata_path))

    return populations, metadatas


def extract_meta_files(csv_paths: list[Path]) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    """Extract the metadata files."""
    metadatas = []
    metaclusters = []

    for csv_path in csv_paths:
        metadata_path = csv_path.parent / "meta_data.csv"
        if not metadata_path.exists():
            raise FileNotFoundError(f"No metadata file found in {csv_path.parent}")

        metadatas.append(pd.read_csv(metadata_path))

        metacluster_path = csv_path.parent / "meta_data_cluster.csv"
        if not metacluster_path.exists():
            raise FileNotFoundError(f"No metadata file found in {csv_path.parent}")

        metacluster = pd.read_csv(metacluster_path).copy()
        """ metacluster["speaker_numbers"] = metacluster["speaker_numbers"].apply(
            lambda x: np.fromstring(str(x).strip("[]"), sep=" ", dtype=int),
        ) """
        metacluster["speaker_numbers"] = metacluster["speaker_numbers"].apply(
            lambda x: np.array(
                [v for v in str(x).strip("[]").replace(",", " ").split() if v != "..."],
                dtype=int,
            ),
        )

        metaclusters.append(metacluster)

    return metadatas, metaclusters


def process_all(directory_path: Path) -> None:
    """Create summarizing data of the leco model output for the nested parameter structure."""
    # Find all gpkg files in the base path
    cluster_file_paths = list(directory_path.rglob("meta_data_cluster.csv"))

    # Extract gpkg files and metadata files
    metadata, metaclusters = extract_meta_files(cluster_file_paths)

    organized_data = organize_meta_by_params_and_seed(cluster_file_paths, metadata, metaclusters)

    step_to_years = 20
    stats_df = write_meta_data(organized_data, step_to_years)
    print(stats_df)

    output_path = directory_path / "spawn_combined_stats.csv"
    stats_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create summary statistics from population output files.",
    )
    parser.add_argument(
        "directory_path",
        nargs="?",
        help="Base directory that contains nested population*.gpkg files.",
    )
    args = parser.parse_args()
    process_all(Path(args.directory_path))
