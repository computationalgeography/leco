"""Functions to create summary plots of leco model output."""

import re
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


def parse_filename(filepath: Path, parameter_name: str, seed: bool = False) -> tuple[int, str]:
    """Extract seed or run number from filename like 'seed_42_run_0001'."""
    if seed:
        match = re.search(r"seed_(\d+)", str(filepath))
    else:
        # Optionally searches for decimals
        match = re.search(rf"/holythree_(\d+)/{parameter_name}_(\d+(?:\.\d+)?)", str(filepath))

    if match:
        return int(match.group(1)), str(match.group(2))
    raise ValueError(f"Could not parse seed nor run from {filepath}")


def organize_by_run(
    populations: list[gpd.GeoDataFrame],
    gpkg_filepaths: list[Path],
    metadata_files: list[pd.DataFrame],
    parameter_name: str,
) -> dict:
    """Organize GeoDataFrames by run and seed for one scenario.

    Returns nested dict: {run: {seed: gdf}}
    """
    scenario_dict = defaultdict(dict)

    for gdf, filepath, metadata in zip(populations, gpkg_filepaths, metadata_files):
        seed, run = parse_filename(filepath, parameter_name)
        scenario_dict[seed][run] = gdf, metadata

    return dict(scenario_dict)


def organize_languages(
    languages: list[pd.Series],
    gpkg_filepaths: list[Path],
    metadata_files: list[pd.DataFrame],
    parameter_name: str,
) -> dict:
    """Organize GeoDataFrames by run and seed for one scenario.

    Returns nested dict: {run: {seed: gdf}}
    """
    scenario_dict = defaultdict(dict)

    for series, filepath, metadata in zip(languages, gpkg_filepaths, metadata_files):
        seed, run = parse_filename(filepath, parameter_name)
        scenario_dict[seed][run] = series, metadata

    return dict(scenario_dict)


def write_data(
    organized_data: dict,
    step_to_years: int,
) -> pd.DataFrame:
    """Calculates only language number"""
    # Get the first actual dataframe to extract time steps
    first_run = next(iter(organized_data.values()))
    first_seed = next(iter(first_run.values()))
    # Pick the meta data file, because it is smaller
    first_meta_data = first_seed[1]

    time_steps = sorted(first_meta_data["time_step"].unique())
    time_steps_years = (np.array(time_steps) * step_to_years).tolist()

    results = []

    # Iterate through the structure
    for seed, runs_dict in organized_data.items():
        print(seed)

        # for run, df in organized_data.items():
        for run, (gdf, metadata) in runs_dict.items():
            print(run)

            number_languages = gdf.groupby("time_step")["language"].nunique().sort_index().tolist()

            # Creates dataframe with languages as columns and time steps as rows
            language_speakers = (
                gdf.groupby(["time_step", "language"])["id"].count().unstack(fill_value=0).sort_index()
            )

            external_changes = metadata["external_change"]
            internal_changes = metadata["internal_change"]

            for i, year in enumerate(time_steps_years):
                # Get speaker counts for this time step (row i)
                speakers_at_timestep = language_speakers.loc[time_steps[i]]
                # Only sample the languages with speakers
                nonzero_speakers = speakers_at_timestep[speakers_at_timestep > 0]
                results.append(
                    {
                        "seed": seed,
                        "variable_value": run,
                        "year": year,
                        "language_count": number_languages[i],
                        "internal_change": internal_changes[i],
                        "external_change": external_changes[i],
                        "speaker_mean": np.mean(nonzero_speakers),
                        "speaker_min": np.min(nonzero_speakers),
                        "speaker_max": np.max(nonzero_speakers),
                    },
                )

    return pd.DataFrame(results)


def extract_files(input_paths: list[Path], type_file: str) -> list[gpd.GeoDataFrame]:
    runs = []
    for input in input_paths:
        if type_file == "gpkg":
            population = gpd.read_file(input)
            # Convert language_profile from string to numpy array
            population["language_profile"] = population["language_profile"].apply(
                lambda x: np.fromstring(str(x).strip("[]"), sep=" ", dtype=int),
            )
            runs.append(population)
        if type_file == "metadata":
            metadata = pd.read_csv(input)
            runs.append(metadata)

    return runs


def extract_files_matched(gpkg_paths: list[Path]) -> tuple[list[gpd.GeoDataFrame], list[pd.DataFrame]]:
    """Extract gpkg files and the metadata files from the same run."""
    populations = []
    metadatas = []

    for gpkg_path in gpkg_paths:
        # Read gpkg file
        population = gpd.read_file(gpkg_path)
        population["language_profile"] = population["language_profile"].apply(
            lambda x: np.fromstring(str(x).strip("[]"), sep=" ", dtype=int),
        )
        populations.append(population)

        # Find metadata.csv file in the same folder
        metadata_path = gpkg_path.parent / "meta_data.csv"

        if not metadata_path.exists():
            raise FileNotFoundError(f"No metadata file found in {gpkg_path.parent}")

        metadata = pd.read_csv(metadata_path)
        metadatas.append(metadata)

    return populations, metadatas


def plot_combined(parameter_name: str) -> None:
    """Create summarizing plots of the leco model output."""
    # Read in the population data across all time steps

    # parameter_name = "diffusion_rate"

    path = Path(f"/scratch/posma002/lecoOutput/morris_ff_average/sensitivity_axelrod/{parameter_name}/")
    gpkg_file_paths = list(path.rglob("*.gpkg"))
    populations, metadata = extract_files_matched(gpkg_file_paths)

    if len(metadata) != len(populations):
        print("BE CAREFUL, FILES NOT RECOGNIZED!")

    output_path = path.parent
    organized_data = organize_by_run(
        populations,
        gpkg_file_paths,
        metadata,
        parameter_name,
    )
    # organized data is dictionary ordered by seed, then run and holds gpkg dataframe and meta data dataframe

    # Then calculate statistics
    step_to_years = 20
    stats_df = write_data(organized_data, step_to_years)
    print(stats_df)

    # Save to CSV
    stats_df.to_csv(
        output_path / f"morris_axel_{parameter_name}.csv",
        index=False,
    )


def loop_through_parameters() -> None:
    parameter_names = [
        "speed",
        "mutation_rate",
        "radius",
        "diffusion_rate",
        "similarity_preference",
    ]

    for parameter in parameter_names:
        plot_combined(parameter)


if __name__ == "__main__":
    loop_through_parameters()
