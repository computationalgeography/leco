"""Functions to create summary data of the leco model output for different cluster methods."""

import argparse
import re
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


def parse_seed(filepath: Path) -> str:
    """Extract seed value from nested path (e.g. seed_42)."""
    seed_match = re.search(r"seed_(\d+)", str(filepath))
    if not seed_match:
        raise ValueError(f"Could not parse seed from {filepath}")
    return seed_match.group(1)


def infer_population_variant(filepath: Path) -> str:
    """Extract population file variant from filenames like population_average_0.2.gpkg."""
    stem = filepath.stem
    match = re.fullmatch(r"population(?:_(.+))?", stem)
    if not match:
        return stem

    variant = match.group(1)
    return variant or "default"


def organize_by_params_and_seed(
    populations: list[gpd.GeoDataFrame],
    gpkg_filepaths: list[Path],
    population_variants: list[str],
) -> dict:
    """Organize GeoDataFrames by seed and population variant.

    Returns nested dictionary: {seed: {variant: gdf}}.
    """
    scenario_dict = defaultdict(dict)

    for gdf, filepath, variant in zip(populations, gpkg_filepaths, population_variants, strict=True):
        seed = parse_seed(filepath)
        scenario_dict[seed][variant] = gdf

    return dict(scenario_dict)


def write_data(
    organized_data: dict,
    step_to_years: int,
) -> pd.DataFrame:
    """Calculate summary statistics across all parameter combinations and seeds."""
    results = []

    for seed, variants_dict in organized_data.items():
        print(f"Processing seed={seed}")
        for variant, gdf in variants_dict.items():
            print(f"  variant={variant}")

            time_steps = sorted(gdf["time_step"].unique())
            time_steps_years = (np.array(time_steps) * step_to_years).tolist()
            last_10_steps = set(time_steps[-10:])

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
                    "population_variant": variant,
                    "year": year,
                    "language_count": number_languages[i],
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


def process_all(directory_path: Path) -> None:
    """Create summarizing data of the leco model output for the nested parameter structure."""
    # Find all clustered population gpkg files in the base path
    cluster_file_paths = [p for p in directory_path.rglob("*.gpkg") if p.name.startswith("population")]
    if not cluster_file_paths:
        raise FileNotFoundError(f"No population*.gpkg files found in {directory_path}")

    populations = [gpd.read_file(path) for path in cluster_file_paths]
    print(f"pops {len(populations)}")
    variants = [infer_population_variant(path) for path in cluster_file_paths]
    print(f"variants: {variants}")
    organized_data = organize_by_params_and_seed(populations, cluster_file_paths, variants)

    print(f"organized data{organized_data}")

    step_to_years = 20
    stats_df = write_data(organized_data, step_to_years)
    print(stats_df)

    output_path = directory_path / "spawn_cluster_stats.csv"
    stats_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create summary statistics from differently clustered population output files.",
    )
    parser.add_argument(
        "directory_path",
        nargs="?",
        help="Base directory that contains nested population*.gpkg files.",
    )
    args = parser.parse_args()
    process_all(Path(args.directory_path))
