"""Functions to create summary data of the leco model output for different parameter combinations."""

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

PARAMETER_NAMES = [
    "speed",
    "mutation_rate",
    "radius",
    "diffusion_rate",
    "similarity_preference",
]


def parse_path_metadata(filepath: Path, baseline: bool = False) -> tuple[str, dict]:
    """Extract all parameter values and seed from a nested path.

    Parametrized path:
        speed_0.5/mutation_rate_0.01/radius_30.0/diffusion_rate_0.1/similarity_preference_0.8/seed_42/

    Baseline path:
        seed_42/

    Returns:
        seed: the seed value as string
        params: dict of parameter values (all None if baseline=True)

    """
    filepath_str = str(filepath)
    params = {}

    if baseline:
        params = dict.fromkeys(PARAMETER_NAMES)
    else:
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


def organize_meta_by_params_and_seed(
    run_dirs: list[Path],
    metadata_files: list[pd.DataFrame],
    metacluster_files: list[pd.DataFrame],
    baseline: bool = False,
) -> dict:
    """Organize meta data by parameter combination and seed.

    Returns nested dictionary: {param_combo_tuple: {seed: (metadata, metacluster)}}
    For baseline runs, param_combo_tuple is a single entry of all Nones.
    """
    scenario_dict = defaultdict(dict)

    for path, metadata, metacluster in zip(run_dirs, metadata_files, metacluster_files, strict=True):
        seed, params = parse_path_metadata(path, baseline=baseline)
        param_key = tuple(params[p] for p in PARAMETER_NAMES)
        scenario_dict[param_key][seed] = (metadata, metacluster)

    return dict(scenario_dict)


def extract_phylogeny_info(
    run_dirs: list[Path],
    metacluster_files: list[pd.DataFrame],
) -> pd.DataFrame:
    """Extract family-level statistics from phylogeny_information.csv per seed.

    For each seed:
    - Loads phylogeny_information.csv
    - Identifies languages alive at the final time step
    - Aligns with the speaker array from meta_data_cluster.csv (sorted by language ID)
    - Computes per-family language count and speaker count
    - Returns mean/median/min/max across families, plus family count
    """
    results = []

    for run_dir, metacluster in zip(run_dirs, metacluster_files, strict=True):
        seed_match = re.search(r"seed_(\d+)", str(run_dir))
        if not seed_match:
            raise ValueError(f"Could not parse seed from {run_dir}")
        seed = seed_match.group(1)

        phylogeny_path = run_dir / "phylogeny_information.csv"
        if not phylogeny_path.exists():
            raise FileNotFoundError(f"No phylogeny_information.csv found in {run_dir}")

        phylo = pd.read_csv(phylogeny_path)

        # Get the final time step speaker array
        final_row = metacluster.iloc[-1]
        final_step = final_row["time_step"]
        speaker_array = final_row["speaker_numbers"]  # already parsed as np.array

        # Languages alive at final step: t_extinct is NaN or beyond final step
        alive = cast(
            "pd.DataFrame",
            phylo[phylo["t_extinct"].isna() | (phylo["t_extinct"] >= final_step)].copy(),
        )
        alive = alive.sort_values("language").reset_index(drop=True)

        if len(alive) != len(speaker_array):
            raise ValueError(
                f"Seed {seed}: alive language count ({len(alive)}) does not match "
                f"speaker array length ({len(speaker_array)})",
            )

        # Align speaker counts with language IDs
        alive["speaker_count"] = speaker_array

        # Aggregate per family
        family_agg = alive.groupby("root_language").agg(
            language_count=("language", "count"),
            speaker_count=("speaker_count", "sum"),
        )

        results.append(
            {
                "seed": seed,
                "family_count": len(family_agg),
                "languages_per_family_mean": family_agg["language_count"].mean(),
                "languages_per_family_median": np.median(family_agg["language_count"]),
                "languages_per_family_min": family_agg["language_count"].min(),
                "languages_per_family_max": family_agg["language_count"].max(),
                "speakers_per_family_mean": family_agg["speaker_count"].mean(),
                "speakers_per_family_median": np.median(family_agg["speaker_count"]),
                "speakers_per_family_min": family_agg["speaker_count"].min(),
                "speakers_per_family_max": family_agg["speaker_count"].max(),
            },
        )

    return pd.DataFrame(results)


def write_meta_data(
    organized_data: dict,
    step_to_years: int,
    baseline: bool = False,
    expected_seed_count: int = 5,
) -> pd.DataFrame:
    """Calculate summary statistics across all parameter combinations and seeds."""
    first_param_combo = next(iter(organized_data.values()))
    first_seed = next(iter(first_param_combo.values()))
    first_meta_data = first_seed[0]
    time_steps = sorted(first_meta_data["time_step"].unique())
    time_steps_years = (np.array(time_steps) * step_to_years).tolist()

    results = []

    for param_key, seeds_dict in organized_data.items():
        param_dict = dict(zip(PARAMETER_NAMES, param_key, strict=True))

        if not baseline and len(seeds_dict) != expected_seed_count:
            raise ValueError(
                f"Expected {expected_seed_count} seeds for {param_dict}, "
                f"but found {len(seeds_dict)}: {list(seeds_dict.keys())}",
            )

        for seed, (metadata, metacluster) in seeds_dict.items():
            external_changes = metadata["external_change"]
            internal_changes = metadata["internal_change"]
            number_languages = metacluster["language_number"]
            language_speakers = metacluster["speaker_numbers"]
            convergences = metacluster["convergences"]
            divergences = metacluster["divergences"]
            nncor_unnormalized = metacluster["nncor_unnormalized"]
            nncor_normalized = metacluster["nncor_normalized"]
            simpson_baseline = metacluster["simpson_baseline"]

            for i, year in enumerate(time_steps_years):
                row = {
                    "seed": seed,
                    **param_dict,
                    "step": int(year / step_to_years),
                    "year": year,
                    "language_count": number_languages[i],
                    "internal_change": internal_changes[i],
                    "external_change": external_changes[i],
                    "speakers": language_speakers[i].tolist(),
                    "speaker_mean": float(language_speakers[i].mean()),
                    "speaker_median": int(np.median(language_speakers[i])),
                    "speaker_min": int(language_speakers[i].min()),
                    "speaker_max": int(language_speakers[i].max()),
                    "convergences": convergences[i],
                    "divergences": divergences[i],
                    "nncor_unnormalized": nncor_unnormalized[i],
                    "nncor_normalized": nncor_normalized[i],
                    "simpson_baseline": simpson_baseline[i],
                }
                results.append(row)

    return pd.DataFrame(results)


def extract_meta_files(csv_paths: list[Path]) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    """Extract the metadata files."""
    metadatas = []
    metaclusters = []

    for csv_path in csv_paths:
        metadata_path = csv_path / "meta_data.csv"
        if not metadata_path.exists():
            raise FileNotFoundError(f"No metadata file found in {csv_path}")
        metadatas.append(pd.read_csv(metadata_path))

        metacluster_path = csv_path / "meta_data_cluster.csv"
        if not metacluster_path.exists():
            raise FileNotFoundError(f"No metadata file found in {csv_path}")

        metacluster = pd.read_csv(metacluster_path).copy()
        metacluster["speaker_numbers"] = metacluster["speaker_numbers"].apply(
            lambda x: np.array(
                [v for v in str(x).strip("[]").replace(",", " ").split() if v != "..."],
                dtype=int,
            ),
        )
        metaclusters.append(metacluster)

    return metadatas, metaclusters


def process_all(directory_path: Path, baseline: bool = False) -> None:
    """Create summarizing data of the leco model output for the nested parameter structure."""
    cluster_file_paths = list(directory_path.rglob("meta_data_cluster.csv"))
    run_dirs = sorted({path.parent for path in cluster_file_paths})

    metadata, metaclusters = extract_meta_files(run_dirs)
    organized_data = organize_meta_by_params_and_seed(run_dirs, metadata, metaclusters, baseline=baseline)

    step_to_years = 20
    stats_df = write_meta_data(organized_data, step_to_years, baseline=baseline)
    print(stats_df)

    if baseline:
        # Extract phylogeny information from phylogeny_information.csv
        phylo_df = extract_phylogeny_info(run_dirs, metaclusters)
        stats_df["seed"] = stats_df["seed"].astype(str)
        phylo_df["seed"] = phylo_df["seed"].astype(str)
        stats_df = stats_df.merge(phylo_df, on="seed", how="left")
        output_path = directory_path / "spawn_base_stats.csv"
        stats_df.to_csv(output_path, index=False)
    else:
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
    # When adding --baseline it summarizes only the baseline
    # and will not search for the variable names in the folder path
    parser.add_argument(
        "--baseline",
        action="store_true",
        default=False,
        help="""Set if the directory contains a baseline run (seed_x folders only, no parameter subfolders).
        Requires phylogeny_information.csv files for every seed.""",
    )
    args = parser.parse_args()
    process_all(Path(args.directory_path), baseline=args.baseline)
