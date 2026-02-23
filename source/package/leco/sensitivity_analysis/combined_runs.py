"""Functions to create summary plots of leco model output."""

import re
import logging
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


def parse_filename(filepath: Path) -> tuple[int, int]:
    """Extract seed and run number from filename like 'seed_42_run_0001'."""
    parameter_name = "speed"
    match = re.search(rf"seed_(\d+)_{parameter_name}_(\d+(?:\.\d+)?)", str(filepath))

    if match:
        return int(match.group(1)), int(match.group(2))

    return logger.error("Could not parse seed and run from %s", filepath)


def organize_by_scenario_and_run(
    populations: list[gpd.GeoDataFrame],
    filepaths: list[Path],
) -> dict:
    """Organize GeoDataFrames by run and seed for one scenario.

    Returns nested dict: {run: {seed: gdf}}
    """
    scenario_dict = defaultdict(dict)

    for gdf, filepath in zip(populations, filepaths):
        seed, run = parse_filename(filepath)
        scenario_dict[run][seed] = gdf

    return dict(scenario_dict)


def mean_std_languages_per_run(
    gdfs: list[gpd.GeoDataFrame],
    step_to_years: int,
) -> tuple[np.ndarray]:
    """gdfs: all seeds for one.

    Calculate the number of born, extinct and total number of languages per time step
    """
    born_languages_per_seed = []
    extinct_languages_per_seed = []
    language_counts_per_seed = []
    # language_speakers_per_seed = []

    for gdf in gdfs:
        # Get languages present at each time step
        # Store in set for easy comparison
        languages_by_time = gdf.groupby("time_step")["language"].apply(set)

        # Sort by timestep to ensure correct order
        languages_by_time = languages_by_time.sort_index()

        timesteps = languages_by_time.index
        born_langs = []
        extinct_langs = []

        for i, _timestep in enumerate(timesteps):
            if i == 0:
                # First timestep: all languages are "born"
                born_langs.append(len(languages_by_time.iloc[i]))
                extinct_langs.append(0)
            else:
                prev_languages = languages_by_time.iloc[i - 1]
                curr_languages = languages_by_time.iloc[i]

                # born: in current but not in previous
                born = len(curr_languages - prev_languages)
                # Extinct: in previous but not in current
                extinct = len(prev_languages - curr_languages)

                born_langs.append(born)
                extinct_langs.append(extinct)

        born_languages_per_seed.append(born_langs)
        extinct_languages_per_seed.append(extinct_langs)

        # Calculate the total number of languages per time step
        language_counts = languages_by_time.apply(len)  # Count languages in each set
        language_counts_per_seed.append(language_counts.values)

        """ # language_speakers = languages_by_time["id"].count().unstack(fill_value=0)
        language_speakers = (
            gdf.groupby(["time_step", "language"])["id"].count().unstack(fill_value=0)
        )
        print(f"language_speakers fill_value = 0 {language_speakers}")
        language_speakers_per_seed.append(np.mean(language_speakers, axis=0))
        print(f"mean: {np.mean(language_speakers, axis = 0)}") """

    # Convert to arrays and calculate mean and std across seeds
    born_languages_array = np.array(born_languages_per_seed)
    extinct_languages_array = np.array(extinct_languages_per_seed)
    language_counts_array = np.array(language_counts_per_seed)
    # language_speakers_array = np.array(language_speakers_per_seed)

    mean_born = np.mean(born_languages_array, axis=0)
    std_born = np.std(born_languages_array, axis=0)
    mean_extinct = np.mean(extinct_languages_array, axis=0)
    std_extinct = np.std(extinct_languages_array, axis=0)
    mean_counts = np.mean(language_counts_array, axis=0)
    std_counts = np.std(language_counts_array, axis=0)
    # mean_speakers = np.mean(language_speakers_array, axis=0)
    # std_speakers = np.std(language_speakers_array, axis=0)

    # Convert timesteps to years
    timesteps_years = timesteps.values * step_to_years

    return (
        timesteps_years,
        mean_born,
        std_born,
        mean_extinct,
        std_extinct,
        mean_counts,
        std_counts,
        # mean_speakers,
        # std_speakers,
    )


##### Variance
def calculate_heterozygosity(language_profiles: np.array) -> float:
    """Calculates the heterozygosity for a language"""
    n_individuals = language_profiles.shape[0]

    # Count occurrences of each value at each position
    counts_per_position = np.apply_along_axis(
        lambda x: np.bincount(x.astype(int), minlength=int(language_profiles.max()) + 1),
        axis=0,
        arr=language_profiles.astype(int),
    )

    # Calculate heterozygosity per position
    # H = 1 - sum(p_i^2) where p_i is frequency of category i
    frequencies = counts_per_position / n_individuals
    heterozygosity_per_position = 1 - np.sum(frequencies**2, axis=0)

    # Number of categories present at each position
    n_categories_per_position = np.count_nonzero(counts_per_position, axis=0)

    # Normalize: H_norm = H / (1 - 1/K)
    # Avoid division by zero for monomorphic positions (K=1)
    normalized_h = np.zeros_like(heterozygosity_per_position)
    mask = n_categories_per_position > 1
    normalized_h[mask] = heterozygosity_per_position[mask] / (1 - 1 / n_categories_per_position[mask])

    return np.mean(normalized_h)


def within_variance(population: gpd.GeoDataFrame) -> dict[int, float]:
    """Calculate the variance within languages over time."""
    languages = population.groupby("language")["language_profile"]
    zygosity = dict[int, float]()

    for language_id, language_profiles in languages:
        heterozygosity = calculate_heterozygosity(np.stack(language_profiles))
        zygosity[language_id] = heterozygosity

    return zygosity


def compute_fixation_index_per_step(population: gpd.GeoDataFrame) -> dict[str, float]:
    """Function that calculates the fixation index Fst for the total population."""
    heterozygosity_total = calculate_heterozygosity(np.stack(population["language_profile"]))

    # If there's no variation at all, Fst is undefined
    if heterozygosity_total == 0:
        return {
            "fixation_index": 0.0,
            "heterozygosity_total": 0.0,
            "mean_heterozygosity_within": 0.0,
        }

    heterozygosity_per_language = within_variance(population)
    mean_heterozygosity_within = sum(heterozygosity_per_language.values()) / len(heterozygosity_per_language)

    fixation_index = (heterozygosity_total - mean_heterozygosity_within) / heterozygosity_total

    heterozygosity_dict = {
        "fixation_index": fixation_index,
        "heterozygosity_total": heterozygosity_total,
        "mean_heterozygosity_within": mean_heterozygosity_within,
    }

    return heterozygosity_dict


def compute_fixation_indices(
    population: gpd.GeoDataFrame,
) -> np.ndarray[dict[str, float]]:
    """Compute the fixation index across multiple time steps."""
    steps = np.sort(population["time_step"].unique())

    fixation_indices = []

    for step in tqdm(steps):
        population_step = population[population["time_step"] == step]
        heterozygosity_dict = compute_fixation_index_per_step(population_step)
        fixation_indices.append(heterozygosity_dict)

    return np.array(fixation_indices)


def multiple_runs_fixation_index(
    populations: list[gpd.GeoDataFrame],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Function that calculates the mean and standard deviation for fixation_index, heterozygosity_total,
    and mean_heterozygosity_within across multiple populations per timestep.
    """
    metrics = ["fixation_index", "heterozygosity_total", "mean_heterozygosity_within"]
    metrics_per_population = {metric: [] for metric in metrics}

    for pop in populations:
        # Calculate fixation index for all time steps in this population
        fst_array = compute_fixation_indices(pop)
        # Extract all three metric values from each dictionary
        for metric in metrics:
            values = np.array([d[metric] for d in fst_array])
            metrics_per_population[metric].append(values)

    # Calculate mean and std for each metric
    result = {}
    for metric in metrics:
        stacked = np.array(metrics_per_population[metric])
        # Shape: (n_populations, n_timesteps)
        mean_vals = np.mean(stacked, axis=0)  # Shape: (n_timesteps,)
        std_vals = np.std(stacked, axis=0, ddof=1)  # sample std, Shape: (n_timesteps,)
        result[metric] = (mean_vals, std_vals)

    return result


def calculate_stats_all_scenarios(
    organized_data: dict,
    step_to_years: int,
    seed_number: int,
) -> pd.DataFrame:
    """Calculate mean and std for each scenario and run.

    organized_data: {scenario_name: {run: {seed: gdf}}}
    Returns a DataFrame with columns: scenario, run, time_step, years, mean_languages, std_languages
    """
    results = []

    for scenario_name, runs_dict in organized_data.items():
        for run, seeds_dict in runs_dict.items():
            # Skip even runs
            print(int(run))
            """ if int(run) % 2 == 0:
                continue """

            print(int(run))
            # Get all gdfs for this run (all seeds)
            gdfs_for_run = list(seeds_dict.values())

            if len(gdfs_for_run) != seed_number:
                raise (ValueError(f"Number of seeds do not match {seed_number}!"))

            # Calculate stats using your existing function
            (
                years,
                mean_born,
                std_born,
                mean_extinct,
                std_extinct,
                mean_counts,
                std_counts,
                # mean_speakers,
                # std_speakers,
            ) = mean_std_languages_per_run(gdfs_for_run, step_to_years)
            fst_stats = multiple_runs_fixation_index(gdfs_for_run)

            mean_fst, std_fst = fst_stats["fixation_index"]
            mean_het_total, std_het_total = fst_stats["heterozygosity_total"]
            mean_het_within, std_het_within = fst_stats["mean_heterozygosity_within"]

            # Create records for this run
            for (
                t,
                y,
                m_b,
                s_b,
                m_e,
                s_e,
                m_c,
                s_c,
                # m_s,
                # s_s,
                m_f,
                s_f,
                m_ht,
                s_ht,
                m_hs,
                s_hs,
            ) in zip(
                range(len(years)),
                years,
                mean_born,
                std_born,
                mean_extinct,
                std_extinct,
                mean_counts,
                std_counts,
                # mean_speakers,
                # std_speakers,
                mean_fst,
                std_fst,
                mean_het_total,
                std_het_total,
                mean_het_within,
                std_het_within,
            ):
                results.append(
                    {
                        "scenario": scenario_name,
                        "run": run,
                        "time_step": t,
                        "years": y,
                        "mean_born": m_b,
                        "std_born": s_b,
                        "mean_extinct": m_e,
                        "std_extinct": s_e,
                        "mean_languages": m_c,
                        "std_languages": s_c,
                        # "mean_speakers": m_s,
                        # "std_speakers": s_s,
                        "mean_fst": m_f,
                        "std_fst": s_f,
                        "mean_heterozygosity_total": m_ht,
                        "std_heterozygosity_total": s_ht,
                        "mean_heterozygosity_within": m_hs,
                        "std_heterozygosity_within": s_hs,
                    },
                )

    return pd.DataFrame(results)


def group_by_run(gdfs: list[gpd.GeoDataFrame], filenames: list[str]):
    runs = defaultdict(list)

    for gdf, name in zip(gdfs, filenames):
        run_id = int(re.search(r"run_(\d+)", name).group(1))
        runs[run_id].append(gdf)

    return runs


def extract_files(input_paths: list[Path]) -> list[gpd.GeoDataFrame]:
    runs = []
    for input in input_paths:
        population = gpd.read_file(input)
        # Convert language_profile from string to numpy array
        population["language_profile"] = population["language_profile"].apply(
            lambda x: np.fromstring(str(x).strip("[]"), sep=" ", dtype=int),
        )
        runs.append(population)

    return runs


def from_path_to_gdf(path: Path, name: str) -> list[gpd.GeoDataFrame]:
    """Return gdf from list of paths."""
    gpkg = list(path.rglob("*.gpkg"))
    print(type(gpkg))
    return gpkg, extract_files(gpkg)


def plot_combined(seed_number: int = 5) -> None:
    """Create summarizing plots of the leco model output."""
    # Read in the population data across all time steps

    method = "ff"
    linkage = "average"
    # work with dictionary to loop through?

    """ populated = Path(
        f"/scratch/posma002/lecoOutput/holythree_{method}_{linkage}_populations/populated/"
    )
    few = Path(
        f"/scratch/posma002/lecoOutput/holythree_{method}_{linkage}_populations/few/"
    )
    single = Path(
        f"/scratch/posma002/lecoOutput/holythree_{method}_{linkage}_populations/single/"
    )

    populated_gpkg = list(populated.rglob("*.gpkg"))
    few_gpkg = list(few.rglob("*.gpkg"))
    single_gpkg = list(single.rglob("*.gpkg"))

    populated_populations = extract_files(populated_gpkg)
    few_populations = extract_files(few_gpkg)
    single_populations = extract_files(single_gpkg)

    output_path = populated.parent

    organized_data = {
        "populated": organize_by_scenario_and_run(
            populated_populations,
            populated_gpkg,
        ),
        "few": organize_by_scenario_and_run(
            few_populations,
            few_gpkg,
        ),
        "single": organize_by_scenario_and_run(single_populations, single_gpkg),
    } """

    path = Path("/scratch/posma002/lecoOutput/morris_ff_average/sensitivity/speed/populations")
    gpkg = list(path.rglob("*.gpkg"))
    print(type(gpkg))
    populations = extract_files(gpkg)

    output_path = path.parent
    organized_data = {
        "few": organize_by_scenario_and_run(
            populations,
            gpkg,
        ),
    }

    # Then calculate statistics
    step_to_years = 20
    stats_df = calculate_stats_all_scenarios(organized_data, step_to_years, seed_number)
    # Create a figure showing the number of languages over time
    langs = []
    for pop in populations:
        number_languages = pop.groupby("time_step")["language"].nunique().tolist()
        langs.append(number_languages)

    print(stats_df)

    # Save to CSV
    stats_df.to_csv(
        output_path / f"language_statistics_{method}_{linkage}.csv",
        index=False,
    )


if __name__ == "__main__":
    plot_combined()
