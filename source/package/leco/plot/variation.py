"""Test variation."""

from pathlib import Path
from tqdm import tqdm

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def calculate_linguistic_similarity_pairwise(
    profile_a: np.ndarray[int],
    profile_b: np.ndarray[int],
) -> float:
    """Calculate the similarity between two language profiles."""
    # Count the number of meanings with the same form
    matching_meanings = np.sum(profile_a == profile_b)
    # Similarity is the proportion of matching meanings
    return matching_meanings / len(profile_a)


def calculate_variogram(
    spatial_distances: np.array,
    linguistic_distances: np.array,
    n_bins: int = 10,
) -> tuple[np.array, np.array, np.array]:
    """
    Calculate empirical variogram by binning spatial distances.

    Parameters:
    -----------
    spatial_distances : array
        Pairwise spatial distances
    linguistic_distances : array
        Pairwise linguistic distances
    n_bins : int
        Number of distance bins

    Returns:
    --------
    bin_centers : array
        Center of each distance bin
    semivariances : array
        Average semivariance for each bin
    counts : array
        Number of pairs in each bin
    """
    # Calculate semivariance: γ(h) = (1/2) * distance²
    semivariances_all = 0.5 * linguistic_distances**2

    # Create bins
    max_dist = np.max(spatial_distances)
    bins = np.linspace(0, max_dist, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Assign pairs to bins and calculate average semivariance
    digitized = np.digitize(spatial_distances, bins)

    binned_semivariances = []
    binned_counts = []

    for i in range(1, n_bins + 1):
        mask = digitized == i
        if np.sum(mask) > 0:
            binned_semivariances.append(np.mean(semivariances_all[mask]))
            binned_counts.append(np.sum(mask))
        else:
            binned_semivariances.append(np.nan)
            binned_counts.append(0)

    return bin_centers, np.array(binned_semivariances), np.array(binned_counts)


def plot_variogram(spatial_distances, linguistic_distances, n_bins=10, figsize=(12, 5)):
    """Plot the variogram and optionally the variogram cloud."""

    # Calculate binned variogram
    bin_centers, semivariances, counts = calculate_variogram(spatial_distances, linguistic_distances, n_bins)

    fig, ax2 = plt.subplots(1, 1, figsize=(8, 5))

    valid_mask = ~np.isnan(semivariances)
    ax2.plot(
        bin_centers[valid_mask],
        semivariances[valid_mask],
        "o-",
        linewidth=2,
        markersize=8,
        color="#2563eb",
    )
    ax2.set_xlabel("Spatial Distance", fontsize=12)
    ax2.set_ylabel("Semivariance", fontsize=12)
    ax2.set_title("Empirical Variogram", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    for i, (x, y, c) in enumerate(
        zip(bin_centers[valid_mask], semivariances[valid_mask], counts[valid_mask])
    ):
        if c > 0:
            ax2.text(x, y, f"n={int(c)}", fontsize=8, ha="center", va="bottom")

    plt.tight_layout()
    return fig


def distance_to_distance(
    population: gpd.GeoDataFrame,
    output_path: Path,
    cmap: mpl.colors.ListedColormap,
) -> None:
    """Plot linguistic distance against the geographic distance for the last time step."""

    # Calculate pairwise linguistic distances
    n = len(population)
    linguistic_distances = []
    geographic_distances = []

    for i in range(n):
        for j in range(i + 1, n):  # Only upper triangle to avoid duplicates
            # Linguistic similarity -> distance (1 - similarity)
            similarity = calculate_linguistic_similarity_pairwise(
                population.iloc[i]["language_profile"],
                population.iloc[j]["language_profile"],
            )
            linguistic_distances.append(1 - similarity)

            # Geographic distance
            geo_dist = population.geometry.iloc[i].distance(population.geometry.iloc[j])
            geographic_distances.append(geo_dist)

    # Variogram plot
    """ fig = plot_variogram(geographic_distances, linguistic_distances, n_bins=8)
    plt.savefig(output_path / "Variogram.jpeg", dpi=300)
    plt.close(fig) """

    # Scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(geographic_distances, linguistic_distances, alpha=0.5, s=10)
    plt.xlabel("Geographic Distance (km)")
    plt.ylabel("Linguistic distance (Hamming)")

    plt.title("Geographic vs Linguistic Distance for last time step")

    # Save the plot
    plt.savefig(output_path / "GeographicToLinguisticDistance.jpeg", dpi=300)
    plt.close()


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
    """Function that calculates the variance within languages over time."""

    languages = population.groupby("language")["language_profile"]
    zygosity = dict[int, float]()

    for language_id, language_profiles in languages:
        heterozygosity = calculate_heterozygosity(np.stack(language_profiles))
        zygosity[language_id] = heterozygosity

    return zygosity


def compute_fixation_index_per_step(population: gpd.GeoDataFrame) -> float:
    """Function that calculates the fixation index Fst for the total population."""

    heterozygosity_total = calculate_heterozygosity(np.stack(population["language_profile"]))

    heterozygosity_per_language = within_variance(population)
    mean_heterozygosity_within = sum(heterozygosity_per_language.values()) / len(heterozygosity_per_language)

    fixation_index = (heterozygosity_total - mean_heterozygosity_within) / heterozygosity_total

    return fixation_index


def compute_fixation_indices(population: gpd.GeoDataFrame) -> np.ndarray:
    """Compute the fixation index across multiple time steps."""
    steps = np.sort(population["time_step"].unique())

    fixation_indices = []

    for step in tqdm(steps):
        population_step = population[population["time_step"] == step]
        fixation_index = compute_fixation_index_per_step(population_step)
        fixation_indices.append(fixation_index)

    return np.array(fixation_indices)


def multiple_runs_fixation_index(
    populations: list[gpd.GeoDataFrame],
) -> tuple[np.ndarray, np.ndarray]:
    """Function that calculates the fixation index mean and standard deviation across
    multiple populations per timestep."""

    fixation_indices_per_population = []

    for pop in populations:
        # Calculate fixation index for all time steps in this population
        fst_array = compute_fixation_indices(pop)
        fixation_indices_per_population.append(fst_array)

    # Stack all arrays
    # Shape: (n_populations, n_timesteps)
    stacked_fst = np.array(fixation_indices_per_population)

    # Calculate mean and std per timestep (across populations)
    mean_fst = np.mean(stacked_fst, axis=0)  # Shape: (n_timesteps,)
    std_fst = np.std(stacked_fst, axis=0, ddof=1)  # sample std, Shape: (n_timesteps,)

    return mean_fst, std_fst


def extract_seed_run(file_path: Path) -> tuple[int, int]:
    """Extract seed and run number from path input"""
    parts = file_path.stem.split("_")

    seed = int(parts[parts.index("seed") + 1])
    run = int(parts[parts.index("run") + 1])

    return (seed, run)


def analyze_variance() -> None:
    """Create summarizing plots of the leco model output."""
    # Read in the population data across all time steps

    input_file = Path(
        "/scratch/posma002/lecoOutput/holythree_ff_average_populations/few/population_seed_44_run_0003.gpkg"
    )

    population = gpd.read_file(input_file)

    # Convert language_profile from string to numpy array
    population["language_profile"] = population["language_profile"].apply(
        lambda x: np.fromstring(str(x).strip("[]"), sep=" ", dtype=int)
    )

    output_path = input_file.parent

    # Create a figure showing linguistic to geographic distance correlation at the last time step
    # last_step = population["time_step"].max()
    # population_last_step = population[population["time_step"] == last_step]

    # heterozygosity_per_language = within_variance(population_last_step)
    fixation_index = compute_fixation_indices(population)
    print(fixation_index)
    print(output_path)

    # Extract the seed and run number from input file and save in fst
    seed, run = extract_seed_run(input_file)
    np.savetxt(output_path / f"fst_{seed}_{run}.csv", fixation_index, delimiter=",")

    # distance_to_distance(population_last_step, output_path, cmap)


if __name__ == "__main__":
    analyze_variance()
