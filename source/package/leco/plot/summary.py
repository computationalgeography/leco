"""Functions to create summary plots of leco model output."""

from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def calculate_tick_intervals(min_val: int, max_val: int, max_ticks: int = 10) -> list[int]:
    """Calculate appropriate tick intervals to keep plots readable."""
    range_val = max_val - min_val

    if range_val == 0:
        return [min_val]

    # Calculate rough step size
    rough_step = range_val / max_ticks

    # Find a "nice" step size (powers of 10, 2, 5)
    magnitude = 10 ** np.floor(np.log10(rough_step))
    normalized_step = rough_step / magnitude

    if normalized_step <= 1:
        nice_step = 1 * magnitude
    elif normalized_step <= 2:
        nice_step = 2 * magnitude
    elif normalized_step <= 5:
        nice_step = 5 * magnitude
    else:
        nice_step = 10 * magnitude

    # Generate ticks
    first_tick = np.ceil(min_val / nice_step) * nice_step
    ticks = []
    tick = first_tick
    while tick <= max_val:
        ticks.append(int(tick))
        tick += nice_step

    # Ensure we have at least min_val and max_val if they're not already included
    if min_val not in ticks:
        ticks.insert(0, int(min_val))
    if max_val not in ticks:
        ticks.append(int(max_val))

    return sorted(set(ticks))


def language_number_plot(
    output_path: Path,
    number_languages: list[int],
    step_to_years: int = 20,
) -> None:
    """Plot number of languages over time."""
    nr_steps = len(number_languages)

    # Scale the index to represent years for each time step
    years = [i * step_to_years for i in range(nr_steps)]

    plt.plot(years, number_languages, marker="o")
    plt.xlabel("Years")
    plt.ylabel("Number of Languages")

    # Calculate tick interval based on number of time steps
    # x_ticks = calculate_tick_intervals(0, years, max_ticks=10)
    # plt.xticks(x_ticks, [str(i) for i in x_ticks])

    # Calculate tick interval based on number of languages
    max_languages = max(number_languages)
    y_ticks = calculate_tick_intervals(0, max_languages, max_ticks=10)

    plt.yticks(y_ticks, [str(i) for i in y_ticks])
    plt.title("Number of Languages Over Time")

    # Save the plot
    plt.savefig(output_path / "Number_of_Languages.jpeg", dpi=300)
    plt.close()


def language_speakers_plot(
    language_speakers: gpd.GeoDataFrame,
    output_path: Path,
    cmap: mpl.colors.ListedColormap,
    lang_to_index: dict[int, int],
    step_to_years: int = 20,
) -> None:
    """Plot number of agents speaking a language over time."""
    # Sort the languages by most to least spoken at the first time step,
    # so most spoken languages are shown on the bottom
    sorted_cols = language_speakers.iloc[0].sort_values(ascending=False).index
    language_speakers = language_speakers[sorted_cols]

    # Scale the index to represent years for each time step (assuming each time step is 20 years)
    language_speakers.index = language_speakers.index * step_to_years

    # Create color list that matches sorted language columns
    colors = [cmap.colors[lang_to_index[lang]] for lang in language_speakers.columns]

    _fig, ax = plt.subplots(figsize=(10, 6))

    # Create a stacked area plot for the number of agents per language over time
    language_speakers.plot.area(
        ax=ax,
        stacked=True,
        color=colors,
        linewidth=0,
    )

    ax.get_legend().remove()  # Remove the legend for clarity
    ax.set_xlim(language_speakers.index.min(), language_speakers.index.max())
    ax.set_xlabel("Year", size=18)
    ax.set_ylabel("Number of Agents", size=18)
    ax.tick_params(axis="both", labelsize=14)
    ax.set_title("Number of Agents per Language over Time", size=22)

    # Save the plot
    plt.savefig(output_path / "Agents_per_Language.jpeg", dpi=300)
    plt.close()


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
                population.iloc[i]["language_profile"], population.iloc[j]["language_profile"]
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


def plot_summaries(
    input_file: Path,
    cmap: mpl.colors.ListedColormap | None,
    lang_to_index: dict[int, int] | None,
) -> None:
    """Create summarizing plots of the leco model output."""
    # Read in the population data across all time steps
    population = gpd.read_file(input_file)

    output_path = input_file.parent

    # Create a figure showing the number of languages over time
    number_languages = population.groupby("time_step")["language"].nunique().tolist()
    language_number_plot(output_path, number_languages)

    # Create a figure showing the number of agents speaking a language over time
    language_speakers = population.groupby(["time_step", "language"])["id"].count().unstack(fill_value=0)
    language_speakers_plot(language_speakers, output_path, cmap, lang_to_index)

    # Create a figure showing linguistic to geographic distance correlation at the last time step
    last_step = population["time_step"].max()
    population_last_step = population[population["time_step"] == last_step]
    distance_to_distance(population_last_step, output_path, cmap)
