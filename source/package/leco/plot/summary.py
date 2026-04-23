"""Functions to create summary plots of leco model output."""

from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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

    # Calculate tick interval based on number of languages
    max_languages = max(number_languages)
    y_ticks = calculate_tick_intervals(0, max_languages, max_ticks=10)

    plt.yticks(y_ticks, [str(i) for i in y_ticks])
    plt.title("Number of Languages Over Time")

    # Save the plot
    plt.savefig(output_path / "Number_of_Languages.png", dpi=150)
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
    plt.savefig(output_path / "Agents_per_Language.png", dpi=150)
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


def speaker_distribution_plot(
    final_population: gpd.GeoDataFrame,
    output_path: Path,
    speaker_bin_size: int = 10,
) -> None:
    """Create a histogram of the frequency of languages by number of speakers at final time step."""
    # Calculate the speakers_per_language
    speakers_per_language = final_population.groupby("language")["id"].count().to_numpy()
    if speakers_per_language.size == 0:
        return

    max_speakers = int(speakers_per_language.max())
    bins = np.arange(0, max_speakers + speaker_bin_size, speaker_bin_size)
    if bins.size < 2:
        bins = np.array([0, speaker_bin_size])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(speakers_per_language, bins=bins, color="steelblue", alpha=0.75, edgecolor="black")
    ax.set_title("Speaker Distribution at Final Timestep", size=16)
    ax.set_xlabel("Number of Agents Speaking a Language", size=14)
    ax.set_ylabel("Frequency (Number of Languages)", size=14)
    ax.tick_params(axis="both", labelsize=11)
    plt.tight_layout()
    fig.savefig(output_path / "speaker_distribution_final_step.png", dpi=150)
    plt.close()


def plot_last_time_step(
    final_population: gpd.GeoDataFrame,
    output_path: Path,
    final_step: int,
    cmap: mpl.colors.ListedColormap,
    lang_to_index: dict[int, int] | None,
) -> None:
    """Create a static spatial plot for the final time step.

    Perhaps this function would fit better in the animation file.
    """
    _fig, ax = plt.subplots()

    # Map language IDs to indices for color mapping, using lang_to_index if provided
    if lang_to_index is not None:
        indices = final_population["language"].map(lang_to_index).to_numpy()
    else:
        indices, _ = pd.factorize(final_population["language"])

    ax.scatter(
        final_population.geometry.x,
        final_population.geometry.y,
        c=indices,
        cmap=cmap,
        vmin=0,
        vmax=indices.max() if indices.size > 0 else 0,
        s=80,
        marker=".",
    )

    """ ax.set_xlim(0, space[0])
    ax.set_ylim(0, space[1])
    ax.set_xlabel("X Position (km)", size=20)
    ax.set_ylabel("Y Position (km)", size=20)
    ax.tick_params(axis="both", labelsize=16) """
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_title(f"Year {final_step * 20}", size=20)

    plt.savefig(output_path / "LastStep.png", bbox_inches="tight")
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
    ## For now, not informative
    """language_speakers = population.groupby(["time_step", "language"])["id"].count().unstack(fill_value=0)
    language_speakers_plot(language_speakers, output_path, cmap, lang_to_index)"""

    # Create a figure showing the frequency distribution of agents speaking a language at last time step
    final_step = population["time_step"].max()
    last_step_population = population[population["time_step"] == final_step]
    speaker_distribution_plot(last_step_population, output_path)

    # Create a figure of the spatial distribution of agents at last time step
    plot_last_time_step(last_step_population, output_path, final_step, cmap, lang_to_index)
