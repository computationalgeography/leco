import os
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path


def calculate_tick_intervals(
    min_val: int, max_val: int, max_ticks: int = 10
) -> list[int]:
    """Calculate appropriate tick intervals to keep plots readable"""
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

    return sorted(list(set(ticks)))


def language_number_plot(
    output_path: str,
    languagenumber: list[int],
) -> None:
    """Plot number of languages over time"""

    nr_steps = len(languagenumber)

    plt.plot(languagenumber, marker="o")
    plt.xlabel("Timestep")
    plt.ylabel("Number of Languages")

    # Calculate tick interval based on number of time steps
    x_ticks = calculate_tick_intervals(0, nr_steps, max_ticks=10)
    plt.xticks(x_ticks, [str(i) for i in x_ticks])

    # Calculate tick interval based on number of languages
    max_languages = max(languagenumber)
    y_ticks = calculate_tick_intervals(0, max_languages, max_ticks=10)

    plt.yticks(y_ticks, [str(i) for i in y_ticks])
    plt.title("Number of Languages Over Time")

    # Save the plot
    plt.savefig(os.path.join(output_path, "Number_of_Languages.pdf"))
    plt.close()


def language_counts_plot(
    language_counts: gpd.GeoDataFrame,
    output_path: str,
    cmap: matplotlib.colors.ListedColormap,
    lang_to_index: dict[int, int],
) -> None:
    """Plot number of agents speaking a language over time"""

    # Sort the languages by most to least spoken at the first timestep, so most spoken languages are shown on the bottom
    sorted_cols = language_counts.iloc[0].sort_values(ascending=False).index
    language_counts = language_counts[sorted_cols]

    # Scale the index to represent years for each timstep (assuming each timestep is 20 years)
    language_counts.index = language_counts.index * 20

    # Create color list that matches sorted language columns
    colors = [cmap.colors[lang_to_index[lang]] for lang in language_counts.columns]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Create a stacked area plot for the number of agents per language over time
    language_counts.plot.area(
        ax=ax,
        stacked=True,
        color=colors,
        linewidth=0,
    )

    ax.get_legend().remove()  # Remove the legend for clarity
    ax.set_xlim(language_counts.index.min(), language_counts.index.max())
    ax.set_xlabel("Year", size=18)
    ax.set_ylabel("Number of Agents", size=18)
    ax.tick_params(axis="both", labelsize=14)
    ax.set_title("Number of Agents per Language over Time", size=22)

    # Save the plot
    plt.savefig(os.path.join(output_path, "Agents_per_Language.jpeg"), dpi=300)
    plt.close()


def plot_summaries(
    input_file: str,
    cmap: matplotlib.colors.ListedColormap | None,
    lang_to_index: dict[int, int] | None,
) -> None:
    """Create summarizing plots of the leco model output"""

    # Read in the population data across all timesteps
    population = gpd.read_file(input_file)

    output_path = Path(input_file).parent

    # Create a figure showing the number of languages over time
    languagenumber = population.groupby("timestep")["language"].nunique().tolist()
    language_number_plot(output_path, languagenumber)

    # Create a figure showing the number of agents speaking a language over time
    language_counts = (
        population.groupby(["timestep", "language"])["id"].count().unstack(fill_value=0)
    )
    language_counts_plot(language_counts, output_path, cmap, lang_to_index)
