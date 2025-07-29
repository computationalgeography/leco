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


def create_colormap(nr_languages: int) -> matplotlib.colors.ListedColormap:
    """Create a colormap for the languages"""
    nr_colors = 20  # number of colors to extract from each of the base_cmaps below
    base_cmaps = ["Greys", "Purples", "Reds", "Blues", "Oranges", "Greens", "RdPu"]

    # Sample from linspace 0.2 to 0.8 to avoid having overly dark and light shades
    raw_colors = np.concatenate(
        [plt.get_cmap(name)(np.linspace(0.2, 0.8, nr_colors)) for name in base_cmaps]
    )

    nr_unique_colors = len(raw_colors)
    np.random.shuffle(raw_colors)  # Shuffle colors

    if nr_languages <= nr_unique_colors:
        selected_colors = raw_colors[0:nr_languages]
    else:
        # If not enough unique colors, repeat some colors
        print(
            f"Not enough colors ({nr_unique_colors}) for {nr_languages} languages: some colors will be used multiple times."
        )
        # Evenly distribute reused colors to avoid repetition at the same time
        selected_colors = []
        for i in range(nr_languages):
            color_idx = i % nr_unique_colors
            selected_colors.append(raw_colors[color_idx])
        selected_colors = np.array(selected_colors)

    return matplotlib.colors.ListedColormap(selected_colors)  # Create a colormap


def language_number_plot(output_path: str, languagenumber: list, nr_steps: int) -> None:
    """Plot number of languages over time"""

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
    plt.savefig(os.path.join(output_path, "Number_of_Languages.pdf"))
    plt.close()


def language_counts_plot(
    language_counts: gpd.GeoDataFrame,
    output_path: str,
) -> None:
    """Plot number of agents speaking a language over time"""

    # Sort the languages by most to least spoken at the first timestep, so most spoken languages are shown on the bottom
    sorted_cols = language_counts.iloc[0].sort_values(ascending=False).index
    language_counts = language_counts[sorted_cols]

    cmap = create_colormap(len(language_counts.columns))

    fig, ax = plt.subplots(figsize=(12, 6))

    # Create a stacked area plot for the number of agents per language over time
    language_counts.plot.area(
        ax=ax,
        stacked=True,
        colormap=cmap,
        linewidth=0,
    )

    ax.get_legend().remove()  # Remove the legend for clarity
    ax.set_xlim(language_counts.index.min(), language_counts.index.max())
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Number of Agents")
    ax.set_title("Number of Agents per Language over Time")

    # Save the plot
    plt.savefig(os.path.join(output_path, "Agents_per_Language.pdf"))
    plt.close()


def plot_summaries(input_file: str) -> None:
    """Create summarizing plots of the leco model output"""

    # Read in the population data across all timesteps
    population = gpd.read_file(input_file)

    output_path = Path(input_file).parent

    # Create a figure showing the number of languages over time
    languagenumber = population.groupby("timestep")["language"].nunique().tolist()
    nr_steps = population["timestep"].max()
    language_number_plot(output_path, languagenumber, nr_steps)

    # Create a figure showing the number of agents speaking a language over time
    language_counts = (
        population.groupby(["timestep", "language"])["id"].count().unstack(fill_value=0)
    )
    language_counts_plot(language_counts, output_path)
