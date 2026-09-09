"""Functions to create summary plots of leco model output."""

from pathlib import Path

import geopandas as gpd
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
    small_step = 2
    medium_step = 5
    magnitude = 10 ** np.floor(np.log10(rough_step))
    normalized_step = rough_step / magnitude

    if normalized_step <= 1:
        nice_step = 1 * magnitude
    elif normalized_step <= small_step:
        nice_step = 2 * magnitude
    elif normalized_step <= medium_step:
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


def plot_summaries(
    directory: Path,
    population: gpd.GeoDataFrame,
) -> None:
    """Create summarizing plots of the leco model output."""
    # Create a figure showing the number of languages over time
    number_languages = population.groupby("time_step")["language"].nunique().tolist()
    language_number_plot(directory, number_languages)
