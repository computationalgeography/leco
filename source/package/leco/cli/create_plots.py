import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib


def point_plotje(
    output_run: str, population: pd.DataFrame, timestep: int, x_max: int, y_max: int
):
    """Plot agents position colored by language for every timestep"""
    unique_languages = sorted(
        np.unique(population["language"])
    )  # Get unique langauges in sorted order
    nr_languages = len(unique_languages)

    ## From protomodel
    nr_colors = 20  # number of colors to extract from each of the base_cmaps below
    base_cmaps = ["Greys", "Purples", "Reds", "Blues", "Oranges", "Greens", "RdPu"]

    # n_base = len(base_cmaps)
    # we go from 0.2 to 0.8 below to avoid having several whites and blacks in the resulting cmaps
    colors = np.concatenate(
        [plt.get_cmap(name)(np.linspace(0.2, 0.8, nr_colors)) for name in base_cmaps]
    )

    colors = colors[0:nr_languages]
    cmap = matplotlib.colors.ListedColormap(colors)

    plt.scatter(
        population.geometry.x.values,
        population.geometry.y.values,
        c=population["language"],  # Convert language to list for coloring
        cmap=cmap,
        vmin=0,  # vmin and vmax define the range of the colormap
        vmax=nr_languages - 1,
        s=80,
        marker=".",
    )
    plt.xlim(0, x_max)
    plt.ylim(0, y_max)
    plt.colorbar()
    plt.title(f"Timestep {timestep}")
    plt.savefig(os.path.join(output_run, "Point_t" + str(timestep) + ".pdf"))
    plt.close()


def calculate_tick_intervals(min_val, max_val, max_ticks=10):
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


def language_number_plotje(
    output_run: str, languagenumber: list, nr_agents: int, nr_steps: int
):
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
    plt.savefig(os.path.join(output_run, "Number_of_Languages.pdf"))
    plt.close()
