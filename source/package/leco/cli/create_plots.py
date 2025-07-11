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


def language_number_plotje(
    output_run: str, languagenumber: list, nr_agents: int, nr_steps: int
):
    """Plot number of languages over time"""
    plt.plot(languagenumber, marker="o")
    plt.xlabel("Timestep")
    plt.xticks(range(0, nr_steps + 1, 50), [str(i) for i in range(0, nr_steps + 1, 50)])
    plt.ylabel("Number of Languages")
    plt.yticks(
        range(0, nr_agents + 1, 50), [str(i) for i in range(0, nr_agents + 1, 50)]
    )
    plt.title("Number of Languages Over Time")
    plt.savefig(os.path.join(output_run, "Number_of_Languages.pdf"))
    plt.close()
