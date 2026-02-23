"""Create a 3D interactive plot of the leco model output."""

from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt


def plot_3d_fig(
    input_file: Path,
    cmap: mpl.colors.ListedColormap,
) -> None:
    """Create a 3D interactive plot of the leco model output."""
    # Read in the population data across all time steps
    population = gpd.read_file(input_file)

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    ax.scatter(
        population.geometry.x,
        population.geometry.y,
        population.time_step,
        c=population["language"],
        cmap=cmap,
        s=1,
    )

    # Add lines to visualize the evolution of individual agents over time
    for _pid, group in population.groupby("id"):
        # Sort by time step to ensure correct line plotting
        group_sorted = group.sort_values("time_step")
        ax.plot(
            group_sorted.geometry.x,
            group_sorted.geometry.y,
            group_sorted.time_step,
            color="gray",
            linewidth=0.5,
            alpha=0.5,
        )

    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")
    ax.set_zlabel("Time step")

    plt.show()
