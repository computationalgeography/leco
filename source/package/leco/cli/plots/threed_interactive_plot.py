import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib


def plot_3d_fig(
    input_file: str,
    cmap: matplotlib.colors.ListedColormap,
) -> None:
    """Create a 3D interactive plot of the leco model output"""
    # Read in the population data across all timesteps
    population = gpd.read_file(input_file)

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    ax.scatter(
        population.geometry.x,
        population.geometry.y,
        population.timestep,
        c=population["language"],
        cmap=cmap,
        s=1,
    )

    # Add lines to visualize the evolution of individual agents over time
    for pid, group in population.groupby("id"):
        # Sort by timestep to ensure correct line plotting
        group = group.sort_values("timestep")
        ax.plot(
            group.geometry.x,
            group.geometry.y,
            group.timestep,
            color="gray",
            linewidth=0.5,
            alpha=0.5,
        )

    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")
    ax.set_zlabel("Timestep")

    plt.show()
