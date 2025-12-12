"""Create an animation of the leco model output."""

import ast
from pathlib import Path

import geopandas as gpd
import logging
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from shapely import Polygon

from ..model.initialization import initialize_barrier


def prepare_animation_data(population: gpd.GeoDataFrame) -> list[dict]:
    """Create a list of dictionaries (frames) containing for each time_step the population data."""
    animation_data = []
    for time_step, group in population.groupby("time_step"):
        frame_data = {"time_step": time_step, "population": group.copy()}
        animation_data.append(frame_data)

    return animation_data


def create_scatter_frame(
    data: gpd.GeoDataFrame,
    ax: plt.subplot,
    cmap: mpl.colors.ListedColormap,
    nr_languages: int,
    space: list[float, float],
    barrier: Polygon | None,
) -> list:
    """Create a scatter plot for each frame of the animation."""
    time_step = data["time_step"]
    time_step = time_step * 20  # Scale to represent years (assuming each time_step is 20 years)
    population = data["population"]

    ax.clear()

    # If a barrier is present in the simulation, add this to the plot
    if barrier:
        coords = list(barrier.exterior.coords)
        poly = mpl.patches.Polygon(
            coords,
            fill=True,
            color="peru",
            alpha=0.7,
            linewidth=None,
        )
        ax.add_patch(poly)

    scatter = ax.scatter(
        population.geometry.x,
        population.geometry.y,
        c=population["language"],
        cmap=cmap,
        vmin=0,
        vmax=nr_languages - 1,
        s=80,
        marker=".",
    )

    ax.set_xlim(0, space[0])
    ax.set_ylim(0, space[1])
    ax.set_xlabel("X Position (km)", size=13)
    ax.set_ylabel("Y Position (km)", size=13)
    ax.tick_params(axis="both", labelsize=10)
    ax.set_title(f"Year {time_step}", size=16)

    return [scatter]


def create_animation(
    animation_data: list[dict],
    output_path: Path,
    cmap: mpl.colors.ListedColormap,
    space: list[float, float],
    barrier: Polygon | None,
    filename: str = "animation.gif",
) -> None:
    """Create an animated gif file of agents positions over time colored by language."""
    fig, ax = plt.subplots()

    # Extract unique languages
    all_languages = set()
    for data in animation_data:
        all_languages.update(data["population"]["language"])

    unique_languages = sorted(all_languages)
    nr_languages = len(unique_languages)

    # Create the animation using the FuncAnimation class
    anim = FuncAnimation(
        fig,
        create_scatter_frame,  # Function to create each frame
        fargs=(
            ax,
            cmap,
            nr_languages,
            space,
            barrier,
        ),  # Arguments for the frame function
        frames=animation_data,  # Pass the data for each frame to the function
        interval=50,  # The number of milliseconds between frames, only for python environments
        blit=False,
        repeat=True,  # Repeat the animation
    )

    # Save the animation as a GIF file
    gif_path = output_path / filename
    logging.debug(gif_path)
    anim.save(gif_path, writer="pillow", fps=4)  # fps is frames per second
    plt.close(fig)
    logging.debug(f"Animation saved: {gif_path}")


def string_to_float_list(string: str) -> list[float]:
    """Change string to a list of float values."""
    values = ast.literal_eval(string)
    a, b = float(values[0]), float(values[1])

    return [a, b]


def plot_animation(
    input_file: Path,
    parameters: dict,
    cmap: mpl.colors.ListedColormap,
) -> None:
    """Create an animation of the leco model output."""
    # Read in the population data across all time_steps
    population = gpd.read_file(input_file)

    # Access parameters from the output directory
    output_path = input_file.parent

    # Initialize a spatial barrier if specified
    barrier = None
    if parameters["barrier"]["present"] == "True":
        barrier = initialize_barrier(
            parameters["space"]["shape"],
            string_to_float_list(parameters["barrier"]["x_extend"]),
            string_to_float_list(parameters["barrier"]["y_extend"]),
        )

    animation_data = prepare_animation_data(population)
    create_animation(
        animation_data,
        output_path,
        cmap,
        parameters["space"]["shape"],
        barrier,
    )
