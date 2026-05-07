"""Create an animation of the leco model output."""

import ast
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
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
    space: list[float, float],
    barrier: Polygon | None,
    lang_to_index: dict[int, int] | None,
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
        vmax=len(lang_to_index) - 1,
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
    lang_to_index: dict[int, int] | None,
    filename: str = "animation.gif",
) -> None:
    """Create an animated gif file of agents positions over time colored by language."""
    fig, ax = plt.subplots()

    # Extract unique languages
    all_languages = set()
    for data in animation_data:
        all_languages.update(data["population"]["language"])

    # Create the animation using the FuncAnimation class
    anim = FuncAnimation(
        fig,
        create_scatter_frame,  # Function to create each frame
        fargs=(
            ax,
            cmap,
            space,
            barrier,
            lang_to_index,
        ),  # Arguments for the frame function
        frames=animation_data,  # Pass the data for each frame to the function
        interval=40,  # The number of milliseconds between frames, only for python environments
        blit=False,
        repeat=True,  # Repeat the animation
    )

    # Save the animation as a GIF file
    gif_path = output_path / filename
    anim.save(gif_path, writer="pillow", fps=8)  # fps is frames per second
    plt.close(fig)


def string_to_float_list(string: str) -> list[float]:
    """Change string to a list of float values."""
    values = ast.literal_eval(string)
    a, b = float(values[0]), float(values[1])

    return [a, b]


def plot_animation(
    input_file: Path,
    parameters: dict,
    cmap: mpl.colors.ListedColormap,
    lang_to_index: dict[int, int] | None,
) -> None:
    """Create an animation of the leco model output."""
    # Read in the population data across all time_steps
    population = gpd.read_file(input_file)
    max_step = population["time_step"].max()
    cutoff = max_step - 499
    population = population[population["time_step"] >= cutoff]

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
        lang_to_index,
    )


def plot_last_time_step_bold(
    input_file: Path,
    parameters: dict,
    cmap: mpl.colors.ListedColormap,
    lang_to_index: dict[int, int] | None,
) -> None:
    """Create a static scatter plot for the last simulated time step."""
    # Read all time steps and select the last one
    population = gpd.read_file(input_file)
    max_step = population["time_step"].max()
    last = population[population["time_step"] == max_step]

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

    _fig, ax = plt.subplots()

    # Draw barrier if present
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

    # Map language IDs to indices for color mapping, using lang_to_index if provided
    if lang_to_index is not None:
        indices = last["language"].map(lang_to_index).to_numpy()
    else:
        indices, _ = pd.factorize(last["language"])

    ax.scatter(
        last.geometry.x,
        last.geometry.y,
        c=indices,
        cmap=cmap,
        vmin=0,
        vmax=indices.max() if indices.size > 0 else 0,
        s=80,
        marker=".",
    )

    # Overlay for language 1 — larger with edge
    lang_id = 112014
    lang1_mask = last["language"] == lang_id
    print(last["language"])
    if lang1_mask.any():
        lang1_indices = indices[lang1_mask.to_numpy()]
        ax.scatter(
            last[lang1_mask].geometry.x,
            last[lang1_mask].geometry.y,
            c=lang1_indices,
            cmap=cmap,
            vmin=0,
            vmax=indices.max() if indices.size > 0 else 0,
            s=82,
            marker="o",
        )

    # Use the same spatial extent as in the animation
    """ ax.set_xlim(0, space[0])
    ax.set_ylim(0, space[1])
    ax.set_xlabel("X Position (km)", size=20)
    ax.set_ylabel("Y Position (km)", size=20)
    ax.tick_params(axis="both", labelsize=16) """
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_title(f"Year {max_step * 20}", size=20)

    plt.savefig(output_path / f"LastStep_{lang_id}.pdf", bbox_inches="tight")
    plt.close()


def plot_last_time_step(
    input_file: Path,
    parameters: dict,
    cmap: mpl.colors.ListedColormap,
    lang_to_index: dict[int, int] | None,
) -> None:
    """Create a static scatter plot for the last simulated time step."""
    # Read all time steps and select the last one
    population = gpd.read_file(input_file)
    max_step = population["time_step"].max()
    last = population[population["time_step"] == max_step]

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

    _fig, ax = plt.subplots()

    # Draw barrier if present
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

    # Map language IDs to indices for color mapping, using lang_to_index if provided
    if lang_to_index is not None:
        indices = last["language"].map(lang_to_index).to_numpy()
    else:
        indices, _ = pd.factorize(last["language"])

    ax.scatter(
        last.geometry.x,
        last.geometry.y,
        c=indices,
        cmap=cmap,
        vmin=0,
        vmax=indices.max() if indices.size > 0 else 0,
        s=80,
        marker=".",
    )

    # Use the same spatial extent as in the animation
    """ ax.set_xlim(0, space[0])
    ax.set_ylim(0, space[1])
    ax.set_xlabel("X Position (km)", size=20)
    ax.set_ylabel("Y Position (km)", size=20)
    ax.tick_params(axis="both", labelsize=16) """
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_title(f"Year {max_step * 20}", size=20)

    plt.savefig(output_path / "LastStep.pdf", bbox_inches="tight")
    plt.close()


def plot_first_time_step(
    input_file: Path,
    parameters: dict,
    cmap: mpl.colors.ListedColormap,
    lang_to_index: dict[int, int] | None,
) -> None:
    """Create a static scatter plot for the last simulated time step."""
    # Read all time steps and select the last one
    population = gpd.read_file(input_file)
    min_step = population["time_step"].min()
    last = population[population["time_step"] == min_step]

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

    _fig, ax = plt.subplots()

    # Draw barrier if present
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

    # Map language IDs to indices for color mapping, using lang_to_index if provided
    if lang_to_index is not None:
        indices = last["language"].map(lang_to_index).to_numpy()
    else:
        indices, _ = pd.factorize(last["language"])

    ax.scatter(
        last.geometry.x,
        last.geometry.y,
        c=indices,
        cmap=cmap,
        vmin=0,
        vmax=indices.max() if indices.size > 0 else 0,
        s=80,
        marker=".",
    )

    # Use the same spatial extent as in the animation
    """ ax.set_xlim(0, space[0])
    ax.set_ylim(0, space[1])
    ax.set_xlabel("X Position (km)", size=20)
    ax.set_ylabel("Y Position (km)", size=20)
    ax.tick_params(axis="both", labelsize=16) """
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_title(f"Year {min_step * 20}", size=20)

    plt.savefig(output_path / "FirstStep.pdf", bbox_inches="tight")  # , dpi=300)
    plt.close()
