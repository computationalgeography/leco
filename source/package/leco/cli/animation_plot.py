import os
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
from pathlib import Path


def prepare_animation_data(population: gpd.GeoDataFrame) -> list[dict]:
    """Create a list of dictionaries (frames) containing for each timestep the population data"""
    animation_data = []
    for timestep, group in population.groupby("timestep"):
        frame_data = {"timestep": timestep, "population": group.copy()}
        animation_data.append(frame_data)

    return animation_data


def create_scatterframe(data, ax, cmap, nr_languages, x_max, y_max) -> None:
    """Create a scatter plot for each frame of the animation"""
    timestep = data["timestep"]
    population = data["population"]

    ax.clear()
    scatter = ax.scatter(
        population.geometry.x,
        population.geometry.y,
        c=population.language,
        cmap=cmap,
        vmin=0,
        vmax=nr_languages - 1,
        s=80,
        marker=".",
    )

    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_title(f"Timestep {timestep}")

    return [scatter]


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


def create_animation(
    animation_data: list[dict],
    output_path: str,
    x_max: int,
    y_max: int,
    filename: str = "animation.gif",
) -> None:
    """Create an animated gif file of agents positions over time colored by language"""

    fig, ax = plt.subplots()

    # Extract unique languages
    all_languages = set()
    for data in animation_data:
        all_languages.update(data["population"]["language"])
    unique_languages = sorted(list(all_languages))
    nr_languages = len(unique_languages)

    # Create a colormap
    cmap = create_colormap(nr_languages)

    # Create the animation using the FuncAnimation class
    anim = FuncAnimation(
        fig,
        create_scatterframe,  # Function to create each frame
        fargs=(
            ax,
            cmap,
            nr_languages,
            x_max,
            y_max,
        ),  # Arguments for the frame function
        frames=animation_data,  # Pass the data for each frame to the function
        interval=500,  # The number of milliseconds between frames
        blit=False,
        repeat=True,  # Repeat the animation
    )

    # Save the animation as a GIF file
    gif_path = os.path.join(output_path, filename)
    print(gif_path)
    anim.save(gif_path, writer="pillow", fps=2)
    plt.close(fig)
    print(f"Animation saved: {gif_path}")


def plot_animation(input_file: str, parameters: dict) -> None:
    """Create an animation of the leco model output"""
    # Read in the population data across all timesteps
    population = gpd.read_file(input_file)

    # Access parameters from the output directory
    output_path = Path(input_file).parent

    if type(parameters["x_max"]) is not int:
        parameters["x_max"] = int(parameters["x_max"])
        parameters["y_max"] = int(parameters["y_max"])

    animation_data = prepare_animation_data(population)
    create_animation(
        animation_data,
        output_path,
        parameters["x_max"],
        parameters["y_max"],
    )
