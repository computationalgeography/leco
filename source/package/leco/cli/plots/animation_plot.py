import os
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
from pathlib import Path
from shapely import Polygon, box


def prepare_animation_data(population: gpd.GeoDataFrame) -> list[dict]:
    """Create a list of dictionaries (frames) containing for each timestep the population data"""
    animation_data = []
    for timestep, group in population.groupby("timestep"):
        frame_data = {"timestep": timestep, "population": group.copy()}
        animation_data.append(frame_data)

    return animation_data


def create_scatterframe(
    data: gpd.GeoDataFrame,
    ax: plt.subplot,
    cmap: matplotlib.colors.ListedColormap,
    nr_languages: int,
    x_max: int,
    y_max: int,
    barrier: Polygon | None,
) -> None:
    """Create a scatter plot for each frame of the animation"""
    timestep = data["timestep"]
    population = data["population"]

    ax.clear()

    # If a barrier is present in the simulation, add this to the plot
    if barrier:
        coords = list(barrier.exterior.coords)
        poly = matplotlib.patches.Polygon(
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


def create_animation(
    animation_data: list[dict],
    output_path: str,
    cmap: matplotlib.colors.ListedColormap,
    x_max: int,
    y_max: int,
    barrier: Polygon | None,
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
            barrier,
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


def initialize_barrier(
    max_coor: float,
    bar_x: float,
    bar_y: float,
    bar_x_radius: float,
    bar_y_radius: float,
) -> Polygon:
    """Initialize a barrier in the continuous space"""

    barrier = Polygon(
        [
            (bar_x - bar_x_radius, bar_y - bar_y_radius),
            (bar_x + bar_x_radius, bar_y - bar_y_radius),
            (bar_x + bar_x_radius, bar_y + bar_y_radius),
            (bar_x - bar_x_radius, bar_y + bar_y_radius),
        ]
    )

    # Bounding box for valid space
    bounds = box(0, 0, max_coor, max_coor)

    # Clip the barrier to fit inside the bounds
    barrier_clipped = barrier.intersection(bounds)

    # Check if clipping occurred
    if not barrier_clipped.equals(barrier):
        print(
            f"Warning: Barrier at ({bar_x:.2f}, {bar_y:.2f}) was clipped to fit within "
            f"bounds [0, {max_coor}]"
        )

    return barrier_clipped


def plot_animation(
    input_file: str, parameters: dict, cmap: matplotlib.colors.ListedColormap
) -> None:
    """Create an animation of the leco model output"""
    # Read in the population data across all timesteps
    population = gpd.read_file(input_file)

    # Access parameters from the output directory
    output_path = Path(input_file).parent

    if type(parameters["x_max"]) is not int:
        parameters["x_max"] = int(parameters["x_max"])
        parameters["y_max"] = int(parameters["y_max"])

    # Initialize a spatial barrier if specified
    barrier = None
    if parameters["barrier"] == "True":
        barrier = initialize_barrier(
            parameters["x_max"],
            float(parameters["bar_x"]),
            float(parameters["bar_y"]),
            float(parameters["bar_x_radius"]),
            float(parameters["bar_y_radius"]),
        )

    animation_data = prepare_animation_data(population)
    create_animation(
        animation_data,
        output_path,
        cmap,
        parameters["x_max"],
        parameters["y_max"],
        barrier,
    )
