import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib

from .summary_plots import plot_summaries
from .animation_plot import plot_animation
from .threed_interactive_plot import plot_3d_fig


def create_colormap(
    nr_languages: int, rng: np.random.default_rng
) -> matplotlib.colors.ListedColormap:
    """Create a colormap for the languages"""
    nr_colors = 20  # number of colors to extract from each of the base_cmaps below
    base_cmaps = ["Greys", "Purples", "Reds", "Blues", "Oranges", "Greens", "RdPu"]

    # Sample from linspace 0.2 to 0.8 to avoid having overly dark and light shades
    raw_colors = np.concatenate(
        [plt.get_cmap(name)(np.linspace(0.2, 0.8, nr_colors)) for name in base_cmaps]
    )

    nr_unique_colors = len(raw_colors)
    rng.shuffle(raw_colors)  # Shuffle colors

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


def plot(
    input_file: str,
    parameters: dict | None,
    summaries: bool,
    animation: bool,
    interactive: bool,
) -> None:
    """Create plots of the leco model output"""

    # Initialize seed
    if parameters is not None:
        rng = np.random.default_rng(int(parameters["seed"]))
    else:
        rng = np.random.default_rng(42)

    population = gpd.read_file(input_file)
    cmap = create_colormap(population.language.nunique(), rng)

    if summaries:
        print("Create summarizing plots of the leco model output")
        plot_summaries(input_file, cmap)

    if animation:
        print("Create animation of the leco model output")
        plot_animation(input_file, parameters, cmap)

    if interactive:
        print("Create 3D interactive plot of the leco model output")
        plot_3d_fig(input_file, cmap)
