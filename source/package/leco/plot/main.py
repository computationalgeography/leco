"""Create plots of the leco model output."""

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from .animation import plot_animation
from .summary import plot_summaries
from .threed_interactive import plot_3d_fig


def create_colormap(
    languages: gpd.GeoSeries,
    seed: int,
) -> tuple[mpl.colors.ListedColormap, dict[int, int]]:
    """Create a consistent colormap for the languages present in the simulation output."""
    rng = np.random.default_rng(seed)
    nr_colors = 20
    base_cmaps = ["Greys", "Purples", "Reds", "Blues", "Oranges", "Greens", "RdPu"]

    raw_colors = np.concatenate([plt.get_cmap(name)(np.linspace(0.2, 0.8, nr_colors)) for name in base_cmaps])

    rng.shuffle(raw_colors)
    # Create a deterministic assignment based on language names
    language_ids = sorted(languages.unique())

    # Create a hash-based assignment for consistent colors
    language_colors = {}
    for _i, language in enumerate(language_ids):
        # Use hash of language name to get consistent color index
        hash_val = hash(str(language)) % len(raw_colors)
        language_colors[language] = raw_colors[hash_val]

    # Create colormap from the assigned colors
    selected_colors = [language_colors[lang] for lang in language_ids]
    cmap = mpl.colors.ListedColormap(selected_colors)

    # Create a mapping from language to index for plotting
    lang_to_index = {lang: idx for idx, lang in enumerate(language_ids)}

    return cmap, lang_to_index  # , language_colors  # Return both for later use


def plot(
    input_file: list[str] | str,
    parameters: dict,
) -> None:
    """Create plots of the leco model output."""
    population = gpd.read_file(input_file)

    cmap, lang_to_index = create_colormap(population.language, parameters["initialization"]["seed"])

    print("Create summarizing plots of the leco model output")
    plot_summaries(input_file, cmap, lang_to_index)

    print("Create animation of the leco model output")
    plot_animation(input_file, parameters, cmap)

    print("Create 3D interactive plot of the leco model output")
    plot_3d_fig(input_file, cmap)
