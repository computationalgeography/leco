"""Create plots of the leco model output."""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from .phylogenetic_plots import visualize_phylogenies
from .phylogenetic_tree import create_phylogeny
from .summary import plot_summaries


def create_colormap(
    languages: gpd.GeoSeries,
    seed: int,
) -> tuple[ListedColormap, dict[int, int]]:
    """Create a consistent colormap for the languages present in the simulation output."""
    rng = np.random.default_rng(seed)
    nr_colors = 20
    base_colour_maps = ["Greys", "Purples", "Reds", "Blues", "Oranges", "Greens", "RdPu"]

    raw_colors = np.concatenate(
        [plt.get_cmap(name)(np.linspace(0.2, 0.8, nr_colors)) for name in base_colour_maps],
    )

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
    cmap = ListedColormap(selected_colors)

    # Create a mapping from language to index for plotting
    lang_to_index = {lang: idx for idx, lang in enumerate(language_ids)}

    return cmap, lang_to_index  # , language_colors  # Return both for later use


def plot(
    directory: Path,
    gpkg_file_name: str,
    parameters: dict,
    k_steps: int = 4,
) -> None:
    """Create plots of the leco model output."""
    population = gpd.read_file(directory / gpkg_file_name)

    plot_summaries(directory, population)

    create_phylogeny(directory, population, parameters["initialization"]["steps"])
    # Select equally distributed steps that you want to visualize spatial distribution of based on k_steps
    steps = np.round(np.linspace(0, parameters["initialization"]["steps"], num=k_steps)).astype(int).tolist()
    visualize_phylogenies(directory, population, steps)
