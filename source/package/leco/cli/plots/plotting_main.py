import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib

from .summary_plots import plot_summaries
from .animation_plot import plot_animation
from .threed_interactive_plot import plot_3d_fig


def create_colormap(
    languages: gpd.GeoSeries,
) -> tuple[matplotlib.colors.ListedColormap, dict[int, int]]:
    nr_colors = 20
    base_cmaps = ["Greys", "Purples", "Reds", "Blues", "Oranges", "Greens", "RdPu"]

    raw_colors = np.concatenate(
        [plt.get_cmap(name)(np.linspace(0.2, 0.8, nr_colors)) for name in base_cmaps]
    )

    # Create a deterministic assignment based on language names
    language_ids = sorted(languages.unique())

    # Create a hash-based assignment for consistent colors
    language_colors = {}
    for i, language in enumerate(language_ids):
        # Use hash of language name to get consistent color index
        hash_val = hash(str(language)) % len(raw_colors)
        language_colors[language] = raw_colors[hash_val]

    # Create colormap from the assigned colors
    selected_colors = [language_colors[lang] for lang in language_ids]
    cmap = matplotlib.colors.ListedColormap(selected_colors)

    # Create a mapping from language to index for plotting
    lang_to_index = {lang: idx for idx, lang in enumerate(language_ids)}

    return cmap, lang_to_index  # , language_colors  # Return both for later use


def plot(
    input_file: str,
    parameters: dict,
    summaries: bool,
    animation: bool,
    interactive: bool,
) -> None:
    """Create plots of the leco model output"""

    population = gpd.read_file(input_file)

    cmap, lang_to_index = create_colormap(population.language)

    if summaries:
        print("Create summarizing plots of the leco model output")
        plot_summaries(input_file, cmap, lang_to_index)

    if animation:
        print("Create animation of the leco model output")
        plot_animation(input_file, parameters, cmap)

    if interactive:
        print("Create 3D interactive plot of the leco model output")
        plot_3d_fig(input_file, cmap)
