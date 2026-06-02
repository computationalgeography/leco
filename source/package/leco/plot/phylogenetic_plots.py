"""Create phylogenetic visualizations of language evolution over time."""

import logging
from collections import defaultdict
from io import StringIO
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from Bio import Phylo
from Bio.Phylo import BaseTree
from matplotlib.axes._axes import Axes
from matplotlib.lines import Line2D

logger = logging.getLogger(__name__)


def plot_phylogenies(
    per_root_trees: dict[str, str],
    output_path: Path,
    speaker_size_extant: dict[int, int],
    show_singleton_nodes: bool = True,
) -> None:
    """Plot phylogenetic trees; optionally show singleton (non-split) internal nodes."""
    n_trees = len(per_root_trees)
    fig, axes = plt.subplots(n_trees, 1, figsize=(10, 3.5 * n_trees), constrained_layout=True)
    if n_trees == 1:
        axes = [axes]

    parsed: list[tuple[str, BaseTree.Tree]] = []
    for root_label, newick in per_root_trees.items():
        tree = Phylo.read(StringIO(newick), "newick")  # type: ignore[arg-type]
        tree.ladderize()
        parsed.append((root_label, tree))

    def _max_depth(tree: BaseTree.Tree) -> float:
        return max(tree.distance(tip) for tip in tree.get_terminals())

    global_max_depth = max(_max_depth(tree) for _, tree in parsed)

    for ax, (root_label, tree) in zip(axes, parsed, strict=True):
        root_int = int(root_label.rsplit("_", maxsplit=1)[-1])
        color = ROOT_FAMILY_COLORS[root_int % len(ROOT_FAMILY_COLORS)]
        _draw_phylo_ax(
            ax,
            tree,
            root_label,
            color,
            global_max_depth,
            speaker_size_extant,
            show_singleton_nodes,
        )

    suffix = "_with_nodes" if show_singleton_nodes else ""
    fig.savefig(output_path / f"Phylogenies{suffix}.png", bbox_inches="tight", dpi=150)
    plt.close(fig)


def draw_clade(
    clade: BaseTree.Clade,
    x_start: float,
    ax: Axes,
    color: tuple[float, float, float, float],
    lw: float,
    global_max_depth: float,
    speaker_size_extant: dict[int, int],
    show_singleton_nodes: bool,
    tip_y: dict[int, float],
) -> None:
    """Draw a single clade onto ax with consistent x-axis scaling."""
    branch_len = clade.branch_length if clade.branch_length is not None else 0.0
    x_end = x_start + branch_len
    y = tip_y[id(clade)]
    ax.plot([x_start, x_end], [y, y], color=color, lw=lw, solid_capstyle="butt")

    if clade.is_terminal():

        def normalize_name(name: str | None) -> int:
            if name is None:
                return 0
            # Remove version values to match with speaker size dictionary
            base = name.split("_", maxsplit=1)[0]
            # Return as an integer
            return int(base)

        n_speakers = int(speaker_size_extant.get(normalize_name(clade.name), 0))
        ax.text(
            x_end + global_max_depth * 0.01,
            y,
            str(n_speakers),
            va="center",
            ha="left",
            fontsize=8,
            color=color,
        )
    else:
        is_singleton = len(clade.clades) == 1
        if show_singleton_nodes and is_singleton:
            ax.scatter([x_end], [y], s=18, color=color, zorder=3)

        child_ys = [tip_y[id(c)] for c in clade.clades]
        ax.plot([x_end, x_end], [min(child_ys), max(child_ys)], color=color, lw=lw, solid_capstyle="butt")
        for child in clade.clades:
            draw_clade(
                child,
                x_end,
                ax,
                color,
                lw,
                global_max_depth,
                speaker_size_extant,
                show_singleton_nodes,
                tip_y,
            )


def _draw_phylo_ax(
    ax: Axes,
    tree: BaseTree.Tree,
    root_label: str,
    color: tuple[float, float, float, float],
    global_max_depth: float,
    speaker_size_extant: dict[int, int],
    show_singleton_nodes: bool = True,
) -> None:
    """Draw a single phylogenetic tree onto ax with consistent x-axis scaling."""
    terminals = tree.get_terminals()
    n_tips = len(terminals)
    tip_y: dict[int, float] = {id(tip): i for i, tip in enumerate(terminals)}

    def assign_y(clade: BaseTree.Clade) -> float:
        if clade.is_terminal():
            return tip_y[id(clade)]
        child_ys = [assign_y(c) for c in clade.clades]
        y = sum(child_ys) / len(child_ys)
        tip_y[id(clade)] = y
        return y

    assign_y(tree.root)

    def get_y(clade: BaseTree.Clade) -> float:
        return tip_y[id(clade)]

    lw = 1.5

    root = tree.root
    child_ys = [get_y(c) for c in root.clades]
    ax.plot([0, 0], [min(child_ys), max(child_ys)], color=color, lw=lw, solid_capstyle="butt")
    for child in root.clades:
        draw_clade(
            child,
            0.0,
            ax,
            color,
            lw,
            global_max_depth,
            speaker_size_extant,
            show_singleton_nodes,
            tip_y,
        )

    ax.set_xlim(-global_max_depth * 0.05, global_max_depth * 1.1)
    ax.set_ylim(-0.8, n_tips - 0.2)
    ax.set_title(f"Root: {root_label}", loc="left", fontsize=11, fontweight="bold", color=color, pad=4)
    ax.set_yticks([])
    ax.spines[["left", "top", "right"]].set_visible(False)
    ax.spines["bottom"].set_color("lightgrey")
    ax.tick_params(axis="x", labelsize=8, colors="grey")
    ax.set_xlabel("Time steps", fontsize=8, color="grey")


ROOT_FAMILY_COLORS = [
    (0.894, 0.102, 0.110, 1.0),  # vivid red
    (0.216, 0.494, 0.722, 1.0),  # strong blue
    (0.302, 0.686, 0.290, 1.0),  # vivid green
    (1.000, 0.498, 0.000, 1.0),  # vivid orange
    (0.596, 0.306, 0.639, 1.0),  # purple
    (1.000, 1.000, 0.200, 1.0),  # yellow
    (0.651, 0.337, 0.157, 1.0),  # brown
    (0.969, 0.506, 0.749, 1.0),  # pink
]

LANG_MARKERS = [
    "o",  # circle
    "s",  # square
    "^",  # triangle up
    "D",  # diamond
    "v",  # triangle down
    "P",  # plus (filled)
    "X",  # x (filled)
    "*",  # star
    "h",  # hexagon 1
    "p",  # pentagon
    "<",  # triangle left
    ">",  # triangle right
    "H",  # hexagon 2
    "d",  # thin diamond
    "8",  # octagon
    "1",  # tri down (Y-shape)
    "2",  # tri up
    "3",  # tri left
    "4",  # tri right
    "+",  # plus (unfilled)
    "x",  # x (unfilled)
    "|",  # vline
    "_",  # hline
]


def colors_and_markers_for_languages_by_root(
    lang_to_root: dict,
) -> tuple[dict, dict]:
    """Assign each language a color (same per root) and a unique marker shape (per language within root)."""
    roots_ordered = sorted(set(lang_to_root.values()), key=str)
    by_root: dict = defaultdict(list)
    for lang, root in lang_to_root.items():
        by_root[root].append(lang)
    for langs in by_root.values():
        langs.sort(key=str)

    lang_color: dict = {}
    lang_marker: dict = {}

    if len(ROOT_FAMILY_COLORS) < len(roots_ordered):
        logger.warning("Not enough colors for the number of root languages.")

    for root in roots_ordered:
        color = ROOT_FAMILY_COLORS[root % len(ROOT_FAMILY_COLORS)]
        if len(LANG_MARKERS) < len(by_root):
            logger.warning("Not enough markers for the number of languages.")
        for lang_idx, lang in enumerate(by_root[root]):
            lang_color[lang] = color
            lang_marker[lang] = LANG_MARKERS[lang_idx % len(LANG_MARKERS)]

    return lang_color, lang_marker


def plot_spatial_families(
    population: gpd.GeoDataFrame,
    output_path: Path,
    time_step: int,
    lang_to_root: dict[int, int],
) -> None:
    """Scatter agents at final step; hue by phylogenetic root, shape by language within root."""
    lang_color, lang_marker = colors_and_markers_for_languages_by_root(lang_to_root)

    default_color = (0.6, 0.6, 0.6, 1.0)
    default_marker = "o"

    _fig, ax = plt.subplots(figsize=(8, 8))

    # Group by (color, marker) and scatter each group together
    groups: dict = defaultdict(list)
    for _, row in population.iterrows():
        lang = row["language"]
        color = lang_color.get(lang, default_color)
        marker = lang_marker.get(lang, default_marker)
        groups[(lang, color, marker)].append((row.geometry.x, row.geometry.y))

    for (lang, color, marker), coords in groups.items():
        xs, ys = zip(*coords, strict=True)
        ax.scatter(xs, ys, c=[color], s=80, marker=marker, label=lang)

    # Build legend: one section per root with color swatch + per-language shape entries
    roots_ordered = sorted(set(lang_to_root.values()), key=str)
    by_root: dict = defaultdict(list)
    for lang, root in lang_to_root.items():
        by_root[root].append(lang)

    legend_handles = []
    for root_idx, root in enumerate(roots_ordered):
        color = ROOT_FAMILY_COLORS[root_idx % len(ROOT_FAMILY_COLORS)]
        langs = sorted(by_root[root], key=str)
        # Root label (filled square as color swatch)
        legend_handles.append(
            Line2D(
                [],
                [],
                marker="s",
                color="w",
                markerfacecolor=color,
                markersize=10,
                label=f"Root {root}",
                markeredgecolor="grey",
            ),
        )
        # One entry per language showing its shape
        for lang_idx, lang in enumerate(langs):
            marker = LANG_MARKERS[lang_idx % len(LANG_MARKERS)]
            legend_handles.append(
                Line2D(
                    [],
                    [],
                    marker=marker,
                    color="w",
                    markerfacecolor=color,
                    markersize=6,
                    label=f"  {lang}",
                ),
            )

    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_title("Agents colored by phylogenetic root (shape = language)", size=14)

    plt.savefig(output_path / f"Phylospatial_step{time_step}.png", bbox_inches="tight", dpi=150)
    plt.close()


def read_newick_trees(input_dir: Path) -> dict[str, str]:
    """Read Newick tree format and sets to dictionary."""
    return {
        path.stem: path.read_text(encoding="utf-8").strip() for path in sorted(input_dir.glob("*.newick"))
    }


def visualize_phylogenies(input_dir: Path, time_steps: list[int], population: gpd.GeoDataFrame) -> None:
    """Visualize the phylogenetic relations in phylogenies and spatial plots."""
    # Read in the Newick trees
    per_root_trees = read_newick_trees(input_dir)
    phylogeny_information = pd.read_csv(input_dir / "phylogeny_information.csv")

    for step in time_steps:
        step_population = gpd.GeoDataFrame(
            population[population["time_step"] == step],
            geometry=population.geometry.name,
            crs=population.crs,
        )

        lang_to_root = (
            phylogeny_information[
                (phylogeny_information["t_birth"] <= step) & (phylogeny_information["t_extinct"] >= step)
            ]
            .set_index("language")["root_language"]
            .to_dict()
        )
        plot_spatial_families(step_population, input_dir, step, lang_to_root)

        # If this is final step, create the phylogenies of the extant languages
        if step == max(time_steps):
            # Find speaker sizes of extant languages (present at final time step)
            speaker_size_extant = step_population.groupby("language")["id"].count().to_dict()
            plot_phylogenies(per_root_trees, input_dir, speaker_size_extant, True)
            plot_phylogenies(per_root_trees, input_dir, speaker_size_extant, False)
