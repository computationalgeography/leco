"""Create phylogenetic visualizations of language evolution over time."""

import logging
from pathlib import Path

import geopandas as gpd
from matplotlib.image import imread
import matplotlib.pyplot as plt
import pandas as pd

import ete3

# from ete3 import Tree, TreeStyle, NodeStyle
import networkx as nx


def postprocess_phylogeny(population: gpd) -> pd.DataFrame:
    records = []
    for lang, group in population.groupby("language"):
        t_birth = group["time_step"].min()
        t_extinct = group["time_step"].max()
        if t_birth == 0:
            parent_languages = ""
            # This should be the -1 parent_id
        else:
            # Get the parent ids of agents speaking this language at t_birth
            parent_ids = (
                group[group["time_step"] == t_birth]["id"].dropna().unique().tolist()
            )
            # Find what languages those parents spoke at t_birth - 1
            parent_languages = (
                population[
                    (population["time_step"] == t_birth - 1)
                    & (population["id"].isin(parent_ids))
                ]["language"]
                .unique()
                .tolist()
            )

        records.append(
            {
                "language": lang,
                "t_birth": t_birth,
                "t_extinct": t_extinct,
                "parents": parent_languages,
            }
        )

    return pd.DataFrame(records)


def test_ete3(df, OutputPath: Path):
    # Build graph first
    G = nx.DiGraph()
    for row in df.itertuples(index=False):
        G.add_node(row.language, t_birth=row.t_birth, t_extinct=row.t_extinct)
    for row in df.itertuples(index=False):
        for p in row.parents:
            G.add_edge(p, row.language)

    # Find roots and their subgraphs
    roots = [n for n in G.nodes if G.in_degree(n) == 0]

    def build_newick_subtree(node, graph, internal_node_id=[1000]):
        children = list(graph.successors(node))
        if not children:
            return str(node)
        else:
            subtrees = [
                build_newick_subtree(c, graph, internal_node_id) for c in children
            ]
            # Create an internal node label for this node to keep track
            internal_label = f"InternalNode{internal_node_id[0]}"
            internal_node_id[0] += 1
            # Return Newick string with internal node label
            return "(" + ",".join(subtrees) + ")" + internal_label

    # Ensure output directory exists
    OutputPath.mkdir(parents=True, exist_ok=True)

    for root in roots:
        newick = build_newick_subtree(root, G) + ";"
        print(f"Rendering root {root} with newick:\n{newick}\n")

        tree = Tree(newick, format=1)

        # Add node styles and features if needed
        for node in tree.traverse():
            if node.is_leaf():
                # Only leaf nodes have language IDs
                lang_id = int(node.name)
                if lang_id in G.nodes:
                    node.add_feature("t_birth", G.nodes[lang_id]["t_birth"])
                    node.add_feature("t_extinct", G.nodes[lang_id]["t_extinct"])
            else:
                node.add_feature("t_birth", 0)
                node.add_feature("t_extinct", 100)

            node_style = NodeStyle()
            if node.is_leaf():
                node_style["fgcolor"] = "darkgreen"
                node_style["shape"] = "sphere"
                node_style["size"] = 8
            else:
                node_style["fgcolor"] = "darkblue"
                node_style["shape"] = "circle"
                node_style["size"] = 10
            node.set_style(node_style)

        ts = TreeStyle()
        ts.show_leaf_name = True
        ts.show_branch_length = False
        ts.scale = 300  # Control scale of tree (adjust as needed)
        ts.title.add_face(
            ete3.TextFace(f"Phylogeny from root {root}", fsize=14), column=0
        )
        ts.show_scale = True

    # Create figure with subplots (one per root)
    fig, axes = plt.subplots(1, len(roots), figsize=(5 * len(roots), 8))
    if len(roots) == 1:
        axes = [axes]

    for i, root in enumerate(sorted(roots)):
        ax = axes[i]

        # Build Newick for this root's subgraph only
        newick = build_newick_subtree(root, G) + ";"
        tree = Tree(newick, format=1)

        # Safe node processing (only leaves get lang_id lookup)
        for node in tree.traverse():
            if node.is_leaf():
                try:
                    lang_id = int(node.name)
                    if lang_id in G.nodes:
                        node.add_feature("t_birth", G.nodes[lang_id]["t_birth"])
                        node.add_feature("t_extinct", G.nodes[lang_id]["t_extinct"])
                except Exception as e:
                    logging.error(f"! Error adding features to node {node.name}: {e}")
            else:
                node.add_feature("t_birth", 0)
                node.add_feature("t_extinct", 100)

            # Style nodes
            node_style = NodeStyle()
            if node.is_leaf():
                node_style["fgcolor"] = "darkgreen"
                node_style["size"] = 8
            else:
                node_style["fgcolor"] = "darkblue"
                node_style["size"] = 6
            node.set_style(node_style)

        # ETE3 TreeStyle
        ts = TreeStyle()
        ts.show_leaf_name = True
        ts.scale = 400
        ts.show_branch_length = False

        # Render THIS tree to THIS subplot axis
        out_file = OutputPath / f"temp_root_{root}.png"
        tree.render(str(out_file), tree_style=ts, w=500, h=600)

        img = imread(str(out_file))
        ax.imshow(img, aspect="auto")
        ax.set_title(f"Root {root}", fontsize=14, fontweight="bold")
        ax.axis("off")  # Hide axes

    plt.suptitle("Language Phylogenies by Root (ETE3)", fontsize=16)
    plt.tight_layout()
    plt.savefig(
        OutputPath / "combined_ete3_phylogenies.png", dpi=300, bbox_inches="tight"
    )


def create_phylogeny(input_file: Path):
    # Read in the population data across all time steps
    population = gpd.read_file(input_file)
    output_path = input_file.parent

    evolution = postprocess_phylogeny(population)
    logging.debug(print(evolution))

    test_ete3(evolution, output_path)
