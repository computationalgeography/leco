import os
from pathlib import Path
import geopandas as gpd
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

## UNDER DEVELOPMENT


def postprocess_phylogeny(population: gpd) -> pd.DataFrame:
    records = []
    for lang, group in population.groupby("language"):
        t_birth = group["time_step"].min()
        t_extinct = group["time_step"].max()
        if t_birth == 0:
            parents = "init"
            roots = lang  # The language itself is the root if it starts at time_step 0
        else:
            agent_ids_t_birth = group.loc[group["time_step"] == t_birth, "id"].tolist()
            parents = population.loc[
                (population["time_step"] == t_birth - 1) & (population["id"].isin(agent_ids_t_birth)),
                "language",
            ].tolist()
            roots = None  # We'll fill this in later

        records.append(
            {
                "language": lang,
                "t_birth": t_birth,
                "t_extinct": t_extinct,
                "parents": parents,
                "roots": roots,
            }
        )

    languages = pd.DataFrame(records)

    # Fill in the roots for non-init languages by tracing back to origins
    def find_root(lang_name, languages_df):
        """Recursively find the root ancestor of a language"""
        lang_data = languages_df[languages_df["language"] == lang_name]
        if lang_data.empty:
            return None

        parents = lang_data.iloc[0]["parents"]
        if parents == "init":
            return lang_name  # This language is itself a root
        elif isinstance(parents, list) and len(parents) > 0:
            # For languages with multiple parents, we'll use the first parent's root
            # You might want to modify this logic based on your specific needs
            return find_root(parents[0], languages_df)
        else:
            return None

    # Update roots for all non-init languages
    for idx, row in languages.iterrows():
        if row["roots"] is None:
            languages.at[idx, "roots"] = find_root(row["language"], languages)

    return languages


def visualize_phylogeny(languages: pd.DataFrame):
    G = nx.DiGraph()

    for _, row in languages.iterrows():
        lang = row["language"]
        t_birth = row["t_birth"]
        t_extinct = row["t_extinct"]
        parents = row["parents"]

        G.add_node(lang, birth=t_birth, death=t_extinct)

        if parents != "init":
            for p in parents:
                G.add_edge(p, lang)

    """ # Draw the graph
    pos = nx.spring_layout(G)  # or use a tree layout
    nx.draw(G, pos, with_labels=True, node_size=800, node_color="skyblue", arrows=True)
    plt.title("Language Phylogeny")
    plt.show() """

    # --- Manual layout: y = t_birth, x = index or hash for spacing --- IS NOT WORKING UGHH
    pos = {}
    used_x = {}
    for i, node in enumerate(G.nodes):
        birth = G.nodes[node].get("t_birth", 0)

        # Assign x position based on number of existing nodes with the same birth time
        if birth not in used_x:
            used_x[birth] = 0
        x = used_x[birth]
        used_x[birth] += 1

        pos[node] = (x, -birth)  # negative birth time → top = older

    # --- Draw the graph ---
    plt.figure(figsize=(12, 8))

    # Draw edges and nodes
    nx.draw(
        G,
        pos,
        with_labels=True,
        arrows=True,
        node_color="skyblue",
        node_size=800,
        font_size=10,
    )

    # Add a time axis
    y_ticks = sorted(set(-G.nodes[n]["t_birth"] for n in G.nodes if G.nodes[n]["t_birth"] is not None))
    plt.yticks(y_ticks, labels=[-y for y in y_ticks])
    plt.xlabel("Lineage")
    plt.ylabel("Time (time step)")
    plt.title("Language Phylogeny with Time Axis")
    plt.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_language_phylogeny(languages):
    """
    Create a phylogenetic visualization with temporal evolution and tree-like branching
    """
    fig, ax = plt.subplots(figsize=(14, 10))

    # Create hierarchical positioning based on parent-child relationships
    y_positions = {}

    def assign_position_hierarchically(lang_data, languages_df):
        """Recursively assign positions keeping families together"""
        # Start with root languages (those with "init" parents)
        roots = languages_df[languages_df["parents"] == "init"].copy()
        current_y = 0

        # Process each root and its descendants
        for _, root in roots.iterrows():
            current_y = assign_subtree_positions(root["language"], languages_df, current_y)

        return current_y

    def assign_subtree_positions(lang, languages_df, start_y):
        """Assign positions for a language and all its descendants"""
        if lang in y_positions:
            return start_y

        # Assign position to current language
        y_positions[lang] = start_y
        current_y = start_y + 1

        # Find all children of this language
        children = []
        for _, row in languages_df.iterrows():
            if row["parents"] != "init" and isinstance(row["parents"], list):
                if lang in row["parents"]:
                    children.append(row["language"])

        if not children:
            return current_y

        # Sort children by birth time - LATEST births get placed farthest from parent
        children_data = languages_df[languages_df["language"].isin(children)].sort_values(
            "t_birth", ascending=False
        )  # Reverse order

        # Get current language data for comparison
        current_lang_data = languages_df[languages_df["language"] == lang].iloc[0]

        # Assign positions sequentially to avoid overlaps, maintaining birth order
        child_positions = []
        for i, (_, child_row) in enumerate(children_data.iterrows()):
            if child_row["language"] not in y_positions:
                # Check if child birth matches parent extinction
                if child_row["t_birth"] == current_lang_data["t_extinct"] + 1:
                    # Place child at same y position as parent
                    child_positions.append((child_row["language"], y_positions[lang]))
                    y_positions[child_row["language"]] = y_positions[lang]
                else:
                    # Assign sequential positions starting from current_y
                    child_positions.append((child_row["language"], current_y))
                    y_positions[child_row["language"]] = current_y
                    current_y += 1

        # Now process descendants for each child in the order they were positioned
        for child_lang, child_y in child_positions:
            current_y = assign_subtree_positions(child_lang, languages_df, current_y)

        return current_y

    # Assign positions hierarchically
    assign_position_hierarchically(languages, languages)

    # Handle any unassigned languages (shouldn't happen with proper data)
    current_max = max(y_positions.values()) + 1 if y_positions else 0
    for _, row in languages.iterrows():
        if row["language"] not in y_positions:
            y_positions[row["language"]] = current_max
            current_max += 1

    n_languages = len(y_positions)

    # Color assignment: each root language gets its own color family
    root_languages = languages[languages["parents"] == "init"]["language"].tolist()
    root_colors = plt.cm.Set1(np.linspace(0, 1, len(root_languages)))

    # Create color mapping for all languages based on their root ancestor
    def get_root_ancestor(lang, languages_df):
        """Find the root ancestor of a language"""
        current_lang = lang
        visited = set()

        while current_lang not in visited:
            visited.add(current_lang)
            lang_data = languages_df[languages_df["language"] == current_lang]
            if lang_data.empty:
                break
            parents = lang_data.iloc[0]["parents"]
            if parents == "init":
                return current_lang
            elif isinstance(parents, list) and len(parents) > 0:
                current_lang = parents[0]  # Follow first parent
            else:
                break
        return current_lang

    color_map = {}
    for i, root in enumerate(root_languages):
        root_color = root_colors[i]
        color_map[root] = root_color

        # Assign same color to all descendants
        for _, row in languages.iterrows():
            lang = row["language"]
            if get_root_ancestor(lang, languages) == root:
                color_map[lang] = root_color

    # First, draw all the branching connections
    for idx, row in languages.iterrows():
        lang = row["language"]
        y = y_positions[lang]
        birth = row["t_birth"]
        extinct = row["t_extinct"]

        # Draw parent connections as tree branches
        if row["parents"] != "init" and isinstance(row["parents"], list):
            for parent in row["parents"]:
                if parent in y_positions:
                    parent_y = y_positions[parent]

                    # Find parent's data
                    parent_info = languages[languages["language"] == parent]
                    if not parent_info.empty:
                        parent_extinct = parent_info.iloc[0]["t_extinct"]

                        # Branch point is at the child's birth time or slightly before
                        branch_x = birth

                        # Get color for this lineage
                        branch_color = color_map.get(lang, "black")

                        # Draw the branching connection
                        # Horizontal line from parent timeline to branch point
                        ax.plot(
                            [parent_extinct, branch_x],
                            [parent_y, parent_y],
                            color=branch_color,
                            linewidth=2,
                            alpha=0.8,
                        )

                        # Vertical line from parent level to child level
                        ax.plot(
                            [branch_x, branch_x],
                            [parent_y, y],
                            color=branch_color,
                            linewidth=2,
                            alpha=0.8,
                        )

                        # Optional: Add a small circle at branch points
                        ax.plot(
                            branch_x,
                            parent_y,
                            "o",
                            color=branch_color,
                            markersize=4,
                            alpha=0.8,
                        )

    # Then, plot language lifespans as horizontal lines
    for idx, row in languages.iterrows():
        lang = row["language"]
        y = y_positions[lang]
        birth = row["t_birth"]
        extinct = row["t_extinct"]

        # Get color for this language based on root ancestor
        color = color_map.get(lang, "blue")

        # Draw lifespan as a thick horizontal line
        ax.plot(
            [birth, extinct],
            [y, y],
            color=color,
            linewidth=4,
            alpha=0.8,
            solid_capstyle="round",
        )

        # Add language label next to the line
        label_x = extinct + 0.2
        ax.text(
            label_x,
            y,
            lang,
            ha="left",
            va="center",
            fontsize=10,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
        )

    # Customize the plot
    ax.set_xlabel("Time Step", fontsize=14)
    ax.set_ylabel("Languages", fontsize=14)
    ax.set_title("Language Phylogeny with Tree-like Branching", fontsize=16, fontweight="bold")

    # Set y-axis to show language names in hierarchical order
    y_labels = [""] * n_languages
    for lang, pos in y_positions.items():
        if pos < len(y_labels):
            y_labels[pos] = lang

    # Hide y-axis
    ax.set_yticks([])
    ax.set_yticklabels([])

    # Set reasonable x-axis limits
    if not languages.empty:
        x_min = languages["t_birth"].min() - 1
        x_max = languages["t_extinct"].max() + 2
        ax.set_xlim(x_min, x_max)

    ax.set_ylim(-0.5, n_languages - 0.5)

    # Add grid for better readability
    ax.grid(True, axis="x", alpha=0.3, linestyle="--")

    plt.tight_layout()
    return fig, ax


def create_phylogeny(input_file: Path):
    # Read in the population data across all time steps
    population = gpd.read_file(input_file)
    output_path = input_file.parent

    languages = postprocess_phylogeny(population)
    # visualize_phylogeny(languages)
    print(languages)

    # Usage
    fig, ax = plot_language_phylogeny(languages)
    plt.savefig(os.path.join(output_path, "Phylogeny.jpeg"))
