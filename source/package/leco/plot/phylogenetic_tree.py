"""Create phylogenetic visualizations of language evolution over time."""

import re
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import pandas as pd


def postprocess_phylogeny(population: gpd.GeoDataFrame) -> pd.DataFrame:
    """Generate dataframe on phylogenetic information for every language.

    This information includes time of birth, time of death and the most common parent language.
    """
    records = []
    for language, group in population.groupby("language"):
        t_birth = group["time_step"].min()
        t_extinct = group["time_step"].max()
        if t_birth == 0:
            # Root language
            parent_language = ""
        else:
            # Get the ids of agents speaking this language at t_birth
            start_speakers = group[group["time_step"] == t_birth]["id"].dropna().unique().tolist()
            # Find which languages they spoke and how many at time step t_birth - 1
            parent_language_counts = population[
                (population["time_step"] == t_birth - 1) & (population["id"].isin(start_speakers))
            ]["language"].value_counts()
            # Only select the language with the highest speaker count as parent language
            parent_language = [parent_language_counts.index[0]] if not parent_language_counts.empty else []

        records.append(
            {
                "language": language,
                "t_birth": t_birth,
                "t_extinct": t_extinct,
                "parent": parent_language,
            },
        )

    return pd.DataFrame(records)


def relabel_parent_continuation(
    chain: list,
    parent: str | int,  # language label as saved in t_switch dictionary
    switch: int,
    parent_version: str,  # new version label for language
) -> list:
    """Relabel the continuation of parent language after switch and add the split node.

    New labels include the parent + "_vn" with n being an integer.
    """
    # Find all nodes in the ancestry chain that come from parent: parent and later versions
    # Nodes contain label and time step
    parent_label = str(parent)
    parent_nodes = [
        node
        for node in chain
        if node[0] == parent
        or (isinstance(node[0], str) and (node[0] == parent_label or node[0].startswith(f"{parent_label}_v")))
    ]

    # We want to alter the original parent node associated with the latest version of the parent nodes
    original_parent_node = parent_nodes[-1]
    # Get the time step of the original parent node
    time_parent_node = original_parent_node[1]

    # Update the original parent node with a new version label
    updated_chain = [
        (parent_version, time_parent_node) if node == original_parent_node else node for node in chain
    ]

    # Add a new node with the original parent name at time of current switch
    if (original_parent_node[0], switch) not in updated_chain:
        updated_chain.append((original_parent_node[0], switch))

    # Sort timed nodes by switch time and return
    return sorted(
        [node for node in updated_chain if isinstance(node, tuple) and len(node) == 2],
        key=lambda item: item[1],
    )


def continuous_to_split(ancestry_chains: dict, t_final: int) -> dict:
    """Languages can continue after another language has split of.

    To create a phylogeny, these continuation of languages need to be translated in nodes.
    The original language id is assigned a split and the continued language receives a version number.
    """
    # Store all time switches and children per language
    lang_t_switches = defaultdict(set)
    # Store all parent and children per switch time
    t_switches = defaultdict(lambda: defaultdict(list))
    for leaf, chain in ancestry_chains.items():
        for node in chain:
            if not isinstance(node, tuple) or len(node) != 2:
                # Root entries can be plain language ids (len(node)=1) without switch time.
                continue
            lang, t_switch = node
            lang_t_switches[lang].add((t_switch, leaf))
            t_switches[t_switch][lang].append(leaf)

    # Initialize a dictionary that tracks added versions of languages
    parent_version_counter = defaultdict(int)
    parent_version_by_switch = {}

    # Sort t_switches on chronological order
    t_switches = dict(sorted(t_switches.items()))

    # Loop through switches, starting with the earliest split
    for switch in (s for s in t_switches if s < t_final):
        for parent in t_switches[switch]:
            parent_life_time = max(lang_t_switches[parent], key=lambda pair: pair[0])[0]
            # Check if there is continuation
            if parent_life_time > switch:
                # The rest group include all languages that continue in the original language
                rest_group = [leaf for t, leaf in lang_t_switches[parent] if t > switch]

                version_key = (parent, switch)
                # Add new version of this parent language
                if version_key not in parent_version_by_switch:
                    parent_version_counter[parent] += 1
                    parent_version_by_switch[version_key] = f"{parent}_v{parent_version_counter[parent]}"
                # Access new version name
                parent_version = parent_version_by_switch[version_key]

                # Add for each of the continuing chains an extra node
                for child in rest_group:
                    ancestry_chains[child] = relabel_parent_continuation(
                        ancestry_chains[child],
                        parent,
                        switch,
                        parent_version,
                    )

    return ancestry_chains


def build_ancestry_chains(phylogeny_df: pd.DataFrame, t_final: int, t_birth_map: dict) -> dict:
    """Create ancestry chains for every extant language (leaf) starting from root.

    Every chain consists of nodes, whereby a node includes the ancestor and time of switch to next ancestor.
    """
    # Create a mapping that gives the parent per language for fast and recurring look up
    parent_map = {}
    for _, row in phylogeny_df.iterrows():
        parents = row["parent"]
        if isinstance(parents, list) and len(parents) > 0:
            parent_map[row["language"]] = parents[0]
        else:
            parent_map[row["language"]] = None  # root language

    # Find extant languages (present at final time step)
    extant = phylogeny_df[phylogeny_df["t_extinct"] == t_final]["language"].tolist()

    # For each extant language, walk up the parent chain and store in ancestry_chains
    ancestry_chains = {}
    for language in extant:
        # Build timed ancestor nodes only; leaf is added once at t_final below.
        chain = []
        current = language
        # Add the tip as the first node
        chain.append((current, t_final))
        # Get the parent language and make the parent language the current, then repeat
        # While loop stops when None parent is reached = root
        while parent_map.get(current) is not None:
            p = parent_map[current]
            # Save the ancestor id with the time step of the switch to their daughter language
            chain.append((p, t_birth_map[current]))
            current = p

        # Store the ancestry chain from root to leaf and include terminal switch at t_final.
        chain_root_to_leaf = list(reversed(chain))
        ancestry_chains[language] = chain_root_to_leaf

    # Add additional splits for continuous languages and return
    return continuous_to_split(ancestry_chains, t_final)


def _build_rooted_newick_components(ancestry_chains: dict) -> list[tuple[str, str]]:
    """Build Newick components per root, each anchored with branch length from synthetic time 0."""
    # We keep timed nodes distinct, because a language label can appear at multiple switch times.
    # Node id: (language_label, switch_time)

    # Initialize dictionary for children nodes, roots, and root switch times.
    children_by_node: dict[tuple, dict[tuple, int]] = defaultdict(dict)
    roots: set[tuple] = set()
    root_switch_times: dict[tuple, int] = {}

    # Loop through ancestry chains and
    for normalized_chain in ancestry_chains.values():
        root_node = normalized_chain[0]
        roots.add(root_node)
        # Root node carries its own switch time in the chain.
        root_switch_times[root_node] = int(root_node[1])

        # Loop through normalized chain and collect parent children pairs
        for i in range(len(normalized_chain) - 1):
            parent, child = normalized_chain[i], normalized_chain[i + 1]
            branch_length = child[1] - parent[1]
            # Store the parent-child branch length
            children_by_node[parent][child] = branch_length

    def node_sort_key(node: tuple) -> tuple[str, int]:
        """Return consistent ordering, sorted first by language label (node[0]), then by switch_time [1]."""
        return (str(node[0]), int(node[1]))

    def render_node(node: tuple) -> str:
        """Build Newick tree structure, starting from root successively up to leaf."""
        label = str(node[0])
        children = children_by_node.get(node, {})
        if not children:
            return label

        # Collect for every child their child and write in newick tree structure
        rendered_children = [
            f"{render_node(child)}:{children[child]}" for child in sorted(children, key=node_sort_key)
        ]
        return f"({','.join(rendered_children)}){label}"

    sorted_roots = sorted(roots, key=node_sort_key)

    # Create newick tree structure per root node
    components: list[tuple[str, str]] = []
    for root in sorted_roots:
        root_label = str(root[0])
        components.append((root_label, f"{render_node(root)}:{root_switch_times[root]}"))
    return components


def build_newick_trees_per_root(ancestry_chains: dict) -> dict[str, str]:
    """Build one Newick tree per disconnected root component.

    Each per-root tree is anchored to a synthetic start node at time 0, so the root
    lineage is visible from time step 0 until its first split.
    """
    components = _build_rooted_newick_components(ancestry_chains)
    trees = {}
    for root_label, component in components:
        trees[root_label] = f"({component})ROOT_START:0;"
    return trees


def create_phylogeny(input_file: Path, time_steps: int) -> None:
    """Create a phylogenetic tree for a single run based on vertical transmission."""
    # Read in the population data across all time steps
    population = gpd.read_file(input_file)
    output_path = input_file.parent

    phylogeny_info = postprocess_phylogeny(population)
    t_birth_map = dict(zip(phylogeny_info["language"], phylogeny_info["t_birth"], strict=True))
    # Save phylogenetic data to csv file
    phylogeny_info.to_csv(output_path / "phylogeny_information.csv", index=False)
    ancestry_chains = build_ancestry_chains(phylogeny_info, time_steps, t_birth_map)

    # Build Newick trees for different roots
    per_root_trees = build_newick_trees_per_root(ancestry_chains)
    for root_label, root_newick in per_root_trees.items():
        clean_root = re.sub(r"[^A-Za-z0-9_-]+", "_", str(root_label)).strip("_")
        if clean_root == "":
            clean_root = "unknown_root"
        file_name = f"phylogenetic_tree_root_{clean_root}.newick"
        (output_path / file_name).write_text(root_newick, encoding="utf-8")
