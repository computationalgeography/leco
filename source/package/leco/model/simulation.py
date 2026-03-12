"""Runs the leco model."""

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .initialization import initialize_barrier, initialize_population
from .interaction import interact
from .movement import move
from .population_dynamics import population_dynamics


def mutate_profile(
    language_profiles: np.ndarray[int],
    profile_attributes: dict[str, int | float],
    rng: np.random.default_rng,
) -> list[np.ndarray[int], int]:
    """Mutate language profile of agents."""
    # Get the current number of agents and the number of meanings
    nr_agents, nr_meanings = language_profiles.shape

    # Generate mutation masks for all agents at once
    mutation_probabilities = rng.random((nr_agents, nr_meanings))

    mutation_mask = mutation_probabilities < profile_attributes["mutation_rate"]

    # Generate the new forms
    mutated_forms = rng.integers(
        0,
        profile_attributes["forms"],
        size=(nr_agents, nr_meanings),
    )

    # Mutate the forms if mask is true
    mutated_profiles = np.where(mutation_mask, mutated_forms, language_profiles)

    # Count the number of mutations that have occurred to track internal change
    mutations_count = np.sum(language_profiles != mutated_profiles)

    return list(mutated_profiles), mutations_count


def simulate(p: dict, directory: Path, sensitivity: bool = False) -> None | float:
    """Run the LECo model of Linguistic Evolutionary COmputations."""
    # Initialize seed
    rng = np.random.default_rng(p["initialization"]["seed"])

    # Initialize a spatial barrier if specified
    barrier = None
    if p["barrier"]["present"]:
        barrier = initialize_barrier(
            p["space"]["shape"],
            p["barrier"]["x_extent"],
            p["barrier"]["y_extent"],
        )

    # Initialize a population of agents
    population, max_id = initialize_population(
        p["initialization"]["agents"],
        p["space"]["shape"],
        p["initialization_subset_area"],
        p["initialization"]["nr_start_languages"],
        p["language"]["forms"],
        p["language"]["meanings"],
        rng,
    )

    # Create a dataframe to store data on the origin of changes to the language profile
    meta_data = pd.DataFrame(
        {
            "time_step": [0],
            "internal_change": [0],
            "external_change": [0],
        },
    )

    # Write initialization dataframe to a .geoparquet file
    population.to_parquet(directory / "output000.geoparquet")

    for step in tqdm(range(1, p["initialization"]["steps"] + 1)):
        # Add the current time step to the population dataframe
        population["time_step"] = step

        # Apply birth and death events to the population
        population, max_id = population_dynamics(
            population,
            p["population_dynamics"],
            p["initialization"]["agents"],
            step,
            max_id,
            rng,
        )

        # Movement of the agents within space
        population.geometry = move(
            population.geometry,
            barrier,
            p["barrier"]["impermeability"],
            p["movement"]["speed"],
            p["space"]["shape"],
            rng,
        )

        # Mutations of the agents' language profiles
        population["language_profile"], internal_change = mutate_profile(
            np.stack(population["language_profile"]),
            p["language"],
            rng,
        )

        # Interaction between nearby agents during which linguistic features can be adopted
        population["language_profile"], external_change = interact(
            np.stack(population["language_profile"]),
            p["interaction"],
            p["language"]["meanings"],
            population.geometry,
            rng,
        )

        # Record meta data for this time step
        meta_data.loc[len(meta_data)] = [step, internal_change, external_change]

        # Save output per time step to geoparquet file
        population.to_parquet(
            directory / f"output{step:03d}.geoparquet",
        )

    # Save meta data to csv file
    meta_data.to_csv(directory / "meta_data.csv", index=False)

    if sensitivity:
        # Calculate proportion of external changes compared to internal changes
        md = meta_data.copy()

        # Calculate the proportion of external changes compared to total changes per step
        total_changes_per_step = md["internal_change"] + md["external_change"]

        # Exclude steps with no changes at all
        mask = total_changes_per_step != 0
        proportion_per_step = (md["external_change"] / total_changes_per_step)[mask]

        return proportion_per_step.mean()

    return None
