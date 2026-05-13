"""Runs the leco model."""

from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from tqdm import tqdm

from .initialization import initialize_barrier, initialize_population
from .interaction import interact
from .movement import move
from .population_dynamics import population_dynamics


def mutate_profile(
    language_profiles: npt.NDArray[np.int64],
    profile_attributes: dict[str, int | float],
    rng: np.random.Generator,
) -> tuple[list[npt.NDArray[np.int64]], int]:
    """Mutate language profile of agents."""
    # Get the current number of agents and the number of meanings
    nr_agents, nr_meanings = language_profiles.shape

    # Generate mutation masks for all agents at once
    mutation_probabilities = rng.random((nr_agents, nr_meanings))

    mutation_mask = mutation_probabilities < profile_attributes["mutation_rate"]

    # Generate the new forms
    mutated_forms = rng.integers(
        0,
        int(profile_attributes["forms"]),
        size=(nr_agents, nr_meanings),
    )

    # Mutate the forms if mask is true
    mutated_profiles = np.where(mutation_mask, mutated_forms, language_profiles)

    # Count the number of mutations that have occurred to track internal change
    mutations_count = int(np.sum(language_profiles != mutated_profiles))

    return list(mutated_profiles), mutations_count


def simulate(p: dict, directory: Path) -> None | float:
    """Run the LECo model of Linguistic Evolutionary COmputations."""
    # Initialize seed
    rng = np.random.default_rng(p["initialization"]["seed"])
    steps = p["initialization"]["steps"]
    write_interval = p["initialization"]["write_interval"]

    # Initialize buffer to store population data for writing to geoparquet file
    population_buffer = []

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
        p["initialization_subset_area"]["present"],
        p["initialization_subset_area"]["x_extent"],
        p["initialization_subset_area"]["y_extent"],
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
    population.to_parquet(directory / "steps_0000.geoparquet")

    for step in tqdm(range(1, steps + 1)):
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
            # Transform Pandas serie to list and specify numpy array type
            np.stack(cast("list[npt.NDArray[np.int64]]", population["language_profile"].to_list())),
            p["language"],
            rng,
        )

        # Interaction between nearby agents during which linguistic features can be adopted
        population["language_profile"], external_change = interact(
            # Transform Pandas serie to list and specify numpy array type
            np.stack(cast("list[npt.NDArray[np.int64]]", population["language_profile"].to_list())),
            p["interaction"],
            p["language"]["meanings"],
            population.geometry,
            rng,
        )

        # Record meta data for this time step
        meta_data.loc[len(meta_data)] = [step, internal_change, external_change]

        # Add current population to buffer
        population_buffer.append(population.copy())

        # Save output per write_interval to geoparquet file
        if step % write_interval == 0:
            combined_population = pd.concat(population_buffer, ignore_index=True)
            combined_population.to_parquet(
                directory / f"steps_{step - write_interval + 1:04d}_{step:04d}.geoparquet",
            )
            population_buffer = []

    # Write any remaining steps
    if population_buffer:
        last_written_end = (steps // write_interval) * write_interval
        pd.concat(population_buffer, ignore_index=True).to_parquet(
            directory / f"steps_{last_written_end + 1:04d}_{steps:04d}.geoparquet",
        )

    # Save meta data to csv file
    meta_data.to_csv(directory / "meta_data.csv", index=False)
