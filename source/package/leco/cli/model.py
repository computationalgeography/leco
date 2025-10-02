import numpy as np
import time
import os

from .model_popdynamics import population_dynamics
from .model_movement import move
from .model_initialization import initialize_population, initialize_barrier
from .model_interaction import interact


def mutate_profile(
    language_profiles: np.ndarray[int],
    nr_forms: int,
    mutation_rate: float,
    rng: np.random.default_rng,
) -> np.ndarray[int]:
    """Mutate language profile of agents"""

    # Get the current number of agents and the number of meanings
    nr_agents, nr_meanings = language_profiles.shape

    # Generate mutation masks for all agents at once
    mutation_prob = rng.random((nr_agents, nr_meanings))
    mutation_mask = mutation_prob < mutation_rate

    # Generate the new forms
    mutated_forms = rng.integers(0, nr_forms, size=(nr_agents, nr_meanings))

    # Mutate the forms if mask is true
    mutated_profiles = np.where(mutation_mask, mutated_forms, language_profiles)

    return list(mutated_profiles)


def run_model(p: dict, output_run: str):
    """Run the LECo model of Linguistic Evolutionary COmputations"""

    # Initialize seed
    rng = np.random.default_rng(p["seed"])

    # Start time to track model run time
    start_time = time.time()

    # Initialize a spatial barrier if specified
    barrier = None
    if p["barrier"]:
        barrier = initialize_barrier(
            p["x_max"],
            p["y_max"],
            p["bar_x"],
            p["bar_y"],
        )

    # Initialize a population of agents
    population = initialize_population(
        p["agents"],
        p["x_max"],
        p["y_max"],
        p["init_subset_area"],
        p["init_x"],
        p["init_y"],
        p["nr_start_languages"],
        p["forms"],
        p["meanings"],
        rng,
    )

    if output_run:
        # Write initialization dataframe to a .geoparquet file
        population.to_parquet(os.path.join(output_run, "output000.geoparquet"))

    # Keep track of the maximum ID for agent births
    max_id = population["id"].max()

    for step in range(1, p["steps"] + 1):
        print(step)
        # Add the current timestep to the population dataframe
        population["timestep"] = step

        # Apply birth and death events to the population
        population, max_id = population_dynamics(
            population,
            p["death_rate"],
            p["birth_rate"],
            p["logistic_growth"],
            p["end_growth_time"],
            p["multiplier"],
            p["agents"],
            step,
            max_id,
            rng,
        )

        # Movement of the agents within space
        population.geometry = move(
            population.geometry,
            barrier,
            p["bar_impermeability"],
            p["speed"],
            p["x_max"],
            p["y_max"],
            rng,
        )

        # Mutations of the agents' language profiles
        population["language_profile"] = mutate_profile(
            np.stack(population["language_profile"]),
            p["forms"],
            p["mutation_rate"],
            rng,
        )

        # Interaction between nearby agents during which linguistic features can be adopted
        population["language_profile"] = interact(
            np.stack(population["language_profile"]),
            p["int_partner_prob"],
            p["diffusion_rate"],
            population.geometry,
            p["int_radius"],
            p["bar_impermeability"],
            barrier,
            rng,
        )

        # Save output per timestep to geoparquet file
        if output_run:
            population.to_parquet(
                os.path.join(output_run, f"output{step:03d}.geoparquet")
            )

    print("--- %s seconds ---" % (time.time() - start_time))
    print("--- %s minutes ---" % ((time.time() - start_time) / 60))


# %%
