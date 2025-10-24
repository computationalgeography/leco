"""Functions for population dynamics: growth and death of agents."""

import geopandas as gpd
import numpy as np
import pandas as pd


def compute_r(
    init_pop: int,
    carrying_capacity: int,
    duration: int,
    target_ratio: float = 0.99,
) -> float:
    """Compute the intrinsic growth rate r for logistic growth."""
    """
    r = -(1/duration) * ln((1 - target_ratio)/target_ratio * (N0/(K - N0)))
    """
    # Reach a target ratio of K in a given time period
    numerator = (1 - target_ratio) / target_ratio
    denominator = init_pop / (carrying_capacity - init_pop)
    return -(1 / duration) * np.log(numerator * denominator)


def effective_growth(r: float, carrying_capacity: int, nr_agents: int) -> float:
    """Calculate the effective growth rate per step."""
    return r * (1 - nr_agents / carrying_capacity)


def set_birth_rate(
    pop_dynamics: dict[float, float, bool, int, float],
    init_population: int,
    nr_agents: int,
    timestep: int,
) -> float:
    """Calculate the birth rate of the current timestep."""
    if pop_dynamics["logistic_growth"] is False:
        # if logistic growth is not applied, birth rate is constant
        return pop_dynamics["birth_rate"]

    # If the end of the logistic growth period is reached, return the constant birth rate
    if pop_dynamics["end_growth_time"] <= timestep:
        return pop_dynamics["birth_rate"]

    # If logistic growth is applied and end of growth period is not reached, calculate the dynamic birth rate
    carrying_capacity = init_population * pop_dynamics["multiplier"]  # Calculate the carrying capacity
    # Compute intrinsic growth rate, r, which is constant based on initial population,
    # carrying capacity and duration of growth period
    r = compute_r(init_population, carrying_capacity, pop_dynamics["end_growth_time"])
    # Calculate the effective growth rate at the current population size
    effective_growth_rate = effective_growth(r, carrying_capacity, nr_agents)
    # The effective growth rate represent the difference between birth and death rate
    return pop_dynamics["death_rate"] + effective_growth_rate


def population_dynamics(
    population: gpd.GeoDataFrame,
    pop_dynamics: dict[float, float, bool, int, float],
    init_population: int,
    timestep: int,
    max_id: int,
    rng: np.random.default_rng,
) -> gpd.GeoDataFrame:
    """Update population based on death and birth rates."""
    nr_agents = len(population)

    # Determine for every agent whether it will die based on the deathrate
    death_masks = rng.choice(
        [False, True],
        size=nr_agents,
        p=[1 - pop_dynamics["death_rate"], pop_dynamics["death_rate"]],
    )

    # Remove the agents that die while remaining consecutive row count number
    survivors = population[~death_masks].copy().reset_index(drop=True)

    # Determine the current birth rate, which may depend on logistic growth
    current_birth_rate = set_birth_rate(
        pop_dynamics,
        init_population,
        nr_agents,
        timestep,
    )

    # Determine for every agent whether it will reproduce based on the birthrate
    birth_masks = rng.choice(
        [False, True],
        size=len(survivors),
        p=[1 - current_birth_rate, current_birth_rate],
    )

    # Handle births: duplicate the agents that give birth
    if birth_masks.any():
        num_births = birth_masks.sum()  # Number of offspring to be born
        new_ids = np.arange(
            max_id + 1,
            max_id + 1 + num_births,
        )  # Generate new unique IDs for offspring
        max_id = max(new_ids)  # Update the maximum ID

        # Get parent indices for births
        birth_indices = np.where(birth_masks)[0]

        # Create offspring by taking parent data and updating IDs
        offspring = population.iloc[birth_indices].copy()
        offspring["id"] = new_ids

        # Combine survivors with offspring
        new_population = pd.concat([survivors, offspring], ignore_index=True)
    else:
        new_population = survivors

    return new_population, max_id
