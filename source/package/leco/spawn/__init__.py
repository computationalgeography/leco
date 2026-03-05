"""Code related to spawning multiple leco runs."""

import concurrent
import copy
import os
from collections.abc import Generator
from decimal import Decimal, getcontext
from pathlib import Path

import tomllib
from frozendict import deepfreeze, frozendict

from ..model.simulation import simulate as model_simulate

__all__ = ["default_max_nr_workers", "spawn"]


getcontext().prec = 6


def default_max_nr_workers() -> int:
    """Return default maximum number of workers (processes using a single CPU core) to use."""
    return (os.cpu_count() or 2) // 2


def expand_range(parameter: dict) -> Generator[int | float, None, None]:
    """Yield each value in parameter["range"]."""
    value, stop, step = parameter["range"]

    if value > stop:
        raise ValueError("Start value of range must not be larger than the stop value")

    while value < stop:
        yield value
        value += step


def expand_set(parameter: dict) -> Generator[int | float, None, None]:
    """Yield each value in parameter["set"]."""
    values = set(parameter["set"])

    for value in values:
        assert isinstance(value, (int, float))
        yield value


expand = {
    "range": expand_range,
    "set": expand_set,
}


def as_string(value: float) -> str:
    string = str(Decimal(value) * Decimal(1))

    if string.find(".") != -1:
        string = string.rstrip("0")

    return string


def expand_parameter(
    default_configuration: dict,
    parameter: tuple[str, str, dict],
    directory_pathname_pattern: str,
    cwd: Path,
) -> list[tuple[dict, Path]]:
    """Return as many copies of configurations as there are values in the parameter passed in.

    :param default_configuration: Configuration to tweak for parameter value
    :param parameter: A list with: section name, parameter name, parameter value
    :param directory_pathname_pattern: Template for creating unique output directory pathnames
    :param cwd: Current working directory
    :return: List of tuples, each of which contains a configuration and an output directory path

    The parameter value can be represented by a range or a set of values. For each real value in these
    collections, a configuration is returned, along with a directory to store results in.
    """
    configurations = []
    section_name, parameter_name, value = parameter

    for value_ in expand[next(iter(value.keys()))](value):
        configuration = copy.deepcopy(default_configuration)
        parameter_value = float(as_string(value_)) if isinstance(value_, float) else int(as_string(value_))
        configuration[section_name][parameter_name] = parameter_value
        directory_pathname = directory_pathname_pattern.replace(f"{{{parameter_name:}}}", as_string(value_))
        directory_path = cwd / directory_pathname

        configurations.append((configuration, directory_path))

    return configurations


def substitute_default_values(
    directory_pathname_pattern: str,
    default_configuration: dict,
    variable_parameter_values: dict,
    overridden_parameter: tuple[str, str],
) -> str:
    """Substitute parameter value placeholders in a pathname by their default values."""
    section, parameter = overridden_parameter
    directory_pathname = directory_pathname_pattern

    for a_section, a_parameters in variable_parameter_values.items():
        for a_parameter in a_parameters:
            if not (a_section == section and a_parameter == parameter):
                # Replace {parameter} by its default value
                default_value = default_configuration[a_section][a_parameter]
                directory_pathname = directory_pathname.replace(
                    f"{{{a_parameter}}}",
                    as_string(default_value),
                )

    return directory_pathname


def merge_configurations(
    default_configuration: dict,
    spawn_configuration: dict,
    cwd: Path,
) -> set[tuple[frozendict, Path]]:
    """Tweak default Leco configurations given a spawn configuration.

    :return: A list of tuples, each of which contains a unique configuration and a unique output directory
             path
    """
    configurations = []

    directory_pathname_pattern = os.path.expandvars(
        Path(spawn_configuration["directory_pattern"]).expanduser(),
    )

    assert "sensitivity" in spawn_configuration  # For now
    assert spawn_configuration["sensitivity"]["run"]["method"] == "ofat"  # For now

    sensitivity_run_configuration = spawn_configuration["sensitivity"]["run"]

    # Find parameters to vary. These are located in the initialization, movement, language,
    # population_dynamics, interaction sections
    variable_parameter_values = {}

    for section in "initialization", "movement", "language", "population_dynamics", "interaction":
        if section in sensitivity_run_configuration:
            variable_parameter_values[section] = sensitivity_run_configuration[section]

    # Seed is special. If a range (or a set) of seeds is passed in, all runs need to be executed for each of
    # these. If no such range (or set) is passed in, then don't do anything fancy.
    seed_parameter = None

    if (
        "initialization" in variable_parameter_values
        and "seed" in variable_parameter_values["initialization"]
    ):
        seed_parameter = variable_parameter_values["initialization"]["seed"]
        del variable_parameter_values["initialization"]["seed"]

    for section, parameters in variable_parameter_values.items():
        for parameter, value in parameters.items():
            assert not (section == "initialization" and parameter == "seed")

            # Tweak the default configuration for the current parameter
            directory_pathname = substitute_default_values(
                directory_pathname_pattern,
                default_configuration,
                variable_parameter_values,
                (section, parameter),
            )

            configurations_ = expand_parameter(
                default_configuration,
                (section, parameter, value),
                directory_pathname,
                cwd,
            )

            if seed_parameter is not None:
                # Tweak the current configurations for seed. Store results in different collection.
                configurations__ = []
                for configuration, directory_path in configurations_:
                    configurations__ += expand_parameter(
                        configuration,
                        ("initialization", "seed", seed_parameter),
                        str(directory_path),
                        cwd,
                    )
                # Overwrite original collection
                configurations_ = configurations__

            # All paths should be unique
            assert len({tuple_[1] for tuple_ in configurations_}) == len(configurations_), configurations_

            # Update overall result collection
            configurations += configurations_

    unique_configurations = {(deepfreeze(tuple_[0]), tuple_[1]) for tuple_ in configurations}

    # All paths should be unique
    assert len({tuple_[1] for tuple_ in unique_configurations}) == len(unique_configurations)

    return unique_configurations


def configurations(configuration_file_path: Path) -> set[tuple[frozendict, Path]]:
    """Determine a configuration and where to store the results."""
    with Path.open(configuration_file_path, "rb") as configuration_file:
        spawn_configuration = tomllib.load(configuration_file)

    assert "configuration_file" in spawn_configuration

    cwd = configuration_file_path.parent
    configuration_file_path = cwd / spawn_configuration["configuration_file"]

    with Path.open(configuration_file_path, "rb") as configuration_file:
        run_configuration = tomllib.load(configuration_file)

    return merge_configurations(run_configuration, spawn_configuration, cwd)


def simulate(arguments: dict) -> None | float:
    # TODO: turn off progress
    model_simulate(*arguments)


def spawn(configuration_file_path: Path, *, max_nr_workers: int, continue_on_error: bool) -> None:
    """Spawn Leco runs."""
    # NOTE: We are assuming here that we need to *run* the model. Otherwise add subcommands (run,
    #       postprocess, ...).
    configurations_ = configurations(configuration_file_path)

    for _, directory_path in configurations_:
        directory_path.mkdir(parents=True, exist_ok=False)

    generators = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_nr_workers) as executor:
        generator = executor.map(simulate, configurations_)

        # Obtaining the result raises any exception thrown and stops further processing
        if continue_on_error:
            # Delay obtaining results
            generators.append(generator)
        else:
            # Obtain results now
            list(generator)

    if continue_on_error:
        for generator in generators:
            list(generator)
