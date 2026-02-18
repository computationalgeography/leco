"""
Code related to spawning multiple leco runs
"""

import concurrent
import copy
from decimal import Decimal, getcontext
import os
from pathlib import Path
import tomllib
from typing import Generator

from frozendict import deepfreeze, frozendict

from ..model.simulation import simulate as model_simulate


__all__ = ["default_max_nr_workers", "spawn"]


getcontext().prec = 6


def default_max_nr_workers() -> int:
    """
    Return default maximum number of workers (processes using a single CPU core) to use
    """
    return (os.cpu_count() or 2) // 2


def expand_range(parameter: dict) -> Generator[int | float, None, None]:
    """
    Yield each value in parameter["range"]
    """
    value, stop, step = parameter["range"]
    assert value <= stop

    while value < stop:
        assert isinstance(value, (int, float))
        yield value
        value += step


def expand_set(parameter: dict) -> Generator[int | float, None, None]:
    """
    Yield each value in parameter["set"]
    """
    values = set(parameter["set"])

    for value in values:
        assert isinstance(value, (int, float))
        yield value


def as_string(value: int | float) -> str:
    string = str(Decimal(value) * Decimal(1))

    if string.find(".") != -1:
        string = string.rstrip("0")

    return string


def expand_parameter(
    default_configuration: dict,
    parameter: list[str],
    directory_pathname_pattern: str,
    cwd: Path,
) -> list[tuple[dict, Path]]:
    """
    Return as many copies of configurations as there are values in the parameter passed in

    :param default_configuration: Configuration to tweak for parameter value
    :param parameter: A list with three strings: section name, parameter name, parameter value
    :param directory_pathname_pattern: Template for creating unique output directory pathnames
    :param cwd: Current working directory
    :return: List of tuples, each of which contains a configuration and an output directory path

    The parameter value can be represented by a range or a set of values. For each real value in these
    collections, a configuration is returned, along with a directory to store results in.
    """
    parameter_values = {
        "range": expand_range,
        "set": expand_set,
    }
    configurations = []
    section_name, parameter_name, value = parameter

    for value in parameter_values[list(value.keys())[0]](value):
        configuration = copy.deepcopy(default_configuration)
        configuration[section_name][parameter_name] = float(as_string(value))
        directory_pathname = directory_pathname_pattern.replace(f"{{{parameter_name:}}}", as_string(value))
        directory_path = cwd / directory_pathname

        configurations.append((configuration, directory_path))

    return configurations


def substitute_default_values(
    directory_pathname_pattern: str,
    default_configuration: dict,
    variable_parameter_values: dict,
    overridden_parameter: tuple[str, str],
) -> str:
    """
    Substitute parameter value placeholders in a pathname by their default values
    """
    section, parameter = overridden_parameter
    directory_pathname = directory_pathname_pattern

    for a_section, a_parameters in variable_parameter_values.items():
        for a_parameter in a_parameters.keys():
            if not (a_section == section and a_parameter == parameter):
                # Replace {parameter} by its default value
                default_value = default_configuration[a_section][a_parameter]
                directory_pathname = directory_pathname.replace(
                    f"{{{a_parameter}}}", as_string(default_value)
                )

    return directory_pathname


def merge_configurations(
    default_configuration: dict, spawn_configuration: dict, cwd: Path
) -> list[tuple[frozendict, Path]]:
    """
    Tweak default Leco configurations given a spawn configuration

    :return: A list of tuples, each of which contains a unique configuration and a unique output directory
             path
    """
    configurations = []

    directory_pathname_pattern = os.path.expandvars(
        os.path.expanduser(spawn_configuration["directory_pattern"])
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

    for section, parameters in variable_parameter_values.items():
        for parameter, value in parameters.items():
            directory_pathname = substitute_default_values(
                directory_pathname_pattern,
                default_configuration,
                variable_parameter_values,
                (section, parameter),
            )
            configurations += expand_parameter(
                default_configuration,
                [section, parameter, value],
                directory_pathname,
                cwd,
            )

    unique_configurations = set([(deepfreeze(tuple_[0]), tuple_[1]) for tuple_ in configurations])

    # All paths should be unique
    assert len(set(tuple_[1] for tuple_ in unique_configurations)) == len(unique_configurations)

    return unique_configurations


def configurations(configuration_file_path: Path) -> list[tuple[dict, Path]]:
    """
    "Compute" a configuration and determine where to store the results
    """
    with Path.open(configuration_file_path, "rb") as configuration_file:
        spawn_configuration = tomllib.load(configuration_file)

    assert "configuration_file" in spawn_configuration

    cwd = configuration_file_path.parent
    configuration_file_path = cwd / spawn_configuration["configuration_file"]

    with Path.open(configuration_file_path, "rb") as configuration_file:
        run_configuration = tomllib.load(configuration_file)

    return merge_configurations(run_configuration, spawn_configuration, cwd)


def simulate(arguments):
    # TODO: turn off progress
    model_simulate(*arguments)


def spawn(configuration_file_path: Path, max_nr_workers: int) -> None:

    # NOTE: We are assuming here that we need to *run* the model. Otherwise add subcommands (run,
    #       postprocess, ...).
    configurations_ = configurations(configuration_file_path)

    for _, directory_path in configurations_:
        directory_path.mkdir(parents=True, exist_ok=False)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(simulate, configurations_)
