"""Command line interface for leco model."""

import logging
import shutil
import sys
from pathlib import Path

import docopt
import tomllib

from ..cluster.feed_forward import diversify
from ..model.simulation import simulate
from ..plot.create import plot
from ..version import __version__ as version
from .main import main_function


def check_intermediate_start(intermediate_start: Path, intermediate_step: int) -> None:
    """Check if intermediate_start file contains the step of interest."""
    _, steps_range = Path(intermediate_start).stem.split("steps_", 1)
    start, end = map(int, steps_range.split("_"))
    if not (start <= intermediate_step <= end):
        raise ValueError(
            f"intermediate_step {intermediate_step} not in range [{start}, {end}] "
            f"of '{Path(intermediate_start).name}'.",
        )


def run_leco(
    arguments: dict,
    configuration: dict,
    configuration_file_path: Path,
) -> None:
    """Run the leco model with specified configuration."""
    directory = create_directory(arguments["<directory>"])
    # Store a copy of the configuration file for documentation and reproducibility
    shutil.copy2(configuration_file_path, directory / "configuration.toml")

    intermediate_start = arguments["--start_geoparquet"]
    intermediate_step = int(arguments["--intermediate_step"]) if arguments["--intermediate_step"] else 0

    # Check if intermediate_start file contains the step of interest
    if intermediate_start is not None:
        check_intermediate_start(intermediate_start, intermediate_step)

    # Run the leco model
    simulate(configuration, directory, intermediate_start, intermediate_step)


def cluster_languages(arguments: dict, configuration: dict) -> None:
    """Run language classification on the leco model output for a specified method."""
    linkage = arguments["--linkage"]
    distance_threshold = float(arguments["--distance"])
    directory = Path(arguments["<directory>"])
    gpkg_file_name = arguments["<gpkg_file_name>"]

    diversify(
        directory,
        gpkg_file_name,
        distance_threshold,
        linkage,
        int(configuration["initialization"]["steps"]),
        float(configuration["interaction"]["radius"]),
        float(configuration["interaction"]["diffusion_rate"]),
        int(configuration["initialization"]["write_interval"]),
        arguments["--start_gpkg"],
        int(arguments["--intermediate_step"]) if arguments["--intermediate_step"] else 0,
    )


def plot_results(arguments: dict, configuration: dict) -> None:
    """Create plots of the leco model output."""
    gpkg_file_path = Path(arguments["<gpkg_file>"])
    plot(gpkg_file_path, configuration)


def load_configuration(configuration_file_path: Path) -> dict:
    """Load TOML configuration file with error handling."""
    with Path.open(configuration_file_path, "rb") as configuration_file:
        return tomllib.load(configuration_file)


def create_directory(directory_pathname: str) -> Path:
    """Create directory with name provided by user input."""
    directory_path = Path(directory_pathname)
    directory_path.mkdir(parents=True, exist_ok=False)

    return directory_path


def usage() -> str:
    """Return usage string of the leco command."""
    command = Path(sys.argv[0]).name
    return f"""\
Run leco model

Usage:
    {command} run [--debug] [--start_geoparquet <geoparquet_file>]
        [--intermediate_step <step>] <configuration_file> <directory>
    {command} cluster [--debug] [--linkage <single|average|complete>]
        [--distance <distance_threshold>] [--start_gpkg <gpkg_file>]
        [--intermediate_step <step>] <configuration_file> <directory>
        <gpkg_file_name>
    {command} plot [--debug] <configuration_file> <gpkg_file>

Options:
-h --help                             Show this screen and exit
--version                             Show version and exit
<configuration_file>                  Path to a TOML configuration file
--debug                               Enable debug logging
<directory>                           Directory to store/read the output
--distance <distance_threshold>       Distance threshold to set clusters
                                      [default: 0.3]
<gpkg_file>                           Path to a gpkg file created during
                                      clustering
<gpkg_file_name>                      Name of the gpkg file to create
                                      which will be stored in the
                                      <directory>
--start_gpkg <gpkg_file>              Path to file with intermediate
                                      population configuration
--start_geoparquet <geoparquet_file>  Path to file with intermediate
                                      population configuration
--intermediate_step <step>            Time step in the intermediate
                                      geoparquet_file to start on
--linkage <single|average|complete>   Clustering linkage to use
                                      [default: average]

Typical workflow:
    {command} run configuration.toml results
    {command} cluster --distance 0.2 results/configuration.toml results population.gpkg
    {command} plot results/configuration.toml results/population.gpkg
"""


@main_function
def main() -> None:
    """Command line interface for leco model."""
    arguments = docopt.docopt(usage(), sys.argv[1:], version=version)

    if arguments["--debug"]:
        logging.basicConfig(level=logging.DEBUG)

    configuration_file_path = Path(arguments["<configuration_file>"])
    configuration = load_configuration(configuration_file_path)

    command_to_function = {
        "run": lambda args, config: run_leco(args, config, configuration_file_path),
        "cluster": cluster_languages,
        "plot": plot_results,
    }

    # Determine the subcommand and call the corresponding function
    active_command = next(cmd for cmd in command_to_function if arguments[cmd])
    command_to_function[active_command](arguments, configuration)
