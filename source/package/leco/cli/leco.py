"""Command line interface for leco model."""

import logging
import shutil
import sys
from pathlib import Path

import docopt
import tomllib

from ..cluster.all import classify_all
from ..cluster.feed_forward import diversify
from ..cluster.per_step import diversify_stepwise
from ..model.simulation import simulate
from ..plot.create import plot
from ..sensitivity_analysis.morris import analyze
from ..version import __version__ as version


def run_leco(arguments: dict, configuration: dict, configuration_file_path: Path) -> None:
    """Run the leco model with specified configuration."""
    directory = create_directory(arguments["<directory>"])
    # Store a copy of the configuration file for documentation and reproducibility
    shutil.copy2(configuration_file_path, directory / "configuration.toml")
    # Run the leco model
    simulate(configuration, directory)


def cluster_languages(arguments: dict, configuration: dict) -> None:
    """Run language classification on the leco model output for a specified method."""
    method = arguments["--method"]
    linkage = arguments["--linkage"]
    distance_threshold = float(arguments["--distance"])
    directory = Path(arguments["<directory>"])

    # Dictionary maps methods to their corresponding functions
    classification_by_method = {
        "all": classify_all,  # 3D clustering over all time_steps
        "feed_forward": diversify,  # Feed-forward diversification-based clustering
        "step": diversify_stepwise,  # Cluster per time step independently
    }

    # Call the corresponding function with appropriate parameters
    if method == "feed_forward":
        classification_by_method[method](
            directory,
            distance_threshold,
            linkage,
            configuration["initialization"]["steps"],
            configuration["interaction"]["radius"],
        )
    else:
        classification_by_method[method](directory, distance_threshold, linkage)


def plot_results(arguments: dict, configuration: dict) -> None:
    """Create plots of the leco model output."""
    gpkg_file_path = Path(arguments["<gpkg_file>"])
    plot(gpkg_file_path, configuration)


def sensitivity_analysis(arguments: dict, configuration: dict) -> None:
    """Run sensitivity analysis on the leco model."""
    method = arguments["--method"]
    distance_threshold = float(arguments["--distance"])
    directory = create_directory(arguments["<directory>"])

    analyze(method, distance_threshold, configuration, directory)


def load_configuration(configuration_file_path: Path) -> dict:
    """Load TOML configuration file with error handling."""
    with Path.open(configuration_file_path, "rb") as configuration_file:
        return tomllib.load(configuration_file)


def create_directory(directory_path: str) -> Path:
    """Create directory with name provided by user input."""
    directory_path = Path(directory_path)
    directory_path.mkdir(parents=True, exist_ok=False)

    return directory_path


def main() -> None:
    """Command line interface for leco model."""
    command = Path(sys.argv[0]).name
    usage = f"""\
Run leco model

Usage:
    {command} run [--debug] <configuration_file> <directory>
    {command} cluster [--debug] [--method <all|feed_forward|step>]
        [--linkage <single|average|complete>] [--distance <distance_threshold>]
        <configuration_file> <directory>
    {command} plot [--debug] <configuration_file> <gpkg_file>
    {command} sensitivity [--debug] [--method <all|feed_forward>]
        [--distance <distance_threshold>] <configuration_file> <directory>

Options:
  -h --help                             Show this screen and exit
  --version                             Show version and exit
  <configuration_file>                  Path to a TOML configuration file
  --debug                               Enable debug logging
  --distance <distance_threshold>       Distance threshold to set clusters
                                        [default: 0.3]
  <gpkg_file>                           Path to a gpkg file after clustering
  <directory>                           Directory to store/read the output
  --linkage <single|average|complete>   Clustering linkage to use
                                        [default: average]
  --method <all|feed_forward|step>      Clustering method to use
                                        [default: feed_forward]

Typical workflow:
    {command} run configuration.toml results
    {command} cluster --method all --distance 0.2 configuration.toml results
    {command} plot configuration.toml population.gpkg
    {command} sensitivity configuration.toml sensitivity_results
"""

    arguments = docopt.docopt(usage, sys.argv[1:], version=version)

    if arguments["--debug"]:
        logging.basicConfig(level=logging.DEBUG)

    configuration_file_path = Path(arguments["<configuration_file>"])
    configuration = load_configuration(configuration_file_path)

    command_to_function = {
        "run": lambda args, config: run_leco(args, config, configuration_file_path),
        "cluster": cluster_languages,
        "plot": plot_results,
        "sensitivity": sensitivity_analysis,
    }

    # Determine the subcommand and call the corresponding function
    active_command = next(cmd for cmd in command_to_function if arguments[cmd])
    command_to_function[active_command](arguments, configuration)
