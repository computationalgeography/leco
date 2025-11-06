"""Command line interface for leco model."""

import logging
import shutil
import sys
from pathlib import Path

import docopt
import tomllib

from ..cluster.language_classification import classify_all
from ..cluster.per_timestep import classify_single
from ..cluster.speciation import speciate
from ..model.simulation import simulate
from ..plot.create import plot
from ..version import __version__ as version

from .main import main_function


@main_function
def run_leco(arguments: dict) -> None:
    """Run the leco model with specified configuration."""
    configuration_path = Path(arguments["<configfile>"])
    # Load the parameters in dictionary from configuration file
    configuration = load_config(configuration_path)
    # Create a directory to store the results
    directory = create_directory(arguments["<directory>"])

    # Store a copy of the configuration file iin the run directory
    shutil.copy2(configuration_path, directory / "configuration.toml")

    # Run the leco model
    simulate(configuration, directory)


def cluster_languages(arguments: dict) -> None:
    """Run language classification on the leco model output for a specified method."""
    method = arguments["--method"] or "all"
    distance_threshold = float(arguments["--distance"] or 0.3)
    directory = Path(arguments["<directory>"])

    # Dictionary maps methods to their corresponding functions
    classification_by_method = {
        "all": classify_all,  # 3D clustering over all timesteps
        "single": classify_single,  # 2D clustering per timestep [Note: under development]
        "speciation": speciate,  # Feed-forward speciation-based clustering [Note: under development]
    }

    # Check if method is valid and call the corresponding function
    classification_by_method[method](directory, distance_threshold)


def plot_results(arguments: dict) -> None:
    """Create plots of the leco model output."""
    configuration = load_config(arguments["<configfile>"])
    gpkg_file = Path(arguments["<gpkgfile>"])
    plot(gpkg_file, configuration)


def load_config(config_file: Path) -> dict:
    """Load TOML config with error handling."""
    with Path.open(config_file, "rb") as f:
        return tomllib.load(f)


def create_directory(directory_path: str) -> Path:
    """Create directory with name provided by user input."""
    directory = Path(directory_path)
    directory.mkdir(parents=True, exist_ok=False)

    return directory


def main() -> None:
    """Command line interface for leco model."""
    command = Path(sys.argv[0]).name
    usage = f"""\
Run leco model

Usage:
    {command} run [--debug] <configfile> <directory>
    {command} cluster [--debug] [--method <all|speciation|single>] [--distance <distancethreshold>] <directory>
    {command} plot [--debug] <configfile> <gpkgfile>

Options:
  -h --help                         Show this screen and exit
  --version                         Show version and exit
  <configfile>                      Path to the configuration TOML file
  --debug                           Enable debug logging
  --distance <distthreshold>        Distance threshold to set clusters [default: 0.3]
  <gpkgfile>                        Path to a gpkg file created during the clustering
  <directory>                       Directory to store/read the model output
  --method <all|speciation|single>  Clustering method to use

Typical workflow:
    {command} run configuration.toml results_20251023
    {command} cluster --method speciation --distance 0.2 results_20251023
    {command} plot configuration.toml population.gpkg
"""

    arguments = docopt.docopt(usage, sys.argv[1:], version=version)

    if arguments["--debug"]:
        logging.basicConfig(level=logging.DEBUG)

    command_to_function = {
        "run": run_leco,
        "cluster": cluster_languages,
        "plot": plot_results,
    }

    # Check if method is valid and call the corresponding function
    active_command = next(cmd for cmd in command_to_function if arguments[cmd])
    command_to_function[active_command](arguments)
