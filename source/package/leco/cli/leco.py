"""Command line interface for leco model."""

import shutil
import sys
from datetime import datetime
from pathlib import Path

import docopt
import tomllib
from leco.cluster.language_classification import run_classification
from leco.model.main import run_model
from leco.plot.main import plot
from leco.version import __version__ as version

from .main import main_function


@main_function
def run_leco(config_file: str, output_path: str) -> None:
    """Run the leco model with specified configuration."""
    config = load_config(config_file)
    # Create a subdirectory for the specific run in the output path
    output_dir = create_run_dir(output_path)
    # Store a copy of the configuration file iin the run directory
    shutil.copy2(config_file, Path(output_dir) / "config.toml")
    # Run the leco model
    run_model(config, output_dir)


def lang_classification(input_dir: str, dist_threshold: float) -> None:
    """Run language classification on the leco model output."""
    run_classification(input_dir, dist_threshold)


def plot_results(data: str, config_file: str) -> None:
    """Create plots of the leco model output."""
    config = load_config(config_file)
    plot(data, config)


def load_config(config_file: str) -> dict:
    """Load TOML config with error handling."""
    with Path.open(config_file, "rb") as f:
        return tomllib.load(f)


def create_run_dir(outputpath: str, suffix: str | None = None) -> str:
    """Create output directory for specific run and store parameter values in a text file."""
    # Create the directory if it does not exist yet
    Path.mkdir(outputpath, exist_ok=True, parents=True)

    # Create a subdirectory for each run named after date and time
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_dir_run = Path(outputpath) / (
        f"results_{timestamp}_{suffix}" if suffix else f"results_{timestamp}",
    )
    Path.mkdir(output_dir_run, exist_ok=True, parents=True)

    return output_dir_run


def get_dist_threshold(threshold: str) -> float:
    """Get distance threshold from command line arguments or return default value."""
    return float(threshold) if threshold else 0.3


def main() -> None:
    """Command line interface for leco model."""
    command = Path(sys.argv[0]).name
    usage = f"""\
Run leco model

Usage:
    {command} run --config <configfile> --output <outputdirectory>
    {command} cluster --input <inputdirectory> [--dist <distancethreshold>]
    {command} plot --gpkg <gpkgfile> --config <configfile>

Options:
  -h --help                      Show this screen and exit
  --version                      Show version and exit
  -c --config <configfile>       Path to the configuration TOML file
  -d --dist <distthreshold>      Distance threshold to set clusters [default: 0.3]
  -g --gpkg <gpkgfile>           Path to a gpkg file created during the cluster step
  -i --input <inputdirectory>    Input directory containing the .geoparquet files created during the run
  -o --output <outputdirectory>  Output directory

Typical workflow:
    {command} run -c C:/home/PhD/leco_model/configuration.toml -o C:/home/PhD/leco_model/output/
    {command} cluster -i C:/home/PhD/leco_model/output/results_20251023 -d 0.2
    {command} plot -g C:/home/PhD/leco_model/output/results_20251023/population.gpkg
                   -c C:/home/PhD/leco_model/output/results_20251023/config.toml
"""
    arguments = docopt.docopt(usage, sys.argv[1:], version=version)
    print(arguments)

    if arguments["run"]:
        config_file = arguments["--config"]
        output_path = arguments["--output"]
        run_leco(config_file, output_path)

    if arguments["cluster"]:
        directory = arguments["--input"]
        dist_threshold = get_dist_threshold(arguments["--dist"])
        lang_classification(directory, dist_threshold)

    if arguments["plot"]:
        file = arguments["--gpkg"]
        config = arguments["--config"]
        plot_results(file, config)
