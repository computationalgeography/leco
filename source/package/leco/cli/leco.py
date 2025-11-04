"""Command line interface for leco model."""

import shutil
import sys
from datetime import datetime
from pathlib import Path

import docopt
import tomllib

# Use relative imports within the whole package! This makes it much easier to move things around later.
#
# Nitpicking:
# - run_classification → classify
# - run_model → simulate (?)
# - ...
#
# Then:
#
# from ..cluster import classify, classify_single, speciate
#
# A function or module named main is special. Don't use the name elsewhere. It is not needed. Rename to
# simulate.py?

from leco.cluster.language_classification import run_classification
from leco.cluster.per_timestep import run_classification_single
from leco.cluster.speciation import run_speciation
from leco.model.main import run_model
from leco.plot.main import plot
from leco.version import __version__ as version

from .main import main_function


# Understand the role of the main_function decorator and where to put it
# Spell-check and lint
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


def lang_classification(input_dir: str, method: str, dist_threshold: float) -> None:
    """Run language classification on the leco model output for a specified method."""
    # I know that method can only be one of these strings, but if in the future method contains another value
    # the code will not break but just do nothing and someone has to dive int to figure out why nothing
    # is happening. In this case, a map from method to function can be useful, e.g.:
    #     classification_by_method[method](input_dir, dist_threshold)
    # This pattern can also be used in the main function below: call a sub-main function per sub-command and
    # pass in all parsed arguments. Each of these sub-mains can then grab the values it needs and continue.
    if method == "all":
        # 3D clustering over all timesteps
        run_classification(input_dir, dist_threshold)

    if method == "single":
        # 2D clustering per timestep [Note: under development]
        run_classification_single(input_dir, dist_threshold)

    if method == "speciation":
        # Feed-forward speciation-based clustering [Note: under development]
        run_speciation(input_dir, dist_threshold)


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
    base = Path(outputpath)
    base.mkdir(exist_ok=True, parents=True)

    # Create a subdirectory for each run named after date and time
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    dir_name = f"results_{timestamp}_{suffix}" if suffix else f"results_{timestamp}"
    output_dir_run = base / dir_name
    output_dir_run.mkdir(parents=True, exist_ok=True)

    return output_dir_run


def main() -> None:
    """Command line interface for leco model."""
    command = Path(sys.argv[0]).name
    # Great, I can now see that there are three sub-commands to use, each with a separate set of
    # arguments.
    # Try to keep the line length of the usage string <= 80 chars
    # Get rid of unnecessary paths in examples
    # Store paths as Python Path instances, not strings. You can update lots of places where you create a Path
    # instance. If a string represents a path, make it a Path. If you need a string, convert the Path instance
    # to a string: str(my_path). You will likely not need to do this often.
    # What's with the --gpkg option. Can it be left out (is it *option*al)? If not, make it a
    # positional. Same for some of the other arguments.
    usage = f"""\
Run leco model

Usage:
    {command} run --config <configfile> --output <outputdirectory>
    {command} cluster --input <inputdirectory> [--method <all|speciation|single>] [--dist <distancethreshold>]
    {command} plot --gpkg <gpkgfile> --config <configfile>

Options:
  -h --help                         Show this screen and exit
  --version                         Show version and exit
  --config <configfile>             Path to the configuration TOML file
  --dist <distthreshold>            Distance threshold to set clusters [default: 0.3]
  --gpkg <gpkgfile>                 Path to a gpkg file created during the clustering
  --input <inputdirectory>          Input directory containing the .geoparquet files created during the run
  --method <all|speciation|single>  Clustering method to use
  --output <outputdirectory>        Output directory

Typical workflow:
    {command} run --config C:/home/PhD/leco_model/configuration.toml --output C:/home/PhD/leco_model/output/
    {command} cluster --input C:/home/PhD/leco_model/output/results_20251023 --method speciation --dist 0.2
    {command} plot --gpkg C:/home/PhD/leco_model/output/results_20251023/population.gpkg --config C:/home/PhD/leco_model/output/results_20251023/config.toml
"""

    arguments = docopt.docopt(usage, sys.argv[1:], version=version)

    # Great, short and easy
    # Nitpicking: prefer not to use abbreviations (dist, lang, config, ...). Here and elsewhere. Also in
    # configuration file. Variable names, function names, file names, ... are all part of the
    # documentation.
    if arguments["run"]:
        config_file = arguments["--config"]
        output_path = arguments["--output"]
        run_leco(config_file, output_path)

    if arguments["cluster"]:
        directory = arguments["--input"]
        # Get method and distance threshold if given, oterwise use defaults
        method = arguments.get("--method", "all")
        dist_threshold = float(arguments.get("--dist", 0.3))
        lang_classification(directory, method, dist_threshold)

    if arguments["plot"]:
        file = arguments["--gpkg"]
        config = arguments["--config"]
        plot_results(file, config)
