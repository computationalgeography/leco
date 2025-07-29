import os.path
import sys
from datetime import datetime
import docopt
import tomllib
from pathlib import Path

from ..version import __version__ as version
from .main import main_function
from .array_main import run_model
from .language_classification import run_classification
from .summary_plots import plot_summaries
from .animation_plot import plot_animation


@main_function
def leco(config: dict, output_dir: str) -> None:
    print("Run leco model")
    run_model(config, output_dir)


def lang_classification(input_dir: str, dist_threshold: float) -> None:
    print("Run language classifiation")
    run_classification(input_dir, dist_threshold)


def plot_sums(input_file: str) -> None:
    print("Create summarizing plots of the leco model output")
    plot_summaries(input_file)


def plot_anim(input_file: str, parameters: dict) -> None:
    print("Create animation of the leco model output")
    plot_animation(input_file, parameters)


def create_run_dir(outputpath: str, params: dict) -> str:
    """Create output directory for specific run and store parameter values in a text file"""
    os.makedirs(
        outputpath, exist_ok=True
    )  # Create the directory if it does not exist yet

    # Create a subdirectory for each run named after date and time
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir_run = os.path.join(outputpath, f"results_{timestamp}")
    os.makedirs(output_dir_run, exist_ok=True)

    # Write parameter values to text file and store in the ouput directory of the specific run
    with open(os.path.join(output_dir_run, "parameters.txt"), "w") as f:
        for key, value in params.items():
            f.write(f"{key} = {value}\n")

    return output_dir_run


def open_parameters(output_path: str) -> dict:
    """Read parameters in as strings from a text file (parameters.txt) which is automatically created during leco run and stored in the output directory"""
    params = {}
    param_file = os.path.join(output_path, "parameters.txt")
    if os.path.exists(param_file):
        with open(param_file, "r") as f:
            for line in f:
                key, value = line.strip().split(" = ")
                params[key] = value
    else:
        print(
            f"Parameter file {param_file} not found. Please ensure the dataframe is stored in the same directory as the parameters file created during the leco run."
        )
    return params


def main() -> None:
    command = os.path.basename(sys.argv[0])
    usage = f"""\
Run leco model

Usage:
    {command} -i <configfile> -o <outputpath> [-g <growthratefile>]
    {command} --classify <outputpath/resultsdir> [-d <distthreshold>]
    {command} --classify-after -i <configfile> -o <outputpath> [-g <growthratefile>] [-d <distthreshold>]
    {command} --plot-summaries <inputfile>
    {command} --plot-animation <inputfile>

Arguments:
    -i <configfile>       Specify path to TOML format configuration file
    -o <outputpath>       Specify path to output directory (does not have to exist yet)
    -g <growthratefile>   Specify path to file that contains growth rate per timestep
    -d <distthreshold>    Specify distance threshold for language classification (default: 0.2)

Options:
    -h --help                                               Show this screen and exit
    --version                                               Show version and exit
    --classify <outputpath/resultsdir>                      Run language classification on existing leco output
    --classify-after                                        Run language classification directly after running leco model
    --plot-summaries <outputpath/resultsdir/df.gpkg>        Create summarizing plots of the leco model output
    --plot-animation <outputpath/resultsdir/df.gpkg>        Create animation of the leco model output.

Examples:
    {command} -i config.toml -o results/
    {command} --classify results/run_001/ -d 0.1
    {command} --classify-after -i config.toml -o results/ -d 0.1
    {command} --plot-summaries results/run_001/population.gpkg
    {command} --plot-animation results/run_001/population.gpkg
"""
    arguments = docopt.docopt(usage, sys.argv[1:], version=version)
    print(f"command-line arguments: {arguments}")

    if arguments["--classify"]:
        input_dir = arguments["--classify"]
        dist_threshold = float(arguments["-d"]) if arguments["-d"] else 0.2
        return lang_classification(input_dir, dist_threshold)

    if arguments["--plot-summaries"]:
        input_file = arguments["--plot-summaries"]
        return plot_sums(input_file)

    if arguments["--plot-animation"]:
        input_file = arguments["--plot-animation"]
        param_file = open_parameters(Path(input_file).parent)
        return plot_anim(input_file, param_file)

    config_file = arguments["-i"]
    # Parse the TOML file and make sure the right format is used
    try:
        with open(config_file, "rb") as g:
            config = tomllib.load(g)
    except Exception as e:
        print(f"Error: {e}. Please provide a configuration file in TOML format")
        exit(1)

    output = arguments["-o"]  # Will be None if not provided

    if output:
        # Create a subdirectory for the specific run in the output path and store the parameter values
        output = create_run_dir(output, config)

    # Run the leco model
    leco(config, output)

    # Run language classification directly after running the leco model
    if arguments["--classify-after"]:
        print("Running language classification on leco results")
        dist_threshold = (
            float(arguments["--distthreshold"]) if arguments["--distthreshold"] else 0.2
        )
        lang_classification(output, dist_threshold)
