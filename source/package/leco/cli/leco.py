import os.path
import sys
from datetime import datetime
import docopt
import tomllib
from pathlib import Path

from ..version import __version__ as version
from .main import main_function
from .model import run_model
from .language_classification import run_classification

from .plots.plotting_main import plot


@main_function
def leco(config: dict, output_dir: str) -> None:
    print("Run leco model")
    run_model(config, output_dir)


def lang_classification(input_dir: str, dist_threshold: float) -> None:
    print("Run language classifiation")
    run_classification(input_dir, dist_threshold)


def load_config(config_file):
    """Load TOML config with error handling"""
    try:
        with open(config_file, "rb") as f:
            return tomllib.load(f)
    except Exception as e:
        print(f"Error: {e}. Please provide a configuration file in TOML format")
        exit(1)


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
    {command} --plot <inputfile>
    {command} --plot-summaries <inputfile>
    {command} --plot-animation <inputfile>
    {command} --plot-3d <inputfile>
    {command} --all -i <configfile> -o <outputpath>

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
    --plot <outputpath/resultsdir/df.gpkg>                  Creates plots of the leco model output: summary, animated and 3D interactive plots
    --plot-summaries <outputpath/resultsdir/df.gpkg>        Create summarizing plots of the leco model output
    --plot-animation <outputpath/resultsdir/df.gpkg>        Create animation of the leco model output
    --plot-3d <outputpath/resultsdir/df.gpkg>               Create 3D interactive plot of the leco model output
    --all                                                   Run leco model, language classification and create plots in one go

Examples:
    {command} -i config.toml -o results/
    {command} --classify results/run_001/ -d 0.1
    {command} --classify-after -i config.toml -o results/ -d 0.1
    {command} --plot results/run_001/population.gpkg
    {command} --plot-summaries results/run_001/population.gpkg
    {command} --plot-animation results/run_001/population.gpkg
    {command} --plot-3d results/run_001/population.gpkg
    {command} --all -i config.toml -o results/
"""
    arguments = docopt.docopt(usage, sys.argv[1:], version=version)
    print(f"command-line arguments: {arguments}")

    # Helper functions
    def get_dist_threshold() -> float:
        """Get distance threshold from command line arguments or return default value"""
        return float(arguments["-d"]) if arguments["-d"] else 0.2

    def get_param_file(input_file: str) -> str:
        """Get parameter file from command line arguments or return None"""
        # if arguments["--plot-animation"] or arguments["--plot-3d"] or arguments["--plot"]:
        return open_parameters(Path(input_file).parent)
        # return None

    # Command dispatch table of commands that don't require the leco model to run
    commands = {
        "--classify": lambda: lang_classification(
            arguments["--classify"], get_dist_threshold()
        ),
        "--plot-summaries": lambda: plot(
            arguments["--plot-summaries"], None, True, False, False
        ),
        "--plot-animation": lambda: plot(
            arguments["--plot-animation"],
            get_param_file(arguments["--plot-animation"]),
            False,
            True,
            False,
        ),
        "--plot-3d": lambda: plot(arguments["--plot-3d"], None, False, False, True),
        "--plot": lambda: plot(
            arguments["--plot"], get_param_file(arguments["--plot"]), True, True, True
        ),
    }

    # Execute single commands
    for cmd, func in commands.items():
        if arguments[cmd]:
            return func()

    # Handle main workflow (leco model run and therefore a config file is required)
    if not arguments["-i"]:
        return print(
            "Please provide a configuration file in TOML format. See --help for usage."
        )

    # Load the configuration file
    config = load_config(arguments["-i"])
    # Create a subdirectory for the specific run in the output path and store the parameter values
    output = create_run_dir(arguments["-o"], config) if arguments["-o"] else None

    # Run main leco model
    try:
        leco(config, output)
    except Exception as e:
        print(f"Error running leco model: {e}")
        print("Terminating execution - skipping post-processing")
        exit(1)

    # Post-processing pipeline
    if arguments["--classify-after"] or arguments["--all"]:
        print("Running language classification on leco results")
        lang_classification(output, get_dist_threshold())

    if arguments["--all"]:
        plot(
            os.path.join(output, "population.gpkg"),
            open_parameters(output),
            True,
            True,
            True,
        )
