import os.path
import sys
from datetime import datetime
import docopt
import tomllib
from pathlib import Path
import traceback
import glob

from ..version import __version__ as version
from .main import main_function
from .model import run_model
from .language_classification import run_classification
from .phylogeny_largeclustering_ETE import create_phylo

from .plots.plotting_main import plot, plot_sensitivity


@main_function
def leco(config: dict, output_dir: str) -> None:
    print("Run leco model")
    try:
        run_model(config, output_dir)
    except Exception as e:
        print(f"Error running leco model: {e}")
        traceback.print_exc()
        print("Terminating execution of the leco model")
        exit(1)


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


def create_run_dir(outputpath: str, params: dict, suffix: str | None = None) -> str:
    """Create output directory for specific run and store parameter values in a text file"""
    os.makedirs(
        outputpath, exist_ok=True
    )  # Create the directory if it does not exist yet

    # Create a subdirectory for each run named after date and time
    timestamp = datetime.now().strftime("%Y%m%d")  # _%H%M%S")
    output_dir_run = os.path.join(
        outputpath,
        f"results_{timestamp}_{suffix}" if suffix else f"results_{timestamp}",
    )
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


def run_sensitivity_analysis(
    base_config: dict,
    param_name: str,
    values: list,
    output_base: str,
    dist_threshold: float,
):
    """Run the leco model for a range of values for a given parameter and classify the languages."""
    for val in values:
        print(f"Running sensitivity analysis: {param_name}={val}")
        config = base_config.copy()
        config[param_name] = val  # or deeper nesting depending on your TOML structure
        output_path = create_run_dir(output_base, config, suffix=f"{param_name}_{val}")
        leco(config, output_path)

        # Classify languages after each run
        lang_classification(output_path, dist_threshold)


def main() -> None:
    command = os.path.basename(sys.argv[0])
    usage = f"""\
Run leco model

Usage:
    {command} --classify=<resultsdir>
    {command} --phylo=<gpkgfile>
    {command} --plot=<gpkgfile>
    {command} --plot-summaries=<gpkgfile>
    {command} --plot-animation=<gpkgfile>
    {command} --plot-3d=<gpkgfile>
    {command} --sensruns -i <configfile> -o <outputdir> -p <param> -v <values> [-d <distthreshold>]
    {command} --sensplot [--null=<pathpattern>] [--point=<pathpattern>] [--barrier=<pathpattern>]
    {command} -i <configfile> [-o <outputdir>] [--classify-after] [--all] [-d <distthreshold>]

Options:
  -h --help                                      Show this screen and exit.
  --version                                      Show version and exit.
  -i <configfile>                                Path to the configuration TOML file.
  -o <outputdir>                                 Output directory.
  -d <distthreshold>                             Distance threshold [default: 0.3].
  --classify=<resultsdir>                        Run language classification on existing leco output.
  --classify-after                               Run language classification after leco model.
  --phylo=<gpkgfile>                             Create phylogeny of languages from a .gpkg file.
  --plot=<gpkgfile>                              Create all plots from a .gpkg file.
  --plot-summaries=<gpkgfile>                    Create summary plots from leco output.
  --plot-animation=<gpkgfile>                    Create animation from leco output.
  --plot-3d=<gpkgfile>                           Create interactive 3D plot from leco output.
  --all                                          Run full post-processing pipeline after leco.
  --sensruns                                     Run sensitivity analysis.
  -p <param>                                     Parameter name to vary during sensitivity runs.
  -v <values>                                    Comma-separated values for sensitivity parameter.
  --sensplot                                     Create summary plots for multiple sensitivity runs given a glob pattern to match directories.
  --null=<pathpattern>                            Glob pattern to match directories for null scenario sensitivity runs.
  --point=<pathpattern>                           Glob pattern to match directories for point scenario sensitivity runs.
  --barrier=<pathpattern>                         Glob pattern to match directories for barrier scenario sensitivity runs.

Examples:
    {command} -i config.toml -o results/
    {command} --classify results/run_001/ -d 0.1
    {command} --classify-after -i config.toml -o results/ -d 0.1
    {command} --phylo results/run_001/population.gkpg
    {command} --plot results/run_001/population.gpkg
    {command} --plot-summaries results/run_001/population.gpkg
    {command} --plot-animation results/run_001/population.gpkg
    {command} --plot-3d results/run_001/population.gpkg
    {command} --all -i config.toml -o results/ -d 0.3
    {command} --sensruns -i config.toml -o results/ -p seed -v 1,2,3,4,5
    {command} --sensplot --null "results/sensruns_output*/population.gpkg"
"""
    arguments = docopt.docopt(usage, sys.argv[1:], version=version)
    print(f"Arguments parsed: {arguments}")

    # Helper functions
    def get_dist_threshold() -> float:
        """Get distance threshold from command line arguments or return default value"""
        return float(arguments["-d"]) if arguments["-d"] else 0.3

    def get_param_file(input_file: str) -> str:
        """Get parameter file from command line arguments or return None"""
        # if arguments["--plot-animation"] or arguments["--plot-3d"] or arguments["--plot"]:
        return open_parameters(Path(input_file).parent)
        # return None

    def get_gpkg_files(pattern: str) -> list[str]:
        """Expand a glob pattern to a list of .gpkg files across directories."""
        matched_files = []
        for path in glob.glob(pattern):
            matched_files.extend(str(f) for f in Path(path).rglob("*.gpkg"))
        if not matched_files:
            print(f"No .gpkg files found for pattern: {pattern}")
        return matched_files

    # Command dispatch table of commands that don't require the leco model to run
    commands = {
        "--classify": lambda: lang_classification(
            arguments["--classify"], get_dist_threshold()
        ),
        "--phylo": lambda: create_phylo(arguments["--phylo"]),
        "--plot-summaries": lambda: plot(
            arguments["--plot-summaries"],
            None,
            True,
            False,
            False,
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
        "--sensplot": lambda: plot_sensitivity(
            get_gpkg_files(arguments["--null"]),
            None,  # get_gpkg_files(arguments["--point"]),
            get_gpkg_files(arguments["--barrier"]),
        ),
    }

    # Execute single commands
    for cmd, func in commands.items():
        if arguments[cmd]:
            return func()

    # Handle sensitivity analysis
    if arguments["--sensruns"]:
        config = load_config(arguments["-i"])
        if not arguments["-v"]:
            return print("Please provide values -v with comma-separated list.")
        param = arguments["-p"]
        values = [int(v) for v in arguments["-v"].split(",")]
        output_base = arguments["-o"] if arguments["-o"] else "sensruns_output"
        print(output_base)
        return run_sensitivity_analysis(
            config, param, values, output_base, get_dist_threshold()
        )

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
    leco(config, output)

    # Post-processing pipeline
    if arguments["--classify-after"] or arguments["--all"]:
        print("Running language classification on leco results")
        lang_classification(output, get_dist_threshold())

    if arguments["--all"]:
        create_phylo(os.path.join(output, "population.gpkg"))
        plot(
            os.path.join(output, "population.gpkg"),
            open_parameters(output),
            True,
            True,
            True,
        )
