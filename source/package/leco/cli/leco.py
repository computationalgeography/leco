import os.path
import sys

# import os
from datetime import datetime

import docopt

import tomllib

from ..version import __version__ as version
from .main import main_function
from .array_main import run_model


@main_function
def leco(config: dict, output_dir: str) -> None:
    print("Run leco model")
    run_model(config, output_dir)


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


def main() -> None:
    command = os.path.basename(sys.argv[0])
    usage = f"""\
Run leco model

Usage:
    {command} -i <configfile> [-o <outputpath>] [-g <growthratefile>]

Arguments:
    -i <configfile>       Specify path to TOML format configuration file
    -o <outputpath>       Specify path to output directory (does not have to exist yet)
    -g <growthratefile>   Specify path to file that contains growth rate per timestep

Options:
    -h --help          Show this screen and exit
    --version          Show version and exit

Examples:
    {command} -i config.toml -o results/
"""
    arguments = docopt.docopt(usage, sys.argv[1:], version=version)
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

    # TODO Handle command-line arguments
    print(f"command-line arguments: {arguments}")

    return leco(config, output)
