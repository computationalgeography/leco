"""Command line interface for the script to spawn concurrent leco model runs."""

import sys
from pathlib import Path

import docopt

from .. import spawn
from ..version import __version__ as version


def main() -> None:
    """Command line interface for spawn command."""
    command = Path(sys.argv[0]).name
    usage = f"""\
Spawn concurrent leco model runs

Usage:
    {command}
        [--max_nr_workers=<nr_workers>] [--continue_on_error]
        <configuration_file>

Options:
    -h --help                      Show this screen and exit
    --version                      Show version and exit
    <configuration_file>           Path to a TOML configuration file
    --max_nr_workers=<nr_workers>  Maximum number of processes to use
                                   [default: {spawn.default_max_nr_workers()}]
                                   This number must likely not be larger than
                                   the number of physical cores in the node.
    --continue_on_error            Continue spawning processes when an error
                                   occurs

The configuration file determines what kind of runs will be spawned.
"""
    arguments = docopt.docopt(usage, sys.argv[1:], version=version)
    max_nr_workers = int(arguments["--max_nr_workers"])
    continue_on_error = arguments["--continue_on_error"]
    configuration_file_path = Path(arguments["<configuration_file>"])

    if max_nr_workers <= 0:
        raise ValueError("--max_nr_workers must be larger than zero")

    if not configuration_file_path.exists():
        raise ValueError(f"Configuration file {configuration_file_path} does not exist")

    spawn.spawn(configuration_file_path, max_nr_workers=max_nr_workers, continue_on_error=continue_on_error)
