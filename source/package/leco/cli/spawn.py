"""Command line interface for the script to spawn concurrent leco model runs."""

import sys
from pathlib import Path

import docopt

from .. import spawn
from ..version import __version__ as version
from .main import main_function

__all__ = ["main"]


def spawn_run(
    configuration_file_path: Path,
    *,
    max_nr_workers: int,
    continue_on_error: bool,
    arguments: list[str],
) -> None:
    spawn.run(
        configuration_file_path,
        max_nr_workers=max_nr_workers,
        continue_on_error=continue_on_error,
        arguments=arguments,
    )


def spawn_cluster(
    configuration_file_path: Path,
    *,
    max_nr_workers: int,
    continue_on_error: bool,
    arguments: list[str],
) -> None:
    spawn.cluster(
        configuration_file_path,
        max_nr_workers=max_nr_workers,
        continue_on_error=continue_on_error,
        arguments=arguments,
    )


@main_function
def main() -> None:
    """Command line interface for spawn command."""
    command = Path(sys.argv[0]).name
    usage = f"""\
Spawn concurrent leco model runs

Usage:
    {command} (run | cluster)
        [--max_nr_workers=<nr_workers>] [--continue_on_error]
        <configuration_file> [-- <argument>...]

Options:
    -h --help                      Show this screen and exit
    --version                      Show version and exit
    <configuration_file>           Path to a TOML configuration file
    <argument>                     Subcommand-specific arguments.
                                   These are the ones supported by the leco
                                   cli, without the subcommand itself.
    --max_nr_workers=<nr_workers>  Maximum number of processes to use
                                   [default: {spawn.default_max_nr_workers()}]
                                   This number must likely not be larger than
                                   the number of physical cores in the node.
    --continue_on_error            Continue spawning processes when an error
                                   occurs

Example workflow:

python {command} run --max_nr_workers=4 documentation/spawn.toml

python {command} cluster --max_nr_workers=4 documentation/spawn.toml -- \
    --method feed_forward --linkage average --distance 0.3
"""
    arguments = docopt.docopt(usage, sys.argv[1:], version=version)
    max_nr_workers = int(arguments["--max_nr_workers"])
    continue_on_error = arguments["--continue_on_error"]
    configuration_file_path = Path(arguments["<configuration_file>"])

    subcommand_arguments = arguments["<argument>"]

    if max_nr_workers <= 0:
        raise ValueError("--max_nr_workers must be larger than zero")

    if not configuration_file_path.exists():
        raise ValueError(f"Configuration file {configuration_file_path} does not exist")

    command_to_function = {
        "run": spawn_run,
        "cluster": spawn_cluster,
    }

    active_command = next(cmd for cmd in command_to_function if arguments[cmd])
    command_to_function[active_command](
        configuration_file_path,
        max_nr_workers=max_nr_workers,
        continue_on_error=continue_on_error,
        arguments=subcommand_arguments,
    )
