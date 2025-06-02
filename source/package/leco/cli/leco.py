import os.path
import sys

import docopt

from ..version import __version__ as version
from .main import main_function


@main_function
def leco() -> None:
    print("TODO: Run leco model")


def main() -> None:
    command = os.path.basename(sys.argv[0])
    usage = f"""\
Run leco model

Usage:
    {command}

Options:
    -h --help          Show this screen and exit
    --version          Show version and exit

Examples:
    {command}
"""
    arguments = docopt.docopt(usage, sys.argv[1:], version=version)

    # TODO Handle command-line arguments
    print(f"command-line arguments: {arguments}")

    return leco()
