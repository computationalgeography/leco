import tempfile
from pathlib import Path

import docopt
import tomli_w

from ..cli.leco import cluster_languages, usage
from .subcommand import spawn_subcommand

__all__ = ["cluster"]


def cluster_(arguments: tuple[tuple[dict, Path], list[str]]) -> None:
    configuration_path_tuple, subcommand_arguments = arguments
    configuration, directory_path = configuration_path_tuple

    with tempfile.NamedTemporaryFile(delete_on_close=False) as file:
        tomli_w.dump(configuration, file)
        file.close()
        configuration_file_path = Path(file.name)
        subcommand_arguments = docopt.docopt(
            usage(),
            [
                "cluster",
                *subcommand_arguments,
                str(configuration_file_path),
                str(directory_path),
            ],
        )

        cluster_languages(subcommand_arguments, configuration)


def cluster(
    configuration_file_path: Path,
    *,
    max_nr_workers: int,
    continue_on_error: bool,
    arguments: list[str],
) -> None:
    """Spawn clustering commands."""
    spawn_subcommand(
        cluster_,
        configuration_file_path,
        max_nr_workers=max_nr_workers,
        continue_on_error=continue_on_error,
        arguments=arguments,
    )
