import tempfile
from pathlib import Path

import docopt
import tomli_w

from ..cli.leco import plot_results, usage
from .subcommand import spawn_subcommand

__all__ = ["plot"]


def plot_(arguments: tuple[tuple[dict, Path], list[str]]) -> None:
    configuration_path_tuple, subcommand_arguments = arguments
    configuration, directory_path = configuration_path_tuple
    # TODO: This name must either be passed in or programmatically determined
    # NOTE: See also: https://github.com/computationalgeography/leco/issues/29
    geopackage_path = directory_path / "population_average.gpkg"

    if not geopackage_path.exists():
        raise ValueError(f"GeoPackage {geopackage_path} does not exist")

    # geopackage_pathname,

    with tempfile.NamedTemporaryFile(delete_on_close=False) as file:
        tomli_w.dump(configuration, file)
        file.close()
        configuration_file_path = Path(file.name)
        subcommand_arguments = docopt.docopt(
            usage(),
            [
                "plot",
                *subcommand_arguments,
                str(configuration_file_path),
                str(geopackage_path),
            ],
        )

        plot_results(subcommand_arguments, configuration)


def plot(
    configuration_file_path: Path,
    *,
    max_nr_workers: int,
    continue_on_error: bool,
    arguments: list[str],
) -> None:
    """Spawn plotting of model results."""
    spawn_subcommand(
        plot_,
        configuration_file_path,
        max_nr_workers=max_nr_workers,
        continue_on_error=continue_on_error,
        arguments=arguments,
    )
