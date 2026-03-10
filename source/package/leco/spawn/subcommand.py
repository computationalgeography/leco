import concurrent
from collections.abc import Callable
from pathlib import Path

from .configuration import run_configurations

__all__ = ["spawn_subcommand"]


def spawn_subcommand(
    function: Callable,
    configuration_file_path: Path,
    *,
    max_nr_workers: int,
    continue_on_error: bool,
    arguments: list[str],
) -> None:
    """Spawn function passed in."""
    configurations_ = run_configurations(configuration_file_path)
    generators = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_nr_workers) as executor:
        generator = executor.map(
            function,
            zip(configurations_, len(configurations_) * [arguments], strict=True),
        )

        # Obtaining the result raises any exception thrown and stops further processing
        if continue_on_error:
            # Delay obtaining results
            generators.append(generator)
        else:
            # Obtain results now
            list(generator)

    if continue_on_error:
        for generator in generators:
            list(generator)
