import sys
import traceback
from collections.abc import Callable


def main_function(function: Callable) -> int:
    def wrapper(*args, **kwargs) -> int:
        try:
            status = function(*args, **kwargs)
        except Exception as error:  # pylint: disable=broad-exception-caught
            traceback.print_exception(error)

            sys.stderr.write(f"Error occurred:\n{error!s}\n")
            status = 1

        return status

    return wrapper
