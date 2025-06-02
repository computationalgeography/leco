import sys
import traceback
from typing import Callable


def main_function(function: Callable):
    def wrapper(*args, **kwargs) -> int:
        try:
            status = function(*args, **kwargs)
        except Exception as error:  # pylint: disable=broad-exception-caught
            traceback.print_exception(error)

            sys.stderr.write(f"Error occurred:\n{str(error)}\n")
            status = 1

        return status

    return wrapper
