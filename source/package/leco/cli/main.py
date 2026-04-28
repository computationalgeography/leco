import sys
import traceback
from collections.abc import Callable


def main_function(function: Callable) -> Callable[..., int]:
    def wrapper(*args, **kwargs) -> int:
        status = 1

        try:
            result = function(*args, **kwargs)
            status = result if isinstance(result, int) else 0
        except Exception as error:
            traceback.print_exception(error)
            sys.stderr.write(f"Error occurred:\n{error!s}\n")

        return status

    return wrapper
