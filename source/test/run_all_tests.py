#!/usr/bin/env python
import os.path
import sys
import unittest


def main():
    start_directory_pathname = os.path.dirname(__file__)
    pattern = "*_test.py"
    verbosity = 2

    test_suite = unittest.defaultTestLoader.discover(start_directory_pathname, pattern)
    test_runner = unittest.TextTestRunner(verbosity=verbosity)
    result = test_runner.run(test_suite)

    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
