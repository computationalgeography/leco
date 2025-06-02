import unittest

import leco


class PackageTest(unittest.TestCase):
    def test_version_attribute(self):
        self.assertTrue(hasattr(leco, "__version__"))
        self.assertTrue(isinstance(leco.__version__, str))
