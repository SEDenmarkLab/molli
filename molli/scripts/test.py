"""
This module performs unit tests.
"""

from argparse import ArgumentParser
import unittest as ut
import molli as ml
import molli_test as mltest

arg_parser = ArgumentParser(
    "molli test",
    description="This module is included as a testing playground. Don't expect anything serious here!",
)

testsuite = ut.TestSuite()
testsuite.addTest(ut.makeSuite(mltest.BasicInstallTC))


def molli_main(args, config=None, output=None, **kwargs):
    runner = ut.TextTestRunner(verbosity=1)
    runner.run(testsuite)
