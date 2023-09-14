# ================================================================================
# This file is part of
#      -----------
#      MOLLI 1.0.0
#      -----------
# (C) 2021 Alexander S. Shved and the Denmark laboratory
# University of Illinois at Urbana-Champaign, Department of Chemistry
# ================================================================================


"""
This configures proper installation of the molli package
and enables pip install .
"""

from setuptools import setup, find_packages, Extension
from glob import glob
import os

# Include directories for c++ source compilation
import pybind11 as pb11
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "molli_xt",
        sources=glob("molli_xt/*.cpp"),
    ),
]

include_dirs = [
    pb11.get_include(),
    "molli_xt",
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    include_dirs=include_dirs,
)
