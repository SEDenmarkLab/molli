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
    name="molli",
    packages=find_packages(),
    package_data={"molli.data": ["*.yml", "*.yaml"], "molli.files": ["*"]},
    version="1.0.0a4",
    author="Alexander S. Shved",
    author_email="shvedalx@illinois.edu",
    install_requires=[
        "numpy>=1.22.3",
        "scipy>=1.9.1",
        "attrs~=22.1.0",
        "packaging~=22.0",
        "bidict~=0.21.2",
        "requests~=2.28",
        "PyYAML~=5.3",
        "msgpack~=1.0.3",
        "scipy>=1.4.1",
        "colorama>=0.4.4",
        "networkx>=2.8.7",
        "tqdm~=4.64.0",
        "h5py~=3.7.0",
        "pyvista~=0.37"
    ],
    python_requires=">=3.10,<3.11",
    entry_points={"console_scripts": ["molli = molli.__main__:main"]},
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    include_dirs=include_dirs,
)
