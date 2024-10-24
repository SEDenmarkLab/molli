[![Upload Python Package](https://github.com/SEDenmarkLab/molli/actions/workflows/python-publish.yml/badge.svg)](https://github.com/SEDenmarkLab/molli/actions/workflows/python-publish.yml)
[![Upload Anaconda Package](https://github.com/SEDenmarkLab/molli/actions/workflows/anaconda-deploy.yml/badge.svg)](https://github.com/SEDenmarkLab/molli/actions/workflows/anaconda-deploy.yml)

![Anaconda version](https://anaconda.org/esalx/molli/badges/version.svg)
![Anaconda license](https://anaconda.org/esalx/molli/badges/license.svg)
![Anaconda last updated](https://anaconda.org/esalx/molli/badges/latest_release_relative_date.svg)
![Anaconda platforms](https://anaconda.org/esalx/molli/badges/platforms.svg)

<img src="docs/imgs/molli_logo.svg" width="400">

# `molli`: Molecular Toolbox Library

https://github.com/SEDenmarkLab/molli

Developed by:

- Alexander S. Shved
- Blake E. Ocampo   
- Elena S. Burlova  
- Casey L. Olen 
- N. Ian Rinehart   

[S. E. Denmark Laboratory, University of Illinois, Urbana-Champaign](https://denmarkgroup.illinois.edu/)

For all work using `molli`, please cite the primary publication: Shved, A. S.; Ocampo, B. E.; Burlova, E. S.; Olen, C. L.; Rinehart, N. I.; Denmark. S. E.; *J. Chem. Inf. Mod.* **2024**, [**DOI**: 10.1021/acs.jcim.4c00424](https://pubs.acs.org/doi/10.1021/acs.jcim.4c00424)

Supplementary Materials and Libraries can be found on the Zenodo Repository: [**DOI**: 10.5281/zenodo.10719790](https://zenodo.org/records/10719791)

Copyright 2022-2023 The Board of Trustees of the University of Illinois.
All Rights Reserved.

# About `molli`

Molli is a cross-platform toolbox written in modern Python (3.10+) that provides a convenient API for molecule manipulations, combinatorial library generation with stereochemical fidelity from plain CDXML files, as well as parallel computing interface. The main feature of molli is the full representation of molecular graphs, geometries and geometry ensembles with no implicit atoms. Additionally, a compact and extensible format for molecular library storage make it a useful tool for *in silico* library generation. `molli` is cross-platform code that runs on a wide range of hardware from laptops and workstations to distributed memory clusters. 

# Installation and Building

Molli is available as the source code distribution on GitHub. Additionally, convenient installation is provided in the form of a PyPi package and conda package.

**Note**: We routinely test the package on Linux and Windows OS. OSX support is tested upon the pull request submission using GitHub workflows. We can only offer limited support for that OS at this time.

## Install using `pip`

### Installation from PyPI

The easiest way to obtain molli is to obtain the latest [PyPI package](https://pypi.org/project/molli/). 
```bash
pip install molli
```
Upon a successful installation of molli, one can test the installation by running the following commands
```bash
molli --VERSION\
molli test -vv
```
which will provide the current version (it is obtained dynamically from the Git tags and determined at the installation time) and run the full test suite to guarantee that the core functionality performs correctly.

### Install from source

Installation from source can offer a few advantages, such as the editable installation, or installing . This is convenient for users who wish to significantly alter their `molli` experience by modifying the core functionality

```bash
pip install git+https://github.com/SEDenmarkLab/molli.git

# or

pip install -e git+https://github.com/SEDenmarkLab/molli.git#egg=molli
```
**Editable installation**: (Assumes that the repository source code was cloned onto the hard drive using Github tools into `./molli/` folder) We have noticed that development with VSCode is not greatly compatible with the most recent version of the 

```bash
pip install -e molli/ --config-settings editable_mode=compat
```

## Install as a `conda` package

Molli can be installed from a conda repository:

**Note**: Conda setup is not fully configured yet, so you may expect that there will be slight changes to the syntax. For more information about the current installation instructions, please visit the [Anaconda repository](https://anaconda.org/esalx/molli)

```bash
conda install molli
```

# Testing the installation

## Core functions

**Note**: By default, `molli` only tests the core functionality: the functions that *do not* depend on external computational or chemoinformatics packages, such as OpenBabel, RDKit, Orca, XTB and CREST. These tests are considered extended (see below).

There are two syntaxes that allow to test the functionality of `molli`, of which one (`molli test`) is a more convenient alias for another (`python -m unittest`). See [unittest documentation](https://docs.python.org/3/library/unittest.html) for additional arguments, which can be applied to both. 

```bash
python -m unittest molli_test # additional args
molli test # additional args
```

## Extended tests

These tests are *automatically* invoked if the corresponding packages are either importable (that is, installed in the same `conda` environment or the corresponding Python virtual environment), or the corresponding executables can be located. An example of such extended test is found below (if [Environment Modules](https://modules.readthedocs.io/en/latest/) configures the packages on your machine)

```bash
module load xtb/6.4.1
module load crest/2.11.1
molli test -vv # Now this tests XTB and CREST driver
```
