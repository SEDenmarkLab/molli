![Anaconda version](https://anaconda.org/esalx/molli/badges/version.svg)
![Anaconda license](https://anaconda.org/esalx/molli/badges/license.svg)
![Anaconda last updated](https://anaconda.org/esalx/molli/badges/latest_release_relative_date.svg)
![Anaconda platforms](https://anaconda.org/esalx/molli/badges/platforms.svg)

# `molli 1.0`: Molecular Toolbox Library

https://github.com/SEDenmarkLab/molli

Developed by:

- Alexander S. Shved
- Blake E. Ocampo   
- Elena S. Burlova  
- Casey L. Olen 
- N. Ian Rinehart   

[S. E. Denmark Laboratory, University of Illinois, Urbana-Champaign](https://denmarkgroup.illinois.edu/)

Copyright 2022-2023 The Board of Trustees of the University of Illinois.
All Rights Reserved.

# Installation and Building

Molli is designed to serve as both a pip package and a conda package.
The main difference comes from the fact that several required components are unavailable from a normal pip distribution (openbabel, xtb/crest, etc). 

Molli is both a pip and conda package. Technically, it is installable with both, but there are large differences described below. They boil down to pip's inability to install certain secondary dependencies.

## Install as a `pip` package

This mode is not preferred, but allows easier debugging in certain cases. Please note that this installation may leave your conda environment (if you have one) in a broken state.

1. Download or clone the source code to *`<path_to_molli_src>`*
2. `pip install` *`<path_to_molli_src>`*

This way of installing also compiles molli external C++ code correctly.

## Install as a `conda` package

TBD when conda channel becomes functional


## Building
1. Create a new environment
2.
```
conda config --set channel_priority flexible
conda config --append channels conda-forge
```

## Testing installation

1. Run the following command from your terminal.
```
molli info
```
2. Perform unit tests (-v for verbose mode is optional):
```
python -m unittest molli_test -v
```

## Install in development mode

```
pip install -e . --config-settings editable_mode=compat
```

# Hello, World!

This section describes your first steps in molli package

# Package structure

# Benchmarks
