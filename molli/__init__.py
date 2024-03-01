# ================================================================================
# This file is part of `molli 1.0`
# (https://github.com/SEDenmarkLab/molli)
#
# Developed by:
#     - Alexander S. Shved <shvedalx@illinois.edu>,
#     - Blake E. Ocampo
#     - Elena S. Burlova
#     - Casey L. Olen
#     - N. Ian Rinehart
#
# S. E. Denmark Laboratory, University of Illinois, Urbana-Champaign
# https://denmarkgroup.illinois.edu/
#
# Copyright 2022-2023 The Board of Trustees of the University of Illinois.
# All Rights Reserved.
#
# Licensed under the terms MIT License
# The License is included in the distribution as LICENSE file.
# You may not use this file except in compliance with the License.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
# ================================================================================


"""
------------------------------------
 `MOLLI`: Molecular Library Toolbox
------------------------------------

`Molli`, molecular library toolbox, is a python 3.10+ library that supports
small to medium size full molecular structure manipulations, combinatorial molecular 
library generation, structure manipulations, and feature Extraction.
Molli offsers an efficient molecular and conformer library storage format.

It also implements a lot of command line tools (run `molli --HELP` or `molli list` for more details)

Copyright 2022-2023 The Board of Trustees of the University of Illinois. 
All Rights Reserved.

Developed by: 
    - Alexander S. Shved <shvedalx@illinois.edu>,
    - Blake E. Ocampo
    - Elena S. Burlova
    - Casey L. Olen
    - N. Ian Rinehart

S. E. Denmark Laboratory, University of Illinois, Urbana-Champaign
https://denmarkgroup.illinois.edu/

Licensed under the terms MIT License 
The License is included in the distribution as LICENSE file.
You may not use this file except in compliance with the License. 

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.

"""


from . import config

__version__ = config.VERSION

# Believe it or not, on Windows `aux` is not a valid file/folder name
from . import _aux as aux

try:
    import molli_xt as xt
except:
    MOLLI_USING_EXTENSIONS = False
else:
    MOLLI_USING_EXTENSIONS = True


from . import data
from . import math
from . import parsing
from . import storage

# from . import chem
from . import external
from . import files
from . import storage
from . import pipeline
from . import visual

from .chem import *
from .ftypes import *
from . import descriptor

from .reader import *
from .writer import *
