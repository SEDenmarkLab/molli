# ================================================================================
# This file is part of
#      -----------
#      MOLLI 1.0.0
#      -----------
# (C) 2022 Alexander S. Shved and the Denmark laboratory
# University of Illinois at Urbana-Champaign, Department of Chemistry
# ================================================================================
"""
---------------
# `MOLLI 1.0.0a7`
---------------
(C) 2022 Alexander S. Shved and the Denmark laboratory  
University of Illinois at Urbana-Champaign, Department of Chemistry

Molli, molecular library toolbox, is a library that supports
small to medium size full molecular structure manipulations.

It also implements a lot of command line tools (run `molli --HELP` or `molli list` for more details)

"""

__version__ = "1.0.0a7"

# Determine whether molli C++ extensions are available
# if not, pure python analogs should be provided

try:
    import molli_xt as xt
except:
    MOLLI_USING_EXTENSIONS = False
else:
    MOLLI_USING_EXTENSIONS = True

from . import config

# Believe it or not, on Windows `aux` is not a valid file/folder name
from . import _aux as aux

from . import data
from . import math
from . import parsing
from . import storage

# from . import chem
from . import descriptor
from . import external
from . import files
from . import storage

from .chem import *
from .ftypes import *
