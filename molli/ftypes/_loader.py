# ================================================================================
# This file is part of
#      -----------
#      MOLLI 1.0.0
#      -----------
# (C) 2023 Alexander S. Shved and the Denmark laboratory
# University of Illinois at Urbana-Champaign, Department of Chemistry
# ================================================================================


"""
This file contains definitions of universal loader system
"""
from . import Promolecule
from pathlib import Path
from typing import Callable, T


def loads(
    cls: type[Promolecule],
    path: str | Path,
    *,
    ftype: str = None,
    loader: Callable = None,
    name: str = None
):
    pass
