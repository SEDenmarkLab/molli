from glob import glob
from typing import Callable, Iterable
from pathlib import Path
from warnings import warn
from . import Molecule, ConformerEnsemble, StructureLike


def sglob(pattern: str, loader: Callable[[str | Path], StructureLike], strict: bool = True):
    """# `sglob`
    This function provides a simple way of shallow globbing over files while simultaneously reading them as molli objects.
    
    ## Example

    ```python
    for mol in mglob("*.mol2", Molecule.load_mol2, strict=False):
        do_smth_with_molecule(mol)
    ```

    ## Parameters

    `pattern: str`
        See the definition of glob function
        Statement with simple wildcards that can be expanded to a list of files
    `loader: Callable[[str  |  Path], StructureLike]`
        A function which is then called on the file name to convert it to a molli object
    `strict: bool`, optional, default: `True`
        If strict, any errors in loading files are converted to exceptions.
        If not, a warning is issued, but the iteration is allowed to proceed.

    ## Yields

    `_type_`
        _description_

    ## Raises

    `xc.with_traceback`
        _description_
    """    
    all_files = glob(pattern)

    for fn in all_files:
        try:
            yield loader(fn)
        except Exception as xc:
            if strict:
                raise xc.with_traceback()
            else:
                warn(
                    f"An error occurred while loading {fn}: {xc}. Skipping the file..."
                )


def dglob(pattern: str, loader: Callable[[str | Path], Iterable[StructureLike]], strict: bool = True):
    """# `dglob`
    This function provides a simple way of shallow globbing over files while simultaneously reading them as molli objects.
    
    ## Example

    ```python
    for mol in mglob("*.mol2", Molecule.load_mol2, strict=False):
        do_smth_with_molecule(mol)
    ```

    ## Parameters

    `pattern: str`
        See the definition of glob function
        Statement with simple wildcards that can be expanded to a list of files
    `loader: Callable[[str  |  Path], StructureLike]`
        A function which is then called on the file name to convert it to a molli object
    `strict: bool`, optional, default: `True`
        If strict, any errors in loading files are converted to exceptions.
        If not, a warning is issued, but the iteration is allowed to proceed.

    ## Yields

    `_type_`
        _description_

    ## Raises

    `xc.with_traceback`
        _description_
    """    
    all_files = glob(pattern)

    for fn in all_files:
        try:
            yield from loader(fn)
        except Exception as xc:
            if strict:
                raise xc.with_traceback()
            else:
                warn(
                    f"An error occurred while loading {fn}: {xc}. Skipping the file..."
                )