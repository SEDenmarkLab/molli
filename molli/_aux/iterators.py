from glob import glob
from typing import Callable, Iterable, TypeVar, Generator
from pathlib import Path
from warnings import warn
import sys

T = TypeVar("T")


def sglob(
    pattern: str,
    loader: Callable[[str | Path], T],
    strict: bool = True,
) -> Generator[T, None, None]:
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
                raise xc
            else:
                warn(
                    f"An error occurred while loading {fn}: {xc}. Skipping the"
                    " file..."
                )


def dglob(
    pattern: str,
    loader: Callable[[str | Path], Iterable[T]],
    strict: bool = True,
) -> Generator[T, None, None]:
    """# `dglob`
    This function provides a simple way of deep globbing over files while simultaneously reading them as molli objects.

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
                raise xc
            else:
                warn(
                    f"An error occurred while loading {fn}: {xc}. Skipping the"
                    " file..."
                )


if sys.version_info >= (3, 12):
    from itertools import batched
else:
    try:
        from more_itertools import batched
    except:

        def batched(iterable, n):
            """This is from python documentation website.
            This version is a fallback in case `more_itertools` is not installed
            and python is not >= 3.12
            """
            from itertools import islice

            if n < 1:
                raise ValueError("n must be at least one")
            it = iter(iterable)
            while batch := tuple(islice(it, n)):
                yield batch


def len_batched(iterable: Iterable | int, n: int) -> int:
    L = iterable if isinstance(iterable, int) else len(iterable)
    return L // n + (1 if L % n else 0)
