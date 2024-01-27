# ================================================================================
# This file is part of
#      -----------
#      MOLLI 1.0.0
#      -----------
# (C) 2021 Alexander S. Shved and the Denmark laboratory
# University of Illinois at Urbana-Champaign, Department of Chemistry
# ================================================================================


"""
Auxiliary elements that make the life of `molli`'s users easier.
"""
import colorama

from datetime import datetime, timedelta
from time import perf_counter_ns
from dataclasses import asdict, is_dataclass
import gc
from uuid import uuid1
from pathlib import Path
import types
import importlib.machinery as impm
import sys

# MOLLI_DEFAULT_DATA_LOCATION = Path("~/.molli").expanduser()


class ForeColor:
    COLORS = {
        "red": colorama.Fore.RED,
        "green": colorama.Fore.GREEN,
        "blue": colorama.Fore.BLUE,
        "ltred": colorama.Fore.LIGHTRED_EX,
        "ltblue": colorama.Fore.LIGHTBLUE_EX,
        "yellow": colorama.Fore.YELLOW,
        "magenta": colorama.Fore.MAGENTA,
        "default": colorama.Fore.WHITE,
    }

    def __init__(self, color: str = "default"):
        self.color = color.lower()

    def __enter__(self):
        c = self.__class__.COLORS[self.color]
        print(c, end="")
        # return self

    def __exit__(self, *args):
        print(colorama.Style.RESET_ALL, end="")


class timeit:
    """
    Easy to use performance indicator for code.

    example usage:
    ```python
    with timeit("my description", print_on_exit=True):
        # Code to be timed goes in here.
        ...
    ```
    """

    def __init__(self, desc: str = "", print_on_exit=True):
        self.desc = desc
        self.print_on_exit = print_on_exit
        self.state = "not started"

    @property
    def td_ns(self):
        if hasattr(self, "start"):
            if hasattr(self, "finish"):
                return self.finish - self.start
            else:
                return perf_counter_ns() - self.start
        else:
            raise AttributeError("This timer has not started yet")

    @property
    def td(self):
        return timedelta(microseconds=self.td_ns / 1000)

    def __enter__(self):
        self.start = perf_counter_ns()
        return self

    def __exit__(self, exc_type: type[Exception], *args):
        self.finish = perf_counter_ns()

        if exc_type is None:
            self.state = "done"
        else:
            self.state = f"error: {exc_type.__name__}"

        if self.print_on_exit:
            print(self)

    def __str__(self) -> str:
        return f"{self.desc}: {self.td} ({self.state})"


class gc_suspend:
    def __enter__(self):
        gc.disable()

    def __exit__(self, *args):
        gc.enable()


class catch_interrupt:
    def __enter__(self):
        return self

    def __exit__(self, exc, *args):
        if exc is None:
            pass
        elif exc == KeyboardInterrupt:
            with ForeColor("ltred"):
                print("Keyboard interrupt")
            exit(1)
        else:
            raise exc


def unique_path(_path: Path | str, *, idx_format: str = "0>3") -> Path:
    """ """

    proposed = Path(_path)

    while proposed.is_file():
        stem = proposed.stem
        stem1, *stem2 = stem.rsplit("___", maxsplit=1)
        try:
            idx = int(stem2[0])
        except:
            idx = 1
        else:
            idx += 1

        proposed = proposed.with_stem(f"{stem1}___{idx:{idx_format}}")

        # fn, first, *rest = name.split(".")
        # if len(first) == 32:
        #     # This is probably uuid then
        #     sfx = (fn, uuid1().hex, *rest)
        # else:
        #     sfx = (fn, uuid1().hex, first, *rest)

        # new_name = ".".join(sfx)
        # return proposed.with_name(new_name)

    return proposed


def load_external_module(fpath: Path | str, modname: str):
    """Loads external python code. Very convenient."""
    loader = impm.SourceFileLoader(modname, fpath)
    mod = types.ModuleType(loader.name)
    loader.exec_module(mod)
    return mod
