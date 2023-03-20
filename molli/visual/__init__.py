# from . import backend_pyvista
from .primitives import color_cpk, wireframe, test_ballnstick
from typing import Type, Callable
from molli import Structure


def view(
    *objects, plotter=None, backend="pyvista", plotting_fx: dict[Type, Callable] = None
):
    ...


Structure._repr_html_ = test_ballnstick
