from typing import Literal
import molli as ml


def configure(
    backend: Literal["py3dmol", "pyvista"] = "py3dmol",
    theme: str = "dark",
    bgcolor: str = "black",
    width: int | str = None,
    height: int = None,
    style: Literal["ballnstick", "wireframe"] = "ballnstick",
    style_conf: Literal["ballnstick", "wireframe"] = "wireframe",
    **kwargs,
):
    """
    This configures the *default* visualization protocol in Jupyter notebooks.
    Note that individual functions from the corresponding submodules can still be used.
    """

    match backend:
        case "py3dmol":
            from . import _py3dmol

            _py3dmol.BGCOLOR = bgcolor

            if height is not None:
                _py3dmol.HEIGHT = height

            if width is not None:
                _py3dmol.WIDTH = width

            match style:
                case "ballnstick":
                    _py3dmol.STYLE = _py3dmol.STYLE_BALLNSTICK
                case "wireframe":
                    _py3dmol.STYLE = _py3dmol.STYLE_WIREFRAME

            if style and not style_conf:
                style_conf = style

            match style_conf:
                case "ballnstick":
                    _py3dmol.STYLE_CONF = _py3dmol.STYLE_BALLNSTICK
                case "wireframe":
                    _py3dmol.STYLE_CONF = _py3dmol.STYLE_WIREFRAME

            ml.Structure._repr_html_ = _py3dmol.view_structure
            ml.ConformerEnsemble._repr_html_ = _py3dmol.view_ensemble

        case "pyvista":
            import pyvista as pv
            from . import _pyvista

            _pyvista.BGCOLOR = bgcolor
            _pyvista.THEME = theme
            _pyvista.WIDTH = width
            _pyvista.HEIGHT = height

            pv.set_plot_theme(theme)

            ml.Structure._repr_html_ = _pyvista.draw_ballnstick
            ml.ConformerEnsemble._repr_html_ = _pyvista.draw_wireframe

        case _:
            raise NotImplementedError(f"Backend {backend!r} is not supported.")
