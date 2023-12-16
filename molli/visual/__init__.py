from typing import Literal
import molli as ml


def configure(
    backend: Literal["py3dmol", "pyvista"] = "py3dmol",
    theme: str = "dark",
    bgcolor: str = "black",
    height: int = 500,
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
            _py3dmol.HEIGHT = height

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
            pass

        case _:
            raise NotImplementedError(f"Backend {backend!r} is not supported.")
