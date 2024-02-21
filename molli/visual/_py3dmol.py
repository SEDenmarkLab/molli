import molli as ml
from io import StringIO
import py3Dmol
import shlex

BGCOLOR = "black"
WIDTH = "100%"
HEIGHT = 500
STYLE_BALLNSTICK = {"stick": {"radius": 0.1}, "sphere": {"scale": 0.15}}
STYLE_WIREFRAME = {"line": {}}
STYLE = STYLE_BALLNSTICK
STYLE_CONF = STYLE_BALLNSTICK


def _dump_mol2(s: ml.Structure | ml.ConformerEnsemble, _stream: StringIO = None):
    """
    This version is specifically designed to produce the smallest files possible.
    It is also adapting the output to what 3DMol.js expects
    """
    if _stream is None:
        stream = StringIO()
    else:
        stream = _stream

    if not isinstance(s, ml.ConformerEnsemble):
        stream.write(
            f"@<TRIPOS>MOLECULE\n{s.name}\n{s.n_atoms} {s.n_bonds} 0 0"
            " 0\nSMALL\nNO_CHARGES\n\n"
        )

        stream.write("@<TRIPOS>ATOM\n")
        for i, a in enumerate(s.atoms):
            x, y, z = s.coords[i]
            elt = a.element.symbol
            lbl = a.label or "None"
            stream.write(f"{i+1} {lbl} {x:.4f} {y:.4f} {z:.4f} {elt}\n")

        stream.write("@<TRIPOS>BOND\n")
        for i, b in enumerate(s.bonds):
            a1, a2 = s.atoms.index(b.a1), s.atoms.index(b.a2)
            btype = b.get_mol2_type()
            stream.write(f"{i+1} {a1+1} {a2+1} {btype}\n")

        stream.write("\n")

    else:
        for c in s:
            _dump_mol2(c, stream)

    if _stream is None:
        return stream.getvalue()


def _dump_xyz(s: ml.Structure | ml.ConformerEnsemble, _stream: StringIO = None):
    """
    This version is specifically designed to produce the smallest files possible.
    It is also adapting the output to what 3DMol.js expects
    """
    if _stream is None:
        stream = StringIO()
    else:
        stream = _stream

    if not isinstance(s, ml.ConformerEnsemble):
        stream.write(f"{s.n_atoms}\n\n")

        for i, a in enumerate(s.atoms):
            x, y, z = s.coords[i]
            elt = a.element.symbol
            stream.write(f"{elt} {x:.4f} {y:.4f} {z:.4f}\n")

    else:
        for c in s:
            _dump_xyz(c, stream)

    if _stream is None:
        return stream.getvalue()


def view_structure(
    m: ml.Structure,
    bgcolor: str = None,
    height: int = None,
    style: dict = None,
    view=None,
):
    bgcolor = bgcolor or BGCOLOR
    height = height or HEIGHT
    style = style or STYLE

    if view is None:
        v = py3Dmol.view(width="100%", height=height)
    else:
        v = view
    v.addModel(_dump_mol2(m), "mol2")

    if style is not None:
        v.setStyle(style)

    v.setHoverable(
        {},
        True,
        """
        function(atom,viewer,event,container) {
            if(!atom.label) {
                atom.label = viewer.addLabel(atom.elem + '(' + atom.index + ')', {position: atom, backgroundColor: 'silver', fontColor:'black'});
            }
        }
        """,
        """
        function(atom,viewer) { 
            if(atom.label) {
                viewer.removeLabel(atom.label);
                delete atom.label;
            }
        }
        """,
    )
    v.zoomTo()
    v.setBackgroundColor(bgcolor)
    v.show()
    v.update()


def view_ensemble(
    ens: ml.ConformerEnsemble,
    bgcolor: str = None,
    height: int = None,
    style: dict = None,
    style_conf: dict = None,
    view=None,
):
    bgcolor = bgcolor or BGCOLOR
    height = height or HEIGHT
    style = style or STYLE
    style_conf = style_conf or STYLE_CONF

    if view is None:
        v = py3Dmol.view(width="100%", height=500)
    else:
        v = view

    v.setBackgroundColor(bgcolor)

    for i, c in enumerate(ens):
        v.addModel(_dump_mol2(c), "mol2", {"style": style_conf if i else style})

    # v.animate({"loop": "forward"})

    v.zoomTo()
    v.show()
    v.update()


def view_ensemble_animated(
    ens: ml.ConformerEnsemble,
    bgcolor: str = None,
    height: int = None,
    style: dict = None,
    view=None,
):
    pass


from IPython.core.magic import register_line_magic


@register_line_magic
def mlib_view(line: str):
    """This line magic allows to view mlib with a simple syntax"""
    mlib_path, key, *rest = shlex.split(line)
    v = py3Dmol.view(width="100%", height=HEIGHT)

    mlib = ml.MoleculeLibrary(mlib_path, readonly=True)
    with mlib.reading():
        view_structure(mlib[key], view=v)

        for k in rest:
            v.addModel(
                _dump_mol2(mlib[k]),
                "mol2",
            )

    v.setStyle(STYLE)
    v.update()


@register_line_magic
def clib_view(line: str):
    """This line magic allows to view mlib with a simple syntax"""
    clib_path, key = shlex.split(line)
    v = py3Dmol.view(width="100%", height=HEIGHT)

    mlib = ml.ConformerLibrary(clib_path, readonly=True)
    with mlib.reading():
        view_ensemble(mlib[key], view=v)

    v.update()
