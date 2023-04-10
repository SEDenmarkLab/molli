# from . import backend_pyvista
from .primitives import color_cpk, wireframe, test_ballnstick
from typing import Type, Callable
from molli import Structure
from warnings import warn

try:
    import py3Dmol
except:
    warn(
        "`py3Dmol` must be installed for molecule visualization. Skipping dependent functions."
    )
else:

    def view(m: Structure):
        v = py3Dmol.view(width=1000, height=500)
        v.addModel(m.dumps_mol2(), "mol2")
        v.setStyle({"stick": {"radius": 0.1}, "sphere": {"scale": 0.15}})
        v.setHoverable(
            {},
            True,
            """
            function(atom,viewer,event,container) {
                if(!atom.label) {
                    atom.label = viewer.addLabel(atom.elem + atom.serial, {position: atom, backgroundColor: 'mintcream', fontColor:'black'});
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
        v.setBackgroundColor(None)
        v.show()

    Structure._repr_html_ = view
