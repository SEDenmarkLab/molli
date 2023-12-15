import molli as ml

import py3Dmol

BGCOLOR = "black"
WIDTH = 1000
HEIGHT = 500
STYLE_BALLNSTICK = {"stick": {"radius": 0.1}, "sphere": {"scale": 0.15}}
STYLE_WIREFRAME = {}


def view_structure(m: ml.Structure, view=None):
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
    v.setBackgroundColor(BGCOLOR)
    v.show()


def view_ensemble_animated(ens: ml.ConformerEnsemble, view=None):
    pass


def view_ensemble(ens: ml.ConformerEnsemble, view=None):
    pass
