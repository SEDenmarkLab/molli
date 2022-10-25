"""

# `molli.parsing.legacy` module

Left for backwards compatibility reasons.

"""

# from xml.dom import minidom as xmd
import xml.etree.cElementTree as cET
import re
from warnings import warn
from io import StringIO, BytesIO
from . import ConformerEnsemble, Molecule, Atom, Bond, Element


def ensemble_from_molli_old_xml(f: StringIO | BytesIO) -> ConformerEnsemble:
    # This auxiliary function parses an old molli collection
    """

    # `read_molli_old_xml`
    parse an old version of the collection

    This function is primarily intented for backwards compatibility reasons with the old molli version

    ## Parameters

    `f : StringIO`
        xml file stream

    ## Returns

    `ConformerEnsemble`
        Ensemble of conformers as written in the xml file.
        Note: if no conformer geometries are given, default geometry will be imported as 0th conformer.
    """
    tree = cET.parse(f)
    mol = tree.getroot()
    name = mol.attrib["name"]

    xatoms = mol.findall("./atoms/a")
    xbonds = mol.findall("./bonds/b")
    xgeom = mol.find("./geometry/g")
    xconfs = mol.findall("./conformers/g")

    atoms = []
    bonds = []
    conformers = []

    n_atoms = len(xatoms)
    n_conformers = len(xconfs)

    ens = ConformerEnsemble(n_conformers=len(xconfs), n_atoms=len(xatoms), name=name)

    for i, a in enumerate(xatoms):
        aid, s, l, at = a.attrib["id"], a.attrib["s"], a.attrib["l"], a.attrib["t"]
        ens.atoms[i].element = Element[s]
        ens.atoms[i].label = l

    for j, b in enumerate(xbonds):
        ia1, ia2 = map(int, b.attrib["c"].split())
        ord, ar = (
            (1.5, True) if b.attrib["t"] == "ar" else (float(b.attrib["t"]), False)
        )
        ens.append_bond(Bond(ens.atoms[ia1 - 1], ens.atoms[ia2 - 1], ord, aromatic=ar))

    for k, g in enumerate(xconfs):
        m = re.match(r"#(?P<L>[0-9]+),(?P<D>[0-9]+):(?P<G>.+);", g.text)

        L = int(m.group("L"))
        D = int(m.group("D"))
        G = m.group("G")

        assert D == 3, "Only 3d coordinates supported for now"

        coord = []

        for a, xyz in enumerate(G.split(";")):
            x, y, z = map(float, xyz.split(","))

            ens._coords[(k, a)] = (x, y, z)

    return ens
