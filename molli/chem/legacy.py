# ================================================================================
# This file is part of `molli 1.0`
# (https://github.com/SEDenmarkLab/molli)
#
# Developed by Alexander S. Shved <shvedalx@illinois.edu>
#
# S. E. Denmark Laboratory, University of Illinois, Urbana-Champaign
# https://denmarkgroup.illinois.edu/
#
# Copyright 2022-2023 The Board of Trustees of the University of Illinois.
# All Rights Reserved.
#
# Licensed under the terms MIT License
# The License is included in the distribution as LICENSE file.
# You may not use this file except in compliance with the License.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
# ================================================================================


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


def read_geom(k, g, ens):
    m = re.match(r"#(?P<L>[0-9]+),(?P<D>[0-9]+):(?P<G>.+);", g.text)
    L = int(m.group("L"))
    D = int(m.group("D"))
    G = m.group("G")

    assert D == 3, "Only 3d coordinates supported for now"

    coord = []
    for a, xyz in enumerate(G.split(";")):
        x, y, z = map(float, xyz.split(","))
        if isinstance(ens, ConformerEnsemble):
            ens._coords[(k, a)] = (x, y, z)
        elif isinstance(ens, Molecule):
            ens._coords[a] = (x, y, z)
    return ens


def ensemble_from_molli_old_xml(
    f: StringIO | BytesIO, mol_lib=False
) -> ConformerEnsemble | Molecule:
    """Parses an old version of the collection.
    This function is primarily intended for backwards compatibility
    reasons with the old molli version. It is best to use the
    `recollect` command rather than directly interact with this function

    Parameters
    ----------
    f : StringIO | BytesIO
        xml file stream
    mol_lib : bool, optional
        Returns `ConformerEnsemble` if True, by default False

    Returns
    -------
    ConformerEnsemble | Molecule
        Returns Conformer Ensemble or Molecule

    Notes
    -----
    If no conformer geometries are given, default geometry will be imported
    as the 0th conformer.
    """

    tree = cET.parse(f)
    mol = tree.getroot()
    name = mol.attrib["name"]

    xatoms = mol.findall("./atoms/a")
    xbonds = mol.findall("./bonds/b")
    xgeom = mol.findall("./geometry/g")
    xconfs = mol.findall("./conformers/g")

    atoms = []
    bonds = []
    conformers = []

    n_atoms = len(xatoms)

    if len(xconfs) == 0:
        n_conformers = len(xgeom)
    else:
        n_conformers = len(xconfs)

    if mol_lib:
        ens = Molecule(n_atoms=n_atoms, name=name)
    else:
        ens = ConformerEnsemble(n_conformers=n_conformers, n_atoms=n_atoms, name=name)

    for i, a in enumerate(xatoms):
        aid, s, l, at = a.attrib["id"], a.attrib["s"], a.attrib["l"], a.attrib["t"]
        ens.atoms[i].element = Element[s]
        ens.atoms[i].label = l
        ens.atoms[i].set_mol2_type(at)

    for j, b in enumerate(xbonds):
        ia1, ia2 = map(int, b.attrib["c"].split())
        ens.append_bond(_b := Bond(ens.atoms[ia1 - 1], ens.atoms[ia2 - 1]))
        _b.set_mol2_type(b.attrib["t"])

    if len(xconfs) == 0:
        for k, g in enumerate(xgeom):
            ens = read_geom(k, g, ens)
    else:
        for k, g in enumerate(xconfs):
            ens = read_geom(k, g, ens)

    return ens
