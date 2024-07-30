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
This file provides the necessary functionality to parse a CDXML file with stereochemical hinting
"""

from xml.etree import cElementTree as et
from attrs import define, field
from pathlib import Path
from typing import List, Dict
from itertools import chain
import numpy as np
from warnings import warn
from ..chem import *
from ..math import rotate_2dvec_outa_plane, mean_plane
from math import radians
from scipy.spatial import KDTree


def position(elt: et.Element):
    if "BoundingBox" in elt.attrib:
        l, t, r, b = map(float, elt.attrib["BoundingBox"].split())
        return (l + r) / 2, (t + b) / 2
    elif "p" in elt.attrib:
        l, t = map(float, elt.attrib["p"].split())
        return l, t
    else:
        raise ValueError(f"Element {elt} does not have positional attributes")


class CDXMLSyntaxWarning(SyntaxWarning):
    pass


def _cdxml_3dify_(s: StructureLike, _a1: AtomLike, _a2: AtomLike, *, sign=1):
    """
    Bend the coordinates:
        sign: tuple[int, int]
        where sign of int signifies the direction of out-of-plane bending
        sign[1] > sign[0]
    """
    assert abs(sign) <= 2, "Sign can only be +/-1"
    b = s.lookup_bond(_a1, _a2)
    a1, a2 = s.get_atoms(_a1, _a2)
    i1, i2 = s.get_atom_indices(a1, a2)

    if b is None:
        raise CDXMLSyntaxWarning(f"{a1} and {a2} are not bonded. Aborting")
    if abs(sign) == 1:
        if s.is_bond_in_ring(b):
            # Fun stuff happens here
            # this is under a few assumptions
            s.substructure((a1, a2)).coords += sign * np.array(
                [[0.0, 0.5, 0.75], [0.0, 0.5, 1.5]]
            )

            for _b in s.bonds_with_atom(a1):
                if not s.is_bond_in_ring(_b):
                    s.substructure(s.yield_bfs(a1, _b % a1)).translate(
                        sign * np.array([0.0, 0.5, 0.75])
                    )

            for _b in s.bonds_with_atom(a2):
                if not s.is_bond_in_ring(_b):
                    s.substructure(s.yield_bfs(a2, _b % a2)).translate(
                        sign * np.array([0.0, 0.5, 1.5])
                    )

        else:
            # We need to define the rotation matrix first.
            normal = mean_plane(s.coord_subset(s.connected_atoms(a1)))
            angle = sign * (
                radians(+90) if s.n_bonds_with_atom(a1) == 4 else radians(+60)
            )
            rotation = rotate_2dvec_outa_plane(s.vector(i1, i2), angle, normal)
            substruct = s.substructure(s.yield_bfs(a1, a2))
            v = s.get_atom_coord(a1)
            substruct.translate(-v)
            substruct.transform(rotation)
            substruct.translate(v)

    elif abs(sign) == 2:
        s.substructure((a1, a2)).coords += (sign / 2) * np.array([0.0, 0.0, 1.0])

        for _b in s.bonds_with_atom(a1):
            if not s.is_bond_in_ring(_b):
                s.substructure(s.yield_bfs(a1, _b % a1)).translate(
                    (sign / 2) * np.array([0.0, 0.0, 1.0])
                )

        for _b in s.bonds_with_atom(a2):
            if not s.is_bond_in_ring(_b):
                s.substructure(s.yield_bfs(a2, _b % a2)).translate(
                    (sign / 2) * np.array([0.0, 0.0, 1.0])
                )


def validate_label(lbl: et.Element):
    """
    This validator is used to filter out all labels that are *not* intended as the compound labels
    Valid label:
        - only has one text child element
        - that element font "face" attribute must be set to 1 (CDXML lingo: bold face)
    """
    _sub = lbl.findall("./s")
    return len(_sub) == 1 and _sub[0].attrib.get("face", "0") == "1"


def validate_fragment(frag: et.Element):
    """
    This is validator that removes invalid CDXML fragments
    right now it is fairly crude (at least one bond must be present)
    """
    return any(x.tag == "b" for x in frag)


@define(repr=True)
class CDXMLFile:
    path: str | Path = field(repr=True)
    tree: et.ElementTree = field(repr=False, init=False, kw_only=True)
    bond_length: float = field(default=1.0, repr=False, init=False, converter=float)
    xlabels: Dict[str, et.Element] = field(factory=dict, repr=False, init=False)
    xfrags: List[et.Element] = field(factory=list, repr=False, init=False)
    xfrag_cache: Dict[str, et.Element] = field(factory=dict, repr=False, init=False)
    xfrag_kd: KDTree = field(init=False, repr=False)

    def __attrs_post_init__(self):
        # TODO: generalize to text
        self.tree = et.parse(self.path)
        self.bond_length = self.tree.getroot().attrib["BondLength"]

        # this finds all textboxes that fit under the definition
        # of face=1 (bold face, not chemically interpreted)
        for xt in filter(
            validate_label,
            self.tree.findall("./page/t") + self.tree.findall("./page/group/t"),
        ):
            if (lbl := xt[0].text) in self.xlabels:
                warn(
                    (
                        f"CDXML file {self.path} contains redundant label {lbl!r} "
                        "Only the first occurrence will be kept."
                    ),
                    CDXMLSyntaxWarning,
                )
            else:
                self.xlabels[lbl] = xt

        # this lists all fragments that are validated
        # (can have chemical interpretation)
        # note: this avoids placing items that match ".//fragment/fragment" in this list.
        # those are unexpanded substituents!
        self.xfrags.extend(
            filter(
                validate_fragment,
                self.tree.findall("./page/fragment")
                + self.tree.findall("./page/group/fragment"),
            )
        )

        if len(self.xfrags) != len(self.xlabels):
            warn(
                (
                    f"CDXML file {self.path} contains mismatched number of labels"
                    f" ({len(self.xfrags)}) and fragments ({len(self.xlabels)}). Please make sure"
                    " this is intentional."
                ),
                CDXMLSyntaxWarning,
            )

        # KDTree makes locating geometrically close things significantly eaiser
        xfrag_coords = np.array(list(map(position, self.xfrags)))
        self.xfrag_kd = KDTree(xfrag_coords, balanced_tree=True)  # optional args?

    def keys(self):
        return self.xlabels.keys()

    def __len__(self):
        return len(self.keys())

    def __getitem__(self, key: str) -> Molecule:
        # TEMPORARY. DO A BETTER REWRITE.
        if isinstance(key, int):
            key = list(self.keys())[key]

        if key in self.xfrag_cache:
            frag = self.xfrag_cache[key]
        elif (grpf := self.xlabels[key].find("../fragment")) in self.xfrags:
            frag = grpf
            self.xfrag_cache[key] = frag
        else:
            lpos = position(self.xlabels[key])
            frag = None
            _, indices = self.xfrag_kd.query(lpos, k=5, p=1)
            for i in indices:
                fpos = position(self.xfrags[i])
                if (
                    fpos[1] < lpos[1]
                ):  # if the label is below the fragment, which is what we want
                    frag = self.xfrags[i]
                    break

            if frag is None:  # If nothing was found
                raise KeyError(f"Could not locate a viable candidate for label {key}")
            else:
                self.xfrag_cache[key] = frag

        return self._parse_fragment(frag, name=key)

    def _parse_atom_node(self, node: et.Element) -> Atom:
        """
        Returns Atom and atom id as in cdxml
        """
        # Element
        elt = node.get("Element")
        elt = int(elt) if elt else "C"

        # Label
        # Note: there is a possibility that the underlying text may change.
        # Need to investigate!
        lbl = node.get("AtomNumber")

        # Isotope
        isot = node.get("Isotope")
        isot = None if isot is None else int(isot)

        # Formal charge
        charge = int(node.get("Charge", 0))

        # Formal spin
        #   note: molli atomic attribute of spin is the
        spin2 = node.get("Radical")

        # Potentially populate with other things from the cdxml file?
        attrib = {}
        match spin2:
            case "Doublet":
                spin2 = 1

            case "Singlet":
                spin2 = 2

            case _:
                spin2 = 0

        match node.get("NodeType"):
            case "ExternalConnectionPoint":
                elt = Element.Unknown
                atyp = AtomType.AttachmentPoint
                if apn := node.get("ExternalConnectionNum"):
                    lbl = lbl or "AP" + apn
                else:
                    # NOTE: this is for the cases for when **chemdraw is true to itself in being inconsistent**
                    lbl = lbl or "AP0"
            case (
                "Fragment" | "Nickname"
            ):  # The latter is just a monkey patch. May break.
                elt = Element.Unknown
                atyp = AtomType.AttachmentPoint
                lbl = node.get("id")
            case "GenericNickname":
                elt = Element.Unknown
                atyp = AtomType.AttachmentPoint
                lbl = node.get("GenericNickname")

            case "Unspecified":
                elt = Element.Unknown
                atyp = AtomType.AttachmentPoint
                lbl = node.find("./t/s").text

            case _:
                atyp = AtomType.Regular
                # Potentially add because Casey requests.
                # Or at least make it consistent.
                # lbl = node.get("Element") or "C"

        # Number of hydrogens
        if numhs := node.get("NumHydrogens", 0):
            attrib |= {"__implicit_hydrogens": int(numhs)}

        return Atom(
            elt,
            isotope=isot,
            label=lbl,
            atype=atyp,
            formal_charge=charge,
            formal_spin=spin2,
            attrib=attrib,
        )

    def _parse_bond(self, bd: et.Element, atom_idx: dict[str, Atom]) -> Bond:
        a1 = atom_idx[bd.get("B")]
        a2 = atom_idx[bd.get("E")]
        _order = bd.get("Order")

        # Add handler of stereochemistry
        match _order:
            case None:
                btype = BondType.Single

            case "1.5":
                btype = (
                    BondType.Aromatic
                )  # Unclear if this is correct, but for now let it be

            case _:
                btype = BondType(int(_order))

        if bd.get("Display") == "Dash":
            btype = BondType.Ligand

        return Bond(a1, a2, btype=btype)

    def _parse_fragment(self, frag: et.Element, name: str = None) -> Structure:
        atoms = []
        bonds = []
        coords = []
        atom_idx = {}
        bond_idx = {}
        centers = []
        multiattachments = {}

        try:
            # iterate over all nodes
            for node in frag.findall("./n"):
                # This is to handle the case of multi-attachments
                # like in the case of Cp-ligands
                if node.get("NodeType") == "MultiAttachment":
                    multiattachments[node.get("id")] = node.get("Attachments").split()
                    continue

                atom = self._parse_atom_node(node)
                if atom is not None:
                    atoms.append(atom)
                    atom_idx[node.get("id")] = atom
                    coords.append(position(node))

            result = Molecule(atoms, name=name, copy_atoms=False)
            result.coords[:, 2] = 0

            for bd in frag.findall("./b"):
                B, E = map(bd.get, "BE")
                center = None

                if B in multiattachments:
                    center = E
                    attached = multiattachments[B]
                elif E in multiattachments:
                    center = B
                    attached = multiattachments[E]

                if center is not None:
                    f_order = 1 / len(attached)
                    centers.append(center)
                    for t in attached:
                        result.connect(
                            atom_idx[center],
                            atom_idx[t],
                            btype=BondType.Ligand,
                            # btype=BondType.Dummy,
                            f_order=f_order,
                        )
                else:
                    b = self._parse_bond(bd, atom_idx)
                    result.bonds.append(b)
                    bond_idx[bd.get("id")] = b

            result.coords[:, :2] = coords

            # This is to bring the bond length to 1.5A. Not ideal, but should work in most cases.
            result.scale(1.5 / self.bond_length)

            # in chemdraw window coordinates Y-axis points down
            # (opposite to real life)
            # this line inverts this.
            result.coords *= [1, -1, 1]

            # Not that it's necessary, but the structure coordinates are centered here
            result.translate(-result.centroid())

            for bd in frag.findall("./b[@Display]"):
                if (bdid := bd.get("id")) not in bond_idx:
                    continue

                b = bond_idx[bdid]
                i1, i2 = result.get_atom_indices(b.a1, b.a2)

                match bd.get("Display"):
                    case "WedgeBegin":
                        # result.coords[i2][2] += result.coords[i1][2] + 1.1
                        _cdxml_3dify_(result, i1, i2, sign=+1)
                    case "WedgedHashBegin":
                        _cdxml_3dify_(result, i1, i2, sign=-1)
                    case "WedgeEnd":
                        _cdxml_3dify_(result, i2, i1, sign=+1)
                    case "WedgedHashEnd":
                        _cdxml_3dify_(result, i2, i1, sign=-1)
                    case "Bold":
                        _cdxml_3dify_(result, i1, i2, sign=+2)
                    case "Hash":
                        _cdxml_3dify_(result, i1, i2, sign=-2)
                    case _:
                        pass

            # This translates the central atom to the average of z coords
            for center in centers:
                a = atom_idx[center]
                aid = result.index_atom(a)
                a.atype = AtomType.CoordinationCenter
                connected = list(result.connected_atoms(a))
                connected_coords = result.coord_subset(connected)
                avg_connected_z = (
                    np.max(connected_coords[:, 2]) - np.min(connected_coords[:, 2])
                ) / 2

                result.coords[aid, 2] = avg_connected_z

            # this handles composite structures
            for subfrag in frag.findall("./n/[fragment]"):
                substruct = self._parse_fragment(subfrag.find("./fragment"))
                ap = result.get_atom(subfrag.get("id"))
                result = Molecule.join(
                    result,
                    substruct,
                    ap,
                    substruct.attachment_points[0],
                    optimize_rotation=(substruct.n_atoms > 1),
                    name=name,
                )
        except Exception as xc:
            raise SyntaxError(
                f"Invalid syntax encountered in fragment id=\"{frag.get('id')}\""
            ) from xc

        total_charge = sum(a.formal_charge or 0 for a in result.atoms)
        total_spin2 = sum(a.formal_spin or 0 for a in result.atoms)

        result.charge = total_charge
        result.mult = total_spin2 + 1

        return result
