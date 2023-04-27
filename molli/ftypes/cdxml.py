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


@define(repr=True)
class CDXMLFile:
    path: str | Path = field(repr=True)
    tree: et.ElementTree = field(repr=False, init=False, kw_only=True)
    bond_length: float = field(default=1.0, repr=False, init=False, converter=float)
    labels: Dict[str, et.Element] = field(factory=dict, repr=False, init=False)
    fragments: List[et.Element] = field(factory=list, repr=False, init=False)
    fragment_coords: np.ndarray = field(default=None, repr=False, init=False)

    def __attrs_post_init__(self):
        self.tree = et.parse(self.path)
        self.bond_length = self.tree.getroot().attrib["BondLength"]

        # this finds all textboxes that fit under the definition
        for t in self.tree.findall(".//t/s[@face='1']..."):
            if (lbl := t[0].text) in self.labels:
                warn(
                    f"CDXML file {self.path} contains redundant label {lbl!r} "
                    "Only the first occurrence will be kept.",
                    CDXMLSyntaxWarning,
                )
            else:
                self.labels[lbl] = t

        self.fragments.extend(self.tree.findall("./page/fragment"))
        self.fragments.extend(self.tree.findall("./page/group/fragment"))
        # note: this avoids placing items that match ".//fragment/fragment" in this list.
        # this is unwanted: those are unexpanded substituents!

        self.fragment_coords = np.array(list(map(position, self.fragments)))

        if len(self.fragments) != len(self.labels):
            warn(
                f"CDXML file {self.path} contains mismatched number of labels ({len(self.fragments)}) and fragments ({len(self.labels)}). "
                "Please make sure this is intentional.",
                CDXMLSyntaxWarning,
            )

    def keys(self):
        return self.labels.keys()

    def __getitem__(self, key: str) -> Molecule:
        if (grpf := self.labels[key].find("../fragment")) in self.fragments:
            frag = grpf
        else:
            frag = self.fragments[
                np.argmin(
                    np.sum(
                        np.abs(self.fragment_coords - position(self.labels[key])),
                        axis=1,
                    )
                )
            ]

        return self._parse_fragment(frag, Molecule, name=key)

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

        match node.get("NodeType"):
            case "ExternalConnectionPoint":
                elt = Element.Unknown
                atyp = AtomType.AttachmentPoint
                if apn := node.get("ExternalConnectionNum"):
                    lbl = "AP" + apn
                else:
                    # NOTE: this is for the cases for when **chemdraw is true to itself in being inconsistent**
                    lbl = "AP0"
            case "Fragment" | "Nickname":  # The latter is just a monkey patch. May break.
                elt = Element.Unknown
                atyp = AtomType.AttachmentPoint
                lbl = node.get("id")
            case _:
                atyp = AtomType.Regular
                # Potentially add because Casey requests.
                # Or at least make it consistent.
                # lbl = node.get("Element") or "C"

        return Atom(elt, isotope=isot, label=lbl, atype=atyp)

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

        return Bond(a1, a2, btype=btype)

    def _parse_fragment(
        self, frag: et.Element, cls: type[Structure], name: str = None
    ) -> Structure:
        atoms = []
        bonds = []
        coords = []
        atom_idx = {}
        bond_idx = {}

        # iterate over all nodes
        for node in frag.findall("./n"):
            atom = self._parse_atom_node(node)
            atoms.append(atom)
            atom_idx[node.get("id")] = atom
            coords.append(position(node))

        result = cls(atoms, name=name, copy_atoms=False)
        result.coords[:, 2] = 0

        for bd in frag.findall("./b"):
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
            b = bond_idx[bd.get("id")]
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
                case _:
                    pass

        # this handles composite structures
        for subfrag in frag.findall("./n/[fragment]"):
            substruct = self._parse_fragment(subfrag.find("./fragment"), Structure)
            ap = result.get_atom(subfrag.get("id"))
            result = cls.join(
                result,
                substruct,
                ap,
                substruct.attachment_points[0],
                optimize_rotation=(substruct.n_atoms > 1),
                name=name,
            )

        return result
