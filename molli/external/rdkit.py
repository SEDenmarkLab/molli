# ================================================================================
# This file is part of `molli 1.0`
# (https://github.com/SEDenmarkLab/molli)
#
# Developed by Blake E. Ocampo
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
# `molli.external.rdkit`
This module defines necessary functions (albeit not a complete set) to interface with RDKit.
"""

from ..chem import Molecule
from typing import Dict

import numpy as np
import importlib.util


def is_package_installed(pkg_name):
    return importlib.util.find_spec(pkg_name) is not None

if not is_package_installed("rdkit"):
    raise ImportError("RDKit is not installed in this environment")

if not is_package_installed("IPython"):
    raise ImportError("IPython is not installed in this environment")

import molli as ml
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.PropertyMol import PropertyMol
from rdkit.Chem import rdqueries as chemq
from rdkit import DataStructs
from rdkit.Chem import rdCIPLabeler
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Draw import IPythonConsole

from openbabel.pybel import readstring
from rdkit.Chem import MolToMolBlock
from molli.external.openbabel import from_obmol
from molli.external import RDKitException, RDKitKekulizationException

def _rd_problems(m: ml.Molecule, rdmol: PropertyMol, ext:str, raise_kekulize=True) -> PropertyMol:
    '''Checks for problems in created RDKit mol object. Adds "KekulizeException" if this problem is detected.
    This will error if any other error is detected other than "KekulizeException"

    Parameters
    ----------
    m : ml.Molecule
        Molli Molecule
    rdmol : PropertyMol
        RDKit molecule object
    ext : str
        Extension format used
    raise_kekulize : bool
        Can be used to prevent instantiation if Molecule fails to kekulize

    Returns
    -------
    PropertyMol
        Updated RDKit Mol object if applicable
    '''
    try:
        problems = [x.GetType() for x in Chem.DetectChemistryProblems(rdmol)]
    except:
        raise RDKitException(f'{m.name} had a unique argument error that cannot be resolved preventing instantiation of a valid RDKit Mol with {ext} parsing')

    if len(problems) > 1:
        raise RDKitException(f'There are multiple problems in current RDKit Mol for {m.name}: {problems}')

    elif len(problems) == 1:
        if problems[0] != "KekulizeException":
            raise RDKitException(f'{m.name} cannot create a valid RDKit Mol with {ext} parsing due to the following error: {problems[0]}')
        
        else:
            if raise_kekulize:
                raise RDKitKekulizationException(f'{m.name} is failing to kekulize with {ext} parsing')
            else:
                rdmol.SetProp(problems[0], "1")

    return rdmol

def _rd_mol2(m:ml.Molecule, out:str, remove_hs:bool, ext:str, raise_kekulize=True) -> PropertyMol:
    '''Loads Molecule into RDKit with RDKit's MolFromMol2Block. Will process potential problems with kekulization.

    Parameters
    ----------
    m : ml.Molecule
        Molli Molecule
    out : str
        Mol2Block 
    remove_hs : bool
        Will remove hydrogens if specified
    ext : str
        Extension format used
    raise_kekulize : bool
        Can be used to prevent instantiation if Molecule fails to kekulize

    Returns
    -------
    PropertyMol
        New RDKit Mol Object
    '''

    rdmol = PropertyMol(
        Chem.MolFromMol2Block(out, removeHs=remove_hs)
    )
    if rdmol is None:
        rdmol = Chem.MolFromMol2Block(out, removeHs=remove_hs, sanitize=False)

    rdmol = _rd_problems(m, rdmol, ext, raise_kekulize)

    return rdmol

def _rd_xyz(m:ml.Molecule, out:str, remove_hs:bool, ext:str, raise_kekulize=True) -> PropertyMol:
    '''Loads Molecule into RDKit with RDKit's MolFromXYZBlock. Will process potential problems with kekulization.

    Parameters
    ----------
    m : ml.Molecule
        Molli Molecule
    out : str
        XYZBlock 
    remove_hs : bool
        Will remove hydrogens if specified
    ext : str
        Extension format used
    raise_kekulize : bool
        Can be used to prevent instantiation if Molecule fails to kekulize

    Returns
    -------
    PropertyMol
        New RDKit Mol Object
    '''

    rdmol = PropertyMol(
        Chem.MolFromXYZBlock(out)
    )
    if remove_hs:
        Chem.RemoveHs(rdmol)

    rdmol = _rd_problems(m, rdmol, ext, raise_kekulize)
    return rdmol

def _rd_mol(m:ml.Molecule, out:str, remove_hs:bool, ext:str, raise_kekulize=True) -> PropertyMol:
    '''Loads Molecule into RDKit with RDKit's MolFromMolBlock. Will process potential problems with kekulization.

    Parameters
    ----------
    m : ml.Molecule
        Molli Molecule
    out : str
        MolBlock 
    remove_hs : bool
        Will remove hydrogens if specified
    ext : str
        Extension format used
    raise_kekulize : bool
        Can be used to prevent instantiation if Molecule fails to kekulize

    Returns
    -------
    PropertyMol
        New RDKit Mol Object
    '''

    rdmol = PropertyMol(
        Chem.MolFromMolBlock(out, removeHs=remove_hs)
    )
    if rdmol is None:
        rdmol = Chem.MolFromMolBlock(out, removeHs=remove_hs, sanitize=False)

    rdmol = _rd_problems(m, rdmol, ext, raise_kekulize)
    return rdmol

def _rd_pdb(m:ml.Molecule, out:str, remove_hs:bool, ext:str, raise_kekulize=True) -> PropertyMol:
    '''Loads Molecule into RDKit with RDKit's MolFromPDBBlock. Will process potential problems with kekulization.

    Parameters
    ----------
    m : ml.Molecule
        Molli Molecule
    out : str
        PDBBlock 
    remove_hs : bool
        Will remove hydrogens if specified
    ext : str
        Extension format used
    raise_kekulize : bool
        Can be used to prevent instantiation if Molecule fails to kekulize 

    Returns
    -------
    PropertyMol
        New RDKit Mol Object
    '''

    rdmol = PropertyMol(
        Chem.MolFromPDBBlock(out, removeHs=remove_hs)
    )
    if rdmol is None:
        rdmol = Chem.MolFromPDBBlock(out, removeHs=remove_hs, sanitize=False)

    rdmol = _rd_problems(m, rdmol, ext, raise_kekulize)
    return rdmol

def _rd_smi(m:ml.Molecule, out:str, remove_hs:bool, ext:str, raise_kekulize=True) -> PropertyMol:
    '''Loads Molecule into RDKit with RDKit's MolFromSmiles. Will process potential problems with kekulization.
    Also attempts to maintain coordinates from the original Molli object if hydrogens are maintained and atom
    lists appear to be the same.

    Parameters
    ----------
    m : ml.Molecule
        Molli Molecule
    out : str
        SMILES string
    remove_hs : bool
        Will remove hydrogens if specified
    ext : str
        Extension format used
    raise_kekulize : bool
        Can be used to prevent instantiation if Molecule fails to kekulize 

    Returns
    -------
    PropertyMol
        New RDKit Mol Object
    '''

    rdmol = PropertyMol(
        Chem.MolFromSmiles(out, sanitize=remove_hs)
    )

    if rdmol is None:
        rdmol = Chem.MolFromSmiles(out, sanitize=False)

    rdmol = _rd_problems(m, rdmol, ext, raise_kekulize)

    if isinstance(rdmol, PropertyMol):
        if len(rdmol.GetAtoms()) == len(m.atoms):
            Chem.SanitizeMol(rdmol)
            rd_elem = np.array(
                [x.GetSymbol() for x in rdmol.GetAtoms()]
            )
            ml_elem = np.array(
                [atom.element.symbol for atom in m.atoms]
            )
            #Tests if the symbols are maintained
            if np.array_equal(rd_elem, ml_elem):

                from rdkit.Chem import rdDistGeom
                from rdkit.Geometry import Point3D

                rdDistGeom.EmbedMolecule(rdmol)
                conf = rdmol.GetConformer()
                for i in range(conf.GetNumAtoms()):
                    x,y,z = m.get_atom_coord(i).astype(float)
                    ml_3D = Point3D(x,y,z)
                    conf.SetAtomPosition(i, ml_3D)
            else:
                print(f'Atom Order not maintained upon OpenBabel conversion with SMILES, Coordinates not added to {m.name}')
        else:
            print(f'Number of atoms not the same after OpenBabel conversion with SMILES, Coordinates not added to {m.name}')
    else:
        raise RDKitException(f'Incorrect object type created from RDKit Mol for {m.name} during SMILES Parsing: {type(rdmol)}')

    return rdmol

def to_rdmol(m: ml.Molecule, via='sdf', remove_hs=True, set_atts=False, raise_kekulize=True) -> PropertyMol:
    '''This converts an existing Molecule Object to an RDKit Object. This will utilize Molli if the extension is xyz or mol2, 
    otherwise it will default to attempting to parse with Openbabel. Currently supported formats are xyz, mol2, pdb, sdf, mol, 
    and smi. This will also attempt to maintain the coordinates from the original structure when utilizing SMILES.

    Parameters
    ----------
    m : ml.Molecule
        Moleucle object to be converted
    via : str, optional
        Extension format to use, by default 'sdf'
    remove_hs : bool, optional
        Removes hydrogens from the representation when creating RDKit mol, by default True
    set_atts : bool, optional
        Prototype that attempts to set attributes that exist within a Molli Molecule, inlcuding atoms, bonds, and full molecule, by default False
    raise_kekulize : bool
        Can be used to prevent instantiation if Molecule fails to kekulize 

    Returns
    -------
    PropertyMol
        RDKit Mol capable of being serialized
    '''

    if via in ['xyz', 'mol2']:
        out = ml.dumps(m, fmt=via)
    else:
        out = ml.dumps(m, fmt=via, writer='obabel', obflags='h')

    match via:
        case 'mol2':
            rdmol = _rd_mol2(m=m,out=out,remove_hs=remove_hs,ext=via, raise_kekulize=raise_kekulize)  

        case 'xyz':
            rdmol = _rd_xyz(m=m,out=out,remove_hs=remove_hs,ext=via, raise_kekulize=raise_kekulize)

        case 'mol' | 'sdf':
            rdmol = _rd_mol(m=m,out=out,remove_hs=remove_hs,ext=via, raise_kekulize=raise_kekulize)

        case 'pdb':
            rdmol = _rd_pdb(m=m,out=out,remove_hs=remove_hs,ext=via, raise_kekulize=raise_kekulize)

        case 'smi':
            rdmol = _rd_smi(m=m,out=out,remove_hs=remove_hs,ext=via, raise_kekulize=raise_kekulize)

        # case 'fasta'| 'fa'| 'fsa':
        #     rdmol = PropertyMol(
        #         Chem.MolFromFASTA(out)
        #     )
        # case 'mrv':
        #     rdmol = PropertyMol(
        #         Chem.MolFromMrvBlock(out)
        #     )
        # case 'png':
        #     rdmol = PropertyMol(
        #         Chem.MolFromPNGString(out)
        #     )
        #     print('Currently adding Hs not supported when reading from png.')
        # case 'svg':
        #     rdmol = PropertyMol(
        #         Chem.MolFromRDKitSVG(out, removeHs=removeHs)
        #     )
        case _:
            return NotImplementedError(f'{via} not a currently implemented extension in RDKit/Openbabel Conversion')
    
    if rdmol is None:
        raise RDKitException(f'Cannot Create an RDKit mol from {m.name} using {via}')
    
    elif isinstance(rdmol, PropertyMol):
        rdmol.SetProp("_Name", m.name)

        if set_atts:
            #Full Molecule
            for attrib in m.attrib:
                rdmol.SetProp(attrib, str(m.attrib[attrib]))

            #Sets Molli Molecule, Atom, and Bond Attributes to RDKit Molecule
            if not remove_hs:
                #Atoms
                for i,a in enumerate(rdmol.GetAtoms()):
                    ml_atom = m.get_atom(i)
                    for attrib in ml_atom.attrib:
                        a.SetProp(attrib, str(ml_atom.attrib[attrib])) 

                #Bonds (does not maintain bond order upon reading in, so this exists now :) )
                rdbond_list = {b: {b.GetBeginAtomIdx(),b.GetEndAtomIdx()} for b in rdmol.GetBonds()}
                for i,ml_bond in enumerate(m.bonds):
                    if ml_bond.attrib:
                        ml_atoms = {m.get_atom_index(ml_bond.a1), m.get_atom_index(ml_bond.a2)}
                        att_bond = None
                        for rd_bond, atoms in rdbond_list.items():
                            if atoms == ml_atoms:
                                att_bond = rd_bond
                                break
                        att_bond.SetProp(attrib, str(ml_bond.attrib[attrib]))
            else:
                print(f'Unable to map original Molli Atom and Bond attributes to RDKit object Atoms/Bonds when hydrogens are removed. RDKit object will still be returned for {m}')
        return rdmol   
    else:
        raise RDKitException(f'Incorrect object type created from RDKit Mol for {m.name} during SMILES Parsing: {type(rdmol)}')
    
def from_rdmol(rdmol: PropertyMol | Chem.Mol) -> ml.Molecule:
    '''This function converts an RDKit Mol into a Molli Molecule utilizing Openbabel and MolToMolBlock within RDKit

    Parameters
    ----------
    rdmol : PropertyMol | Chem.Mol
        RDKit Molecule Object

    Returns
    -------
    ml.Molecule
        Molli Molecule Object
    '''

    sdf = MolToMolBlock(rdmol)
    pbmol = readstring("sdf", sdf)
    mol = from_obmol(pbmol.OBMol)

    return mol

def rd_visualize(
    file_path: str,
    rd_list: list,
    subImgSize=(250, 250),
    legendFontSize=30,
    molsPerRow=5,
    legend_prop: str = "_Name",
    highlight_atom_prop:str = None,
    highlight_bond_prop:str = None,
) -> None:
    '''This visualizes a list of RDkit Molecule objects as an SVG. Atoms and Bonds can be highlighted if properties are assigned to the indiivdual Atoms and Bonds

    Parameters
    ----------
    file_path : str
        Path for SVG to be written to
    rd_list : list
        List of RDKit Molecule Objects
    subImgSize : tuple, optional
        Controls the size of Individual Molecule Images, by default (250, 250)
    legendFontSize : int, optional
        Font Size of the Legend, by default 30
    molsPerRow : int, optional
        Number of Molecules visualized per row, by default 5
    legend_prop : str, optional
        Property of RDKit Molecule used to label molecules on the grid, by default "_Name"
    highlight_atom_prop : str, optional
        Property of RDKit Atoms used to identify which atoms should be highlighted, by default None
    highlight_bond_prop : str, optional
        Property of RDKit Bonds used to identify which bonds should be highlighted, by default None

    Returns
    -------
    None

    '''
    if len(rd_list) == 0:
        raise ValueError('List of RDKit Molecules is empty!')
    else:
        nRows = len(rd_list) // molsPerRow
        if len(rd_list) % molsPerRow:
            nRows += 1
        fullSize = (molsPerRow * subImgSize[0], nRows * subImgSize[1])

        d2d = rdMolDraw2D.MolDraw2DSVG(
            fullSize[0], fullSize[1], subImgSize[0], subImgSize[1]
        )
        d2d.drawOptions().legendFontSize = legendFontSize
        highlight_atoms_vals = list()
        highlight_bonds_vals = list()

        # This finds the atoms that contain the property if applicable
        # if highlight_atom_prop and highlight_bond_prop:
        for rdmol in rd_list:
            if highlight_atom_prop:
                assert isinstance(highlight_atom_prop, str), f'Atom Property not a string, it is a {type(highlight_atom_prop)}: {highlight_atom_prop}'
                sub_atoms = [int(a.GetIdx()) for a in rdmol.GetAtoms() if a.HasProp(highlight_atom_prop)]
                highlight_atoms_vals.append(sub_atoms)
            if highlight_bond_prop:
                assert isinstance(highlight_bond_prop, str), f'Bond Property not a string, it is a {type(highlight_bond_prop)}: {highlight_bond_prop}'
                sub_bonds = [int(b.GetIdx()) for b in rdmol.GetBonds() if b.HasProp(highlight_bond_prop)]
                highlight_bonds_vals.append(sub_bonds)

        if not highlight_atom_prop and not highlight_bond_prop:
            highlight_atoms_vals = None
            highlight_bonds_vals = None

        legends = [i.GetProp(legend_prop) if i.HasProp(legend_prop) else '' for i in rd_list]
        d2d.DrawMolecules(
            rd_list,
            highlightAtoms=highlight_atoms_vals,
            highlightBonds=highlight_bonds_vals,
            legends=legends,
        )
        d2d.FinishDrawing()

        with open(file_path, "w") as f:
            f.write(d2d.GetDrawingText())

def ml_rd_visualize(
    file_path: str,
    obj: ml.Molecule | ml.MoleculeLibrary | list,
    via:str='sdf',
    remove_hs=True,
    set_atts=True,
    raise_kekulize=True,
    subImgSize=(250, 250),
    legendFontSize=30,
    molsPerRow=5,
    legend_prop: str = "_Name",
    highlight_atom_prop:str = None,
    highlight_bond_prop:str = None,
) -> None:
    '''This visualizes a list, Molecule, or MoleculeLibrary via RDKit's SVG viewer and Molli's implementation. 
    Atoms and Bonds can be highlighted if properties are assigned to the indiivdual Atoms and Bonds of the RDKit object.

    Parameters
    ----------
    file_path : str
        Path for SVG to be written to
    obj : ml.Molecule | ml.MoleculeLibrary | list
        Input to convert to RDKit Object
    via : str, optional
        Extension format to use, by default 'sdf'
    remove_hs : bool, optional
        Removes hydrogens from the representation when creating RDKit mol, by default True
    set_atts : bool, optional
        Prototype that attempts to set attributes that exist within a Molli Molecule, inlcuding atoms, bonds, and full molecule, by default False
    raise_kekulize : bool
        Can be used to prevent instantiation if Molecule fails to kekulize 
    subImgSize : tuple, optional
        Controls the size of Individual Molecule Images, by default (250, 250)
    legendFontSize : int, optional
        Font Size of the Legend, by default 30
    molsPerRow : int, optional
        Number of Molecules visualized per row, by default 5
    legend_prop : str, optional
        Property of RDKit Molecule used to label molecules on the grid, by default "_Name"
    highlight_atom_prop : str, optional
        Property of RDKit Atoms used to identify which atoms should be highlighted, by default None
    highlight_bond_prop : str, optional
        Property of RDKit Bonds used to identify which bonds should be highlighted, by default None
    '''
    cls = type(obj)

    ml_list = []
    # print(list)
    match cls:
        case ml.Molecule:
            ml_list.append(obj)

        case ml.MoleculeLibrary:
            with obj.reading():
                for k in obj:
                    ml_list.append(obj[k])
        case list:
            ml_list.extend(obj)

    rd_list = []
    except_rdlist = []

    for m in ml_list:
        assert isinstance(m, ml.Molecule), f'{m} is not a valid instance of a Molecule!'
        try:
            rdmol = to_rdmol(
            m=m,
            via=via,
            remove_hs=remove_hs,
            set_atts=set_atts,
            raise_kekulize=raise_kekulize
            )
            rd_list.append(rdmol)
        except Exception as e:
            except_rdlist.append((m.name, e))

    if len(except_rdlist) != 0:
        print(f'There were {len(except_rdlist)} that caused exceptions upon processing:\n{except_rdlist}')

    rd_visualize(
        file_path=file_path,
        rd_list=rd_list,
        subImgSize=subImgSize,
        legendFontSize=legendFontSize,
        molsPerRow=molsPerRow,
        legend_prop=legend_prop,
        highlight_atom_prop=highlight_atom_prop,
        highlight_bond_prop=highlight_bond_prop
    )

def canonicalize_rdmol(rdmol:PropertyMol, remove_hs=False) -> PropertyMol:
    '''Returns canonicalized RDKit mol generated from a canonicalized RDKit SMILES string. The current implementation will only add the "_Name" property

    Parameters
    ----------
    rdmol : PropertyMol
        RDKit Mol to canonicalize
    remove_hs : bool, optional
        Removes Hydrogens upon canonicalization, by default False

    Returns
    -------
    PropertyMol
        Returned RDKit Molecule
    '''

    can_smiles = Chem.MolToSmiles(rdmol, canonical=True)
    can_rdmol = Chem.MolFromSmiles(can_smiles, sanitize=remove_hs)

    Chem.SanitizeMol(can_rdmol)

    if rdmol.HasProp("_Name"):
        can_rdmol.SetProp("_Name", rdmol.GetProp("_Name"))

    return can_rdmol

def can_mol_order(rdmol:PropertyMol) -> tuple[PropertyMol, list[int], list[int]]:
    '''This a function tries to match the indexes of the canonicalized smiles string/molecular graph to a Molli Molecule object.
    Any inputs to this function will AUTOMATICALLY ADD HYDROGENS (make them explicit) to the RDKit mol object.

    Important Notes:
    - It will only have "_Kekulize_Issue" if the initial object had this property set (i.e. if it ran into an issue in the in initial instantiation)
    - The canonical rdkit mol object will have the "Canonical SMILES with hydrogens" available as the property: "_Canonical_SMILES_w_H"
    - There may be some properties missing as the PropertyCache is not being updated on the new canonicalized mol object, so consider using rdmol.UpdatePropertyCache() if you want to continue using the mol object
    
    Parameters
    ----------
    rdmol : PropertyMol
        RDKit Mol to change canonicalize

    Returns
    -------
    tuple[PropertyMol, list[int], list[int]]
        `1. Canonical RDKit Mol Object with Hydrogens\n`
        `2. A List for reordering the Atom Indices after canonicalization\n`
        `3. A list for reordering the Bond Indices after canonicalization\n`
    '''

    # This is here to deal with any smiles strings or mol objects that do not get assigned hydrogens
    new_rdmol = Chem.AddHs(rdmol)

    #### This statement is necessary to generate the mol.GetPropertyName "_smilesAtomOutputOrder" and"_smilesBondOutputOrder"######
    Chem.MolToSmiles(new_rdmol, canonical=True)

    # The smiles output order is actually a string of the form "[0,1,2,3,...,12,]", so it requires a start at 1 and end at -2!
    can_atom_reorder = list(
        map(int, new_rdmol.GetProp("_smilesAtomOutputOrder")[1:-2].split(","))
    )
    canonical_bond_reorder_list = list(
        map(int, new_rdmol.GetProp("_smilesBondOutputOrder")[1:-2].split(","))
    )
    can_smiles_w_h = Chem.MolToSmiles(new_rdmol, canonical=True)

    # # # #Allows maintaining of hydrogens when Mol object is created
    can_mol_w_h = PropertyMol(Chem.MolFromSmiles(can_smiles_w_h, sanitize=False))
    # Certain odd molecules result in some odd calculated properties, so this part is remaining commented out for now
    # can_mol_w_h.UpdatePropertyCache()
    all_props_original_rdmol = list(rdmol.GetPropNames())

    # Helps new rdkit object maintain original properties of rdkit mol put in
    for prop in all_props_original_rdmol:
        if not can_mol_w_h.HasProp(prop):
            can_mol_w_h.SetProp(prop, rdmol.GetProp(prop))

    can_mol_w_h.SetProp("_Canonical_SMILES_w_H", f"{can_smiles_w_h}")

    return can_mol_w_h, can_atom_reorder, canonical_bond_reorder_list

def reorder_molecule(
    ml_mol: ml.Molecule,
    can_rdmol_w_h: PropertyMol,
    can_atom_reorder: list,
    can_bond_reorder: list,
) -> dict[ml.Molecule, PropertyMol]:
    '''This is a function that utilizes the outputs of new_mol_order to reorder an existing molecule.
    Currently done in place on the original ml_mol object.

    Parameters
    ----------
    ml_mol : Molecule
        Molli Molecule Object to be reordered
    can_rdmol_w_h : _type_
        Canonical RDKit Object to be matched with
    can_atom_reorder : list
        List of integers associated with the atom reordering
    can_bond_reorder : list
        List of integers associated with the bond reordering

    Returns
    -------
    dict[ml.Molecule, PropertyMol]
        Dictionary linking Molli Molecule Object to RDKit Object 
    '''

    # This reorders the atoms of the molecule object
    molli_atoms_arr = np.array(ml_mol.atoms)
    fixed_atom_order_list = molli_atoms_arr[can_atom_reorder].tolist()
    ml_mol._atoms = fixed_atom_order_list

    # This reorders the bonds of the molecule object
    molli_obj_bonds_arr = np.array(ml_mol.bonds)
    fixed_bond_order_list = molli_obj_bonds_arr[can_bond_reorder].tolist()
    ml_mol._bonds = fixed_bond_order_list

    # This fixes the geometry of the molecule object
    ml_mol.coords = ml_mol.coords[can_atom_reorder]

    # This checks to see if the new rdkit atom order in the canonical smiles matches the new molli order of atoms
    can_rdkit_atoms = can_rdmol_w_h.GetAtoms()
    can_rdkit_atom_elem = np.array([x.GetSymbol() for x in can_rdkit_atoms])

    new_molli_elem = np.array([atom.element.symbol for atom in ml_mol.atoms])
    equal_check = np.array_equal(can_rdkit_atom_elem, new_molli_elem)

    assert (
        equal_check
    ), f"Array of rdkit atoms: {can_rdkit_atom_elem} is not equal to array of molli atoms: {new_molli_elem}"

    return {ml_mol: can_rdmol_w_h}

class atom_filter(PropertyMol):
    """
    These functions are written as numpy arrays to isolate types of atoms very easily with the goal of the structure being:

    isolated_atoms = (aromatic_type() & sp2_type() & carbon)

    All the functions of this class figure create a boolean array the size of the "All Atoms" array, and then
    they define the intersection of this array with a second array (the case of the condition), and return the array: [(1 & 2 & ...)]

    It is recommended that rdkit molecules are canonicalized before utilizing this function.
    """

    def __init__(self, rdmol):
        self.rdmol = rdmol
        self.atoms = rdmol.GetAtoms()
        self.atoms_array = np.array([x.GetIdx() for x in self.atoms])

        if np.all(np.diff(self.atoms_array) >= 0):
            pass
        else:
            raise ValueError(f"The atom IDs are not ordered from least to greatest")

    def sp2_type(self):
        """
        This takes a numpy array of Atom IDs and returns a boolean for where SP2 atoms exist.
        Inputs to this function are built for an ORDERED LIST OF ALL ATOM IDs from LEAST TO GREATEST.
        """
        sp2_atoms = chemq.HybridizationEqualsQueryAtom(Chem.HybridizationType.SP2)
        sp2 = np.array(
            [x.GetIdx() for x in self.rdmol.GetAtomsMatchingQuery(sp2_atoms)]
        )
        sp2_bool = np.in1d(self.atoms_array, sp2)
        return sp2_bool

    def aromatic_type(self):
        """
        This takes a numpy array of Atom IDs and returns a boolean for where AROMATIC atoms exist.
        Inputs to this function are built for an ORDERED LIST OF ALL ATOM IDs from LEAST TO GREATEST.
        """
        aromatic_atoms = chemq.IsAromaticQueryAtom()
        aromatic = np.array(
            [x.GetIdx() for x in self.rdmol.GetAtomsMatchingQuery(aromatic_atoms)]
        )
        aromatic_bool = np.in1d(self.atoms_array, aromatic)
        return aromatic_bool

    def ring_type(self):
        """
        This takes a numpy array of Atom IDs and returns a boolean for where atoms IN A RING exist.
        Inputs to this function are built for an ORDERED LIST OF ALL ATOM IDs from LEAST TO GREATEST.
        """
        ring_atoms = chemq.IsInRingQueryAtom()
        ring = np.array(
            [x.GetIdx() for x in self.rdmol.GetAtomsMatchingQuery(ring_atoms)]
        )
        ring_bool = np.in1d(self.atoms_array, ring)
        return ring_bool

    def carbon_type(self):
        """
        This takes a numpy array of Atom IDs and returns a boolean for where CARBON atoms exist.
        Inputs to this function are built for an ORDERED LIST OF ALL ATOM IDs from LEAST TO GREATEST.
        """
        carbon_atoms = chemq.AtomNumEqualsQueryAtom(6)
        carbon = np.array(
            [x.GetIdx() for x in self.rdmol.GetAtomsMatchingQuery(carbon_atoms)]
        )
        carbon_bool = np.in1d(self.atoms_array, carbon)
        return carbon_bool

    def nitrogen_type(self):
        """
        This takes a numpy array of Atom IDs and returns a boolean for where NITROGEN atoms exist.
        Inputs to this function are built for an ORDERED LIST OF ALL ATOM IDs from LEAST TO GREATEST.
        """
        nitrogen_atoms = chemq.AtomNumEqualsQueryAtom(7)
        nitrogen = np.array(
            [x.GetIdx() for x in self.rdmol.GetAtomsMatchingQuery(nitrogen_atoms)]
        )
        nitrogen_bool = np.in1d(self.atoms_array, nitrogen)
        return nitrogen_bool

    def oxygen_type(self):
        """
        This takes a numpy array of Atom IDs and returns a boolean for where OXYGEN atoms exist.
        Inputs to this function are built for an ORDERED LIST OF ALL ATOM IDs from LEAST TO GREATEST.
        """
        oxygen_atoms = chemq.AtomNumEqualsQueryAtom(8)
        oxygen = np.array(
            [x.GetIdx() for x in self.rdmol.GetAtomsMatchingQuery(oxygen_atoms)]
        )
        oxygen_bool = np.in1d(self.atoms_array, oxygen)
        return oxygen_bool

    def atom_num_less_than(self, number: int):
        """
        This takes a numpy array of Atom IDs and returns a boolean for where the ATOM NUMBER is LESS than the input.
        Inputs to this function are built for an ORDERED LIST OF ALL ATOM IDs from LEAST TO GREATEST.
        """
        num_light_atoms = chemq.AtomNumLessQueryAtom(number)
        num_light_atom = np.array(
            [x.GetIdx() for x in self.rdmol.GetAtomsMatchingQuery(num_light_atoms)]
        )
        num_light_atom_bool = np.in1d(self.atoms_array, num_light_atom)
        return num_light_atom_bool

    def atom_num_equals(self, number: int):
        """
        This takes a numpy array of Atom IDs and returns a boolean for where the ATOM NUMBER is EQUAL to the input.
        Inputs to this function are built for an ORDERED LIST OF ALL ATOM IDs from LEAST TO GREATEST.
        """
        num_equals_atoms = chemq.AtomNumEqualsQueryAtom(number)
        num_equal_atom = np.array(
            [x.GetIdx() for x in self.rdmol.GetAtomsMatchingQuery(num_equals_atoms)]
        )
        num_equal_atom_bool = np.in1d(self.atoms_array, num_equal_atom)
        return num_equal_atom_bool

    def atom_num_greater_than(self, number: int):
        """
        This takes a numpy array of Atom IDs and returns a boolean for where the ATOM NUMBER is GREATER than the input.
        Inputs to this function are built for an ORDERED LIST OF ALL ATOM IDs from LEAST TO GREATEST.
        """
        num_heavy_atoms = chemq.AtomNumGreaterQueryAtom(number)
        num_heavy_atom = np.array(
            [x.GetIdx() for x in self.rdmol.GetAtomsMatchingQuery(num_heavy_atoms)]
        )
        num_heavy_atom_bool = np.in1d(self.atoms_array, num_heavy_atom)
        return num_heavy_atom_bool

    def isotope_type_equals(self, number: int):
        """
        This takes a numpy array of Atom IDs and returns a boolean for where the ISOTOPE NUMBER is EQUAL to the input.
        Inputs to this function are built for an ORDERED LIST OF ALL ATOM IDs from LEAST TO GREATEST.
        """
        isotope_atoms = chemq.IsotopeEqualsQueryAtom(number)
        isotope_atom = np.array(
            [x.GetIdx() for x in self.rdmol.GetAtomsMatchingQuery(isotope_atoms)]
        )
        isotope_atom_bool = np.in1d(self.atoms_array, isotope_atom)
        return isotope_atom_bool

    def charge_type_less_than(self, number: int):
        """
        This takes a numpy array of Atom IDs and returns a boolean for where the FORMAL CHARGE is LESS THAN the input.
        Inputs to this function are built for an ORDERED LIST OF ALL ATOM IDs from LEAST TO GREATEST.
        """
        charge_less_atoms = chemq.FormalChargeLessQueryAtom(number)
        charge_less_atom = np.array(
            [
                x.GetIdx()
                for x in self.rdmol.GetAtomsMatchingQuery(charge_less_atoms)
            ]
        )
        charge_less_atom_bool = np.in1d(self.atoms_array, charge_less_atom)
        return charge_less_atom_bool

    def charge_type_equals(self, number: int):
        """
        This takes a numpy array of Atom IDs and returns a boolean for where the FORMAL CHARGE is EQUAL to the input.
        Inputs to this function are built for an ORDERED LIST OF ALL ATOM IDs from LEAST TO GREATEST.
        """
        charge_equals_atoms = chemq.FormalChargeEqualsQueryAtom(number)
        charge_equals_atom = np.array(
            [
                x.GetIdx()
                for x in self.rdmol.GetAtomsMatchingQuery(charge_equals_atoms)
            ]
        )
        charge_equals_atom_bool = np.in1d(self.atoms_array, charge_equals_atom)
        return charge_equals_atom_bool

    def charge_type_greater_than(self, number: int):
        """
        This takes a numpy array of Atom IDs and returns a boolean for where the FORMAL CHARGE is GREATER THAN the input.
        Inputs to this function are built for an ORDERED LIST OF ALL ATOM IDs from LEAST TO GREATEST.
        """
        charge_greater_atoms = chemq.FormalChargeGreaterQueryAtom(number)
        charge_greater_atom = np.array(
            [
                x.GetIdx()
                for x in self.rdmol.GetAtomsMatchingQuery(charge_greater_atoms)
            ]
        )
        charge_greater_atom_bool = np.in1d(self.atoms_array, charge_greater_atom)
        return charge_greater_atom_bool

    def hcount_less_than(self, number: int):
        """
        This takes a numpy array of Atom IDs and returns a boolean for where the HYDROGEN COUNT is LESS THAN the input.
        Inputs to this function are built for an ORDERED LIST OF ALL ATOM IDs from LEAST TO GREATEST.
        """
        hcount_less_atoms = chemq.HCountLessQueryAtom(number)
        hcount_less_atom = np.array(
            [
                x.GetIdx()
                for x in self.rdmol.GetAtomsMatchingQuery(hcount_less_atoms)
            ]
        )
        hcount_less_atom_bool = np.in1d(self.atoms_array, hcount_less_atom)
        return hcount_less_atom_bool

    def hcount_equals(self, number: int):
        """
        This takes a numpy array of Atom IDs and returns a boolean for where the HYDROGEN COUNT is EQUAL to the input.
        Inputs to this function are built for an ORDERED LIST OF ALL ATOM IDs from LEAST TO GREATEST.
        """
        hcount_equals_atoms = chemq.HCountEqualsQueryAtom(number)
        hcount_equals_atom = np.array(
            [
                x.GetIdx()
                for x in self.rdmol.GetAtomsMatchingQuery(hcount_equals_atoms)
            ]
        )
        hcount_equals_atom_bool = np.in1d(self.atoms_array, hcount_equals_atom)
        return hcount_equals_atom_bool

    def hcount_greater_than(self, number: int):
        """
        This takes a numpy array of Atom IDs and returns a boolean for where the HYDROGEN COUNT is GREATER THAN the input.
        Inputs to this function are built for an ORDERED LIST OF ALL ATOM IDs from LEAST TO GREATEST.
        """
        hcount_greater_atoms = chemq.HCountGreaterQueryAtom(number)
        hcount_greater_atom = np.array(
            [
                x.GetIdx()
                for x in self.rdmol.GetAtomsMatchingQuery(hcount_greater_atoms)
            ]
        )
        hcount_greater_atom_bool = np.in1d(self.atoms_array, hcount_greater_atom)
        return hcount_greater_atom_bool

    def in_ring(self):
        """
        This takes a numpy array of Atom IDs and returns a boolean for where atoms IN A RING exist.
        Inputs to this function are built for an ORDERED LIST OF ALL ATOM IDs from LEAST TO GREATEST.
        """
        ring_atoms = chemq.IsInRingQueryAtom()
        ring = np.array(
            [x.GetIdx() for x in self.rdmol.GetAtomsMatchingQuery(ring_atoms)]
        )
        ring_bool = np.in1d(self.atoms_array, ring)
        return ring_bool

    def ring_size6(self):
        """
        This takes a numpy array of Atom IDs and returns a boolean for where atoms IN A 6-MEMBERED RING exist.
        Inputs to this function are built for an ORDERED LIST OF ALL ATOM IDs from LEAST TO GREATEST.
        """
        ring_6 = chemq.MinRingSizeEqualsQueryAtom(6)
        size6 = np.array(
            [x.GetIdx() for x in self.rdmol.GetAtomsMatchingQuery(ring_6)]
        )
        size6_bool = np.in1d(self.atoms_array, size6)
        return size6_bool

    def ring_size5(self):
        """
        This takes a numpy array of Atom IDs and returns a boolean for where atoms IN A 5-MEMBERED RING exist.
        Inputs to this function are built for an ORDERED LIST OF ALL ATOM IDs from LEAST TO GREATEST.
        """
        ring_5 = chemq.MinRingSizeEqualsQueryAtom(5)
        size5 = np.array(
            [x.GetIdx() for x in self.rdmol.GetAtomsMatchingQuery(ring_5)]
        )
        size5_bool = np.in1d(self.atoms_array, size5)
        return size5_bool

    def in_2_rings(self):
        """
        This takes a numpy array of Atom IDs and returns a boolean for where atoms IN 2 RINGS exist.
        Inputs to this function are built for an ORDERED LIST OF ALL ATOM IDs from LEAST TO GREATEST.
        """
        ring_2 = chemq.InNRingsEqualsQueryAtom(2)
        ring2 = np.array(
            [x.GetIdx() for x in self.rdmol.GetAtomsMatchingQuery(ring_2)]
        )
        ring2_bool = np.in1d(self.atoms_array, ring2)
        return ring2_bool

    def in_1_ring(self):
        """
        This takes a numpy array of Atom IDs and returns a boolean for where atoms IN 1 RING exist.
        Inputs to this function are built for an ORDERED LIST OF ALL ATOM IDs from LEAST TO GREATEST.
        """
        ring_1 = chemq.InNRingsEqualsQueryAtom(1)
        ring1 = np.array(
            [x.GetIdx() for x in self.rdmol.GetAtomsMatchingQuery(ring_1)]
        )
        ring1_bool = np.in1d(self.atoms_array, ring1)
        return ring1_bool

    def het_neighbors_3(self):
        """
        This takes a numpy array of Atom IDs and returns a boolean for where atoms have HETEROATOM NEIGHBORS = 3.
        Inputs to this function are built for an ORDERED LIST OF ALL ATOM IDs from LEAST TO GREATEST.
        """
        het_a_3 = chemq.NumHeteroatomNeighborsEqualsQueryAtom(3)
        heta3 = np.array(
            [x.GetIdx() for x in self.rdmol.GetAtomsMatchingQuery(het_a_3)]
        )
        heta3_bool = np.in1d(self.atoms_array, heta3)
        return heta3_bool

    def het_neighbors_2(self):
        """
        This takes a numpy array of Atom IDs and returns a boolean for where atoms have HETEROATOM NEIGHBORS = 2.
        Inputs to this function are built for an ORDERED LIST OF ALL ATOM IDs from LEAST TO GREATEST.
        """
        het_a_2 = chemq.NumHeteroatomNeighborsEqualsQueryAtom(2)
        heta2 = np.array(
            [x.GetIdx() for x in self.rdmol.GetAtomsMatchingQuery(het_a_2)]
        )
        heta2_bool = np.in1d(self.atoms_array, heta2)
        return heta2_bool

    def het_neighbors_1(self):
        """
        This takes a numpy array of Atom IDs and returns a boolean for where atoms have HETEROATOM NEIGHBORS = 1.
        Inputs to this function are built for an ORDERED LIST OF ALL ATOM IDs from LEAST TO GREATEST.
        """
        het_a_1 = chemq.NumHeteroatomNeighborsEqualsQueryAtom(1)
        heta1 = np.array(
            [x.GetIdx() for x in self.rdmol.GetAtomsMatchingQuery(het_a_1)]
        )
        heta1_bool = np.in1d(self.atoms_array, heta1)
        return heta1_bool

    def het_neighbors_0(self):
        """
        This takes a numpy array of Atom IDs and returns a boolean for where atoms have HETEROATOM NEIGHBORS = 0.
        Inputs to this function are built for an ORDERED LIST OF ALL ATOM IDs from LEAST TO GREATEST.
        """
        het_a_0 = chemq.NumHeteroatomNeighborsEqualsQueryAtom(0)
        heta0 = np.array(
            [x.GetIdx() for x in self.rdmol.GetAtomsMatchingQuery(het_a_0)]
        )
        heta0_bool = np.in1d(self.atoms_array, heta0)
        return heta0_bool

    def het_neighbors_greater_1(self):
        """
        This takes a numpy array of Atom IDs and returns a boolean for where atoms have HETEROATOM NEIGHBORS > 1.
        Inputs to this function are built for an ORDERED LIST OF ALL ATOM IDs from LEAST TO GREATEST.
        """
        het_a_g1 = chemq.NumHeteroatomNeighborsGreaterQueryAtom(0)
        hetag1 = np.array(
            [x.GetIdx() for x in self.rdmol.GetAtomsMatchingQuery(het_a_g1)]
        )
        hetag1_bool = np.in1d(self.atoms_array, hetag1)
        return hetag1_bool

    def het_neighbors_greater_0(self):
        """
        This takes a numpy array of Atom IDs and returns a boolean for where atoms have HETEROATOM NEIGHBORS > 0.
        Inputs to this function are built for an ORDERED LIST OF ALL ATOM IDs from LEAST TO GREATEST.
        """
        het_a_g0 = chemq.NumHeteroatomNeighborsGreaterQueryAtom(0)
        hetag0 = np.array(
            [x.GetIdx() for x in self.rdmol.GetAtomsMatchingQuery(het_a_g0)]
        )
        hetag0_bool = np.in1d(self.atoms_array, hetag0)
        return hetag0_bool

    def aliph_het_neighbors_2(self):
        """
        This takes a numpy array of Atom IDs and returns a boolean for where atoms are ALIPHATIC AND HAVE 2 HETEROATOM NEIGHBORS.
        Inputs to this function are built for an ORDERED LIST OF ALL ATOM IDs from LEAST TO GREATEST.
        """
        a_het_a_2 = chemq.NumAliphaticHeteroatomNeighborsEqualsQueryAtom(2)
        aheta2 = np.array(
            [x.GetIdx() for x in self.rdmol.GetAtomsMatchingQuery(a_het_a_2)]
        )
        aheta2_bool = np.in1d(self.atoms_array, aheta2)
        return aheta2_bool

    def aliph_het_neighbors_1(self):
        """
        This takes a numpy array of Atom IDs and returns a boolean for where atoms are ALIPHATIC AND HAS 1 HETEROATOM NEIGHBORS.
        Inputs to this function are built for an ORDERED LIST OF ALL ATOM IDs from LEAST TO GREATEST.
        """
        a_het_a_1 = chemq.NumAliphaticHeteroatomNeighborsEqualsQueryAtom(1)
        aheta1 = np.array(
            [x.GetIdx() for x in self.rdmol.GetAtomsMatchingQuery(a_het_a_1)]
        )
        aheta1_bool = np.in1d(self.atoms_array, aheta1)
        return aheta1_bool

    def smarts_query(self, smarts: str):
        query = Chem.MolFromSmarts(smarts)
        substructs = self.rdmol.GetSubstructMatches(query)

        idx = np.zeros(len(self.atoms), dtype=bool)
        for s in substructs:
            idx[list(s)] = True

        return idx
