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

import molli as ml
import numpy as np
import importlib.util
from typing import Iterable
from pprint import pprint

'''
This is meant to interface with the morfeus-ml package originally developed
by Kjell Jorner, along with Gabriel dos Passos Gomes, Pascal Friedrich, and Tobias Gensch
Information can be found below:

Github: https://github.com/digital-chemistry-laboratory/morfeus
Documentation: https://digital-chemistry-laboratory.github.io/morfeus/notes.html

Installation:
pip install morfeus-ml
'''

try:
    from morfeus import BuriedVolume, ConeAngle, Sterimol
    from morfeus.typing import ArrayLike1D
except ImportError as xc:
    raise ImportError("morfeus-ml is not installed in this environment") from xc

def draw_3D_cone(
    cone_angle: ConeAngle,
    mlmol: ml.Molecule,
    cone_atom: ml.Atom,
    show_cone_atom: bool=True,
    cone_color: str = "steelblue",
    cone_opacity: float = 0.3,
    vdw_opacity: float= 0.3
) -> None:
    '''Draw a 3D representation of the molecule with the cone. This is 
    heavily taking from the morfeus implementation of `ConeAngle.draw_3D` 
    function with a simple modification to ensure cone inversion at angles
    greater than 180 degrees.

    Parameters
    ----------
    cone_angle : ConeAngle
        ConeAngle instance created by morfeus
    mlmol : ml.Molecule
        Molecule used for descriptor calculation
    cone_atom : ml.Atom
        Atom used for cone angle calculation
    cone_color : str, optional
        Color of cone drawn, by default "steelblue"
    cone_opacity : float, optional
        Opacity of cone drawn, by default 0.3
    show_cone_atom: bool, optional
        Show Atom used for cone angle calculation in visualization, by default True
    vdw_opacity : float, optional
        Opacity of cone drawn, by default 0.3

    Additional Notes
    -----------------
    `ConeAngle.draw_3D` reference can be found here:
    https://github.com/digital-chemistry-laboratory/morfeus/blob/main/morfeus/cone_angle.py

    '''

    from molli.visual._pyvista import draw_vdwsphere
    from pyvista import Cone

    if show_cone_atom:
        p = draw_vdwsphere(mlmol, _tobeshown=False, opacity=vdw_opacity)
    else:
        p = draw_vdwsphere(mlmol, exclude_atoms=[cone_atom], _tobeshown=False, opacity=vdw_opacity)

    # Determines direction and extension of cone
    angle = cone_angle.cone_angle

    coordinates = mlmol.coords
    radii= np.array([atom.vdw_radius for atom in mlmol.atoms])
    cone_atom_coord = mlmol.get_atom_coord(cone_atom)

    #Inverts the normal vector defining the cone when the angle is greater than 180
    if angle > 180:
        normal = -cone_angle._cone.normal
    else:
        normal = cone_angle._cone.normal
    #Projects all the atom coordinates onto the normal vector
    projected = np.dot(normal, coordinates.T) + np.array(radii)

    #Determines the maximum distance necessary to render for the cone
    max_extension = np.max(projected)

    #The "angle" calculated refers to the apex angle of the cone, which 
    #is multiplied by 2. This needs to be divided by 2 regardless of direction
    if angle > 180:
        max_extension += 1
        #Inverts the cone and moves it 
        fix_angle = (360-angle) / 2
    else:
        fix_angle = angle / 2
    
    #Creates the cone where center (i.e. the midpoint of the cone) is shifted
    #to the middle of the atom that was used to define the cone (i.e. the tip)
    cone = Cone(
        center=cone_atom_coord + (max_extension * normal) / 2,
        direction=-normal,
        angle=fix_angle,
        height=max_extension,
        capping=False,
        resolution=100,
        )

    p.add_mesh(cone, opacity=cone_opacity, show_edges=False, color=cone_color)

    p.show()

def buried_volume(
        ml_mol: ml.Molecule, 
        metal_atom: (ml.Atom | int),
        round_coords: bool = False,
        excluded_atoms: Iterable[ml.Atom] = None,
        radii: (str | ArrayLike1D) = "morfeus",
        include_hs: bool = False,
        radius: float = 3.5,
        radii_type: str = 'bondi',
        radii_scale: float = 1.17,
        density = 0.001,
        z_axis_atoms: Iterable[ml.Atom] = None,
        xz_plane_atoms: Iterable[ml.Atom] = None,
        calc_distal_vol: bool = False,
        dist_method: str = 'sasa', 
        dist_octants: bool = False,
        dist_sasa_density: float = 0.01,
        plot: bool = False,
        verbose: bool = False,
        ) -> ml.Molecule:
    '''Buried Volume calculation as done with morfeus. This will store buried volume
    calculation data and return the molecule with calculated data. Distal volume calculation
    must be specified. In order to do octant analysis, `z_axis_atoms` and `xz_plane_atoms` must
    be specified.
    
    Parameters
    ----------
    ml_mol : ml.Molecule
        Molecule to do buried volume calculation
    metal_atom : (ml.Atom | int)
        Metal atom necessary for buried volume calculation. THE NUMBER WILL HAVE 1 ADDED TO IT
        TO MATCH WITH THE 1-INDEXED ATOM LIST OF MORFEUS
    round_coords: bool, optional
        This converts the coordinates to be rounded to 6 floating point values to match
        traditional XYZ format, by default False
    excluded_atoms : Iterable[ml.AtomLike], optional
        Atoms to exclude in buried volume calculation, by default None with metal atom excluded
    radii : (str | ArrayLike1D) , optional
        vdW radii to use. `molli`, `morfeus`, and a separate array can be specified, by default morfeus 
    include_hs : bool, optional
        Indicates if hydrogens will be included, by default False
    radius : float, optional
        Radius of the sphere for buried volume calculation, by default 3.5
    radii_type : str, optional
        Types of radii to be used: `alvarez`, `bondi`, `crc` or `truhlar`, by default 'bondi'
    radii_scale : float, optional
        Scaling factor for radii, by default 1.17
    density : float, optional
        Volume per point int the sphere (Å³), by default 0.001
    z_axis_atoms : Iterable[ml.AtomLike], optional
        Atoms for indicating orientation of z axis, by default None
    xz_plane_atoms : Iterable[ml.AtomLike], optional
        Atoms for indicating orientation of xz plane, by default None
    calc_distal_vol: bool, optional
        Calculate the distal volume, by default False,
    dist_method: str, optional
        Method to use for distal volume calculation: `sasa` (solvent accessible surface area) or `buried_volume`, by default'sasa'
    dist_octants: bool, optional
        Allow octant decomposition for distal volume calculation, but requires the `buried_volume` method, by default False
    dist_sasa_density: float
        Density of points on the SASA surface, by default 0.01,
    plot : bool, optional
        Allows plotting of the buried volume results, by default False
    verbose : bool, optional
        Allows printing of final results 

    Returns
    -------
    ml.Molecule
        Updated Molecule object

    Additional Notes
    ----------------
    - This has been tested with the `MoleculeLibrary` `ml.files.cinchonidine_no_conf` to match exactly with morfeus
    only if `round_coords` is used. If `round_coords` is not used, it matches to a relative tolerance of 1e-3 and absolute 
    tolerance  of 1e-5 as defined by the `numpy.isclose` function
    - The following attributes are created assuming distal volume calculation and octant analysis under the attrib `BuriedVolume`:
        - percent_buried_volume
        - buried_volume
        - free volume
        - distal_volume 
            - Must specify `calc_distal_vol=True`
        - molecular_volume 
            - Must specify `calc_distal_vol=True`
        - Octants 
            - dict of V_bur calculation for octants
            - Must specify `z_axis_atoms` and `xz_plane_atoms`
        - Quadrants 
            - dict of V_bur calculation for octants
            - Must specify `z_axis_atoms` and `xz_plane_atoms`

    Relevant References
    -------------------
    - Similar Buried Volume Algorithm: 
        - ***DOI***: 10.1021/acs.organomet.6b00371 
    '''
    
    elements = [ml.Element.H if a.element == ml.Element.Unknown else a.element for a in ml_mol.atoms]

    #Allows matching of the XYZ file format if necessary
    if round_coords:
        coordinates = np.vectorize(lambda x: float(f"{x:.6f}"))(ml_mol.coords)
    else:
        coordinates = ml_mol.coords



    #morfeus starts counting from 1 instead of 0
    metal_index = ml_mol.get_atom_index(metal_atom) + 1

    exc_idx = None
    z_idx = None
    xz_idx = None

    #morfeus starts counting from 1 instead of 0
    if excluded_atoms is not None:
        exc_idx = [ml_mol.get_atom_index(x) + 1 for x in excluded_atoms]

    if z_axis_atoms is not None:
        z_idx = [ml_mol.get_atom_index(x) + 1 for x in z_axis_atoms]

    if xz_plane_atoms is not None:
        xz_idx = [ml_mol.get_atom_index(x) + 1 for x in xz_plane_atoms]

    #Chooses the dictionary utilized for calculating vdw radius
    match radii:
        case 'morfeus':
            radii = None
        case 'molli':
            radii = [a.vdw_radius for a in ml_mol.atoms]
        case _:
            print('radii not `molli` or `morfeus`, interpreting as 1D array')
    
    #Calculates Buried Volume
    bv = BuriedVolume(
        elements=elements,
        coordinates=coordinates,
        metal_index=metal_index,
        excluded_atoms= exc_idx,
        radii=radii,
        include_hs=include_hs,
        radius=radius,
        radii_type=radii_type,
        radii_scale=radii_scale,
        density=density,
        z_axis_atoms=z_idx,
        xz_plane_atoms=xz_idx
    )
    
    #Basic Buried Volume Descriptors
    res = {
        'percent_buried_volume': bv.fraction_buried_volume,
        'buried_volume': bv.buried_volume,
        'free_volume': bv.free_volume,
    }

    #Allows independent calculation of the distal volume without octant decomposition
    if calc_distal_vol and not dist_octants:
        bv.compute_distal_volume(method=dist_method,octants=dist_octants,sasa_density=dist_sasa_density)
        res['distal_volume'] = bv.distal_volume
        res['molecular_volume'] = bv.molecular_volume
    

    #Adds octant and quadrant analysis
    if (z_axis_atoms is not None) and (xz_plane_atoms is not None):
        bv.octant_analysis()

        #Allows distal volume calculation with octant decomposition (will alter the output to quadrants)
        if calc_distal_vol and dist_octants:
            print('bad')
            bv.compute_distal_volume(method=dist_method,octants=dist_octants,sasa_density=dist_sasa_density)
            res['distal_volume'] = bv.distal_volume
            res['molecular_volume'] = bv.molecular_volume
        
        res['Octants'] = bv.octants
        res['Quadrants'] = bv.quadrants
        
    
    if plot:
        bv.plot_steric_map()

    ml_mol.attrib['BuriedVolume'] = res

    if verbose:
        print(f'The following attributes were added to {ml_mol} under the attribute name `BuriedVolume`')
        pprint(ml_mol.attrib['BuriedVolume'])

    return ml_mol

def cone_angle(
    ml_mol: ml.Molecule, 
    atom: (ml.Atom | int),
    round_coords: bool = False,
    radii: (str | ArrayLike1D) = "morfeus",
    radii_type: str = 'crc',
    method: str = 'libconeangle',
    verbose: bool = False,
    plot: bool = False,
    show_cone_atom: bool = True,
    cone_color: str = "steelblue",
    cone_opacity: float = 0.3,
    vdw_opacity: float= 0.3
    ) -> ml.Molecule:
    '''Cone Angle calculation as done with morfeus. This will store cone angle
    calculation data and return the molecule with calculated data. Visualization
    requires the installation of pyvista. The `libconeangle` must be installed to use
    the method.

    Parameters
    ----------
    ml_mol : ml.Molecule
        Molecule to do cone angle calculation on
    atom : ml.Atom  |  int
        Cone atom for cone angle calculation
    round_coords : bool, optional
        This converts the coordinates to be rounded to 6 floating point values to match
        traditional XYZ format, by default False
    radii : str  |  ArrayLike1D, optional
        vdW radii to use. `molli`, `morfeus`, and a separate array can be specified, by default morfeus 
    radii_type : str, optional
        Types of radii to be used: `alvarez`, `bondi`, `crc` or `truhlar`, by default 'bondi'
    method : str, optional
        Method of calculation: `internal` or `libconeangle, by default 'libconeangle'
    verbose : bool, optional
        Allows printing of final results, by default False
    plot : bool, optional
        Allows plotting of the cone created for the molecule, by default True
    show_cone_atom : bool, optional
        Specifies if the atom used to calculate the angle is shown, by default True
    cone_color : str, optional
        Color fo the plotted cone, by default "steelblue"
    cone_opacity : float, optional
        Opacity of the cone, by default 0.3
    vdw_opacity : float, optional
        Opacity of the van der Waals visualization, by default 0.3

    Returns
    -------
    ml.Molecule
        Molecule with updated attributes

    Additional Notes
    ----------------
    - This has been tested with the a `MoleculeLibrary` created from an MMFF94 optimized `ml.files.buch_phos_cdxml` with a Ni atom
    appended to it. `round_coords` allows less floats to be utilized and make the calculation a bit faster.
    - The following attributes are added to the Molecule under the attrib `ConeAngle`:
        - ConeAngle
        - TangentAtoms
            - This is a list of numbers
        - NormalVec
            - This is the normal vector used to define the cone.

    Relevant References
    -------------------
    - Exact Cone Angle and Algorithm: 
        - ***DOI***: 10.1002/jcc.23217
    - Tolman Cone Angle: 
        - ***DOI***: 10.1021/cr60307a002
    '''


    elements = [ml.Element.H if a.element == ml.Element.Unknown else a.element for a in ml_mol.atoms]

    #Allows matching of the XYZ file format if necessary
    if round_coords:
        coordinates = np.vectorize(lambda x: float(f"{x:.6f}"))(ml_mol.coords)
    else:
        coordinates = ml_mol.coords

    #morfeus starts counting from 1 instead of 0
    atom_idx = ml_mol.get_atom_index(atom) + 1

    #Chooses the dictionary utilized for calculating vdw radius
    match radii:
        case 'morfeus':
            radii = None
        case 'molli':
            radii = [a.vdw_radius for a in ml_mol.atoms]
        case _:
            print('radii not `molli` or `morfeus`, interpreting as 1D array')

    ca = ConeAngle(
        elements=elements,
        coordinates=coordinates,
        atom_1=atom_idx,
        radii=radii,
        radii_type=radii_type,
        method=method
    )

    res = {
        "ConeAngle":ca.cone_angle,
        "TangentAtoms":ca.tangent_atoms,
        "NormalVec":ca._cone.normal
        }

    ml_mol.attrib['ConeAngle'] = res

    if verbose:
        print(f'The following attributes were added to {ml_mol} under the attribute name `BuriedVolume`')
        pprint(ml_mol.attrib['BuriedVolume'])

    if plot:
        if not is_package_installed("pyvista"):
            raise ImportError("pyvista is not installed in this environment to enable visualization")

        draw_3D_cone(
            cone_angle=ca,
            mlmol=ml_mol,
            cone_atom=atom, 
            show_cone_atom=show_cone_atom,
            cone_color=cone_color,
            cone_opacity=cone_opacity,
            vdw_opacity=vdw_opacity)
    
    return ml_mol

def sterimol(
    ml_mol: ml.Molecule, 
    dummy_atom: (ml.Atom | int),
    attached_atom: (ml.Atom | int),
    round_coords: bool = False,
    radii: (str | ArrayLike1D) = "morfeus",
    radii_type: str = 'crc',
    n_rot_vectors=3600,
    excluded_atoms: Iterable[ml.Atom] = None,
    calc_buried: bool = False,
    bury_radius: float = 5.5,
    bury_method: str = 'delete',
    bury_radii_scale: float = 0.5,
    bury_density: float = 0.01,
    verbose: bool = False
) -> ml.Molecule:
    '''Sterimol calculation as done with morfeus. This will store sterimol
    calculation data and return the molecule with calculated data.

    Parameters
    ----------
    ml_mol : ml.Molecule
        Molecule to do sterimol calculation on
    dummy_atom : ml.Atom  |  int
        Dummy atom for sterimol calculation
    attached_atom : ml.Atom  |  int
        Atom attached to full structure for sterimol calculation
    round_coords : bool, optional
        This converts the coordinates to be rounded to 6 floating point values to match
        traditional XYZ format, by default False
    radii : str  |  ArrayLike1D, optional
        vdW radii to use. `molli`, `morfeus`, and a separate array can be specified, by default morfeus 
    radii_type : str, optional
        Types of radii to be used: `alvarez`, `bondi`, `crc` or `truhlar`, by default 'bondi'
    n_rot_vectors : int, optional
        Number of rotational vectors for determining B1 and B5, by default 3600
    excluded_atoms : Iterable[ml.Atom], optional
        Atoms to exclude in buried volume calculation, by default None with metal atom excluded
    calc_buried : bool, optional
        Do a buried sterimol calculation on top of the original sterimol, by default False
    bury_radius : float, optional
        Radius of the sphere for the buried Sterimol calculation, by default 5.5
    bury_method : str, optional
        Method for burying: `delete`, `slice`, or `truncate` available, by default 'delete'
    bury_radii_scale : float, optional
        Scaling factor for radii with the `delete` method calculation, by default 0.5
    bury_density : float, optional
        Area per point on the surface of (Å²), by default 0.01
    verbose : bool, optional
        Allows printing of final results, by default False

    Returns
    -------
    ml.Molecule
        Updated Molecule object

    Additional Notes
    ----------------
    - This has been tested with the a `MoleculeLibrary` of structures created from `ml.files.BOX_4_position` to match with morfeus
    if `round_coords` are used. 
    - The following attributes are created under the attrib `Sterimol` assuming the buried sterimol calculation is done :
        - B_1
        - B_1_Vec
        - B_5
        - B_5_Vec 
        - L 
        - L_uncorr 
        - d(a1-a2) 
            - represents bond length between a1 and a2
        - B_1_Bury
        - B_1_Vec_Bury
        - B_5_Bury
        - B_5_Vec_Bury
        - L_Bury
        - L_uncorr_Bury
    
    Relevant References
    -------------------
    - Sterimol: 
        - ***DOI***: 10.1016/B978-0-12-060307-7.50010-9
        - ***DOI***: 10.1016/B978-0-08-029222-9.50051-2
    - Buried Sterimol: 
        - ***DOI***: 10.1021/jacs.1c09718
    '''

    #Converts any element to hydrogen if it is unknown
    elements = [ml.Element.H if a.element == ml.Element.Unknown else a.element for a in ml_mol.atoms]

    #Allows matching of the XYZ file format if necessary
    if round_coords:
        coordinates = np.vectorize(lambda x: float(f"{x:.6f}"))(ml_mol.coords)
    else:
        coordinates = ml_mol.coords

    #Chooses the dictionary utilized for calculating vdw radius
    match radii:
        case 'morfeus':
            radii = None
        case 'molli':
            radii = [a.vdw_radius for a in ml_mol.atoms]
        case _:
            print('radii not `molli` or `morfeus`, interpreting as 1D array')

    #morfeus starts counting from 1 instead of 0
    dummy_idx = ml_mol.get_atom_index(dummy_atom) + 1
    attached_idx = ml_mol.get_atom_index(attached_atom) + 1

    exc_idx = None
    if excluded_atoms is not None:
        exc_idx = [ml_mol.get_atom_index(x) + 1 for x in excluded_atoms]

    ster = Sterimol(
    elements=elements, 
    coordinates=coordinates, 
    dummy_index=dummy_idx, 
    attached_index=attached_idx,
    radii=radii,
    radii_type=radii_type,
    n_rot_vectors=n_rot_vectors,
    excluded_atoms=exc_idx,
    calculate=True)

    res = {
            'B_1': ster.B_1_value,
            'B_1_Vec': ster.B_1,
            'B_5': ster.B_5_value,
            'B_5_Vec': ster.B_5,
            'L': ster.L_value,
            'L_Vec':ster.L,
            'L_uncorr':ster.L_value_uncorrected,
            'd(a1-a2)':ster.bond_length,
        }
    
    if calc_buried:
        ster.bury(
            sphere_radius=bury_radius,
            method=bury_method,
            radii_scale=bury_radii_scale,
            density=bury_density
        )
        res['B_1_Bury'] = ster.B_1_value
        res['B_1_Vec_Bury'] = ster.B_1
        res['B_5_Bury'] = ster.B_5_value
        res['B_5_Vec_Bury'] = ster.B_5
        res['L_Bury'] = ster.L_value
        res['L_Vec_Bury'] = ster.L
        res['L_uncorr_Bury'] = ster.L_value_uncorrected

    ml_mol.attrib['Sterimol'] = res

    if verbose:
        print(f'The following attributes were added to {ml_mol} under the attribute name `Sterimol`')
        pprint(ml_mol.attrib['Sterimol'])
    
    return ml_mol