import molli as ml
import numpy as np
import importlib.util
from typing import Iterable
from pprint import pprint

def is_package_installed(pkg_name):
    return importlib.util.find_spec(pkg_name) is not None

if not is_package_installed("morfeus"):
    raise ImportError("morfeus-ml is not installed in this environment")

'''
This is meant to interface with the morfeus-ml package originally developed
by Kjell Jorner, along with Gabriel dos Passos Gomes, Pascal Friedrich, and Tobias Gensch
Information can be found below:

Github: https://github.com/digital-chemistry-laboratory/morfeus
Documentation: https://digital-chemistry-laboratory.github.io/morfeus/notes.html

Installation:
pip install morfeus-ml
'''

from morfeus import BuriedVolume
from morfeus.typing import ArrayLike1D

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
    
    This utilizes an algorithm similar to:
    - *Organometallics* **2016**, *35*, 2286
    - ***DOI***: 10.1021/acs.organomet.6b00371  
    
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
        Types of radii to be used: `alvarez`, `bondi`, `crc` or `truhlar, by default 'bondi'
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
    - The following attributes are created assuming distal volume calculation and octant analysis is done:
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
    '''
    
    elements = [a.element.symbol for a in ml_mol.atoms]

    #Allows matching of the XYZ file format if necessary
    if round_coords:
        coordinates = np.vectorize(lambda x: float(f"{x:.6f}"))(ml_mol.coords)
    else:
        coordinates = ml_mol.coords

    #morfeus starts counting from 1 instead of 0
    if isinstance(metal_atom, ml.Atom):
        metal_index = ml_mol.get_atom_index(metal_atom) + 1

    elif isinstance(metal_atom, int):
        metal_index = metal_atom + 1

    else:
        raise ValueError(f'Metal atom not an atom instance or integer! :  {metal_atom}')

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
