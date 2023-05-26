import molli as ml  # from Ethan
from molli.external import _rdkit
from openbabel import openbabel as ob
import openbabel.pybel as pb
import time
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdmolfiles   # from Ethan
from rdkit.Chem.PropertyMol import PropertyMol # from Ethan
from rdkit.Chem import AllChem
from rdkit.Chem.AllChem import AlignMol, EmbedMolecule, EmbedMultipleConfs
from rdkit.Chem.rdForceFieldHelpers import UFFGetMoleculeForceField
# from rdkit.Chem import Draw,PyMol
# from rdkit.Chem.Draw import IPythonConsole
import random
import os
from tqdm import tqdm
from multiprocessing import Pool
import shutil

        #in_dir = '../main_library/in_silico_library_uff_min_checked1/'
in_dir = 'in_combinatorial/combinatorial_lib.mlib'

out_dir = 'out_conformers1/'           #'out1/out_conformers1_separated/'
        #out_dir_mxyz = 'out1/out_conformers1_mxyz/'
        #reference_structure_file = 'alignment_core.mol'


catalysts_used_in_modelling = [
  '6_1_1_1', '1_1_1_1', 'aa_1', '5_1_1_2', '6_1_1_21', '1_1_1_3',
  '190_1_1_2', '3_1_1_21', '1_1_1_22', '1_1_1_26', '1_1_1_24', '1_1_1_7',
  '11_1_1_30', '1_1_1_30', '1_1_1_27', '6_1_1_14', '1_1_1_29', '1_1_1_15',
  '1_1_1_20', '1_1_1_18', 'aa_18', '171_2_2_17', '90_1_1_17', 
  '7_1_2_2', '14_1_1_13', '281_4_4_2', '22_4_4_28', '249_4_4_3', 
  '227_2_2_2', '254_2_2_11', '16_3_1_9', '200_3_1_21',
  '73_3_1_29', '14_1_2_14', '56_2_1_1', '7_2_1_2', '1_4_1_2',
  '187_1_4_30','187_1_4_2', '185_2_1_10', '185_1_2_2', '154_1_2_15', '3_1_2_18',
  '250_1_3_12', '252_1_1_8', '225_1_1_13', '225_1_1_2'
]


def get_oxazaline_alignment_atoms(mol):
    """ Function that takes in a free bisoxazoline ligand and identifies and returns the atom indices of the nitrogen in one ring, C2 of that ring,
    the bridging methylene, the C2 of the other ring, and the nitrogen of the other ring, in that order. 

    argument mol: RDKit molecule

    returns: list of atom indices of the nitrogen in one ring, C2 of that ring,
    the bridging methylene, the C2 of the other ring, and the nitrogen of the other ring, in that order. 

    """
    # find all the nitrogens, we will use different cases of these to identify the structure
    nitrogens = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == 'N']
    # print(len(nitrogens))
    first_N, second_N, c2, bridge, other_c2 = None, None, None, None, None
    oxazoline_Ns = []

    # the standard case, when the only two nitrogens in a molecule are the oxazolines
    if len(nitrogens) == 2:
        oxazoline_Ns = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == 'N']
        
    # the case when the bridging group is nitrile
    elif len(nitrogens) == 3:
        # print('here')
        # the nitrile N only has one neighbor
        for atom in nitrogens:
            if len(mol.GetAtomWithIdx(atom).GetNeighbors()) != 1:
                oxazoline_Ns.append(atom)
        

    # the case where we have nitrogens in other parts of the molecule, e.g. pyridine rings
    elif len(nitrogens) == 4:
        oxazoline_Ns = []
        # iterate through nitrogens
        for atom in nitrogens:
            # print('here1')
            neighbors = mol.GetAtomWithIdx(atom).GetNeighbors()
            # iterate through nitrogen neighbors
            for neighbor in neighbors:
                # oxazoline C2 should be SP2
                if str(neighbor.GetHybridization()) == 'SP2':
                    # oxazoline C2 should have exactly 3 neighbors, C bridge, N, and O
                    if sorted([i.GetSymbol() for i in neighbor.GetNeighbors()]) == sorted(['C', 'N', 'O']):
                        # print('here2')
                        oxazoline_Ns.append(atom)
                        break
    
    # the cases where we have oxazoline nitrogens, nitrogens at 4-position, and a nitrile bridging group, and the case
    # with oxazoline nitrogens, nitrogens at 4-position, and bridging group 29 with 2 nitrogens can be treated similarly (5 and 6 total nitrogens)
    elif (len(nitrogens) == 5) or (len(nitrogens) == 6):
        oxazoline_Ns = []
        # iterate through nitrogens
        for atom in nitrogens:
            neighbors = mol.GetAtomWithIdx(atom).GetNeighbors()
            # this is the nitrile, skip it
            if len(neighbors) == 1:
                continue
            # iterate through nitrogen neighbors
            for neighbor in neighbors:
                # oxazoline C2 should be SP2
                if str(neighbor.GetHybridization()) == 'SP2':
                    # oxazoline C2 should have exactly 3 neighbors, C bridge, N, and O
                    if sorted([i.GetSymbol() for i in neighbor.GetNeighbors()]) == sorted(['C', 'N', 'O']):
                        # print('here2')
                        oxazoline_Ns.append(atom)
                        break

    else:
        raise Exception( f'{mol.name} failed! nitrogen count: {len(nitrogens)}' )


    assert len(oxazoline_Ns) == 2
    first_N, second_N = oxazoline_Ns
    # get c2 next to first N
    
    # this logic still works for the cbr2 linkers (which are CSp3)
    c2 = [atom.GetIdx() for atom in mol.GetAtomWithIdx(first_N).GetNeighbors() if str(atom.GetHybridization()) == 'SP2'][0]
    other_c2 = [atom.GetIdx() for atom in mol.GetAtomWithIdx(second_N).GetNeighbors() if str(atom.GetHybridization()) == 'SP2'][0]
    # print([i.GetSymbol() for i in mol.GetAtomWithIdx(c2).GetNeighbors()])
    # get bridge
    bridge = set([atom.GetIdx() for atom in mol.GetAtomWithIdx(c2).GetNeighbors()]).intersection(set([atom.GetIdx() for atom in mol.GetAtomWithIdx(other_c2).GetNeighbors()]))
    assert len(bridge) == 1
    bridge = bridge.pop()

    assert second_N in [i.GetIdx() for i in mol.GetAtomWithIdx(other_c2).GetNeighbors()]
    
    return first_N, c2, bridge, other_c2, second_N

""" 
Should only be called through conformer_generation
"""
## see https://www.rdkit.org/UGM/2012/Ebejer_20110926_RDKit_1stUGM.pdf
def _conf_gen(args) -> ml.ConformerEnsemble:
    # separate our arguments. Arguments should always be passed, if not then error in conformer_generation
    reference = args[0] # reference rd_mol object
    in_mol = args[1] # molli Molecule
    ref_alignment_atoms = args[2] # alignment atoms from ref
    n_confs = args[3]
    maxIterations = args[4]
    threshold = args[5] # check for bool or float before energy calculation

# Start by checking the arguments and making sure they're passed correctly
    try: # try to read in the file, calc num rotatable bonds
        mol_dict = _rdkit.create_rdkit_mol(in_mol)
        mol = mol_dict[in_mol]

        Chem.SanitizeMol(mol)
        AllChem.AssignCIPLabels(mol)
        # get the number of rotatable bonds, this will determine how many confs to make
        num_rotatable_bonds = AllChem.CalcNumRotatableBonds(mol)
        if num_rotatable_bonds >=8:
            n_confs *= 3
    except Exception as exp:
        print(f'Failure for {mol} sanitization/CIP labeling/rotatable bond calculation: {exp!s}')
        return
    
    try: # try to embed conformers
        # set some embedding parameters
        ps = AllChem.ETKDGv2()
        ps.maxIterations = maxIterations
        ps.randomSeed = 42
        ps.enforceChirality = True
        ps.useRandomCoords = True
        # add Hs
        mol = Chem.AddHs(mol, addCoords = True)
        # generate n_conformers
        conf_ids = Chem.rdDistGeom.EmbedMultipleConfs(mol, numConfs=n_confs, params=ps)
    except Exception as exp:
        print(f'Failure for {mol} on conformer embedding: {exp!s}')
        return

    try: # try to optimize conformer distribution
        # optimize with MMFF
        AllChem.MMFFSanitizeMolecule(mol)
        AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=1, maxIters=200) # issue here
    except Exception as exp:
        print(f'Failure for {mol} on conformer geometry optimization: {exp!s}')
        return

    try: # try to align conformers with reference
        # align the conformers
        mol_alignment_atoms = list(get_oxazaline_alignment_atoms(mol))
        mol_map = [i for i in zip(mol_alignment_atoms, ref_alignment_atoms)]
        AllChem.AlignMol(mol, reference, atomMap = mol_map) # this line aligns the first conf to the reference mol
        AllChem.AlignMolConformers(mol ,atomIds = mol_alignment_atoms) # this line aligns mol confs to first conf
    except Exception as exp:
        print(f'Failure for {mol} on alignment: {exp!s}')
        return

    do_calc = True
    if isinstance(threshold, bool):
        if threshold == False:
            do_calc = False
        else:
            threshold = 15.0

    if do_calc:
        try: # try to calculate energies for conformers
            # sort the conformer list by increasing energy
            energy_list = []
            mol_props = AllChem.MMFFGetMoleculeProperties(mol)
            for cid in list(conf_ids):
                # print(f'minimming conf {cid}')
                ff = AllChem.MMFFGetMoleculeForceField(mol, mol_props, confId = cid) 
                energy_list.append(ff.CalcEnergy())
        except Exception as exp:
            print(f'Failure for {mol} on energy calculation: {exp!s}')
            return

        # sorted conformer energy list
        sorted_conformers = sorted(zip(list(conf_ids), energy_list), key = lambda x: x[1], reverse = False)
        min_energy = min([i[1] for i in sorted_conformers])
        # filter conformers based on energy threshold
        filtered_conformers = [i for i in sorted_conformers if (i[1] - min_energy < threshold)]
    else:
        filtered_conformers = zip(list(conf_ids))

    try: # convert conformers back to mol2 and put into conformer library
#        make_lib = ml.ConformerLibrary.new(out_dir + "_conformers.clib")  
#        make_lib.open(False)
        mol2_string = ''                                                       # initialize string to hold all conformer mol2 data
        for conf in enumerate(filtered_conformers):                            # iterate through all filtered conformer ids
            temp = rdmolfiles.MolToMolBlock(mol, confId=conf[0])               # turn each conformer in mol into temporary molblock string
            ob_temp = pb.readstring('mol', temp)                               # read that string into open_babel molecule
            mol2_string += ob_temp.write(format='mol2')                        # write out ob_temp as mol2 string, append to mol2_string
        ensemble = ml.ConformerEnsemble.loads_mol2(mol2_string)                # create ensemble with string mol2 data
#        clib.append(in_mol.name, ensemble)                                                      
#        make_lib.close()                                                       # conformers are not named, important to fix or no?         
    except Exception as exp:
        print(f'Failure for {mol} on creating conformer ensemble: {exp!s}')
        return

    return ensemble

# takes an mlib, either file path or string, and optional parameters
def conformer_generation(
    mlib: str | Path | ml.MoleculeLibrary,
    n_confs: int = 50,
    maxIterations: int = 10000,
    threshold: float | bool = 15.0,
    file_output: str | Path = None,
    num_processes: int = 100
    ):
    '''
    Generates a ConformerLibrary for a passed MoleculeLibrary.
    MoleculeLibrary can be passed as the object or as its string or Path representation of its filepath.
    Optional parameters for max conformers per molecule, max iterations for conformer embedding, and energy threshold for conformer filtering.
    If threshold is set to False, conformers will not be filtered, if true, will set threshold back to default of 15.0.
    If no out file path is given, ConformerLibrary will be written to the directory that conformer_generation was called from.
    num_processes initializes size of pool of worker processes with the Pool class. Setting num_processes to None defaults to os.cpu_count()
    '''
    try:    # initialize or copy MoleculeLibrary
        if isinstance(mlib, str | Path):
            lib = ml.MoleculeLibrary(mlib)
        else:
            lib = mlib
    except Exception as exp:
        print(f'Invalid MoleculeLibrary: {exp!s}')
        return
    
    ref_mol = lib[0]    
    reference_dict = _rdkit.create_rdkit_mol(ref_mol)
    reference_molecule = reference_dict[ref_mol] 
    alignment_atoms = list(get_oxazaline_alignment_atoms(reference_molecule))

    length = len(lib)
    conf_list = [n_confs] * length
    iter_list = [maxIterations] * length
    thresh_list = [threshold] * length
    #args should be [ref_mol, mol, [alignment_atoms], n_conf, maxIterations, threshold]
    args = [i for i in zip([reference_molecule for i in range(length)], lib, [alignment_atoms for i in range(length)], conf_list, iter_list, thresh_list)]

    try:    # create output file
        if file_output is not None:
            if isinstance(file_output, Path):
                file_output / 'conformers.mlib'
            else:
                file_output += 'conformers.mlib'
        else:
            file_output = 'conformers.mlib'
        clib = ml.ConformerLibrary.new(file_output)
    except Exception as exp:
        print(f'Error with output file creation: {exp!s}')
        return

    ensembles = None    # generate conformer ensembles, append to conformer library
    with Pool(num_processes) as p:
        ensembles = p.map(_conf_gen, args)
    with clib:
        for i in ensembles:
            clib.append(i.name, i)    

# DELETE BELOW IN FINAL VERSION
if __name__ == '__main__':

#    if not os.path.exists(out_dir):
#        os.makedirs(out_dir)
#    if not os.path.exists(out_dir_mxyz):
#        os.makedirs(out_dir_mxyz)
    mlib = ml.MoleculeLibrary(in_dir)
    # it will save us some time to call this only once here

    ref_mol = mlib[0]
    reference_dict = _rdkit.create_rdkit_mol(ref_mol)
    reference_molecule = reference_dict[ref_mol] 
    alignment_atoms = list(get_oxazaline_alignment_atoms(reference_molecule))
    args = [i for i in zip([reference_molecule for i in range(len(mlib))], mlib, [alignment_atoms for i in range(len(mlib))])]

    start = time.time()
#    with Pool(100) as p:                   
#       p.map(pool_handler, args)           
#    for arg in args:
#        conf_generate(arg)
#    ens = conf_generate(args[0])
#    end = time.time()
    print('Time elapsed: ' + str(end - start) + " s")
    print('Success!')
    # job id: 1082045


