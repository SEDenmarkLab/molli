import molli as ml
from tqdm import tqdm
import os
import subprocess
from multiprocessing import Pool

# file of labelled cores, with bridging group in place (2d), attachment points labelled, A1: R4, A2: R5 syn, A3: R2 anti.
cf_cores_with_bridge = ml.ftypes.CDXMLFile('chemdraws/cores-with-bridge-linker-fragments.cdxml')
# 4 position fragments
cf_4position = ml.ftypes.CDXMLFile('chemdraws/library_R4_ap.cdxml')
# 5-position fragments
cf_5position = ml.ftypes.CDXMLFile('chemdraws/library_R5_ap.cdxml')
# indbox structures - these were drawn manually, see chemdraw
indbox_cores = ml.ftypes.CDXMLFile('chemdraws/indbox_core.cdxml')

# this is where we save the files
out_mol2_dir = 'in_silico_library/'
out_dir_uff_min = 'in_silico_library_uff_min/'

def make_combinatorial_library():
    # 5,5' position fragments
    hydro = cf_5position['1']
    methyl = cf_5position['2']
    ipr = cf_5position['3']
    ph = cf_5position['4']

    print('Making combinatorial library...\n')

    for item in tqdm(cf_cores_with_bridge.keys()):

        core = cf_cores_with_bridge[item]

        for subs in cf_4position.keys():

            pos4 = cf_4position[subs]

            # make core structure - A1 are the attachment points for 4 position
            m1 = ml.Molecule.join(core, pos4, 'A1', 'AP1', optimize_rotation=True)
            core_with_bridge = ml.Molecule.join(m1, pos4, 'A1', 'AP1', optimize_rotation=True)


            # # make 5,5'-dihydro
            m3 = ml.Molecule.join(core_with_bridge, hydro, 'A2', 'AP0', optimize_rotation=True)
            m4 = ml.Molecule.join(m3, hydro, 'A2', 'AP0', optimize_rotation=True)
            m5 = ml.Molecule.join(m4, hydro, 'A3', 'AP0', optimize_rotation=True)
            m6 = ml.Molecule.join(m5, hydro, 'A3', 'AP0', optimize_rotation=True)
            m6.add_implicit_hydrogens() # add hydrogens
            # save file - naming convention: "R4_syn5_anti5_bridge"
            with open(f'{out_mol2_dir}{pos4.name}_{hydro.name}_{hydro.name}_{core.name}.mol2', 'w') as o:
                m6.dump_mol2(o)

            # # make 5,5'-dimethyl, save
            m3 = ml.Molecule.join(core_with_bridge, methyl, 'A2', 'AP0', optimize_rotation=True)
            m4 = ml.Molecule.join(m3, methyl, 'A2', 'AP0', optimize_rotation=True)
            m5 = ml.Molecule.join(m4, methyl, 'A3', 'AP0', optimize_rotation=True)
            m6 = ml.Molecule.join(m5, methyl, 'A3', 'AP0', optimize_rotation=True)
            m6.add_implicit_hydrogens() # add hydrogens
            # save file - naming convention: "R4_syn5_anti5_bridge"
            with open(f'{out_mol2_dir}{pos4.name}_{methyl.name}_{methyl.name}_{core.name}.mol2', 'w') as o:
                m6.dump_mol2(o)

            # make diphenyl, save
            m3 = ml.Molecule.join(core_with_bridge, ph, 'A2', 'AP0', optimize_rotation=True)
            m4 = ml.Molecule.join(m3, ph, 'A2', 'AP0', optimize_rotation=True)
            m5 = ml.Molecule.join(m4, ph, 'A3', 'AP0', optimize_rotation=True)
            m6 = ml.Molecule.join(m5, ph, 'A3', 'AP0', optimize_rotation=True)
            m6.add_implicit_hydrogens() # add hydrogens
            # save file - naming convention: "R4_syn5_anti5_bridge"
            with open(f'{out_mol2_dir}{pos4.name}_{ph.name}_{ph.name}_{core.name}.mol2', 'w') as o:
                m6.dump_mol2(o)

            # make syn methyl, save
            m3 = ml.Molecule.join(core_with_bridge, methyl, 'A2', 'AP0', optimize_rotation=True)
            m4 = ml.Molecule.join(m3, hydro, 'A3', 'AP0', optimize_rotation=True)
            m5 = ml.Molecule.join(m4, methyl, 'A2', 'AP0', optimize_rotation=True)
            m6 = ml.Molecule.join(m5, hydro, 'A3', 'AP0', optimize_rotation=True)
            m6.add_implicit_hydrogens() # add hydrogens
            # save file - naming convention: "R4_syn5_anti5_bridge"
            with open(f'{out_mol2_dir}{pos4.name}_{methyl.name}_{hydro.name}_{core.name}.mol2', 'w') as o:
                m6.dump_mol2(o)

            # make anti methyl, save
            m3 = ml.Molecule.join(core_with_bridge, hydro, 'A2', 'AP0', optimize_rotation=True)
            m4 = ml.Molecule.join(m3, methyl, 'A3', 'AP0', optimize_rotation=True)
            m5 = ml.Molecule.join(m4, hydro, 'A2', 'AP0', optimize_rotation=True)
            m6 = ml.Molecule.join(m5, methyl, 'A3', 'AP0', optimize_rotation=True)
            m6.add_implicit_hydrogens() # add hydrogens
            # save file - naming convention: "R4_syn5_anti5_bridge"
            with open(f'{out_mol2_dir}{pos4.name}_{hydro.name}_{methyl.name}_{core.name}.mol2', 'w') as o:
                m6.dump_mol2(o)

            # make syn ipr, save
            m3 = ml.Molecule.join(core_with_bridge, ipr, 'A2', 'AP0', optimize_rotation=True)
            m4 = ml.Molecule.join(m3, hydro, 'A3', 'AP0', optimize_rotation=True)
            m5 = ml.Molecule.join(m4, ipr, 'A2', 'AP0', optimize_rotation=True)
            m6 = ml.Molecule.join(m5, hydro, 'A3', 'AP0', optimize_rotation=True)
            m6.add_implicit_hydrogens() # add hydrogens
            # save file - naming convention: "R4_syn5_anti5_bridge"
            with open(f'{out_mol2_dir}{pos4.name}_{ipr.name}_{hydro.name}_{core.name}.mol2', 'w') as o:
                m6.dump_mol2(o)

            # make anti ipr, save
            m3 = ml.Molecule.join(core_with_bridge, hydro, 'A2', 'AP0', optimize_rotation=True)
            m4 = ml.Molecule.join(m3, ipr, 'A3', 'AP0', optimize_rotation=True)
            m5 = ml.Molecule.join(m4, hydro, 'A2', 'AP0', optimize_rotation=True)
            m6 = ml.Molecule.join(m5, ipr, 'A3', 'AP0', optimize_rotation=True)
            m6.add_implicit_hydrogens() # add hydrogens
            # save file - naming convention: "R4_syn5_anti5_bridge"
            with open(f'{out_mol2_dir}{pos4.name}_{hydro.name}_{ipr.name}_{core.name}.mol2', 'w') as o:
                m6.dump_mol2(o)

            # # make syn ph, save
            m3 = ml.Molecule.join(core_with_bridge, ph, 'A2', 'AP0', optimize_rotation=True)
            m4 = ml.Molecule.join(m3, hydro, 'A3', 'AP0', optimize_rotation=True)
            m5 = ml.Molecule.join(m4, ph, 'A2', 'AP0', optimize_rotation=True)
            m6 = ml.Molecule.join(m5, hydro, 'A3', 'AP0', optimize_rotation=True)
            m6.add_implicit_hydrogens() # add hydrogens
            # save file - naming convention: "R4_syn5_anti5_bridge"
            with open(f'{out_mol2_dir}{pos4.name}_{ph.name}_{hydro.name}_{core.name}.mol2', 'w') as o:
                m6.dump_mol2(o)

            # make anti ph
            m3 = ml.Molecule.join(core_with_bridge, hydro, 'A2', 'AP0', optimize_rotation=True)
            m4 = ml.Molecule.join(m3, ph, 'A3', 'AP0', optimize_rotation=True)
            m5 = ml.Molecule.join(m4, hydro, 'A2', 'AP0', optimize_rotation=True)
            m6 = ml.Molecule.join(m5, ph, 'A3', 'AP0', optimize_rotation=True)
            m6.add_implicit_hydrogens() # add hydrogens
            # save file - naming convention: "R4_syn5_anti5_bridge"
            with open(f'{out_mol2_dir}{pos4.name}_{hydro.name}_{ph.name}_{core.name}.mol2', 'w') as o:
                m6.dump_mol2(o)
    return 1


def make_indbox_library():

    print('Making indbox library...\n')

    # these are drawn explicitly in the chemdraw, so just parse and save
    for item in tqdm(indbox_cores.keys()):

        core = indbox_cores[item]

        with open(f'{out_mol2_dir}{core.name}.mol2', 'w') as o:
                core.add_implicit_hydrogens()
                core.dump_mol2(o)
    return 1

# we will use pool to make this faster
def minimize_obabel(pool_size = 64):

    print('Minimizing library...')
    args = os.listdir(out_mol2_dir)
        
    with Pool(pool_size) as p:
        p.map(pool_handler_minimize_obabel, args)
    
    return 1

# the pool handler uses obabel command line for speed
def pool_handler_minimize_obabel(arg):
    if arg in os.listdir(out_dir_uff_min):
        return
    else:
        print(f'processing file {arg}...')
        # usinf uff force field
        with open(out_dir_uff_min + arg, 'w') as o:
            subprocess.run(['obminimize', '-ff', 'uff', '-o', 'mol2', f'{out_mol2_dir}{arg}'], stdout= o) 
        return 
    


if __name__ == '__main__':

    # make directories if we need them
    if not os.path.exists(out_mol2_dir):
        os.makedirs(out_mol2_dir)
    if not os.path.exists(out_dir_uff_min):
        os.makedirs(out_dir_uff_min)


    if make_combinatorial_library() == 1:
        print("Successfully made main combinatorial library!")

    if make_indbox_library() == 1:
        print("Successfully made indbox library!")

    if minimize_obabel(pool_size=128) == 1:
        print("Successfully minimized combinatorial library!")



    print('Successfully made and minimized in silico library!')
            