#!/bin/env python3

import molli as ml
import molli.visual
import subprocess
import os

# For logging
import sys
import time
import logging

# This is a failsafe in case openbabel aint installed
import openbabel

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

# TODO: Read everything below from envs

# INPUT ARGS: number of cores / concurrent jobs
thread_concurrency = 2
job_concurrency = 10

# INPUT ARGS: clustering parameters
clustering_removed_variance_columns = 0
clustering_cutoff = 0.8

# INPUT FILES: chemdraw files that the user passes in
cores = os.getenv('CORES_INPUT_FILE', ml.files.box_cores_test_1)
subs = os.getenv('SUBS_INPUT_FILE' ml.files.box_substituents_test_1)

# OUTPUT FILES: output directory
out_dir = os.getenv('JOB_OUTPUT_DIR', '/molli/ncsa-testing-output/')

def parse_chemdraw():
    logging.info("=== Parsing ChemDraw Files ===")
    logging.info(f"Using cores: {cores}")
    logging.info(f"Using substituents: {subs}")

    # parse the files
    subprocess.run(['molli', 'parse', '--hadd', f'{cores}', '-o', f'{out_dir}/BOX_cores_new_env.mlib', "--overwrite"])
    subprocess.run(['molli', 'parse', '--hadd', f'{subs}', '-o', f'{out_dir}/BOX_subs_new_env.mlib', "--overwrite"])

    m_core = ml.MoleculeLibrary(f'{out_dir}/BOX_cores_new_env.mlib')
    logging.debug(len(m_core))
    # you can index fragments directly with the string they are lablled with in the chemdraw
    m_core['1']

    m_subs = ml.MoleculeLibrary(f'{out_dir}/BOX_subs_new_env.mlib')
    logging.debug(len(m_subs))
    m_subs['3']



# Combine step (takes ~2 minutes)
def combinatorial_expansion():
    logging.info("=== Performing Combinatorial Expansion ===")
    subprocess.run(
        [
            'molli',
            'combine',
            f'{out_dir}/BOX_cores_new_env.mlib',
            '-s',
            f'{out_dir}/BOX_subs_new_env.mlib',
            '-j',
            f'{thread_concurrency}', 
            '-o', 
            f'{out_dir}/test_combine_new_env.mlib', 
            '-a', 
            'A1', 
            '--obopt', 
            'uff',
            '-m',
            'same',
            "--overwrite"
        ]
    )


    combined = ml.MoleculeLibrary(f'{out_dir}/test_combine_new_env.mlib')
    logging.debug(len(combined))
    # you index full catalysts structures with the concatenated core_substituent_substituent string
    combined["1_3_3"]

    combined["3_6_6"]

# Conformers step (takes ~4 minutes)
def generate_conformers():
    logging.info("=== Generating Conformers ===")
    subprocess.run(['molli', 
                    'conformers', 
                    f'{out_dir}/test_combine_new_env.mlib', 
                    '-n', 
                    f'{job_concurrency}', 
                    '-o', 
                    f'{out_dir}/test_conformers_new_env.mlib', 
                    '-t', 
                    '-j', ### !!!!!! Number of jobs. Please scale down if host system has fewer cores. defaults to os.cpu_count()//2  !!!!! ###
                    f'{thread_concurrency}',
                    "--overwrite"
                    ])


    clib = ml.ConformerLibrary(f'{out_dir}/test_conformers_new_env.mlib')
    logging.debug(len(clib))

    i = 0
    for conf in clib:
        i += conf.n_conformers
        # logging.debug(conf)
    logging.info(str(i) + ' conformers in library')

    # many of these conformers ar redundant - redundant confs thrown out during aso calculation

    logging.debug(clib[0])

    clib['1_3_3'][0]

    clib['1_3_3'][1]

    clib['3_6_6'][0]

    clib['3_6_6'][24]
    

def aso_descriptor():
    logging.info("=== Generating ASO Descriptor ===")
    # first we make a grid for calculating aso
    subprocess.run(['molli', 
                    'grid', 
                    '--mlib', 
                    f'{out_dir}/test_conformers_new_env.mlib', 
                    '-o', 
                    f'{out_dir}/grid_new_env.npy'
                    ])
    # calculate aso
    subprocess.run(['molli', 
                    'gbca', 
                    'aso', 
                    f'{out_dir}/test_conformers_new_env.mlib', 
                    '-g', 
                    f'{out_dir}/grid_new_env.npy', 
                    '-o', 
                    f'{out_dir}/aso_new_env.h5'
                    ])
    # tqdm looks messed up

    
def post_processing():
    logging.info('=== Running Post-Processing ===')
    subprocess.run(         # check functionality for plotting and pca
         [                   # should be better way to implement post_processing stuff
             'molli', 
             'cluster', 
             f'{out_dir}/aso_new_env.h5', 
             '-o', 
             f'{out_dir}/new_env_data3', 
             '-v', # variance threshold before doing clustering
             f'{clustering_removed_variance_columns}', # remove 0 variance columns
             '-c', # correlation cutoff before clustering
             f'{clustering_cutoff}', # 0.8 by default
         ]
     )
    
    
def main():
    logging.info('=== Starting Job ===')
    start_time = time.time()

    # create output directory if it doesn't exist
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Run each stage of the workflow
    parse_chemdraw()
    combinatorial_expansion()
    generate_conformers()
    aso_descriptor()
    post_processing()

    duration = time.time() - start_time
    logging.info('=== Job Complete in %s seconds ===' % duration)

if __name__ == "__main__":
    main()
