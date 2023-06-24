#!/bin/env python3

import molli as ml
import molli.visual
import subprocess
import threading
import os

import json

import logging
import sys
import time

# This is a failsafe in case openbabel aint installed
#import openbabel
from molli.external import openbabel

def obsvg(mol: ml.Molecule, *, display_name: bool = False, alias: bool = True, color_by_element: bool = True, fgcolor: str = "black", bgcolor: str = "white",):
    from openbabel import openbabel as ob
    obmol = openbabel.to_obmol(mol)
    conv = ob.OBConversion()
    conv.SetOutFormat("svg")

    obmol.DeleteHydrogens()

    if not color_by_element:
        conv.AddOption("u", ob.OBConversion.OUTOPTIONS)

    if alias:
        conv.AddOption("A", ob.OBConversion.OUTOPTIONS)

    if not display_name:
        conv.AddOption("d", ob.OBConversion.OUTOPTIONS)

    conv.AddOption("b", ob.OBConversion.OUTOPTIONS, bgcolor)
    conv.AddOption("B", ob.OBConversion.OUTOPTIONS, fgcolor)


    return conv.WriteString(obmol)

# TODO: Read everything below from envs

# INPUT ARGS: number of cores / concurrent jobs
thread_concurrency = 2
job_concurrency = 10

# INPUT ARGS: clustering parameters
clustering_removed_variance_columns = 0
clustering_cutoff = 0.8

# INPUT FILES: chemdraw files that the user passes in
cores = os.getenv('CORES_INPUT_FILE', ml.files.box_cores_test_1)
subs = os.getenv('SUBS_INPUT_FILE', ml.files.box_substituents_test_1)

# OUTPUT FILES: output directory
out_dir = os.getenv('JOB_OUTPUT_DIR', '/molli/ncsa-testing-output/')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"{out_dir}/log"),
        logging.StreamHandler(sys.stdout)
    ]
)

# Pipe for passing subprocess output to logging
class LogPipe(threading.Thread):

    def __init__(self, level):
        """Setup the object with a logger and a loglevel
        and start the thread
        """
        threading.Thread.__init__(self)
        self.daemon = False
        self.level = level
        self.fdRead, self.fdWrite = os.pipe()
        self.pipeReader = os.fdopen(self.fdRead)
        self.start()

    def fileno(self):
        """Return the write file descriptor of the pipe
        """
        return self.fdWrite

    def run(self):
        """Run the thread, logging everything.
        """
        for line in iter(self.pipeReader.readline, ''):
            logging.log(self.level, line.strip('\n'))

        self.pipeReader.close()

    def close(self):
        """Close the write end of the pipe.
        """
        os.close(self.fdWrite)

def parse_chemdraw():
    logging.info("=== Parsing ChemDraw Files ===")

    # parse the files
    logpipe = LogPipe(logging.INFO)
    with subprocess.Popen(['molli', 'parse', '--hadd', f'{cores}', '-o', f'{out_dir}/BOX_cores_new_env.mlib', "--overwrite"], stdout=logpipe, stderr=logpipe) as s:
        logpipe.close()

    logpipe = LogPipe(logging.INFO)
    with subprocess.Popen(['molli', 'parse', '--hadd', f'{subs}', '-o', f'{out_dir}/BOX_subs_new_env.mlib', "--overwrite"], stdout=logpipe, stderr=logpipe) as s:
        logpipe.close()

    m_core = ml.MoleculeLibrary(f'{out_dir}/BOX_cores_new_env.mlib')
    logging.debug(len(m_core))
    # TODO enforce maximum number of cores?

    m_subs = ml.MoleculeLibrary(f'{out_dir}/BOX_subs_new_env.mlib')
    logging.debug(len(m_subs))
    # TODO enforce maximum number of substituents?


# Combine step (takes ~2 minutes)
def combinatorial_expansion():
    logging.info("=== Performing Combinatorial Expansion ===")
    logpipe = LogPipe(logging.INFO)
    with subprocess.Popen(
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
        ],
        stdout=logpipe,
        stderr=logpipe
    ) as s:
        logpipe.close()


    combined = ml.MoleculeLibrary(f'{out_dir}/test_combine_new_env.mlib')
    logging.debug(len(combined))
    with open(f'{out_dir}/test_combine_new_env_library.json', 'w') as f:
        json.dump({item.name: {'mol2': item.dumps_mol2(), 'svg': obsvg(item)} for item in combined}, f)

# Conformers step (takes ~4 minutes)
def generate_conformers():
    logging.info("=== Generating Conformers ===")
    logpipe = LogPipe(logging.INFO)
    with subprocess.Popen(['molli', 
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
                    ], stdout=logpipe, stderr=logpipe) as s:
         logpipe.close()


    clib = ml.ConformerLibrary(f'{out_dir}/test_conformers_new_env.mlib')
    logging.debug(len(clib))

    i = 0
    for conf in clib:
        i += conf.n_conformers
        # logging.debug(conf)
    logging.info(str(i) + ' conformers in library')

    # many of these conformers ar redundant - redundant confs thrown out during aso calculation

    logging.debug(clib[0])

    #clib['1_3_3'][0]
    #clib['1_3_3'][1]
    #clib['3_6_6'][0]
    #clib['3_6_6'][24]


def aso_descriptor():
    logging.info("=== Generating ASO Descriptor ===")
    # first we make a grid for calculating aso
    logpipe = LogPipe(logging.INFO)
    with subprocess.Popen(['molli', 
                    'grid', 
                    '--mlib', 
                    f'{out_dir}/test_conformers_new_env.mlib', 
                    '-o', 
                    f'{out_dir}/grid_new_env.npy'
                    ], stdout=logpipe, stderr=logpipe) as s:
        logpipe.close()
    # calculate aso
    logpipe = LogPipe(logging.INFO)
    with subprocess.Popen(['molli', 
                    'gbca', 
                    'aso', 
                    f'{out_dir}/test_conformers_new_env.mlib', 
                    '-g', 
                    f'{out_dir}/grid_new_env.npy', 
                    '-o', 
                    f'{out_dir}/aso_new_env.h5'
                    ], stdout=logpipe, stderr=logpipe) as s:
        logpipe.close()
    # tqdm looks messed up


def post_processing():
    logging.info('=== Running Post-Processing ===')
    logpipe = LogPipe(logging.INFO)
    with subprocess.Popen(         # check functionality for plotting and pca
         [                   # should be better way to implement post_processing stuff
             'molli',
             'cluster',
             f'{out_dir}/aso_new_env.h5',
             '-m',
             'tsne',
             '-o',
             f'{out_dir}/new_env_data3_tsne',
             '-v', # variance threshold before doing clustering
             f'{clustering_removed_variance_columns}', # remove 0 variance columns
             '-c', # correlation cutoff before clustering
             f'{clustering_cutoff}', # 0.8 by default
         ], stdout=logpipe, stderr=logpipe
     ) as s:
        logpipe.close()

    logpipe = LogPipe(logging.INFO)
    with subprocess.Popen(         # check functionality for plotting and pca
         [                   # should be better way to implement post_processing stuff
             'molli',
             'cluster',
             f'{out_dir}/aso_new_env.h5',
             '-m',
             'pca',
             '-o',
             f'{out_dir}/new_env_data3_pca',
             '-v', # variance threshold before doing clustering
             f'{clustering_removed_variance_columns}', # remove 0 variance columns
             '-c', # correlation cutoff before clustering
             f'{clustering_cutoff}', # 0.8 by default
         ], stdout=logpipe, stderr=logpipe
     ) as s:
         logpipe.close()


def main():
    logging.info('=== Starting Job ===')
    start_time = time.time()

    logging.info(f"INPUT cores: {cores}")
    logging.info(f"INPUT substituents: {subs}")
    logging.info(f"OUTPUT folder: {out_dir}")

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
