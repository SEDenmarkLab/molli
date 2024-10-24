import molli as ml
from molli.pipeline.xtb import XTBDriver

source = ml.MoleculeLibrary("kras_inhibitors.mlib")
dest = ml.ConformerLibrary("kras_inhibitors_scan.clib", overwrite=True, readonly=False)

N_CORES = 4
TOTAL_CORES = 4  # increase for a more powerful machine

xtb = XTBDriver(
    "xtb",
    nprocs=N_CORES,
    memory=N_CORES * 1000,
    check_exe=False,  # this is only to repeat the workflows
    find=False,  # This is only to repeat the workflows
)

ml.pipeline.jobmap(
    xtb.scan_dihedral,
    source,
    dest,
    args=(("1", "2", "3", "4"),),
    cache_dir="_xtb_scan",
    kwargs={
        "method": "gfn2",
        "accuracy": 0.2,
        "range_deg": (-15.0, 165.0),
        "n_steps": 36,
    },
    n_workers=TOTAL_CORES // N_CORES,
    log_level="debug",
    strict_hash=False,
)
