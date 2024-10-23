# Both the eq and the TS will be optimized here.
import molli as ml
from molli.pipeline.orca import ORCADriver

N_CORES = 32
TOTAL_CORES = 32  # increase for a more powerful machine

orca = ORCADriver(
    "orca",
    nprocs=N_CORES,
    memory=N_CORES * 4000,
    # check_exe=False,  # Uncheck for actual production ready calculations
    # find=False,  # Uncheck for actual production ready calculations
)


print("=" * 80)
print("Optimizing the equilibrium structures (again)", flush=True)

source = ml.MoleculeLibrary("rb_eq_fix.mlib", readonly=True)
dest = ml.MoleculeLibrary("rb_eq_fix_g1.mlib", readonly=False)

ml.pipeline.jobmap(
    orca.basic_calc_m,
    source=source,
    destination=dest,
    cache_dir="_orca_eq_opt2",
    n_workers=TOTAL_CORES // N_CORES,
    kwargs={"keywords": "rks b97-3c tightopt freq miniprint noprintmos"},
    strict_hash=False,
    log_level="debug",
)
