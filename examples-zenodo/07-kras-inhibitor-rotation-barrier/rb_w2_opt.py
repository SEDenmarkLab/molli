# Both the eq and the TS will be optimized here.
import molli as ml
from molli.pipeline.orca import ORCADriver

N_CORES = 4
TOTAL_CORES = 4  # increase for a more powerful machine

orca = ORCADriver(
    "orca",
    nprocs=N_CORES,
    memory=N_CORES * 4000,
    check_exe=False,  # Uncheck for actual production ready calculations
    find=False,  # Uncheck for actual production ready calculations
)


print("Optimizing the transition structures", flush=True)

source = ml.MoleculeLibrary("rb_ts_g2.mlib", readonly=True)
dest = ml.MoleculeLibrary("rb_ts_g3.mlib", readonly=False)

ml.pipeline.jobmap(
    orca.basic_calc_m,
    source=source,
    destination=dest,
    cache_dir="_orca_ts_opt2",
    n_workers=TOTAL_CORES // N_CORES,
    kwargs={
        "keywords": "rks b97-3c tightopt optts freq miniprint noprintmos",
        "input1": "%geom\n TS_Mode {D 9 2 43 44} end\n Calc_Hess True\n Recalc_Hess 10\nend",
    },
    strict_hash=False,
    log_level="debug",
)
