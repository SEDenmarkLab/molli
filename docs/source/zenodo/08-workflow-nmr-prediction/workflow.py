# This is a separate file just so that we do not run into funny issues with Jupyter and ThreadPoolExecutor problems.
import molli as ml
from molli.pipeline.crest import CrestDriver
from molli.pipeline.orca import ORCADriver

source = ml.MoleculeLibrary("cladosporin_g1.mlib", readonly=True)
confs_gff = ml.ConformerLibrary("cladosporin_g2.clib", readonly=False)
confs_gfn2 = ml.ConformerLibrary("cladosporin_g3.clib", readonly=False)

crest = CrestDriver(
    "crest",
    nprocs=32,
    check_exe=False,  # this is only to repeat the workflows
    find=False,  # This is only to repeat the workflows
)
print("Running conformer generation with:", crest.executable, crest.nprocs)
print("debug: nprocs", crest.conformer_search.nprocs)

print("=" * 100)

ml.pipeline.jobmap(
    crest.conformer_search,
    source=source,
    destination=confs_gff,
    cache_dir="_01_conf_search",
    n_workers=4,
    kwargs={"method": "gfnff -i4"},
    strict_hash=False,  # Not recommended unless you know precisely that outputs are correct.
)

print("=" * 100)

ml.pipeline.jobmap(
    crest.conformer_screen,
    source=confs_gff,
    destination=confs_gfn2,
    cache_dir="_02_conf_screen",
    n_workers=4,
    kwargs={"method": "gfn2"},
    strict_hash=False,  # Not recommended unless you know precisely that outputs are correct.
)


source = ml.ConformerLibrary("cladosporin_g3.clib", readonly=True)

orca_optimized = ml.ConformerLibrary(
    "cladosporin_g4.clib",
    readonly=False,
    overwrite=True,
)

orca = ORCADriver("orca", nprocs=8, memory=4000 * 8, envars={})

print("=" * 100)

ml.pipeline.jobmap(
    orca.optimize_ens,
    source=source,
    cache_dir="_03_dft_geoms",
    destination=orca_optimized,
    kwargs={
        "keywords": "rks b97-3c opt freq tightopt tightscf miniprint",
        # "orca_suffix": "-mca btl ^sm",
    },
    n_workers=128 // 8,
    verbose=True,
    log_level="debug",
    strict_hash=False,
)

source = ml.ConformerLibrary("cladosporin_g4.clib", readonly=True)
orca_conf_nmr = ml.ConformerLibrary(
    "cladosporin_g4_nmr.clib",
    readonly=False,
    overwrite=True,
)

print("=" * 100)

ml.pipeline.jobmap(
    orca.giao_nmr_ens,
    source=source,
    destination=orca_conf_nmr,
    cache_dir="_04_dft_nmr",
    kwargs={
        "keywords": "rks pbe0 pcSseg-2 verytightscf nmr cpcm(chloroform)",
        "elements": ("C",),
    },
    verbose=True,
    log_level="debug",
    strict_hash=False,
    n_workers=128 // 8,
)
