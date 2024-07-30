# This script is to show the process of timing measurement
import molli as ml
import timeit

N = 3

# ml.aux.assert_molli_version_min("1.0.0b2")

_clib = ml.ConformerLibrary("bpa_test.clib")

with _clib.reading():
    ensembles = {k: v for k, v in _clib.items()}


def _dir_col(path, overwrite=False):
    return ml.storage.Collection(
        path,
        backend=ml.storage.DirCollectionBackend,
        value_decoder=lambda x: ml.ConformerEnsemble.loads_mol2(x.decode()),
        value_encoder=lambda x: ml.ConformerEnsemble.dumps_mol2(x).encode(),
        ext=".mol2",
        readonly=False,
        overwrite=overwrite,
    )


def _zip_col(path, overwrite=False):
    return ml.storage.Collection(
        path,
        backend=ml.storage.ZipCollectionBackend,
        value_decoder=lambda x: ml.ConformerEnsemble.loads_mol2(x.decode()),
        value_encoder=lambda x: ml.ConformerEnsemble.dumps_mol2(x).encode(),
        ext=".mol2",
        readonly=False,
        overwrite=overwrite,
    )


def _tar_col(path, overwrite=False):
    return ml.storage.Collection(
        path,
        backend=ml.storage.TarCollectionBackend,
        value_decoder=lambda x: ml.ConformerEnsemble.loads_mol2(x.decode()),
        value_encoder=lambda x: ml.ConformerEnsemble.dumps_mol2(x).encode(),
        ext=".mol2",
        readonly=False,
        overwrite=overwrite,
    )


def _ukv_col(path, overwrite=False):
    return ml.ConformerLibrary(
        path,
        readonly=False,
        overwrite=overwrite,
    )


# Note: bpa_test_deflate5.zip is not here as you cannot write into the compressed format
for prep, path in (
    (_ukv_col, "bpa_test.clib"),
    (_tar_col, "bpa_test.tar"),
    (_zip_col, "bpa_test.zip"),
    # (_dir_col, "bpa_test"),
):
    clib_write_times = timeit.Timer(
        stmt="""with library.writing():\n    for k, v in ensembles.items(): library[k]=v""",
        setup="""library = prep(path, overwrite=True)""",
        globals=globals(),
    ).repeat(5, number=1)

    print("Writing times", path, min(clib_write_times), clib_write_times, flush=True)

# Note: bpa_test_deflate5.zip is written from the compressed "bpa_test" directory created after the first one
for prep, path in (
    (_ukv_col, "bpa_test.clib"),
    (_tar_col, "bpa_test.tar"),
    (_zip_col, "bpa_test.zip"),
    # (_zip_col, "bpa_test_deflate5.zip"),
    # (_dir_col, "bpa_test"),
):
    clib_read_times = timeit.Timer(
        stmt="""with library.reading():\n    for k, v in library.items(): pass""",
        setup="""library = prep(path, overwrite=False)""",
        globals=globals(),
    ).repeat(5, number=1)

    print("Read times", path, min(clib_read_times), clib_read_times, flush=True)
