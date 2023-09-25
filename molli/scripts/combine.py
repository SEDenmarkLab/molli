"""
Inspect a .mlib file.
"""

from argparse import ArgumentParser
import molli as ml
from molli.external import openbabel
from itertools import (
    permutations,
    combinations_with_replacement,
    combinations,
    repeat,
    chain,
    product,
)
from tqdm import tqdm
import multiprocessing as mp
import os
from math import comb, perm, factorial

OS_NCORES = os.cpu_count() // 2

arg_parser = ArgumentParser(
    "molli combine",
    description="Read a molli library and perform some basic inspections",
)

arg_parser.add_argument(
    "cores",
    help="Library file to inspect",
)

arg_parser.add_argument(
    "-s",
    "--substituents",
    action="store",
    metavar="<substituents.mlib>",
    help="Attach the same substituents to each attachment",
    required=True,
)

arg_parser.add_argument(
    "-m",
    "--mode",
    action="store",
    choices=["same", "permutns", "combns", "combns_repl"],
    default="permutns",
    help="Attach the same substituents to each attachment",
)

arg_parser.add_argument(
    "-a",
    "--attachment_points",
    action="append",
    default=None,
    help="Syntax for attachment point selector.",
)

arg_parser.add_argument(
    "-j",
    "--n_jobs",
    action="store",
    metavar=1,
    default=1,
    type=int,
    help="Number of parallel jobs",
)

arg_parser.add_argument(
    "-b",
    "--batch_size",
    action="store",
    metavar=1,
    default=None,
    type=int,
    help="Number of molecules to be processed at a time on a single core",
)

arg_parser.add_argument(
    "-o",
    "--output",
    action="store",
    metavar="<combined.mlib>",
    required=True,
    help="Attach the same substituents to each attachment",
)

arg_parser.add_argument(
    "-sep",
    "--separator",
    action="store",
    default="_",
    help="Name separator",
)

arg_parser.add_argument(
    "--hadd",
    action="store_true",
    help=(
        "Add implicit hydrogen atoms wherever possible. By default this only affects elements in"
        " groups 13-17."
    ),
)

arg_parser.add_argument(
    "--obopt",
    nargs="*",
    metavar="ff maxiter tol disp",
    default=None,
    help=(
        "Perform openbabel optimization on the fly. This accepts up to 4 arguments. Arg 1: the"
        " forcefield (uff/mmff94/gaff/ghemical). Arg 2: is the max number of steps (default=500)."
        " Arg 3: energy convergence criterion (default=1e-4) Arg 4: geometry displacement"
        " (default=False) but values ~0.01-0.1 can help escape planarity."
    ),
)


arg_parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Overwrite the target files if they exist (default is false)",
    default=False,
)


def _ml_init(_progress, _parsed, _outputs: mp.Queue):
    """
    This function is used to initialize workers.
    """
    global progress, library, parsed
    progress, parsed = _progress, _parsed

    pid = mp.current_process().pid
    calc_name = str(parsed.output).removesuffix(".mlib")
    library = ml.MoleculeLibrary.new(f"{calc_name}.{pid}.mlib.tmp")
    _outputs.put(library.path)


def _ml_assemble(core: ml.Molecule, core_aps: tuple[int], substituent_combo: tuple[ml.Molecule]):
    assert len(core_aps) == len(substituent_combo)

    deriv = ml.Molecule(core)
    for i, (ap_i, sub) in enumerate(zip(core_aps, substituent_combo)):
        deriv = ml.Molecule.join(
            deriv, sub, ap_i - i, sub.attachment_points[0], optimize_rotation=True
        )
    deriv.name = parsed.separator.join([core.name] + [sub.name for sub in substituent_combo])

    if parsed.hadd:
        deriv.add_implicit_hydrogens()

    if parsed.obopt is not None:
        match parsed.obopt:
            case []:
                openbabel.obabel_optimize(
                    deriv,
                    inplace=True,
                )

            case [ff]:
                openbabel.obabel_optimize(
                    deriv,
                    ff=ff,
                    inplace=True,
                )

            case [ff, maxiter]:
                openbabel.obabel_optimize(
                    deriv,
                    ff=ff,
                    max_steps=int(maxiter),
                    inplace=True,
                )

            case [ff, maxiter, tol]:
                openbabel.obabel_optimize(
                    deriv,
                    ff=ff,
                    max_steps=int(maxiter),
                    tol=float(tol),
                    inplace=True,
                )

            case [ff, maxiter, tol, disp]:
                openbabel.obabel_optimize(
                    deriv,
                    ff=ff,
                    max_steps=int(maxiter),
                    tol=float(tol),
                    coord_displace=float(disp),
                    inplace=True,
                )

            case _:
                raise ValueError(f"Unsupported arguments for openbabel optimize: {parsed.obopt}")

    with progress.get_lock():
        with library:
            library.append(deriv.name, deriv)
        progress.value += 1


def molli_main(args,  **kwargs):
    parsed = arg_parser.parse_args(args)
    cores: list[ml.Molecule] = ml.MoleculeLibrary(parsed.cores)[:]
    substituents: list[ml.Molecule] = ml.MoleculeLibrary(parsed.substituents)[:]

    # TODO: turn all assertions into more meaningful errors
    assert all(sub.n_attachment_points == 1 for sub in substituents)

    ap_indices = [
        [
            core.index_atom(a)
            for lbl in parsed.attachment_points
            for a in core.yield_atoms_by_label(lbl)
        ]
        for core in cores
    ]
    n_aps = len(ap_indices[0])

    # TODO: turn all assertions into more meaningful errors
    assert n_aps > 0, "Did not find any attachment points"
    assert all(
        len(aps) == n_aps for aps in ap_indices
    ), "Cores must have identical number of attachment points"

    n_cores = len(cores)
    n_subst = len(substituents)

    match parsed.mode:
        case "same":
            subst_iter = zip(*repeat(substituents, n_aps))
            lib_size = n_cores * n_subst
        case "permutns":
            # len = n! / (n - k)!
            subst_iter = permutations(substituents, n_aps)
            lib_size = n_cores * perm(n_subst, n_aps)
        case "combns":
            # len = n! / k! / (n - k)!
            subst_iter = combinations(substituents, n_aps)
            lib_size = n_cores * comb(n_subst, n_aps)
        case "combins_repl":
            # from docs: len = (n+r-1)! / r! / (n-1)!
            subst_iter = combinations_with_replacement(substituents, n_aps)
            lib_size = n_cores * perm(n_subst + n_aps - 1, n_aps) // factorial(n_aps)
        case _:
            raise NotImplementedError(f"Unknown mode: {parsed.mode}")

    print(f"Will create a library of size {lib_size}")

    progress = mp.Value("Q", 0, lock=True)
    outputs = mp.Queue()

    with (
        mp.Pool(parsed.n_jobs, initializer=_ml_init, initargs=(progress, parsed, outputs)) as pool,
        tqdm(range(lib_size)) as pb,
    ):
        lib_iter = ((c, i, s) for (c, i), s in product(zip(cores, ap_indices), subst_iter))
        res = pool.starmap_async(
            _ml_assemble, lib_iter, chunksize=parsed.batch_size, error_callback=lambda xc: print(xc)
        )
        cur = 0
        while not res.ready():
            res.wait(0.3)
            with progress.get_lock():
                pb.update(progress.value - cur)
                cur = progress.value

        dest_files = res.get()

    outputs = [outputs.get() for _ in range(outputs.qsize())]
    lib = ml.MoleculeLibrary.concatenate(parsed.output, outputs, overwrite=parsed.overwrite)
    assert len(lib) == lib_size, (
        f"Something went wrong. Output library size is different: expected {lib_size}, obtained"
        f" {len(lib)}"
    )

    for of in outputs:
        os.remove(of)

    # Now all the files will be concatenated

    # with ml.MoleculeLibrary.new(parsed.output, overwrite=False) as lib:
    #     for core in tqdm(cores, desc="Processing cores", position=0):
    #         n_ap = core.n_attachment_points
    #         ap_idx = core.get_atom_indices(*core.attachment_points)
    #         for substituents_combo in permutations(substituents, n_ap):
    #             deriv = ml.Molecule(core)
    #             for i, (ap_i, sub) in enumerate(zip(ap_idx, substituents_combo)):
    #                 deriv = ml.Molecule.join(
    #                     deriv, sub, ap_i - i, sub.attachment_points[0], optimize_rotation=True
    #                 )
    #             deriv.name = parsed.separator.join(
    #                 [core.name] + [sub.name for sub in substituents_combo]
    #             )
    #             lib.append(deriv.name, deriv)
