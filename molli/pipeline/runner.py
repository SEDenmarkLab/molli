import molli as ml
from pprint import pprint
from argparse import ArgumentParser
from pathlib import Path
from tempfile import TemporaryDirectory
from subprocess import run, DEVNULL
import os
import shlex
import sys

arg_parser = ArgumentParser(
    "molli_run",
    description=__doc__,
    add_help=False,
)

arg_parser.add_argument(
    "job",
    action="store",
    type=Path,
    metavar="<input_file>",
)

arg_parser.add_argument(
    "-o",
    "--output",
    action="store",
    required=True,
    type=Path,
    metavar="<output_file>",
)

arg_parser.add_argument(
    "-n",
    "--nprocs",
    action="store",
    type=int,
    default=1,
    metavar="1",
)

arg_parser.add_argument(
    "-s",
    "--submit",
    action="store",
    type=str.lower,
    default="local",
    choices=["sge", "slurm", "local"],
)

arg_parser.add_argument("--scheduler_params", action="store", type=str)


def run_local():
    parsed = arg_parser.parse_args()

    with open(parsed.job, "rb") as f:
        job = ml.pipeline.JobInput.load(f)

    # 1. Create a scratch directory if that does not exist yet for some reason.
    scratch_dir = Path(job.scratch_dir)
    scratch_dir.mkdir(parents=True, exist_ok=True)

    _cwd_original = os.getcwd()

    with TemporaryDirectory(dir=scratch_dir, prefix=job.jid + "__") as td:
        cwd = Path(td)
        os.chdir(cwd)

        if isinstance(job.files, dict):
            for fn, fc in job.files.items():
                if isinstance(fc, str):
                    with open(fn, "wt") as f:
                        f.write(fc)
                else:
                    with open(fn, "wb") as f:
                        f.write(fc)

        environ = os.environ.copy()

        if job.envars is not None:
            environ |= job.envars

        fail = None
        names = []

        for i, (cmd, name) in enumerate(job.commands):
            if name is not None:
                # Named commands get their stdout and stderr recorded
                names.append(name)

                with (
                    open(f"{name}.out", "wt") as stdout,
                    open(f"{name}.err", "wt") as stderr,
                ):
                    proc = run(
                        shlex.split(cmd),
                        cwd=cwd,
                        env=environ,
                        stderr=stderr,
                        stdout=stdout,
                        encoding="utf8",
                    )

            else:
                proc = run(
                    shlex.split(cmd),
                    cwd=cwd,
                    env=environ,
                    stderr=DEVNULL,
                    stdout=DEVNULL,
                    encoding="utf8",
                )

            if proc.returncode:
                fail = i
                break

        stdouts = {}
        stderrs = {}

        for name in names:
            with (
                open(cwd / f"{name}.out", "rt") as stdout,
                open(cwd / f"{name}.err", "rt") as stderr,
            ):
                stdouts[name] = stdout.read()
                stderrs[name] = stderr.read()

        if proc.returncode == 0:
            retfiles = {
                f: f.read_bytes() for f in map(Path, job.return_files) if f.is_file()
            }

    os.chdir(_cwd_original)

    out = ml.pipeline.JobOutput(
        exitcode=proc.returncode,
        stdouts=stdouts,
        stderrs=stderrs,
        files=retfiles,
    )

    with open(parsed.output, "wb") as f:
        out.dump(f)


def run_sched():
    # so what we are trying to do here is to submit an otherwise "local job" to the queue.
    parsed = arg_parser.parse_args()
    match parsed.submit:
        case "sge":
            _molli_run_path = Path(sys.executable).with_name("_molli_run")
            print(_molli_run_path)
            res = run(
                shlex.split(
                    f"qsub -V -terse -sync yes -cwd -S {_molli_run_path.as_posix()}"
                )
                + sys.argv[1:],
                stderr=DEVNULL,
                stdout=DEVNULL,
            )

        case _:
            raise NotImplementedError

    exit(res.returncode)
