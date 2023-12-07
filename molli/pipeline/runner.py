import molli as ml
from pprint import pprint
from argparse import ArgumentParser
from pathlib import Path
from tempfile import TemporaryDirectory
from subprocess import run
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
    required=True
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

    with TemporaryDirectory(dir=scratch_dir, prefix=job.jid + "__") as td:
        cwd = Path(td)

        if isinstance(job.files, dict):
            for fn, fc in job.files.items():
                if isinstance(fc, str):
                    with open(cwd / fn, "wt") as f:
                        f.write(fc)
                else:
                    with open(cwd / fn, "wb") as f:
                        f.write(fc)

        environ = os.environ.copy()

        if job.envars is not None:
            environ |= job.envars

        fail = None

        for i, cmd in enumerate(job.commands):
            with (
                open(cwd / f"__stdout.{i}.txt", "wt") as stdout,
                open(cwd / f"__stderr.{i}.txt", "wt") as stderr,
            ):
                proc = run(
                    shlex.split(cmd),
                    cwd=cwd,
                    env=environ,
                    stderr=stderr,
                    stdout=stdout,
                    encoding="utf8",
                )

            if proc.returncode:
                fail = i
                break

        stdouts = []
        stderrs = []

        for i in range(len(job.commands) if fail is None else fail + 1):
            with (
                open(cwd / f"__stdout.{i}.txt", "rt") as stdout,
                open(cwd / f"__stderr.{i}.txt", "rt") as stderr,
            ):
                stdouts.append(stdout.read())
                stderrs.append(stderr.read())

        print(fail, job.return_files)

        if proc.returncode == 0:
            retfiles = {
                f: (cwd / f).read_bytes()
                for f in job.return_files
                if (cwd / f).is_file()
            }

    out = ml.pipeline.JobOutput(
        exitcode=proc.returncode,
        stdout=stdouts,
        stderr=stderrs,
        files=retfiles,
    )

    with open(parsed.output, "wb") as f:
        out.dump(f)


def run_scheduled():
    # so what we are trying to do here is to submit an otherwise "local job" to the queue.
    parsed = arg_parser.parse_known_args()
    match parsed.scheduler:
        case "sge":
            res = run(
                shlex.split(f"qsub -terse -sync yes -V - {parsed.nprocs} _molli_run")
                + sys.argv[1:]
            )

        case _:
            raise NotImplementedError

    exit(res.returncode)
