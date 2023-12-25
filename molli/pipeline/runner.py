# ================================================================================
# This file is part of `molli 1.0`
# (https://github.com/SEDenmarkLab/molli)
#
# Developed by Alexander S. Shved <shvedalx@illinois.edu>
#
# S. E. Denmark Laboratory, University of Illinois, Urbana-Champaign
# https://denmarkgroup.illinois.edu/
#
# Copyright 2022-2023 The Board of Trustees of the University of Illinois.
# All Rights Reserved.
#
# Licensed under the terms MIT License
# The License is included in the distribution as LICENSE file.
# You may not use this file except in compliance with the License.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
# ================================================================================


"""
This is an auxiliary file that defines the way that the computational jobs are being executed by molli.
"""

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

arg_parser.add_argument("--scheduler_params", action="store", type=str, default="")

arg_parser.add_argument(
    "--scratch",
    action="store",
    default=None,
)

arg_parser.add_argument(
    "--shared",
    action="store",
    default=None,
)


def run_local():
    parsed = arg_parser.parse_args()

    job = ml.pipeline.JobInput.load(parsed.job)
    job_hash = job.hash

    # 1. Create a scratch directory if that does not exist yet for some reason.
    scratch_dir = Path(parsed.scratch)
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

            if proc.returncode != 0:
                fail = i
                break

        stdouts = {}
        stderrs = {}

        for name in names:
            with (
                open(f"{name}.out", "rt") as stdout,
                open(f"{name}.err", "rt") as stderr,
            ):
                stdouts[name] = stdout.read()
                stderrs[name] = stderr.read()

        retfiles = {
            str(f): f.read_bytes() for f in map(Path, job.return_files) if f.is_file()
        }

        os.chdir(_cwd_original)

        exitcode = None

        out = ml.pipeline.JobOutput(
            input_hash=job_hash,
            exitcode=exitcode or proc.returncode,
            stdouts=stdouts,
            stderrs=stderrs,
            files=retfiles,
        )
        out.dump(parsed.output)

        if fail is not None or set(retfiles) != set(job.return_files):
            exit(1)
        else:
            exit(proc.returncode)


def run_sched():
    # so what we are trying to do here is to submit an otherwise "local job" to the queue.
    parsed = arg_parser.parse_args()
    with TemporaryDirectory(dir=parsed.shared) as cwd:
        match parsed.submit:
            case "sge":
                _molli_run_path = Path(sys.executable).with_name("_molli_run")

                sge_inp = _molli_run_path.as_posix() + " " + shlex.join(sys.argv[1:])

                name = "molli_" + Path(parsed.job).stem

                res = run(
                    shlex.split(
                        f"qsub -N {name} -V -terse -sync yes -wd {cwd} {parsed.scheduler_params}"
                    ),
                    encoding="utf8",
                    input=sge_inp,
                    capture_output=False,
                )

            case _:
                raise NotImplementedError

    exit(res.returncode)
