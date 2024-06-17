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
Description of this module
"""

from typing import Any, Generator, Callable, Literal, Generic, TypeVar, Iterable
import molli as ml
from .. import config
from subprocess import run, PIPE, DEVNULL
from pathlib import Path
import attrs
import shlex
from tempfile import TemporaryDirectory, mkstemp
import msgpack
from hashlib import sha3_512
import base64
from pprint import pprint
from joblib import delayed, Parallel
import numpy as np
import re
import os
import sys
from ..storage import Collection
from joblib import Parallel, delayed
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor, Future, ProcessPoolExecutor
import shutil
from copy import copy
from time import sleep

T_in = TypeVar("T_in")
T_out = TypeVar("T_out")

MOLLI_RUN = Path(sys.executable).with_name("_molli_run")


@attrs.define(repr=True)
class JobInput:
    jid: str  # job identifier
    commands: list[tuple[str, str | None]]
    files: dict[str, bytes] = None
    return_files: tuple[str] = None
    envars: dict[str, str] = None
    timeout: float = None

    @property
    def hash(self) -> str:
        data = msgpack.dumps(attrs.asdict(self))
        return base64.urlsafe_b64encode(sha3_512(data).digest())

    def dump(self, fn):
        with open(fn, "wb") as f:
            msgpack.dump(attrs.asdict(self), f)

    @classmethod
    def load(cls, fn):
        with open(fn, "rb") as f:
            return cls(**msgpack.load(f))


@attrs.define(repr=True)
class JobOutput:
    stdouts: dict[str, str] = None
    stderrs: dict[str, str] = None
    exitcode: int = None
    files: dict[str, bytes] = None
    input_hash: str = None

    def dump(self, fn):
        with open(fn, "wb") as f:
            msgpack.dump(attrs.asdict(self), f)

    @classmethod
    def load(cls, fn):
        with open(fn, "rb") as f:
            return cls(**msgpack.load(f))


class Job(Generic[T_in, T_out]):
    """
    A convenient way to wrap a call to an external program
    """

    def __init__(
        self,
        prep: Callable[[T_in, Any], JobInput] = None,
        *,
        post: Callable[[JobOutput, Any], T_out] = None,
        return_files: list[str] = None,
        executable: str | Path = None,
        nprocs: int = None,
        memory: int = None,
        envars: dict = None,
        name: str = None,
        doc: str = None,
    ):
        """If no post"""
        # self._prep = prep
        # self._post = post
        self.return_files = return_files
        self.executable = executable
        self.nprocs = nprocs
        self.envars = envars or dict()
        self.memory = memory
        self.name = name
        self.__doc__ = doc or ""

        if prep is not None:
            self.prep(prep)
        if post is not None:
            self.post(post)

    def prep(self, func):
        self._prep = func
        self.name = self.name or func.__qualname__
        self.__doc__ = self.__doc__ or func.__doc__
        return self

    def post(self, func):
        self._post = func
        self.name = self.name or func.__qualname__
        self.__doc__ = self.__doc__ or func.__doc__
        return self

    def reduce(self, func):
        self._reduce = func
        self.name = func.__qualname__
        self.__doc__ = func.__doc__
        return self

    def _prepare(self, inp: T_in, *args, **kwargs):
        return self._prep(self, inp, *args, **kwargs)

    def _process(self, output: JobOutput, inp: T_in, *args, **kwargs):
        return self._post(self, output, inp, *args, **kwargs)

    def _prepare_iter(self, inp, *args, **kwargs):
        return (self._prepare(x, *args, **kwargs) for x in inp)

    def _process_iter(self, outputs, inp, *args, **kwargs):
        result_iter = (
            self._process(out, x, *args, **kwargs) for out, x in zip(outputs, inp)
        )
        return self._reduce(self, result_iter, inp, *args, **kwargs)

    prepare = _prepare
    process = _process

    @classmethod
    def vectorize(cls, job: "Job", name: str = None):
        vecjob = cls(
            prep=job._prep,
            post=job._post,
            return_files=job.return_files,
            executable=job.executable,
            nprocs=job.nprocs,
            envars=job.envars,
            name=name or job.name,
        )

        vecjob.prepare = vecjob._prepare_iter
        vecjob.process = vecjob._process_iter

        return vecjob

    def __get__(self, obj, objtype=None):
        self.executable = (
            self.executable
            or getattr(objtype, "executable", None)
            or getattr(obj, "executable", None)
        )
        self.nprocs = (
            self.nprocs
            or getattr(objtype, "nprocs", None)
            or getattr(obj, "nprocs", None)
            or 1
        )
        self.memory = (
            self.memory
            or getattr(objtype, "memory", None)
            or getattr(obj, "memory", None)
            or 1_000
        )
        self.envars = (
            (getattr(objtype, "envars", None) or {})
            | (getattr(obj, "envars", None) or {})
            | (self.envars or {})
        )

        return self

        # return type(self)(
        #     prep=self._prep,
        #     post=self._post,
        #     return_files=self.return_files,
        #     executable=executable,
        #     nprocs=nprocs,
        #     envars=envars,
        # )

    def __call__(
        self,
        obj: T_in,
        scheduler: str = "local",
        scratch_dir: str | Path = None,
        shared_dir: str | Path = None,
        n_jobs: int = 1,
        args: tuple = None,
        kwargs: tuple = None,
    ) -> T_out:
        _name = f"{self.name}_{os.getpid()}"

        if scratch_dir is None:
            scratch_dir = config.SCRATCH_DIR

        args = args or ()
        kwargs = kwargs or {}

        match scheduler.lower():
            case "local":
                _runner = _runner_local
            case "sge-cluster":
                _runner = _runner_sge

        with (
            TemporaryDirectory(
                dir=shared_dir,
                prefix=f"molli_jobmap_worker_",
            ) as td,
            ThreadPoolExecutor(n_jobs) as executor,
        ):
            _prepared = self.prepare(obj, *args, **kwargs)

            if isinstance(_prepared, JobInput):
                _prepared.dump(f"{td}/input")
                fut = executor.submit(
                    _runner,
                    f"{td}/input",
                    f"{td}/output",
                    scratch_dir,
                    shared_dir,
                )
                proc = fut.result()
                if proc.returncode == 0:
                    out = JobOutput.load(f"{td}/output")
                    return self.process(out, obj, *args, **kwargs)
                else:
                    out = JobOutput.load(f"{td}/output")
                    raise RuntimeError(f"{proc}\n{out}")
            else:
                futures = []
                for i, inp in enumerate(_prepared):
                    futures.append(
                        executor.submit(
                            _runner,
                            f"{td}/input.{i}",
                            f"{td}/output.{i}",
                            scratch_dir,
                            shared_dir,
                        )
                    )

                results = [f.result() for f in futures]

                if all(proc.returncode == 0 for proc in results):
                    outputs = map(
                        JobOutput.load,
                        (f"{td}/output.{i}" for i in range(len(futures))),
                    )
                    return self.reduce(outputs, obj, *args, **kwargs)


def worker(
    name: str,
    runner: Callable,
    job: Job,
    source: Collection,
    destination: Collection,
    keys: list[str],
    args: tuple = None,
    kwargs: dict = None,
    n_jobs_per_worker: int = 1,
    cache_dir: str | Path = None,
    error_dir: str | Path = None,
    log_dir: str | Path = None,
    scratch_dir: str | Path = None,
    shared_dir: str | Path = None,
    log_level: str = "debug",
):
    _name = f"{name}_{os.getpid()}"

    if scratch_dir is None:
        scratch_dir = config.SCRATCH_DIR

    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

    if error_dir is not None:
        error_dir = Path(error_dir)
        error_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("molli.pipeline.worker")
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = Path(log_dir) / f"{name}_{os.getpid()}.log"
        if not log_file.exists():
            with open(log_file, "wt") as f:
                f.write(config.SPLASH)
            _handler = logging.FileHandler(log_file)
            logger.addHandler(_handler)
            logger.setLevel(log_level.upper())

    args = args or ()
    kwargs = kwargs or {}

    with (
        TemporaryDirectory(
            dir=shared_dir,
            prefix=f"molli_jobmap_worker_",
        ) as td,
        ThreadPoolExecutor(n_jobs_per_worker) as executor,
    ):
        cwd = Path(td)
        logger.info(
            f"Worker {name} (pid={os.getpid()}) started in temporary directory: {cwd.as_posix()}"
        )
        logger.info(f"Following keys will be processed: {keys}")
        success_expanded = set()
        # 1. Identify keys that have been cached and where the caches are correct
        with source.reading():
            expanded_keys = {}
            futures: dict[str, Future] = {}
            objects = {}
            for key in keys:
                obj = source[key]
                objects[key] = obj

                # this can be a JobInput or a Generator of job inputs
                _prepared = job.prepare(obj, *args, **kwargs)

                if isinstance(_prepared, JobInput):
                    inp_key = key
                    expanded_keys[key] = inp_key
                    inp = _prepared
                    if (
                        cache_dir is not None
                        and (cache_dir / f"{inp_key}.out").is_file()
                    ):
                        out = JobOutput.load(cache_dir / f"{inp_key}.out")
                        # Input and output check
                        if inp.hash == out.input_hash:
                            logger.info(
                                f"Found cached result for {inp_key} (hash={out.input_hash!r})"
                            )
                            out.dump(cwd / f"{inp_key}.out")
                            success_expanded.add(inp_key)
                            continue
                        else:
                            logger.info(
                                f"Cached result hash mismatch for {inp_key}. Ignoring the existing result."
                            )

                    inp.dump(cwd / f"{inp_key}.inp")

                    # This submits the external calculation WITHOUT blocking
                    futures[inp_key] = executor.submit(
                        runner,
                        cwd / f"{inp_key}.inp",
                        cwd / f"{inp_key}.out",
                        scratch_dir,
                        shared_dir,
                    )
                    logger.info(f"Submitted calculation {inp_key}")
                else:
                    for i, inp in enumerate(_prepared):
                        inp_key = f"{key}.{i}"
                        if key in expanded_keys:
                            expanded_keys[key].append(inp_key)
                        else:
                            expanded_keys[key] = [inp_key]

                        if (
                            cache_dir is not None
                            and (cache_dir / f"{inp_key}.out").is_file()
                        ):
                            out = JobOutput.load(cache_dir / f"{inp_key}.out")
                            # Input and output check
                            if inp.hash == out.input_hash:
                                out.dump(cwd / f"{inp_key}.out")
                                success_expanded.add(inp_key)
                                continue

                        # This code is meant to be unreachable if cache is already found
                        inp.dump(cwd / f"{inp_key}.inp")

                        # This submits the external calculation WITHOUT blocking
                        futures[inp_key] = executor.submit(
                            runner,
                            cwd / f"{inp_key}.inp",
                            cwd / f"{inp_key}.out",
                            scratch_dir,
                            shared_dir,
                        )

        # 2. *Blocking* await while futures are being computed
        # This step is only required for futures that are yet to be calculated

        for k, f in futures.items():
            proc = f.result()
            logger.info(f"Finished computing for {k}: code {proc.returncode}.")
            logger.debug(f"{proc.stdout=!s}\n{proc.stderr=!s}")
            # Warn the user if process is not successful.
            # Also cache the output
            if proc.returncode == 0:
                success_expanded.add(k)
                if cache_dir is not None:
                    shutil.copy(cwd / f"{k}.out", cache_dir)
            elif error_dir is not None:
                if (cwd / f"{k}.out").exists():
                    shutil.copy(cwd / f"{k}.out", error_dir)

        success = []
        failure = []

        results = {}
        for key, output_keys in expanded_keys.items():
            if isinstance(output_keys, str):
                if output_keys in success_expanded:
                    success.append(key)
                    output = JobOutput.load(cwd / f"{key}.out")
                    results[key] = job.process(
                        output,
                        objects[key],
                        *args,
                        **kwargs,
                    )
                else:
                    failure.append(key)
            else:
                if all(k in success_expanded for k in output_keys):
                    outputs = map(
                        JobOutput.load,
                        (cwd / f"{k}.out" for k in output_keys),
                    )
                    try:
                        results[key] = job.process(
                            outputs,
                            objects[key],
                            *args,
                            **kwargs,
                        )
                    except Exception as xc:
                        failure.append(key)
                        logger.exception(xc)
                    else:
                        success.append(key)

                else:
                    failure.append(key)

        if results:
            with destination.writing():
                for k, res in results.items():
                    destination[k] = res

        logger.info(f"Success: {success!r}")
        logger.info(f"Failure: {failure!r}")

    return success, failure


def jobmap(
    job: Job,
    source: Collection,
    destination: Collection,
    cache_dir: str | Path = None,
    scratch_dir: str | Path = None,
    shared_dir: str | Path = None,
    n_workers: int = None,
    args: tuple = None,
    kwargs: dict = None,
    verbose: bool = False,
    progress: bool = False,
    log_level: str = "warning",
    strict_hash: bool = True,
):
    """
    This function maps a Job call onto a collection of items.
    This represents the central concept of parallelization
    """
    logger = logging.getLogger("molli.pipeline")

    if cache_dir is None:
        cache_dir = Path(source._path.stem + "." + job.name).absolute()
    else:
        cache_dir = Path(cache_dir).absolute()
    cache_dir.mkdir(exist_ok=True, parents=True)

    # This will configure logging into the file
    log_file = cache_dir / "jobmap.log"
    with open(log_file, "at") as f:
        f.write(config.SPLASH)
    _handler = logging.FileHandler(log_file)
    logger.addHandler(_handler)
    logger.setLevel(log_level.upper())

    logger = logging.getLogger("molli.pipeline.jobmap")

    job_input_dir = cache_dir / "input"
    job_input_dir.mkdir(exist_ok=True, parents=True)

    job_output_dir = cache_dir / "output"
    job_output_dir.mkdir(exist_ok=True, parents=True)

    job_work_dir = cache_dir / "work"
    job_work_dir.mkdir(exist_ok=True, parents=True)

    if scratch_dir is None:
        scratch_dir = config.SCRATCH_DIR
    else:
        scratch_dir = Path(scratch_dir)

    with source.reading():
        all_keys = source.keys()

    with destination.reading():
        skip_keys = destination.keys()

    to_be_done = all_keys ^ skip_keys

    if verbose:
        print("Starting a molli.pipeline.jobmap_sge calculation:")
        print("calculation:", job.name)
        print(job.__doc__)
        print("     source: ", source)
        print("destination: ", destination)
        print("scratch dir: ", scratch_dir)
        print(f"Total number of jobs: {len(all_keys):>8}")
        print(f"Exist in destination: {len(skip_keys):>8}")
        print(f"To be computed:       {len(to_be_done):>8}")

    args = args or ()
    kwargs = kwargs or {}

    jobs_to_run = []
    job_len = {}

    # Step 2. Create all job input files
    with source.reading():
        for k in tqdm(to_be_done, desc="Preparing inputs", disable=not progress):
            obj = source[k]
            _input = job.prepare(obj, *args, **kwargs)
            if isinstance(_input, JobInput):
                job_len[k] = None
                if (_out_fn := job_output_dir / f"{k}.out").is_file():
                    try:
                        _out = JobOutput.load(_out_fn)
                    except Exception as xc:
                        logger.debug(
                            f"Cannot load output file {k}.out. Error: {xc}. The results will be computed."
                        )
                    else:
                        if (
                            not strict_hash or _out.input_hash == _input.hash
                        ) and _out.exitcode == 0:
                            logger.debug(
                                f"Found output file {k} successfully! Expensive calculation will be skipped."
                            )
                            continue
                        else:
                            logger.debug(
                                f"Found output file {k} successfully, but input hash differs: found {_out.input_hash!r} != expected {_input.hash!r}"
                            )
                _input.dump(job_input_dir / f"{k}.inp")
                jobs_to_run.append(job_input_dir / f"{k}.inp")

            else:
                L = 0
                for i, _inp in enumerate(_input):
                    L += 1
                    if (_out_fn := job_output_dir / f"{k}.{i}.out").is_file():
                        try:
                            _out = JobOutput.load(_out_fn)
                        except Exception as xc:
                            logger.debug(
                                f"Cannot load output file {k}.{i}.out. Error: {xc}. The results will be computed."
                            )
                        else:
                            if (
                                not strict_hash or _out.input_hash == _input.hash
                            ) and _out.exitcode == 0:
                                logger.debug(
                                    f"Found output file {k}.{i}.out successfully! Expensive calculation will be skipped."
                                )
                                continue
                            else:
                                logger.debug(
                                    f"Found output file {k}.{i}.out, but the file unsuitable: \nexptd {_inp.hash!r} \nfound {_out.input_hash!r} \nexitcode {_out.exitcode}"
                                )

                    _inp.dump(job_input_dir / f"{k}.{i}.inp")
                    jobs_to_run.append(f"{k}.{i}.inp")

                job_len[k] = L

    if jobs_to_run:
        logger.debug(
            f"Listing jobs to be run:\n  -"
            + "\n  -".join(str(x) for x in jobs_to_run)
            + f"\n-----------\n  total: {len(jobs_to_run)}"
        )

    logger.debug(f"{job_len}")

    futures = []
    with ThreadPoolExecutor(
        max_workers=n_workers,  # thread_name_prefix=job.name
    ) as executor:
        for i, jin in tqdm(
            enumerate(jobs_to_run),
            desc="Submitting jobs",
            total=len(jobs_to_run),
        ):
            f = executor.submit(
                _run_local,
                job_input_dir / jin,
                job_work_dir,
                job_output_dir,
                scratch_dir,
            )
            futures.append(f)

        for f in tqdm(futures, desc="Waiting for jobs"):
            proc = f.result()
            if proc.returncode:
                logger.error(f"Failed for process: {proc}")

    with source.reading(), destination.writing():
        for key in (pb := tqdm(to_be_done, "Finalizing the calculations")):
            if job_len[key] is None:
                try:
                    output = JobOutput.load(job_output_dir / f"{key}.out")
                    result = job.process(
                        output,
                        source[key],
                        *args,
                        **kwargs,
                    )
                except Exception as xc:
                    logger.error(f"Failed to compute for {key=}")
                    logger.exception(xc)
                else:
                    destination[key] = result
                    logger.info(f"Successfully computed for {key=}")
            else:
                try:
                    outputs = map(
                        JobOutput.load,
                        (
                            job_output_dir / f"{key}.{j}.out"
                            for j in range(job_len[key])
                        ),
                    )
                    result = job.process(
                        outputs,
                        source[key],
                        *args,
                        **kwargs,
                    )
                except Exception as xc:
                    logger.error(f"Failed to compute for {key=}")
                    logger.exception(xc)
                    pb.write(f"Error for {key!r}: {xc}")
                else:
                    destination[key] = result
                    logger.info(f"Successfully computed for {key=}")


def _run_local(
    ifn: Path,
    cwd: Path,
    odir: Path,
    sdir: Path,
):
    script = f"""{MOLLI_RUN} {ifn} -o {odir.as_posix()} -s {sdir.as_posix()}"""

    proc = run(
        shlex.split(script),
        cwd=cwd,
        capture_output=True,
        encoding="utf8",
    )

    return proc


def is_running(jid: str):
    proc = run(shlex.split(f"qstat -j {jid}"), capture_output=True)
    if proc.returncode:
        return False
    else:
        return True


def jobmap_sge(
    job: Job,
    source: Collection,
    destination: Collection,
    cache_dir: str | Path = None,
    scratch_dir: str | Path = None,
    qsub_header: str = None,
    args: tuple = None,
    kwargs: dict = None,
    verbose: bool = False,
    progress: bool = False,
    update: float = 10.0,
    log_level: str = "warning",
    strict_hash: bool = True,
):
    """
    This function maps a job using SGE cluster functionality.
    """
    # Step 1. Analyze source and destination.
    # Only keys not present in the destination will be run.
    logger = logging.getLogger("molli.pipeline")

    if cache_dir is None:
        cache_dir = Path(source._path.stem + "." + job.name).absolute()
    else:
        cache_dir = Path(cache_dir).absolute()
    cache_dir.mkdir(exist_ok=True, parents=True)

    # This will configure logging into the file
    log_file = cache_dir / "jobmap.log"
    with open(log_file, "at") as f:
        f.write(config.SPLASH)
    _handler = logging.FileHandler(log_file)
    logger.addHandler(_handler)
    logger.setLevel(log_level.upper())

    logger = logging.getLogger("molli.pipeline.jobmap")

    job_input_dir = cache_dir / "input"
    job_input_dir.mkdir(exist_ok=True, parents=True)

    job_output_dir = cache_dir / "output"
    job_output_dir.mkdir(exist_ok=True, parents=True)

    job_work_dir = cache_dir / "work"
    job_work_dir.mkdir(exist_ok=True, parents=True)

    if scratch_dir is None:
        scratch_dir = config.SCRATCH_DIR
    else:
        scratch_dir = Path(scratch_dir)

    with source.reading():
        all_keys = source.keys()

    with destination.reading():
        skip_keys = destination.keys()

    to_be_done = all_keys ^ skip_keys

    if verbose:
        print("Starting a molli.pipeline.jobmap_sge calculation:")
        print("calculation:", job.name)
        print(job.__doc__)
        print("     source: ", source)
        print("destination: ", destination)
        print("scratch dir: ", scratch_dir)
        print(f"Total number of jobs: {len(all_keys):>8}")
        print(f"Exist in destination: {len(skip_keys):>8}")
        print(f"To be computed:       {len(to_be_done):>8}")

    args = args or ()
    kwargs = kwargs or {}

    jobs_to_run = []
    job_len = {}

    # Step 2. Create all job input files
    with source.reading():
        for k in tqdm(to_be_done, desc="Preparing inputs", disable=not progress):
            obj = source[k]
            _input = job.prepare(obj, *args, **kwargs)
            if isinstance(_input, JobInput):
                job_len[k] = None
                if (_out_fn := job_output_dir / f"{k}.out").is_file():
                    try:
                        _out = JobOutput.load(_out_fn)
                    except Exception as xc:
                        logger.debug(
                            f"Cannot load output file {k}.out. Error: {xc}. The results will be computed."
                        )
                    else:
                        if (
                            not strict_hash or _out.input_hash == _input.hash
                        ) and _out.exitcode == 0:
                            logger.debug(
                                f"Found output file {k} successfully! Expensive calculation will be skipped."
                            )
                            continue
                        else:
                            logger.debug(
                                f"Found output file {k} successfully, but input hash differs: found {_out.input_hash!r} != expected {_input.hash!r}"
                            )
                _input.dump(job_input_dir / f"{k}.inp")
                jobs_to_run.append(job_input_dir / f"{k}.inp")

            else:
                L = 0
                for i, _inp in enumerate(_input):
                    L += 1
                    if (_out_fn := job_output_dir / f"{k}.{i}.out").is_file():
                        try:
                            _out = JobOutput.load(_out_fn)
                        except Exception as xc:
                            logger.debug(
                                f"Cannot load output file {k}.{i}.out. Error: {xc}. The results will be computed."
                            )
                        else:
                            if (
                                not strict_hash or _out.input_hash == _input.hash
                            ) and _out.exitcode == 0:
                                logger.debug(
                                    f"Found output file {k}.{i}.out successfully! Expensive calculation will be skipped."
                                )
                                continue
                            else:
                                logger.debug(
                                    f"Found output file {k}.{i}.out, but the file unsuitable: \nexptd {_inp.hash!r} \nfound {_out.input_hash!r} \nexitcode {_out.exitcode}"
                                )

                    _inp.dump(job_input_dir / f"{k}.{i}.inp")
                    jobs_to_run.append(f"{k}.{i}.inp")

                job_len[k] = L

    if jobs_to_run:
        logger.debug(
            f"Listing jobs to be run:\n  -"
            + "\n  -".join(str(x) for x in jobs_to_run)
            + f"\n-----------\n  total: {len(jobs_to_run)}"
        )

    logger.debug(f"{job_len}")

    # Step 3. Schedule the jobs through qsub
    # if batchsize is specified, jobs will be scheduled in batches
    # to reduce the qsub overhead

    jids = []

    for i, jin in tqdm(
        enumerate(jobs_to_run),
        desc="Submitting jobs",
        total=len(jobs_to_run),
    ):
        script = (
            (qsub_header or "")
            + f"\n\n {MOLLI_RUN} {(job_input_dir / jin).as_posix()} -o {job_output_dir.as_posix()} --s {scratch_dir.as_posix()}"
        )

        proc = run(
            shlex.split(f"qsub -N {job.name}.{i+1} -shell n -V -terse -cwd -j y"),
            input=script,
            cwd=job_work_dir,
            capture_output=True,
            encoding="utf8",
        )

        if proc.returncode == 0:
            jids.append(proc.stdout.strip())
            logger.info(f"Successfully submitted calculation {jids[-1]} for {jin!r}")
        else:
            logger.error(
                f"Failed to submit calculation for {jin!r}\nError: {proc.stderr}"
            )

    prev = len(jobs_to_run)
    with tqdm(desc="Waiting for calculations", total=prev) as pbar:
        while prev > 0:
            sleep(update)
            current = sum(int(is_running(j)) for j in jids)
            pbar.update(prev - current)
            prev = current

    with source.reading(), destination.writing():
        for key in (pb := tqdm(to_be_done, "Finalizing the calculations")):
            if job_len[key] is None:
                try:
                    output = JobOutput.load(job_output_dir / f"{key}.out")
                    result = job.process(
                        output,
                        source[key],
                        *args,
                        **kwargs,
                    )
                except Exception as xc:
                    logger.error(f"Failed to compute for {key=}")
                    logger.exception(xc)
                else:
                    destination[key] = result
                    logger.info(f"Successfully computed for {key=}")
            else:
                try:
                    outputs = map(
                        JobOutput.load,
                        (
                            job_output_dir / f"{key}.{j}.out"
                            for j in range(job_len[key])
                        ),
                    )
                    result = job.process(
                        outputs,
                        source[key],
                        *args,
                        **kwargs,
                    )
                except Exception as xc:
                    logger.error(f"Failed to compute for {key=}")
                    logger.exception(xc)
                    pb.write(f"Error for {key!r}: {xc}")
                else:
                    destination[key] = result
                    logger.info(f"Successfully computed for {key=}")
