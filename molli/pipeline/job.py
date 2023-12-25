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
from subprocess import run, PIPE
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
from ..storage import Collection
from joblib import Parallel, delayed
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor, Future, ProcessPoolExecutor
import shutil
from copy import copy

T_in = TypeVar("T_in")
T_out = TypeVar("T_out")


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
        nprocs: int = 1,
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
        self.envars = envars
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
        )
        self.envars = (
            self.envars
            or getattr(objtype, "envars", None)
            or getattr(obj, "envars", None)
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

    # for key in keys:
    #     with source.reading():
    #         obj = source[key]
    #     # Now checking if the result of the computation is already available
    #     out = None
    #     if cache is not None:
    #         with cache.reading():
    #             if key in cache:
    #                 out = cache[key]
    #                 logger.debug(f"{key}: found a cached result")

    #     if out is None:
    #         # Additional support for multiple job input is in order
    #         inp = job._prep(job, obj, *args, **kwargs)
    #         with open(cwd / (key + ".inp"), "wb") as f:
    #             inp.dump(f)

    #         proc = runner(cwd / (key + ".inp"), cwd / (key + ".out"), scratch_dir)

    #         with open(cwd / (key + ".out"), "rb") as f:
    #             out = JobOutput.load(f)

    #         if proc.returncode == 0:
    #             if cache is not None:
    #                 with cache.writing():
    #                     cache[key] = out
    #                     logger.debug(
    #                         f"{key}: successfully computed and cached the intermediate step."
    #                     )
    #             else:
    #                 logger.debug(
    #                     f"{key}: successfully computed the intermediate result."
    #                 )
    #         else:
    #             if cache_error is not None:
    #                 with cache_error.writing():
    #                     cache_error[key] = out
    #                     logger.debug(
    #                         f"{key}: computation failed. Intermediate result cached."
    #                     )
    #             else:
    #                 logger.debug(f"{key}: computation failed.")
    #             break

    #     # At this point we have either successfully computed
    #     # or retrieved a cached calculation
    #     res = job._post(job, out, obj, *args, **kwargs)

    #     with destination.writing():
    #         destination[key] = res
    #         logger.info(f"{key}: completed! Result written in the destination.")


def batched(iterable, n):
    from itertools import islice

    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


def _runner_local(fin, fout, scratch, shared):
    proc = run(
        [
            "_molli_run",
            str(fin),
            "-o",
            str(fout),
            "--scratch",
            str(scratch),
            "--shared",
            str(shared),
        ],
        capture_output=True,
        encoding="utf8",
    )
    return proc


def _runner_sge(fin, fout, scratch, shared):
    proc = run(
        [
            "_molli_run_sched",
            str(fin),
            "-o",
            str(fout),
            "-s",
            "sge",
            "--scratch",
            str(scratch),
            "--shared",
            str(shared),
        ],
        capture_output=True,
        encoding="utf8",
    )
    return proc


def jobmap(
    job: Job,
    source: Collection,
    destination: Collection,
    cache_dir: str | Path = None,
    error_dir: str | Path = None,
    log_dir: str | Path = None,
    scheduler: Literal["local", "sge-cluster"] = "local",
    scratch_dir: str | Path = None,
    shared_dir: str | Path = None,
    n_workers: int = None,
    n_jobs_per_worker: int = 1,
    batch_size: int = 16,
    args: tuple = None,
    kwargs: dict = None,
    verbose: bool = False,
    progress: bool = False,
):
    """
    This function maps a Job call onto a collection of items.
    This represents the central concept of parallelization
    """
    if scratch_dir is None:
        scratch_dir = config.SCRATCH_DIR
    else:
        scratch_dir = Path(scratch_dir)

    if shared_dir is None:
        shared_dir = config.SHARED_DIR / job.name
    else:
        shared_dir = Path(shared_dir)
    shared_dir.mkdir(parents=True, exist_ok=True)

    if cache_dir is not None:
        cache_dir = Path(cache_dir).absolute()

    if error_dir is not None:
        error_dir = Path(error_dir).absolute()

    if log_dir is None:
        log_dir = Path(log_dir).absolute()

    if log_dir is not None and Path(log_dir).is_dir():
        prevlogs = Path(log_dir).glob(f"{job._prep.__qualname__}_*.log")
        for fn in prevlogs:
            os.unlink(fn)

    n_workers = n_workers or 1

    with source.reading():
        all_keys = source.keys()

    with destination.reading():
        skip_keys = destination.keys()

    to_be_done = all_keys ^ skip_keys

    if verbose:
        print("Starting a molli.pipeline.jobmap calculation:")
        print("calculation:", job.name)
        print(job.__doc__)
        print("input <<", source)
        print("output >>", destination)
        print("Scratch dir: ", scratch_dir)
        print(f"Total number of jobs: {len(all_keys):>8}")
        print(f"Exist in destination: {len(skip_keys):>8}")
        print(f"To be computed:       {len(to_be_done):>8}")

    batches = list(batched(sorted(to_be_done), batch_size))

    parallel = Parallel(n_jobs=n_workers, return_as="generator")

    match scheduler.lower():
        case "local":
            _runner = _runner_local
        case "sge-cluster":
            _runner = _runner_sge

    with (
        tqdm(
            desc=f"{job.name} success",
            total=len(all_keys),
            initial=len(skip_keys),
            position=1,
            # dynamic_ncols=True,
            colour="green",
        ) as pbar_success,
        tqdm(
            desc=f"{job.name} error",
            total=len(all_keys),
            initial=0,
            position=0,
            # dynamic_ncols=True,
            colour="red",
        ) as pbar_error,
    ):
        for success, error in parallel(
            delayed(worker)(
                job.name,
                _runner,
                job,
                source,
                destination,
                batch,
                cache_dir=cache_dir,
                error_dir=error_dir,
                log_dir=log_dir,
                scratch_dir=scratch_dir,
                shared_dir=shared_dir,
                args=args,
                kwargs=kwargs,
                log_level="debug" if verbose else "info",
            )
            for batch in batches
        ):
            pbar_success.update(len(success))
            pbar_error.update(len(error))

            if len(error):
                for k in error:
                    pbar_error.write(f"Error at key {k}")
