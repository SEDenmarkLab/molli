from typing import Any, Generator, Callable, Literal, Generic, TypeVar
from ..chem import Molecule
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

    def dump(self, f):
        msgpack.dump(attrs.asdict(self), f)

    @classmethod
    def load(cls, f):
        return cls(**msgpack.load(f))


@attrs.define(repr=True)
class JobOutput:
    stdouts: dict[str, str] = None
    stderrs: dict[str, str] = None
    exitcode: int = None
    files: dict[str, bytes] = None

    def dump(self, f):
        msgpack.dump(attrs.asdict(self), f)

    @classmethod
    def load(cls, f):
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
    ):
        """If no post"""
        self._prep = prep
        self._post = post
        self.return_files = return_files
        self.executable = executable
        self.nprocs = nprocs
        self.envars = envars

    def prep(self, func):
        self._prep = func
        return self

    def post(self, func):
        self._post = func
        return self

    def __get__(self, obj, objtype=None):
        executable = (
            self.executable
            or getattr(objtype, "executable", None)
            or getattr(obj, "executable", None)
        )
        nprocs = (
            self.nprocs
            or getattr(objtype, "nprocs", None)
            or getattr(obj, "nprocs", None)
        )
        envars = (
            self.envars
            or getattr(objtype, "envars", None)
            or getattr(obj, "envars", None)
        )

        return type(self)(
            prep=self._prep,
            post=self._post,
            return_files=self.return_files,
            executable=executable,
            nprocs=nprocs,
            envars=envars,
        )


def worker(
    name: str,
    runner: Callable,
    job: Job,
    source: Collection,
    destination: Collection,
    keys: list[str],
    cache: Collection[JobOutput] = None,
    cache_error: Collection[JobOutput] = None,
    scratch_dir: str | Path = None,
    log_file: str | Path | None = None,
    args: tuple = None,
    kwargs: dict = None,
):

    _name = f"{name}_{os.getpid()}"

    logger = logging.getLogger("molli.pipeline")
    if log_file is not None:
        with open(log_file, "at") as f:
            f.write(config.SPLASH)
        _handler = logging.FileHandler(log_file)
        logger.addHandler(_handler)
    
    if scratch_dir is None:
        scratch_dir = config.SCRATCH_DIR
    
    args = args or ()
    kwargs = kwargs or {}
    
    with TemporaryDirectory(
        dir=scratch_dir,
        prefix=_name + runner.__name__,
    ) as td:
        cwd = Path(td)
        logger.info(f"Worker {name} (pid={os.getpid()}) started in temporary directory: {cwd.as_posix()}")
        logger.info(f"Following keys will be processed: {keys}")
        for key in keys:
            with source.reading():
                obj = source[key]
            # Now checking if the result of the computation is already available
            out = None
            if cache is not None:
                with cache.reading():
                    if key in cache:
                        out = cache[key]
                        logger.debug(f"{key}: found a cached result")
            
            if out is None:
                inp = job._prep(job, obj, *args, **kwargs)
                with open(cwd / (key + ".inp"), "wb") as f:
                    inp.dump(f)
                proc = runner(cwd / (key + ".inp"), cwd / (key + ".out"), scratch_dir)

                with open(cwd / (key + ".out"), "rb") as f:
                    out = JobOutput.load(f)
            
                if proc.returncode == 0: 
                    if cache is not None:
                        with cache.writing():
                            cache[key] = out
                            logger.debug(f"{key}: successfully computed and cached the intermediate step.")
                    else:
                        logger.debug(f"{key}: successfully computed the intermediate result.")
                else:
                    if cache_error is not None:
                        with cache_error.writing():
                            cache_error[key] = out
                            logger.debug(f"{key}: computation failed. Intermediate result cached.")
                    else:
                        logger.debug(f"{key}: computation failed.")
                    break
            
            # At this point we have either successfully computed
            # or retrieved a cached calculation
            res = job._post(job, out, obj, *args, **kwargs)

            with destination.writing():
                destination[key] = res
                logger.info(f"{key}: completed! Result written in the destination.")                




def batched(iterable, n):
    from itertools import islice

    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


def _runner_local(fin, fout, scratch):
    proc = run(
        ["_molli_run", str(fin), "-o", str(fout), "--scratch", str(scratch)],
        capture_output=True,
    )
    return proc


def _runner_sge(fin, fout, scratch):
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
        ],
        capture_output=True,
    )
    return proc


def jobmap(
    job: Job,
    source: Collection,
    destination: Collection,
    cache: Collection = None,
    cache_error: Collection = None,
    scheduler: Literal["local", "sge-cluster"] = "local",
    scratch_dir: str | Path = None,
    n_workers: int = None,
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

    n_workers = n_workers or 1

    with source.reading():
        all_keys = source.keys()

    if cache is not None:
        with cache.reading():
            skip_keys = cache.keys()
    else:
        skip_keys = set()

    to_be_done = all_keys ^ skip_keys

    print("Starting a molli calculation:")
    print("Total number: ", len(all_keys))
    print("Cached: ", len(skip_keys))
    print("To be computed: ", len(to_be_done))
    print("Scratch dir:", scratch_dir)

    batches = list(batched(sorted(all_keys), batch_size))

    parallel = Parallel(n_jobs=n_workers, verbose=50)

    match scheduler.lower():
        case "local":
            _runner = _runner_local
        case "sge":
            _runner = _runner_sge

    results = parallel(
        delayed(worker)(
            job._prep.__qualname__,
            _runner,
            job,
            source,
            destination,
            batch,
            cache=cache,
            cache_error=cache_error,
            scratch_dir=scratch_dir,
            args=args,
            kwargs=kwargs
        )
        for batch in batches
    )

    n_success = sum(results)
