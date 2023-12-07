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


@attrs.define(repr=True)
class JobInput:
    jid: str  # job identifier
    commands: list[tuple[str, str | None]]
    files: dict[str, bytes] = None
    return_files: tuple[str] = None
    envars: dict[str, str] = None
    scratch_dir: str = None
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


@attrs.define(repr=True, slots=False)
class Job:
    inst: object
    _prep: Callable
    _post: Callable

    def prep(self, *args, **kwds):
        return self._prep(self.inst, *args, **kwds)

    def post(self, out, *args, **kwds):
        return self._post(self.inst, out, *args, **kwds)


class JobMaker:
    """
    A convenient way to wrap a call to an external program
    """

    def __init__(
        self,
        prep: Callable[[Any], JobInput] = None,
        *,
        post: Callable[[JobOutput], Any] = None,
        envars: dict = None,
        return_files: list[str] = None,
    ):
        """If no post"""
        self._envars = envars
        self._return_files = return_files
        self._prep = prep
        self._post = post

    def __get__(self, instance, owner=None):
        j = Job(instance, self._prep, self._post)
        j.envars = self._envars
        j.return_files = self._return_files
        return j

    def prep(self, func):
        self._prep = func
        return self

    def post(self, func):
        self._post = func
        return self


def worker(
    runner: Callable,
    job: Job,
    source: Collection,
    destination: Collection,
    keys: list[str],
    cache: Collection[JobOutput] | None,
    scratch_dir: str | Path,
):
    with TemporaryDirectory(
        dir=scratch_dir,
        prefix=f"molli-{runner.__name__}",
    ) as td:
        cwd = Path(td)
        if cache:
            # This retrieves keys that are already cached
            with cache.reading():
                cached = cache.keys() & set(keys)
                for key in cached:
                    with open(cwd / (key + ".out"), "wb") as f:
                        cached[key].dump(f)
        else:
            cached = set()

        success = []
        with source.reading():
            # This computes the values that are missing
            for key in set(keys) ^ cached:
                inp = job.prep(source[key])
                # Paths to files with input and output
                _inp_path = cwd / (key + ".inp")
                _out_path = cwd / (key + ".out")
                with open(_inp_path, "wb") as f:
                    inp.dump(f)

                # this is where the job is actually submitted
                code = runner(_inp_path, _out_path)

                if code == 0:
                    success.append(key)

        with cache.writing():
            for key in success:
                _out_path = cwd / (key + ".out")
                with open(_out_path, "wb") as f:
                    cache[key] = JobOutput.load(f)

            with destination.writing():
                for key in set(success) & cached:
                    destination[key] = job.post(cache[key])

        return len(success) + len(cached)


def batched(iterable, n):
    from itertools import islice

    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


def _runner_local(fin, fout):
    proc = run(["_molli_run", fin, "-o", fout])
    return proc.returncode


def _runner_sge(fin, fout):
    proc = run(["_molli_run_sched", fin, "-o", fout, "-s", "sge"])
    return proc.returncode


def jobmap(
    job: Job,
    source: Collection,
    destination: Collection,
    cache: Collection = None,
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

    if cache:
        with cache.reading():
            skip_keys = cache.keys()
    else:
        skip_keys = set()

    to_be_done = all_keys ^ skip_keys

    print("Starting a molli calculation:")
    print("Total number: ", len(all_keys))
    print("Cached: ", len(skip_keys))
    print("To be computed: ", len(to_be_done))

    batches = list(batched(sorted(all_keys), batch_size))

    parallel = Parallel(n_jobs=n_workers, return_as="generator")

    results = parallel(
        delayed(worker)(
            _runner_sge,
            job,
            source,
            destination,
            batch,
            cache,
            scratch_dir,
        )
        for batch in batches
    )

    for res in tqdm(results):
        pass
