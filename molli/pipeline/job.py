from typing import Any, Generator, Callable, Literal, Generic, TypeVar
from ..chem import Molecule
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


class Job:
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
        cache_key: str = None,
    ):
        """If no post"""
        self._envars = envars
        self._return_files = return_files
        self._prep = prep
        self._post = post
        self._cache_key = cache_key

    def __get__(self, instance, owner=None):
        self._instance = instance
        self._owner = owner
        if hasattr(self._instance, "get_cache"):
            self.cache = self._instance.get_cache(self._cache_key)
        else:
            self.cache = None
        return self

    def prep(self, func):
        self._prep = func
        self._cache_key = self._cache_key or func.__name__
        return self

    def post(self, func):
        self._post = func
        return self

    def __call__(self, *args, **kwargs):
        inp = self._prep(self._instance, *args, **kwargs)
        if self.cache and inp.jid in self.cache:
            print("Cached result found. yum!")
            out = self.cache[inp.jid]
        else:
            ### CALCULATION HAPPENS HERE ###
            with TemporaryDirectory(
                dir=self._instance.scratch_dir, prefix=f"molli-{inp.jid}"
            ) as td:
                # Prepping the ground
                for f in inp.files:
                    (Path(td) / f).write_bytes(inp.files[f])

                proc = run(
                    shlex.split(inp.command),
                    cwd=td,
                    stdout=PIPE,
                    stderr=PIPE,
                    encoding="utf8",
                )
                out_files = {}
                if self._return_files:
                    for f in self._return_files:
                        file_path = Path(td) / f
                        out_files[f] = (
                            file_path.read_bytes() if file_path.exists() else None
                        )

                out = JobOutput(
                    stdout=proc.stdout,
                    stderr=proc.stderr,
                    exitcode=proc.returncode,
                    files=out_files,
                )

            ### ==== ###
            if self.cache is not None and out.exitcode == 0:
                self.cache[inp.jid] = out

        result = self._post(self._instance, out, *args, **kwargs)

        return result


def local_runner(inp: JobInput) -> JobOutput:
    """This function executes a job on a local machine"""
    pass


def runner_sge_cluster(inp: JobInput) -> JobOutput:
    """This function executes a job on a distributed SGE cluster"""
    pass


def jobmap(
    job: Job,
    source: Collection,
    cache: Collection,
    destination: Collection,
    scheduler: Literal["local", "sge-cluster"] = "local",
    scratch_dir: str | Path = None,
    n_workers: int = None,
    job_kwargs: dict = None,
):
    """This function maps a Job call onto a collection of items"""
    pass
