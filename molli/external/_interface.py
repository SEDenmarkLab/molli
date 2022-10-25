from pathlib import Path
from typing import List, Iterable, Callable, Dict
from functools import wraps
from dataclasses import dataclass
import os
from subprocess import run, PIPE

PathLike = Path | str


@dataclass
class JobOutput:
    stdout: str = None
    stderr: str = None
    exitcode: int = None
    files: Dict[str, bytes] = None


class Interface:
    def __init__(
        self,
        executable: PathLike,
        /,
        scratch_dir: PathLike = "scratch",
        cache_dir: PathLike = "cache",
        envars: dict = ...,
        n_procs: int = 1,
        memory: int = 1024,
        encoding: str = "utf8",
        validate_args: List[str] = ["--help"],
        inherit_envars=True,
    ):
        self.executable = Path(executable)
        self.scratch_dir = Path(scratch_dir).absolute()
        self.cache_dir = Path(cache_dir).absolute()
        self.envars = envars
        self.n_procs = n_procs
        self.memory = memory
        self.validate_args = validate_args
        self.encoding = encoding

        match envars:
            case dict() if inherit_envars:
                self.envars = os.environ | envars
            case dict():
                self.envars = envars

            case Ellipsis if inherit_envars:
                self.envars = dict(os.environ)

            case _:
                self.envars = dict()

    def __repr__(self) -> str:
        return f"Interface({self.executable})"

    def validate(self) -> bool:
        out = self.run(self.validate_args)
        return out.success

    def run(
        self,
        args: Iterable[str],
        input: str = None,
        input_files: Dict[str, bytes] = None,
        output_files: Iterable[str] = None,
        cwd: PathLike = None,
        timeout: int = None,
    ) -> JobOutput:

        _cwd = Path(cwd)

        if not _cwd.exists() or not _cwd.is_dir():
            raise ValueError(f"Directory does not exist: {_cwd=}")

        if isinstance(input, Dict):
            for fn, fc in input_files.items():
                with open(_cwd / fn, "wb") as f:
                    f.write(fc)

        proc = run(
            [self.executable, *args],
            cwd=cwd,
            input=input,
            env=self.envars,
            timeout=timeout,
            stderr=PIPE,
            stdout=PIPE,
            encoding=self.encoding,
        )

        outfiles = {}

        if isinstance(output_files, Iterable):
            for f in output_files:
                try:
                    with open(_cwd / f, "rb") as f:
                        fc = f.read()
                except:
                    fc = None
                finally:
                    outfiles[f] = fc

        return JobOutput(
            stdout=proc.stdout,
            stderr=proc.stderr,
            exitcode=proc.returncode,
            files=outfiles,
        )
