import io
from fasteners import InterProcessLock, InterProcessReaderWriterLock
from glob import glob
from zipfile import ZipFile, is_zipfile
from tarfile import TarFile, TarInfo, is_tarfile
from .ukvfile import UKVFile

import abc
from typing import (
    ItemsView,
    Iterator,
    KeysView,
    TypeVar,
    Generic,
    Callable,
    Generator,
    Sequence,
    Type,
    Any,
    Literal,
)
from pathlib import Path
from contextlib import contextmanager
from collections import deque
from collections.abc import MutableMapping
from deprecated import deprecated
import atexit
from io import UnsupportedOperation
import os
from molli._aux.lock import rwlock

T = TypeVar("T")

__all__ = (
    "CollectionBackendBase",
    "DirCollectionBackend",
    "ZipCollectionBackend",
    "MlibCollectionBackend",
    "UkvCollectionBackend",
)


class CollectionBackendBase(metaclass=abc.ABCMeta):
    """
    This is a base class for all possible Collection backends
    """

    def __init__(
        self,
        path,
        *,
        overwrite: bool = False,
        readonly: bool = True,
        bufsize: int = None,
        **kwargs,
    ) -> None:
        """Bufsize refers to the sum of lengths of all keys AND all values"""
        self._path = Path(path)
        self._readonly = readonly

        self._write_queue = deque()
        self._keys = set[str]()

        if overwrite and readonly:
            raise ValueError("overwrite and readonly are mutually exclusive.")

        self._lock = InterProcessReaderWriterLock(rwlock(self._path))

        self._bufsize = (
            int(bufsize) if bufsize is not None else 131_072
        )  # default buffer size will be 128 MB TODO: make adjustable via config?
        self._usedmem = 0

        self._state = "idle"

        # This registers the call so that once python terminates
        # normally then the contents are safely flushed onto the disk
        atexit.register(self.flush)
        # self.update_keys()

    def __getstate__(self):
        _self_dict = self.__dict__.copy()
        if "_lock" in _self_dict:
            del _self_dict["_lock"]
        return _self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = InterProcessReaderWriterLock(rwlock(self._path))

    def begin_read(self):
        pass

    def end_read(self):
        pass

    def begin_write(self):
        pass

    def end_write(self):
        pass

    @abc.abstractmethod
    def update_keys(self):
        pass

    @contextmanager
    def reading(self, timeout: float = None):
        """Context manager that acquires a read lock"""
        if not self._lock.acquire_read_lock(timeout=timeout):
            raise TimeoutError(f"Could not acquire reading lock within timeout")
        self.begin_read()
        self._state = "reading"

        try:
            self.update_keys()
            yield self
        finally:
            self.end_read()
            self._state = "idle"
            self._lock.release_read_lock()

    @contextmanager
    def writing(self, timeout: float = None):
        """Context manager that acquires a write lock"""
        if self._readonly:
            raise UnsupportedOperation(
                f"Cannot begin writing into a readonly collection backend."
            )
        if not self._lock.acquire_write_lock(timeout=timeout):
            raise TimeoutError(f"Could not acquire reading lock within timeout")
        self.begin_write()
        self._state = "writing"
        try:
            self.update_keys()
            yield self
        finally:
            self.flush()
            self.end_write()
            self._state = "idle"
            self._lock.release_write_lock()

    @abc.abstractmethod
    def _read(self, key: str) -> bytes:
        """Get the bytes based on the key"""

    @abc.abstractmethod
    def _write(self, key: str, value: bytes):
        """Write the bytes into the Collection. Assuming all locks are configured correctly."""

    @abc.abstractmethod
    def _truncate(self):
        """Truncate the collection. This deletes all contents."""

    def _delete(self, key: str) -> bytes:
        """This method is responsible for deleting"""
        raise NotImplementedError(
            f"Collection backend {self.__class__} does not support item deletion"
        )

    def put(self, key: str, value: bytes):
        if self._readonly:
            raise IOError("Cannot write into a readonly Collection.")

        self._write_queue.append((key, value))
        self._keys.add(key)
        self._usedmem += len(key) + len(value)

        if self.used_memory > self._bufsize:
            self.flush()

    def get(self, key: str):
        """The getter is very primitive in the ABC. However, this can be a chance to add caching if needed."""
        return self._read(key)

    def truncate(self):
        with self.writing():
            self._truncate()

    @property
    def used_memory(self):
        return self._usedmem

    def flush(self):
        while self._write_queue:
            key, value = self._write_queue.popleft()
            self._write(key, value)
        self._usedmem = 0

    def keys(self) -> set[str]:
        """This returns a set of mapping keys"""
        return self._keys

    def items(self) -> Generator[tuple[str, bytes], None, None]:
        return ((k, self.get(k)) for k in self.keys())

    def __len__(self):
        return len(self.keys())

    def __repr__(self):
        return f"{self.__class__.__name__}({self._path.as_posix()!r})"

    def __contains__(self, __key):
        return __key in self.keys()


class DirCollectionBackend(CollectionBackendBase):
    def __init__(
        self,
        path,
        *,
        overwrite: bool = False,
        readonly: bool = True,
        ext: str = ".dat",
        bufsize=0,
    ) -> None:
        self.ext = ext or ".dat"

        path = path if isinstance(path, Path) else Path(path)

        if not path.exists():
            path.mkdir(exist_ok=True, parents=True)

        super().__init__(path, bufsize=bufsize, readonly=readonly)

    def update_keys(self):
        self._keys = set(
            x.name.removesuffix(self.ext) for x in self._path.glob("*" + self.ext)
        )

    def keys(self):
        return self._keys

    def get_path(self, key: str):
        return self._path / f"{key}{self.ext}"

    def _write(self, key: str, value: bytes):
        with open(self.get_path(key), "wb") as f:
            return f.write(value)

    def _read(self, key: bytes) -> bytes:
        with open(self.get_path(key), "rb") as f:
            return f.read()

    def _truncate(self):
        for fn in map(self.get_path, self.keys()):
            os.remove(fn)

    def __repr__(self):
        return f"{self.__class__.__name__}({self._path.as_posix()!r}, ext={self.ext!r})"


class ZipCollectionBackend(CollectionBackendBase):
    # pass
    def __init__(
        self,
        path,
        *,
        overwrite: bool = False,
        readonly: bool = True,
        ext: str = ".mol2",
        mode: Literal["r", "w", "a", "x"] = "r",
        bufsize=0,
    ) -> None:
        self.ext = ext
        super().__init__(path, mode=mode, bufsize=bufsize, readonly=readonly)

        with self._lock.write_lock():
            if not self._path.is_file():
                with ZipFile(
                    self._path,
                    mode="x",
                ):
                    pass
            elif overwrite:
                with ZipFile(
                    self._path,
                    mode="w",
                ):
                    pass

    def lock_acquire(self):
        self._plock = InterProcessReaderWriterLock(rwlock(self._path))
        self._plock.acquire()

    def lock_release(self):
        self._zipfile.close()
        self._plock.release()

    def begin_read(self):
        if is_zipfile(str(self._path)):
            self._zipfile = ZipFile(self._path, mode="r")

    def end_read(self):
        self._zipfile.close()

    def begin_write(self):
        self._zipfile = ZipFile(self._path, mode="a")

    def end_write(self):
        self._zipfile.close()

    def get_path(self, key: str):
        return f"{key}"

    def update_keys(self):
        self._keys = {
            name for name in self._zipfile.namelist() if name.endswith(self.ext)
        }

    def _write(self, key: str, value: bytes):
        self._zipfile.writestr(f"{self.get_path(key)}{self.ext}", value)

    def _read(self, key: str) -> bytes:
        with self._zipfile.open(key) as f:
            return f.read()

    def _truncate(self, key: bytes) -> bytes:
        self._zipfile.remove(key)


class TarCollectionBackend(CollectionBackendBase):
    # pass
    def __init__(
        self,
        path,
        *,
        overwrite: bool = False,
        readonly: bool = True,
        ext: str = ".mol2",
        mode: Literal["r", "w", "a", "x"] = "r",
        bufsize=0,
    ) -> None:
        self.ext = ext
        super().__init__(path, mode=mode, bufsize=bufsize, readonly=readonly)

        with self._lock.write_lock():
            if not self._path.is_file():
                with TarFile(
                    self._path,
                    mode="x",
                ):
                    pass
            elif overwrite:
                with TarFile(
                    self._path,
                    mode="w",
                ):
                    pass

    def lock_acquire(self):
        self._plock = InterProcessReaderWriterLock(rwlock(self._path))
        self._plock.acquire()

    def lock_release(self):
        self._tarfile.close()
        self._plock.release()

    def begin_read(self):
        if is_tarfile(str(self._path)):
            self._tarfile = TarFile(self._path, mode="r")

    def end_read(self):
        self._tarfile.close()

    def begin_write(self):
        self._tarfile = TarFile(self._path, mode="a")

    def end_write(self):
        self._tarfile.close()

    def get_path(self, key: str):
        return f"{key}"

    def update_keys(self):
        self._keys = {
            name for name in self._tarfile.getnames() if name.endswith(self.ext)
        }

    def _write(self, key: str, value: bytes):
        tarinfo = TarInfo(name=f"{self.get_path(key)}{self.ext}")
        tarinfo.size = len(value)
        self._tarfile.addfile(tarinfo, io.BytesIO(value))

    def _read(self, key: str) -> bytes:

        try:
            f = self._tarfile.extractfile(key)
            return f.read()
        except:
            print(f"No such file or directory: {key}")

    def _truncate(self, key: bytes) -> bytes:
        with TarFile(
            self._path,
            mode="w",
        ):
            pass


@deprecated(
    "Mlib file format was significantly updated and replaced by UKV file format. Please consider"
    " using `molli recollect` to update your Collection format."
)
class MlibCollectionBackend(CollectionBackendBase):
    pass


class UkvCollectionBackend(CollectionBackendBase):
    def __init__(
        self,
        path,
        *,
        overwrite: bool = False,
        readonly: bool = True,
        comment: str = None,
        h1: bytes = None,
        b0: bytes = None,
        bufsize=0,
        **kwds,
    ) -> None:
        super().__init__(path, bufsize=bufsize, readonly=readonly)
        if comment is None:
            comment = ""

        # This access needs to be very restrictive to prevent collisions
        with self._lock.write_lock():
            if not self._path.is_file():
                with UKVFile(
                    self._path,
                    h1=h1,
                    b0=b0,
                    h2=comment.encode(),
                    mode="x",
                ):
                    pass
            elif overwrite:
                with UKVFile(
                    self._path,
                    h1=h1,
                    b0=b0,
                    h2=comment.encode(),
                    mode="w",
                ):
                    pass

    def begin_read(self):
        if not hasattr(self, "_ukvfile"):
            self._ukvfile = UKVFile(self._path, mode="r")
        else:
            self._ukvfile.open("r")

    def end_read(self):
        self._ukvfile.close()

    def begin_write(self):
        if not hasattr(self, "_ukvfile"):
            self._ukvfile = UKVFile(self._path, mode="a")
        else:
            self._ukvfile.open("a")

    def end_write(self):
        self._ukvfile.close()

    def update_keys(self):
        self._keys = {k.decode() for k in self._ukvfile.keys()}

    def _write(self, key: str, value: bytes):
        self._ukvfile.put(key.encode(), value)

    def _read(self, key: str) -> bytes:
        return self._ukvfile.get(key.encode())

    def _truncate(self):
        self._ukvfile.truncate()
