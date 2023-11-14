from fasteners import InterProcessLock, InterProcessReaderWriterLock
from glob import glob
from zipfile import ZipFile, is_zipfile

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
    Self,
    Any,
)
from pathlib import Path
from contextlib import contextmanager
from collections import deque
from collections.abc import MutableMapping
from deprecated import deprecated
import atexit

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

    def __init__(self, path, *, readonly: bool = True, bufsize=0, **kwargs) -> None:
        """Bufsize refers to the sum of lengths of all keys AND all values"""
        self._path = Path(path)
        self._readonly = readonly
        self._write_queue = deque()
        self._keys = set()
        self._lock = None
        self._bufsize = (
            int(bufsize) if bufsize >= 0 else 131_072
        )  # default buffer size will be 128 MB TODO: make adjustable via config?
        self._usedmem = 0
        # This registers the call so that once python terminates
        # normally then the contents are safely flushed onto the disk
        atexit.register(self.flush)
        self.update_keys()

    @property
    def lock(self):
        """Returns a read lock object. When it is acquired it is safe to read the data structure"""
        if self._lock is None:
            if self._path.is_dir():
                self._lock = InterProcessReaderWriterLock(self._path / "__lock__")
            else:
                self._lock = InterProcessReaderWriterLock(
                    self._path.with_name(self._path.name + ".lock")
                )
        return self._lock

    @lock.deleter
    def lock(self):
        self._lock._do_close()
        self._lock = None

    @abc.abstractmethod
    def update_keys(self):
        """Update the locator dictionary"""

    @abc.abstractmethod
    def read(self, key: str) -> bytes:
        """Get the bytes based on the key"""

    @abc.abstractmethod
    def write(self, key: str, value: bytes):
        """Write the bytes into the Collection. Assuming all locks are configured correctly."""

    def delete(self, key: str) -> bytes:
        """This method is responsible for deleting"""
        raise NotImplementedError(
            f"Collection backend {self.__class__} does not support item deletion"
        )

    def put(self, key: str, value: bytes):
        if self.readonly:
            raise IOError("Cannot write into a readonly Collection.")

        self._write_queue.append((key, value))
        self._keys.add(key)
        self._usedmem += len(key) + len(value)

        if self.used_memory > self._bufsize:
            self.flush()

    def get(self, key: str):
        """The getter is very primitive in the ABC. However, this can be a chance to add caching if needed."""
        with self.lock.read_lock():
            return self.read(key)

    @property
    def readonly(self):
        return self._readonly

    @property
    def used_memory(self):
        return self._usedmem

    def flush(self):
        with self.lock.write_lock():
            self.update_keys()
            while self._write_queue:
                key, value = self._write_queue.popleft()
                self.write(key, value)
            self._usedmem = 0

    def keys(self):
        """This returns a set of mapping keys"""
        return self._keys

    def items(self) -> Generator[tuple[bytes, bytes], None, None]:
        return ((k, self.get(k)) for k in self.keys())

    def __len__(self):
        return len(self.keys())

    def __repr__(self):
        return f"{self.__class__.__name__}({self._path.as_posix()!r})"


class DirCollectionBackend(CollectionBackendBase):
    def __init__(
        self, path, *, ext: str = ".dat", readonly: bool = True, bufsize=0
    ) -> None:
        self.ext = ext

        if not path.exists():
            path.mkdir()
        else:
            assert path.is_dir()

        super().__init__(path, readonly=readonly, bufsize=bufsize)

    def update_keys(self):
        allpaths: list[str] = glob(f"*{self.ext}", root_dir=self._path)
        self._keys = set(map(lambda x: x.removesuffix(self.ext), allpaths))

    def keys(self):
        return self._keys

    def get_path(self, key: str):
        return self._path / f"{key}{self.ext}"

    def write(self, key: str, value: bytes):
        with open(self.get_path(key), "wb") as f:
            return f.write(value)

    def read(self, key: bytes) -> bytes:
        with open(self.get_path(key), "rb") as f:
            return f.read()

    def __repr__(self):
        return f"{self.__class__.__name__}({self._path.as_posix()!r}, ext={self.ext!r})"


class ZipCollectionBackend(CollectionBackendBase):
    def __init__(
        self, path, *, ext: str = ".dat", readonly: bool = True, bufsize=0
    ) -> None:
        self.ext = ext
        super().__init__(path, readonly=readonly, bufsize=bufsize)
        self._zipfile = ZipFile(self._path, mode="a")

    def update_keys(self):
        if is_zipfile(str(self._path)):
            with ZipFile(self._path) as f:
                allnames: list[str] = filter(
                    lambda x: x.endswith(self.ext), f.namelist()
                )

            keys = map(lambda x: x.removesuffix(self.ext).encode(), allnames)
            self._keys = set(keys)

    def lock_acquire(self):
        self._plock = InterProcessLock(self._path.with_name(self._path.name + ".lock"))
        self._plock.acquire()

    def lock_release(self):
        self._zipfile.close()
        self._plock.release()

    def get_path(self, key: bytes):
        s_key = key.decode()
        return f"{s_key}{self.ext}"

    def write(self, key: bytes, value: bytes):
        with self._zipfile.open(self.get_path(key), "w") as f:
            f.write(value)

    def read(self, key: bytes) -> bytes:
        with self._zipfile.open(self.get_path(key), "r") as f:
            return f.read()


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
        h1: bytes = None,
        lb: bytes = None,
        comment: bytes = None,
        readonly: bool = True,
        bufsize=0,
    ) -> None:
        from .ukvfile import UKVFile

        super().__init__(path, readonly=readonly, bufsize=bufsize)
        self._ukvfile = UKVFile(self._path, h1=h1, lb=lb, comment=comment, mode="a")

    def update_keys(self):
        self._ukvfile.map_blocks()

    def lock_acquire(self):
        self._plock = InterProcessLock(self._path.with_name(self._path.name + ".lock"))
        self._plock.acquire()
        self._ukvfile.open()

    def lock_release(self):
        self._ukvfile.close()
        self._plock.release()

    def write(self, key: bytes, value: bytes):
        self._ukvfile.write(key, value)

    def read(self, key: bytes) -> bytes:
        return self._ukvfile.read(key)
