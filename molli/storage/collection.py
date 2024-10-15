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
    ValuesView,
)
from pathlib import Path
from contextlib import contextmanager
from collections import deque
from collections.abc import MutableMapping
from .backends import (
    CollectionBackendBase,
    DirCollectionBackend,
    ZipCollectionBackend,
    UkvCollectionBackend,
    MlibCollectionBackend,
)
from io import UnsupportedOperation
import re

T = TypeVar("T")

__all__ = ("Collection",)


def _do_nothing(x):
    return x


class Collection(MutableMapping[str, T]):
    def __init__(
        self,
        path: Path | str,
        backend: Type[CollectionBackendBase],
        value_encoder: Callable[[T], bytes] | str = None,
        value_decoder: Callable[[bytes], T] | str = None,
        *,
        overwrite: bool = False,
        readonly: bool = True,
        encoding: str = "utf8",
        bufsize: int = -1,
        ext: str = None,
        **kwargs,
    ) -> None:
        self._path = Path(path)

        if not self._path.exists() and readonly:
            raise FileNotFoundError(f"{path!r} is not a valid molli collection.")

        self._backend = backend(
            self._path,
            overwrite=overwrite,
            readonly=readonly,
            bufsize=bufsize,
            ext=ext,
            **kwargs,
        )

        self._value_encoder = value_encoder or _do_nothing
        self._value_decoder = value_decoder or _do_nothing
        self._encoding = encoding

    def reading(self, timeout=None):
        return self._backend.reading(timeout=timeout)

    def writing(self, timeout=None):
        return self._backend.writing(timeout=timeout)

    def __contains__(self, __key: str) -> bool:
        return __key in self.keys()

    def keys(self) -> set[str]:
        return self._backend.keys()

    def items(self) -> Generator[tuple[str, T], None, None]:
        return ((k, self._value_decoder(v)) for k, v in self._backend.items())

    def values(self) -> Generator[T, None, None]:
        yield from map(self.__getitem__, self.keys())

    def __getitem__(self, __key: str) -> T:
        value = self._backend.get(__key)
        return self._value_decoder(value)

    def __setitem__(self, __key: str, __value: T) -> None:
        v = self._value_encoder(__value)
        self._backend.put(__key, v)

    def __delitem__(self, __key: str) -> None:
        self._backend.delete(__key)

    def __len__(self):
        return len(self._backend)

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    def flush(self):
        self._backend.flush()

    @property
    def n_items(self):
        return len(self.keys())

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"backend={self._backend},"
            f" n_items={self.n_items})"
        )

    def __enter__(self):
        self._backend.update_keys()
        return self

    def __exit__(self, *_):
        self.flush()
