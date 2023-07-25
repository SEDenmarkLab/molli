from __future__ import annotations
from struct import pack, unpack, Struct
from pathlib import Path
from typing import TypeVar, Generic, Callable, Iterable
from functools import wraps
from enum import IntFlag, auto
import msgpack
from .._aux import unique_path
from io import BytesIO
from warnings import warn

T = TypeVar("T")

# This describes molli library format
# .mli

# File header consists of 4 mandatory identifying bytes and 28 optional bytes

_FILE_HEADER = Struct(b">4s28s")

# Header of each block of data is an integer that indicates the length of the record
# (uint32): length of record
# (uchar): block flags [RESERVED]
# (uchar): length of record key
_BLOCK_HEADER = Struct(b">BI")


class _Library(Generic[T]):
    """
    This class allows optimal storage of binary objects with a relatively small overhead.
    Items can be retrieved by index or by key.
    It is presumed that the items stored need to be in a messagepack serializable object
    """

    GUARD1 = b"ML10"
    GUARD2 = b"Library"

    default_encoder = msgpack.dumps
    default_decoder = msgpack.loads

    def __init__(
        self,
        path: Path | str = None,
        readonly: bool = True,
        encoder: Callable[[T], bytes] = ...,
        decoder: Callable[[bytes], T] = ...,
    ):
        self.path = path
        self.readonly = readonly
        self.encoder = encoder
        self.decoder = decoder

        with self:
            pass

    @property
    def encoder(self):
        return self._encoder

    @encoder.setter
    def encoder(self, val):
        if val is Ellipsis:
            self._encoder = type(self).default_encoder
        elif isinstance(val, Callable):
            self._encoder = val
        else:
            raise ValueError("This is not a valid encoder")

    @property
    def decoder(self):
        return self._decoder

    @decoder.setter
    def decoder(self, val):
        if val is Ellipsis:
            self._decoder = type(self).default_decoder
        elif isinstance(val, Callable):
            self._decoder = val
        else:
            raise ValueError("This is not a valid decoder")

    @classmethod
    def new(
        cls: type[_Library],
        path: Path | str,
        overwrite: bool = False,
        encoder: Callable[[T], bytes] = ...,
        decoder: Callable[[bytes], T] = ...,
    ) -> _Library[T]:
        _path = Path(path)

        if _path.is_file() and not overwrite:
            _path = unique_path(_path)
            warn(f"Requested path was not available. New file name: {_path}")

        with open(_path, "wb") as f:
            f.write(_FILE_HEADER.pack(cls.GUARD1, cls.GUARD2))

        return cls(_path, readonly=False, encoder=encoder, decoder=decoder)

    @classmethod
    def concatenate(
        cls: type[_Library],
        path: Path | str,
        sources: Iterable[Path | str],
        overwrite: bool = False,
        encoder: Callable[[T], bytes] = ...,
        decoder: Callable[[bytes], T] = ...,
    ) -> _Library[T]:
        _path = Path(path)

        if _path.is_file() and not overwrite:
            _path = unique_path(_path)
            warn(f"Requested path was not available. New file name: {_path}")

        with open(_path, "w+b" if overwrite else "x+b") as out_f:
            out_f.write(_FILE_HEADER.pack(cls.GUARD1, cls.GUARD2))

            for source in sources:
                with open(source, "rb") as src_f:
                    src_f.seek(_FILE_HEADER.size)
                    while chunk := src_f.read(131072):
                        out_f.write(chunk)

        return cls(_path, readonly=False, encoder=encoder, decoder=decoder)

    def __enter__(self):
        self.open()
        self.goto(0)
        return self

    def open(self, readonly: bool = ...) -> _Library[T]:
        if readonly is not Ellipsis:
            self.readonly = readonly

        if self.readonly:
            fs = open(self.path, "rb")
        else:
            fs = open(self.path, "r+b")

        self._stream = fs
        self._isopen = True

        h1, h2 = _FILE_HEADER.unpack(fs.read(_FILE_HEADER.size))

        self._block_offsets = []
        self._block_keys = []

        pos = fs.tell()
        eof = fs.seek(0, 2)
        fs.seek(pos)
        nb = 0

        while pos < eof:
            newpos, key = self._read_block_key()
            self._block_offsets.append(pos)
            self._block_keys.append(key)
            nb += 1
            pos = newpos

    def keys(self):
        return self._block_keys

    def close(self):
        # self._stream.seek(_FILE_HEADER.size - 4)
        # self._stream.write(pack(">I", len(self)))
        self._stream.close()
        self._isopen = False

    def __exit__(self, *args):
        self.close()

    def goto(self, block: int):
        if len(self._block_offsets) > 0:
            self._stream.seek(self._block_offsets[block], 0)
        else:
            self._stream.seek(_BLOCK_HEADER.size)

    def _read_block_key(self) -> tuple[int, str]:
        """
        Get the name and flags from current position
        The cursor moves to the next block
        """
        pos = self._stream.tell()
        key_size, data_size = _BLOCK_HEADER.unpack(self._stream.read(_BLOCK_HEADER.size))
        _key: bytes = self._stream.read(key_size)
        self._stream.seek(data_size, 1)
        return self._stream.tell(), _key.decode("ascii")

    def _read(self) -> bytes:
        key_size, data_size = _BLOCK_HEADER.unpack(self._stream.read(_BLOCK_HEADER.size))
        _key: bytes = self._stream.read(key_size)
        return self._stream.read(data_size)

    # @check_open
    def append(self, key: str, obj: T):
        _eof = self._stream.seek(0, 2)
        self._block_offsets.append(_eof)
        data = self.encoder(obj)
        _key = key.encode("ascii")
        header = _BLOCK_HEADER.pack(len(_key), len(data))

        self._stream.write(header + _key + data)

    # @check_open
    def get(self, i: int) -> T:
        if i < 0 or i > len(self):
            raise IndexError(f"Index {i} is out of range for len(sequence) = {len(self)}")
        self.goto(i)
        return self.decoder(self._read())

    def batch(self, start: int, batch_size: int):
        for i in range(start, start + batch_size):
            try:
                item = self.get(i)
            except IndexError:
                pass
            else:
                yield item

    def yield_in_batches(self, batchsize: int = 256):
        """Read file in chunks"""
        for i in range(len(self) // batchsize + 1):
            # i = chunk index
            batch: list[T] = []
            with self:
                for j in range(i * batchsize, min(len(self), (i + 1) * batchsize)):
                    batch.append(self.get(j))
            yield batch

    def __iter__(self):
        # self._current_block = 0
        self.__enter__()
        return self

    def __next__(self) -> T:
        try:
            data = self._read()
        except:
            self.__exit__()
            raise StopIteration
        else:
            return self.decoder(data)

    def __len__(self):
        return len(self._block_offsets)

    def __getitem__(self, locator: int | str | slice | tuple):
        with self:
            match locator:
                case int() as i:
                    return self.get(i)

                case str() as s:
                    i = self._block_keys.index(s)
                    return self.get(i)

                case slice() as slc:
                    return [self.get(i) for i in range(*slc.indices(len(self)))]

                case [*items]:
                    return [self[item] for item in items]
