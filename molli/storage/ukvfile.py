"""
UKVFile (Micro Key-Value pair database)
"""
from __future__ import annotations
from struct import pack, unpack, Struct
from pathlib import Path
from typing import TypeVar, Generic, Callable, Iterable, Literal
from functools import wraps
from enum import IntEnum
import msgpack
from .._aux import unique_path
from io import UnsupportedOperation
from warnings import warn
from threading import RLock
from attrs import define

_FILE_HEADER = Struct(b">16sHI10x")
"""
A plain byte string that is meant for file identification purposes
- `16  (bytes)`  Header 1. File type identifier.
- `2   (ushort)`   h2 size
- `4   (uint)`   b0 size
- `10   (padding)`
"""

_BLOCK_HEADER = Struct(b">BI")
"""
5 byte record header
- `1 (ubyte)`   Key length
- `4 (uint)`    Record length
"""


@define(slots=True, frozen=True)
class UKVRecord:
    pos: int
    key_len: int
    record_len: int

    @property
    def size(self):
        return _BLOCK_HEADER.size + self.key_len + self.record_len

    @property
    def pos_k(self):
        return _BLOCK_HEADER.size + self.pos

    @property
    def pos_v(self):
        return _BLOCK_HEADER.size + self.pos + self.key_len

    @property
    def end(self):
        return self.pos + self.size


class UKVFile:
    """
    UKV files, micro-key-value storage is highly optimized
    Items can be retrieved by index or by key.
    It is presumed that the
    """

    FILE_H1_DEFAULT = b"ML10UKV01"

    def __init__(
        self,
        path: Path | str = None,
        mode: Literal["r", "w", "a", "x"] = "r",
        *,
        h1: bytes = None,
        h2: bytes = None,
        b0: bytes = None,
    ):
        """
        mode must be in ("r", "x", "a")
        if index is true
        """
        self.path = Path(path)

        if mode in {"r", "x", "w", "a"}:
            self.mode = mode
        else:
            raise ValueError
        self.h1 = h1 or self.__class__.FILE_H1_DEFAULT
        self.h2 = h2 or b""
        self.b0 = b0 or b""
        self._toc: dict[bytes, UKVRecord] = dict()
        self._last: bytes = None
        self._eof = None
        self._closed = True
        self.open()

    def __getstate__(self):
        _self_dict = self.__dict__.copy()
        if "_stream" in _self_dict:
            del _self_dict["_stream"]
        return _self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)

    @property
    def closed(self):
        return self._closed

    @property
    def writable(self):
        return not self.closed and self._stream.writable()

    def open(self, mode=None):
        # self._tlock.acquire()

        # if self.process_safe:
        #     from fasteners import InterProcessLock

        #     self._plock = InterProcessLock(self._lock_path)
        #     self._plock.acquire()

        if not self._closed:
            return

        self.mode = mode or self.mode

        try:
            match self.mode:
                case "r":
                    self._stream = self.path.open("rb")
                    self.read_header()
                    self.map_blocks()

                case "a":
                    self._stream = self.path.open("r+b")
                    self.read_header()
                    self.map_blocks()

                case "x":
                    self._stream = self.path.open("x+b")
                    self.write_header()

                case "w":
                    self._stream = self.path.open("w+b")
                    self.write_header()

        except:
            if hasattr(self, "_stream"):
                self._stream.close()
            # self._tlock.release()
            # if self.process_safe:
            #     self._plock.release()
            raise

        else:
            self._closed = False

    def _pack_write(self, struct: Struct, *args):
        tbw = struct.pack(*args)
        return self._stream.write(tbw)

    def _unpack_read(self, struct: Struct, default=None):
        try:
            tbu = self._stream.read(struct.size)
            res = struct.unpack(tbu)
        except:
            return default
        else:
            return res

    def write_header(self):
        if self.mode in {"w", "x"}:
            self._stream.seek(0)
            self._pack_write(_FILE_HEADER, self.h1, len(self.h2), len(self.b0))
            if len(self.h2) > 0:
                self._stream.write(self.h2)
            if len(self.b0) > 0:
                self._stream.write(self.b0)
            self._eof = self._bof
        else:
            raise Exception(
                "Unable to write header unless the file is only being created."
            )

    def read_header(self):
        self._stream.seek(0)
        self.h1, h2len, b0len = self._unpack_read(_FILE_HEADER)

        self.h2 = self._stream.read(h2len)
        self.b0 = self._stream.read(b0len)

    @property
    def _bof(self):
        return _FILE_HEADER.size + len(self.b0) + len(self.h2)

    def map_blocks(self):
        # File delimiters
        # If the file has already been opened, we do not need to reacquire all keys
        self._stream.seek(0, 2)
        if self._eof == self._stream.tell() and self._eof == (
            self._toc[self._last].end if self._last is not None else self._bof
        ):
            return

        pos = self._bof
        self._stream.seek(pos)

        key = None

        while blk_header := self._unpack_read(_BLOCK_HEADER, None):
            key_len, record_len = blk_header
            key = self._stream.read(key_len)

            # This updates the table of contents
            self._toc[key] = (record := UKVRecord(pos, key_len, record_len))
            pos += record.size

            # Skip over the record for the purposes of mapping blocks
            self._stream.seek(record_len, 1)

        self._last = key
        self._eof = pos

    def get(self, key: bytes):
        if self.closed:
            raise UnsupportedOperation("Cannot read from a closed stream")

        record = self._toc[key]
        self._stream.seek(record.pos_v)
        val = self._stream.read(record.record_len)
        return val

    def put(self, key: bytes, value: bytes):
        if not self.writable:
            raise UnsupportedOperation("Cannot write into a non-writable stream")

        if key in self._toc:
            raise KeyError(f"Key {key} already exists.")
        else:
            self._toc[key] = UKVRecord(self._eof, len(key), len(value))

        self._stream.seek(self._eof)
        self._pack_write(_BLOCK_HEADER, len(key), len(value))
        self._stream.write(key)
        self._stream.write(value)
        self._eof = self._stream.tell()

    def keys(self):
        return self._toc.keys()

    def close(self):
        self._stream.close()
        # self._tlock.release()
        self._closed = True

        if self.mode in {"x", "w"}:
            self.mode = "a"
        # if self.process_safe:
        #     self._plock.release()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()

    def __getitem__(self, key: bytes):
        return self.get(key)

    def __setitem__(self, key: bytes, val: bytes):
        self.put(key, val)

    def values(self):
        return (self.get(key) for key in self.keys())

    def items(self):
        return ((key, self.get(key)) for key in self.keys())

    def copy_items(self, _dest: UKVFile, keys: Iterable[bytes]):
        """
        This function is essential for copying
        """
        for k in keys:
            _dest.put(k, self.get(k))

    def truncate(self):
        self._stream.seek(self._bof)
        self._stream.truncate()

    # @classmethod
    # def concatenate(
    #     cls: type[_Library],
    #     path: Path | str,
    #     sources: Iterable[Path | str],
    #     overwrite: bool = False,
    #     encoder: Callable[[T], bytes] = ...,
    #     decoder: Callable[[bytes], T] = ...,
    # ) -> _Library[T]:
    #     _path = Path(path)

    #     if _path.is_file() and not overwrite:
    #         _path = unique_path(_path)
    #         warn(f"Requested path was not available. New file name: {_path}")

    #     with open(_path, "w+b" if overwrite else "x+b") as out_f:
    #         out_f.write(_FILE_HEADER.pack(cls.GUARD1, cls.GUARD2))

    #         for source in sources:
    #             with open(source, "rb") as src_f:
    #                 src_f.seek(_FILE_HEADER.size)
    #                 while chunk := src_f.read(131072):
    #                     out_f.write(chunk)

    #     return cls(_path, readonly=False, encoder=encoder, decoder=decoder)

    # def __enter__(self):
    #     self.open()
    #     self.goto(0)
    #     return self

    # def open(self, readonly: bool = ...) -> _Library[T]:
    #     if readonly is not Ellipsis:
    #         self.readonly = readonly

    #     if self.readonly:
    #         fs = open(self.path, "rb")
    #     else:
    #         fs = open(self.path, "r+b")

    #     self._stream = fs
    #     self._isopen = True

    #     h1, h2 = _FILE_HEADER.unpack(fs.read(_FILE_HEADER.size))

    #     self._block_offsets = []
    #     self._block_keys = []

    #     pos = fs.tell()
    #     eof = fs.seek(0, 2)
    #     fs.seek(pos)
    #     nb = 0

    #     while pos < eof:
    #         newpos, key = self._read_block_key()
    #         self._block_offsets.append(pos)
    #         self._block_keys.append(key)
    #         nb += 1
    #         pos = newpos

    # def keys(self):
    #     return self._block_keys

    # def close(self):
    #     # self._stream.seek(_FILE_HEADER.size - 4)
    #     # self._stream.write(pack(">I", len(self)))
    #     self._stream.close()
    #     self._isopen = False

    # def __exit__(self, *args):
    #     self.close()

    # def goto(self, block: int):
    #     if len(self._block_offsets) > 0:
    #         self._stream.seek(self._block_offsets[block], 0)
    #     else:
    #         self._stream.seek(_BLOCK_HEADER.size)

    # def _read_block_key(self) -> tuple[int, str]:
    #     """
    #     Get the name and flags from current position
    #     The cursor moves to the next block
    #     """
    #     pos = self._stream.tell()
    #     key_size, data_size = _BLOCK_HEADER.unpack(self._stream.read(_BLOCK_HEADER.size))
    #     _key: bytes = self._stream.read(key_size)
    #     self._stream.seek(data_size, 1)
    #     return self._stream.tell(), _key.decode("ascii")

    # def _read(self) -> bytes:
    #     key_size, data_size = _BLOCK_HEADER.unpack(self._stream.read(_BLOCK_HEADER.size))
    #     _key: bytes = self._stream.read(key_size)
    #     return self._stream.read(data_size)

    # # @check_open
    # def append(self, key: str, obj: T):
    #     _eof = self._stream.seek(0, 2)
    #     self._block_offsets.append(_eof)
    #     data = self.encoder(obj)
    #     _key = key.encode("ascii")
    #     header = _BLOCK_HEADER.pack(len(_key), len(data))

    #     self._stream.write(header + _key + data)

    # # @check_open
    # def get(self, i: int) -> T:
    #     if i < 0 or i > len(self):
    #         raise IndexError(f"Index {i} is out of range for len(sequence) = {len(self)}")
    #     self.goto(i)
    #     return self.decoder(self._read())

    # def batch(self, start: int, batch_size: int):
    #     for i in range(start, start + batch_size):
    #         try:
    #             item = self.get(i)
    #         except IndexError:
    #             pass
    #         else:
    #             yield item

    # def yield_in_batches(self, batchsize: int = 256):
    #     """Read file in chunks"""
    #     for i in range(len(self) // batchsize + 1):
    #         # i = chunk index
    #         batch: list[T] = []
    #         with self:
    #             for j in range(i * batchsize, min(len(self), (i + 1) * batchsize)):
    #                 batch.append(self.get(j))
    #         yield batch

    # def __iter__(self):
    #     # self._current_block = 0
    #     self.__enter__()
    #     return self

    # def __next__(self) -> T:
    #     try:
    #         data = self._read()
    #     except:
    #         self.__exit__()
    #         raise StopIteration
    #     else:
    #         return self.decoder(data)

    # def __len__(self):
    #     return len(self._block_offsets)

    # def __getitem__(self, locator: int | str | slice | tuple):
    #     with self:
    #         match locator:
    #             case int() as i:
    #                 return self.get(i)

    #             case str() as s:
    #                 i = self._block_keys.index(s)
    #                 return self.get(i)

    #             case slice() as slc:
    #                 return [self.get(i) for i in range(*slc.indices(len(self)))]

    #             case [*items]:
    #                 return [self[item] for item in items]
