from __future__ import annotations
from struct import pack, unpack, Struct
from pathlib import Path
from typing import TypeVar, Generic, Callable, Iterable
from functools import wraps
from enum import IntEnum
import msgpack
from .._aux import unique_path
from io import BytesIO
from warnings import warn
from threading import RLock
from attrs import define

_FILE_HEADER = Struct(b">16s255p")
"""
A plain byte string that is meant for file identification purposes
- `16  (bytes)` Header 1. File type identifier.
- `256 (bytes)` Header 2. Comment string.
"""

_BLOCK_HEADER = Struct(b">BI")
"""
5 byte record header
- `1 (uchar)`   Key length 
- `4 (uint)`    Record length
"""

@define(slots=True, frozen=True)
class BMFRecord:
    pos: int
    key_len: int
    record_len: int

    @property
    def size(self):
        return _BLOCK_HEADER.size + self.key_len + self.record_len 
    
    @property
    def pos1(self):
        return _BLOCK_HEADER.size + self.pos

    @property
    def end(self):
        return self.pos + self.size


class BMFFile:
    """
    This class allows optimal storage of binary strings with a small overhead.
    This is roughl
    Items can be retrieved by index or by key.
    It is presumed that the 
    """

    FILE_H1_DEFAULT = b"BMFF001"

    def __init__(
        self,
        path: Path | str = None,
        mode: str = "r",
        *,
        h1: bytes = None,
        h2: bytes = None,
    ):
        """
        mode must be in ("r", "x", "a")
        """
        self.path = Path(path)
        
        if mode in ("r", "x", "w", "a"):
            self.mode = mode 
        else:
            raise ValueError
        self.h1 = h1 
        self.h2 = h2
        self._toc: dict[bytes, BMFRecord] = dict()
        self._lock = RLock()
          
    def open(self):
        self._lock.acquire()
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
            self._lock.release()
            raise
    
    def _pack_write(self, struct: Struct, *args):
        tbw = struct.pack(*args)
        return self._stream.write(tbw)
    
    def _unpack_read(self, struct: Struct, default = None):
        try:
            tbu = self._stream.read(struct.size)
            res = struct.unpack(tbu)
        except:
            return default
        else:
            return res
    
    def write_header(self):
        h1 = self.h1 or self.__class__.FILE_H1_DEFAULT
        self._pack_write(_FILE_HEADER, h1, self.h2)
    
    def read_header(self):
        self._stream.seek(0)
        self.h1, self.h2 = self._unpack_read(_FILE_HEADER)
    
    def map_blocks(self):
        # File delimiters
        pos = _FILE_HEADER.size
        self._stream.seek(pos)

        while (blk_header := self._unpack_read(_BLOCK_HEADER, None)):
            key_len, record_len = blk_header
            key = self._stream.read(key_len)

            # This updates the table of contents
            self._toc[key] = (record := BMFRecord(pos, key_len, record_len))
            pos += record.size

            # Skip over the record for the purposes of mapping blocks
            self._stream.seek(record_len, 1)

    def append(self, key: bytes, value: bytes):
        if len(self._blocks) > 0:
            eof = self._blocks[-1].end
        else:
            eof = _FILE_HEADER.size

        if key in self._toc:
            raise KeyError(f"Key {key} already exists.")
        else:
            self._blocks.append(BMFRecord(eof, len(key), len(value)))
            self._toc[key] = len(self._blocks) - 1
        
        self._stream.seek(eof)
        self._pack_write(_BLOCK_HEADER, len(key), len(value))
        self._stream.write(key)
        self._stream.write(value)
    
    def read_kv(self, _block_id_or_key: int | bytes) -> tuple[bytes, bytes]:
        if isinstance(_block_id_or_key, int):
            block = self._blocks[_block_id_or_key]
        else:
            block = self._blocks[self._toc[_block_id_or_key]]
        
        self._stream.seek(block.pos1)
        key = self._stream.read(block.key_len)
        val = self._stream.read(block.record_len)
        return key, val

    def keys(self):
        return self._toc.keys()            

    def close(self):
        self._stream.close()
        self._lock.release()
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, *args):
        self.close()
    
    def __getitem__(self, key: bytes | Iterable[bytes]):
        if isinstance(key, bytes):
            pass





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
