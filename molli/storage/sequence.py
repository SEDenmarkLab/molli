from __future__ import annotations
from struct import pack, unpack
from pathlib import Path
from typing import TypeVar, Generic, Callable, Sequence
from functools import wraps

T = TypeVar("T")


def check_open(f):
    @wraps(f)
    def inner(self, *args, **kwds):
        if self._isopen:
            return f(self, *args, **kwds)
        else:
            raise IOError("Stream is closed")

    return inner


class _Sequence(Generic[T]):
    """
    This class allows optimal storage of binary objects with basically no overhead.
    Items can be retrieved using getitem
    """

    GUARD = b"ML10SEQN"

    def __init__(
        self,
        path: Path | str,
        /,
        readonly: bool = True,
        encoder: Callable[[T], bytes] = ...,
        decoder: Callable[[bytes], T] = ...,
    ):
        self.path = Path(path)
        self.readonly = readonly
        self.encoder = encoder
        self.decoder = decoder
        self._isopen = False

    @classmethod
    def new(
        cls: type[Sequence],
        path: Path | str,
        overwrite=False,
        safe=True,
        readonly=False,
        encoder=...,
        decoder=...,
    ) -> Sequence[T]:
        _path = Path(path)

        with open(_path, "wb") as f:
            f.write(cls.GUARD)

        return cls(_path, readonly=readonly, encoder=encoder, decoder=decoder)

    def __enter__(self):
        if self.readonly:
            fs = open(self.path, "rb")
        else:
            fs = open(self.path, "r+b")

        if fs.read(8) != type(self).GUARD:
            raise IOError

        self._stream = fs
        self._isopen = True

        pos = 8
        self._block_offsets = []

        while True:
            try:
                newpos = self._skip()
            except:
                break
            else:
                self._block_offsets.append(pos)
                pos = newpos

        self.goto(0)

    def __exit__(self, *args):
        self._stream.close()
        self._isopen = False

    # @check_open
    def goto(self, block: int):
        if len(self._block_offsets) > 0:
            self._stream.seek(self._block_offsets[block], 0)
        else:
            self._stream.seek(8)

    # @check_open
    def _read(self) -> bytes:
        """Read block in current position as bytes"""
        (len_block,) = unpack(">I", self._stream.read(4))
        return self._stream.read(len_block)

    # @check_open
    def _skip(self) -> int:
        """Read block in current position as bytes"""
        (len_block,) = unpack(">I", self._stream.read(4))
        return self._stream.seek(len_block, 1)

    # @check_open
    def append(self, obj: T):
        _eof = self._stream.seek(0, 2)
        self._block_offsets.append(_eof)
        data = self.encoder(obj)
        size = pack(">I", len(data))
        self._stream.write(size + data)

    # @check_open
    def get(self, i: int) -> T:
        if i < 0 or i > len(self):
            raise IndexError(
                f"Index {i} is out of range for len(sequence) = {len(self)}"
            )
        self.goto(i)
        return self.decoder(self._read())

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

    def __getitem__(self, locator: int | slice | tuple):
        with self:
            match locator:
                case int() as i:
                    yield self.get(i)

                case slice() as slc:
                    for i in range(*slc.indices(len(self))):
                        yield self.get(i)

                case [*items]:
                    for item in items:
                        yield from self[item]

        # for loc in locators:
        #     if isinstance(loc, int):
        #         return self.get(loc)
        #     elif isinstance(l)
        #         for i in range(*loc.indices(len(self))):
        #             yield self.get(i)
