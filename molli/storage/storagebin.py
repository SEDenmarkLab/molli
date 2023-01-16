from typing import Generic, TypeVar, NamedTuple
from struct import Struct
from enum import IntEnum, IntFlag, auto
import msgpack

SB_FILE_HEADER = Struct(">4s28s")
"""32 byte identifier
    - (4 byte) descriptor
    - (28 byte) optional
"""

SB_RECORD_HEADER = Struct(">BBHI")
"""8 byte record header: 
    - (uchar)  Record Type Indicator
    - (uchar)  Group ID
    - (ushort) Key length
    - (uint)   Record Length
    >>KKRRRR
"""
T = TypeVar("T")

class SBRecordType(IntEnum):
    unknown = 0
    file_meta = 1
    group_meta = 2
    data_meta = 3
    data_record = 4

    @classmethod
    def _missing_(cls, val):
        return cls.unknown



class StorageBin(Generic[T]):

    encoder = msgpack.dumps
    decoder = msgpack.loads

    def __init_subclass__(cls, encoder, decoder, **kwds) -> None:
        super().__init_subclass__(**kwds)
        cls.encoder = encoder
        cls.decoder = decoder
