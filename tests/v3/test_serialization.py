import dataclasses
import enum

import numcodecs
import numpy as np
import pytest

import zarr.core.array
import zarr.core.chunk_grids
import zarr.core.metadata
from zarr._serialization import (
    decode,
    encode,
    encode_value,
)
from zarr.codecs.transpose import TransposeCodec
from zarr.core.chunk_key_encodings import (
    ChunkKeyEncodingConfiguration,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (np.dtype("i8"), "int64"),
        (np.complex128(1.0, 1.0), [1.0, 1.0]),
        (1j + 1, [1, 1]),
        (np.int64(1), 1),
        (np.datetime64(1, "ns"), 1),
        (np.datetime64(1, "D"), 1),
        (enum.Enum("Dynamic", {"a": "A"}).a, "A"),
        (
            numcodecs.Blosc(),
            {"id": "blosc", "cname": "lz4", "clevel": 5, "shuffle": 1, "blocksize": 0},
        ),
    ],
)
def test_encode_value(value, expected):
    result = encode_value(value)
    assert result == expected


def test_round_trip():
    metadata = zarr.core.array.ArrayV3Metadata(
        shape=(1,),
        data_type=np.dtype("int64"),
        chunk_grid=zarr.core.array.RegularChunkGrid(
            configuration=zarr.core.chunk_grids.RegularChunkGridConfiguration(chunk_shape=(1,))
        ),
        chunk_key_encoding=zarr.core.array.DefaultChunkKeyEncoding(
            configuration=ChunkKeyEncodingConfiguration(separator=".")
        ),
        fill_value=0,
        codecs=(TransposeCodec(order=(0,)),),
        attributes={"a": 1},
        dimension_names=("a",),
    )

    result = decode(encode(metadata), type=zarr.core.array.ArrayV3Metadata)
    assert result == metadata


@dataclasses.dataclass
class WithTuples:
    x: tuple[int, ...]
    y: tuple[int, str]
    z: tuple[int, int, str]
    a: set[int]
    b: list[str]


def test_decode_tuples():
    obj = WithTuples(x=(1, 2), y=(3, "a"), z=(4, 5, "b"), a={0, 1}, b=["c", "d"])
    result = decode(encode(obj), WithTuples)
    assert result == obj
