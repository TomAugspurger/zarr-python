"""
A custom, *simple* reader for Zarr Arrays.

The goals here, in order of priority:

1. Correctness
2. Performance
3. Simplicity

We will allow *zero* unnecessary memory allocations.

For now, we don't allow specifying an output buffer. It leaks into
too many spots to make it worth supporting at the moment.

## Reading Multiple Chunks

We can only support zero-copy reads of multiple chunks when
*each* chunk is contiguous.

In the following diagrams, the numbers indicate chunk IDs.

This is allowed (assuming C-major order):

    0 0 0 0 1 1 1 1
    2 2 2 2 3 3 3 3

This is not (assuming C-major order).

    0 0 0 0 1 1 1 1
    0 0 0 0 1 1 1 1

I think we can summarize this rule as something like: "you can only
zero-copy read multiple chunks if the chunk size matches the array size
in all but the last dimension?" Maybe?
"""

import asyncio
import math
import pathlib
from typing import Any, Literal, cast

import numpy as np
import pytest

import zarr
import zarr.abc.store
import zarr.api.asynchronous
import zarr.codecs
import zarr.codecs.zstd
import zarr.core.indexing
import zarr.storage
from zarr.core.array import AsyncArray
from zarr.core.buffer import BufferPrototype, default_buffer_prototype
from zarr.core.buffer.core import Buffer, NDArrayLikeOrScalar, NDBuffer
from zarr.core.chunk_grids import RegularChunkGrid
from zarr.core.indexing import BasicIndexer, BasicSelection
from zarr.core.metadata.v3 import ArrayV3Metadata

TASK_TYPE = Literal["read", "decode"]


async def get_wrapper(
    store: zarr.abc.store.Store, key: str, cp: zarr.core.indexing.ChunkProjection
) -> tuple[TASK_TYPE, Buffer | None, zarr.core.indexing.ChunkProjection]:
    result = await store.get(key, prototype=default_buffer_prototype())
    return "read", result, cp


async def decode_after_read(
    store: zarr.abc.store.Store,
    key: str,
    cp: zarr.core.indexing.ChunkProjection,
    codec: zarr.codecs.zstd.ZstdCodec,
    out: NDBuffer,
) -> None:
    result = await store.get(key, prototype=default_buffer_prototype())
    await asyncio.to_thread(
        codec._zstd_codec.decode,
        result.as_array_like(),  # type: ignore[attr-defined]
        out.as_ndarray_like()[cp.out_selection],  # type: ignore[arg-type]
    )


def is_contiguous_indexer(indexer: BasicIndexer) -> bool:
    """
    Check whether an indexer induces a contiguous read.

    An indexer induces a contiguous read if:

    - ...

    """
    for dim_indexer, shape in zip(
        reversed(indexer.dim_indexers[1:]), reversed(indexer.shape[1:]), strict=True
    ):
        if dim_indexer.dim_chunk_len < dim_indexer.dim_len and dim_indexer.dim_chunk_len < shape:
            return False

    return True


async def getitem(
    array: AsyncArray[ArrayV3Metadata],
    selection: BasicSelection,
    prototype: BufferPrototype,
) -> NDArrayLikeOrScalar:
    indexer = BasicIndexer(selection, array.metadata.shape, array.metadata.chunk_grid)

    if not all(cp.is_complete_chunk for cp in indexer):
        # We can eventually support contiguous slices off the ends.
        # We can't (ever?) support fancy indexing.
        raise NotImplementedError("Partial chunks are not supported yet")

    # what is this used for?
    _chunk_spec = array.metadata.get_chunk_spec(
        _chunk_coords=None,  # type: ignore[arg-type]
        array_config=array._config,
        prototype=prototype,
    )

    # Get your bytes. Eventually, this will need to support zero allocation readinto, but not yet.
    keys = [array.metadata.encode_chunk_key(cp.chunk_coords) for cp in indexer]
    full_keys = [(array.store_path / key).path for key in keys]

    match array.metadata.codecs:
        case [_]:
            # just bytescodec. Do a zero-copy readinto
            out = prototype.nd_buffer.create(
                shape=indexer.shape,
                dtype=array.metadata.dtype,
                order=array.order,
                fill_value=array.metadata.fill_value,
            )

            if is_contiguous_indexer(indexer):
                # assert isinstance(cp.out_selection, tuple), type(cp.out_selection)
                coros = [
                    array.store.get_into(
                        full_key,
                        out.as_ndarray_like()[cp.out_selection].ravel().view("b"),  # type: ignore[arg-type]
                    )
                    for full_key, cp in zip(full_keys, indexer, strict=True)
                ]
                await asyncio.gather(*coros)

            else:
                coros = [
                    get_wrapper(array.store, full_key, cp)
                    for full_key, cp in zip(full_keys, indexer, strict=True)
                ]
                for result in asyncio.as_completed(coros):
                    (_, tmp_buffer, cp) = await result
                    if tmp_buffer is not None:
                        tmp_array = np.frombuffer(
                            tmp_buffer.as_array_like().ravel().view("b"),  # type: ignore[attr-defined]
                            dtype=array.metadata.dtype,  # type: ignore[attr-defined]
                        ).reshape(cp.shape)
                        out.as_ndarray_like()[cp.out_selection] = tmp_array  # type: ignore[arg-type]

            return out.as_ndarray_like()
        case [_bytes_codec, codec]:
            # bytescodec + something else.
            # We can do a zero-copy readinto for the bytescodec.

            # case 1: each chunk in the user's slice is contiguous in the output.
            out = prototype.nd_buffer.create(
                shape=indexer.shape,
                dtype=array.metadata.dtype,
                order=array.order,
                fill_value=array.metadata.fill_value,
            )
            assert isinstance(codec, zarr.codecs.zstd.ZstdCodec)

            coros = [
                get_wrapper(array.store, full_key, cp)
                for full_key, cp in zip(full_keys, indexer, strict=True)
            ]

            if is_contiguous_indexer(indexer):
                for result in asyncio.as_completed(coros):
                    (_, buffer, cp) = await result
                    if buffer is not None:
                        codec._zstd_codec.decode(
                            buffer.as_array_like(),
                            out.as_ndarray_like()[cp.out_selection],  # type: ignore[arg-type]
                        )
            else:
                # The chunks fill non-contiguous regions of the output.
                # Therefore, a zero-copy readinto is just not possible.
                for result in asyncio.as_completed(coros):
                    (_, buffer, cp) = await result
                    if buffer is not None:
                        tmp_bytes = codec._zstd_codec.decode(
                            buffer.as_array_like(),  # type: ignore[attr-defined]
                            # out.as_ndarray_like()[cp.out_selection],  # type: ignore[arg-type]
                        )
                        tmp_array = np.frombuffer(tmp_bytes, dtype=array.metadata.dtype).reshape(
                            cp.shape
                        )
                        out.as_ndarray_like()[cp.out_selection] = tmp_array  # type: ignore[arg-type]

            return out.as_ndarray_like()
        case _:
            raise NotImplementedError


async def set(array: zarr.AsyncArray, expected: np.ndarray) -> None:
    await array.setitem(slice(None), expected)


async def getitem_nobarrier(
    array: AsyncArray[ArrayV3Metadata],
    selection: BasicSelection,
    prototype: BufferPrototype,
) -> NDArrayLikeOrScalar:
    indexer = BasicIndexer(selection, array.metadata.shape, array.metadata.chunk_grid)

    if not all(cp.is_complete_chunk for cp in indexer):
        # We can eventually support contiguous slices off the ends.
        # We can't (ever?) support fancy indexing.
        raise NotImplementedError("Partial chunks are not supported yet")

    if not is_contiguous_indexer(indexer):
        raise NotImplementedError("Non-contiguous slices are not supported yet")

    # Get your bytes. Eventually, this will need to support zero allocation readinto, but not yet.
    keys = [array.metadata.encode_chunk_key(cp.chunk_coords) for cp in indexer]
    full_keys = [(array.store_path / key).path for key in keys]

    match array.metadata.codecs:
        case [_]:
            # just bytescodec. Do a zero-copy readinto
            out = prototype.nd_buffer.create(
                shape=indexer.shape,
                dtype=array.metadata.dtype,
                order=array.order,
                fill_value=array.metadata.fill_value,
            )
            coros = [
                array.store.get_into(
                    full_key,
                    out.as_ndarray_like()[cp.out_selection].ravel().view("b"),  # type: ignore[arg-type]
                )
                for full_key, cp in zip(full_keys, indexer, strict=True)
            ]
            await asyncio.gather(*coros)
            return out.as_ndarray_like()

        case [_bytes_codec, codec]:
            assert isinstance(codec, zarr.codecs.zstd.ZstdCodec)
            out = prototype.nd_buffer.create(
                shape=indexer.shape,
                dtype=array.metadata.dtype,
                order=array.order,
                fill_value=array.metadata.fill_value,
            )

            # We specifically avoid a barrier between all the "read" tasks and all the
            # "decode" tasks. This lets us start decoding before all the reads are
            # complete, which avoids a single. slow read from blocking the entire
            # process.
            # Let's try using TaskGroups?
            # No: way too slow? Or something. We want more explicit control.

            # What if we just await the final decodes? And each decode would be responsible
            # for await its own read?

            coros = [
                decode_after_read(array.store, full_key, cp, codec, out)
                for full_key, cp in zip(full_keys, indexer, strict=True)
            ]
            await asyncio.gather(*coros)

            return out.as_ndarray_like()
        case _:
            raise NotImplementedError


@pytest.mark.parametrize(
    ("selection", "expected"),
    [
        ((slice(500), slice(500)), True),  # top-left chunk, OK
        ((slice(None), slice(500)), True),  # top-left + bottom-lft, OK
        ((slice(0, 500), slice(None)), False),  # top-left + top-right not OK
    ],
)
def test_is_contiguous_read(selection: BasicSelection, expected: bool) -> None:
    indexer = BasicIndexer(selection, (1000, 1000), RegularChunkGrid(chunk_shape=(500, 500)))
    result = is_contiguous_indexer(indexer)
    assert result is expected


SHAPE = (1000, 1000)
CHUNKS = (500, 500)


@pytest.fixture
def store(tmp_path: pathlib.Path) -> zarr.storage.LocalStore:
    return zarr.storage.LocalStore(tmp_path / "test.zarr")


@pytest.mark.parametrize(
    "filters_compressors",
    [
        ("auto", "auto"),
        ([], []),
    ],
)
@pytest.mark.parametrize(
    "selection",
    [
        (slice(500), slice(500)),  # top-left chunk
        (slice(None), slice(500)),  # top-left + bottom-left
        (slice(0, 500),),  # top-left + top-right, non-contiguous
        (slice(None), slice(None)),  # full array
        (slice(None),),  # top-left + top-right?
    ],
)
async def test_read(
    store: zarr.storage.LocalStore,
    filters_compressors: tuple[Any, Any],
    selection: tuple[slice, ...],
) -> None:
    expected = np.arange(math.prod(SHAPE), dtype="int32").reshape(SHAPE)
    filters, compressors = filters_compressors
    array = await zarr.api.asynchronous.create_array(
        store=store,
        name="uncompressed",
        shape=SHAPE,
        chunks=CHUNKS,
        dtype="int32",
        overwrite=True,
        filters=filters,
        compressors=compressors,
    )
    await set(array, expected)
    prototype = default_buffer_prototype()
    array = cast("AsyncArray[ArrayV3Metadata]", array)
    result = await getitem(array, selection, prototype)
    np.testing.assert_array_equal(result, expected[*selection])


async def test_getitem_nobarrier(store: zarr.storage.LocalStore) -> None:
    CONTIGUOUS_CHUNKS = (250, 1000)

    expected = np.arange(math.prod(SHAPE), dtype="int32").reshape(SHAPE)
    filters = "auto"
    compressors = "auto"
    selection = slice(None), slice(None)
    array = await zarr.api.asynchronous.create_array(
        store=store,
        name="compressed",
        shape=SHAPE,
        chunks=CONTIGUOUS_CHUNKS,
        dtype="int32",
        overwrite=True,
        filters=filters,
        compressors=compressors,
    )
    await set(array, expected)
    prototype = default_buffer_prototype()
    array = cast("AsyncArray[ArrayV3Metadata]", array)
    result = await getitem_nobarrier(array, selection, prototype)
    np.testing.assert_array_equal(result, expected[*selection])
