"""
Some custom implementations of zarr.getitem.

The various implementations differ in:

1. Whether they use `asyncio` vs. a thread pool for decoding.
2. How they allocate the output array.
3. Whether they use `decodeinto` or copy decoded bytes into the output buffer

# Speed of Light analysis

Here are the results for a sample run:

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
┃ Benchmark                    ┃ Duration    ┃ Throughput (GB/s) ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
│ Zarr sync                    │ 0.25 ± 0.01 │ 1.63              │
│ Zarr async                   │ 0.25 ± 0.01 │ 1.59              │
│ Sync direct (fast alloc)     │ 0.07 ± 0.02 │ 5.41              │
│ Sync direct (default alloc)  │ 0.17 ± 0.01 │ 2.36              │
│ Async direct (fast alloc)    │ 0.07 ± 0.00 │ 5.44              │
│ Async direct (default alloc) │ 0.20 ± 0.02 │ 2.01              │
│ async read, sync decode      │ 0.07 ± 0.01 │ 6.10              │
│ Async direct (no decodeinto) │ 0.27 ± 0.01 │ 1.50              │
│ Split                        │ 0.26 ± 0.01 │ 1.54              │
└──────────────────────────────┴─────────────┴───────────────────┘
## Zarr sync / Zarr async

These are using zarr-python. They're roughly the same as expected.

## Sync direct

These intends to measure the overhead from using `asyncio.to_thread`. Uses a
thread pool to run "direct" operations, where "direct" means:

- Use `pathlib.Path.read_bytes` instead of the (async) `Store.get` method
- Use `numcodecs.abc.Codec.decode` instead of the (async) `Codec.decode` method

The "fast alloc" vs. "default alloc" are a couple strategies for how to allocate
the output array. Turns out doing `np.empty` is much faster than `np.full(...,
fill_value)`. Regardless of whether we're doing a `decodeinto` or copying
decoded bytes into the output buffer, we'd prefer to use `np.empty` probably?

## Async direct

Same as "Sync direct" but uses `asyncio.gather` instead of a thread pool. Uses
`Store.get`. Wraps the blocking `codec.decode` method in `asyncio.to_thread`.
This mostly measures the overhead from using async vs. a thread pool by
comparing to "Sync direct".

## No decodeinto

Same as "Async direct", but avoids the `decodeinto` optimization to decode into
the output buffer. This instead decodes each chunk to a temp buffer and then
copies each of those into the output buffer.

This would be required when we want to read a non-contiguous slice.

## Split

Does the work in two stages: 1.) read, 2.) decode. Similar to how zarr works.
Two big `asyncio.gather` calls. This potentially leaves perf on the table since
a slow reader can block us from starting to decode. Sub-optimial since reads and
decodes will typically. stress different parts of the system.

## Strategies

A few high-level strategies, and when they're available.

### Read / decode into

This cuts down on the number of memory allocations and temporary buffers by
doing the last operations (read or decode, depending on the codec pipeline)
directly into the output buffer.

This is available when each chunk is a contiguous slice of the output array
requested by the user.
"""

import asyncio
import concurrent.futures
import math
import pathlib
from typing import Any, cast

import numcodecs.abc
import numpy as np
import pytest
import rich.progress
import rich.table

import zarr.abc.store
import zarr.api.asynchronous
import zarr.codecs.zstd
import zarr.core.buffer
import zarr.core.indexing
import zarr.core.metadata.v3
import zarr.storage
from bench import RecordSet, Run
from zarr.core.common import MemoryOrder

SHAPE = (10_000, 10_000)
CHUNKS = (100, 10_000)
DTYPE = "int32"


"""
| Stage             | Time (us)     |
| ----------------- | ------------- |
| read              |   328 ± 6.26  |
| decode-frombuffer |  2510 ± 111   |
| view,reshape      | 0.658 ± .0213 |

Total: 328 + 2510 + 1 = 2839

| Stage            | Time (us)     |
| ---------------- | ------------- |
| read             |   328 ± 6.26  |
| alloc-deocdeinto |  2510 ± .066  |
| view,reshape     | 0.658 ± .0213 |

Total: 328 + 0.352 + 2530 + 0.658 = 2869
"""

# Comparing decode-frombuffer vs. alloc-decodeinto, there's really no difference
# **for a single chunk**. But for multiple chunks with a single output,
# alloc-decodeinto is faster since we avoid the temporary buffer allocation and
# memcpy (which takes ~ 260us; on the order of the read from disk, and much
# faster than the decode).
#
# So in general, we expect a codec using decodeinto rather than decode to have
# lower peak memory and faster by a bit.


def is_contiguous_indexer(indexer: zarr.core.indexing.BasicIndexer) -> bool:
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


def read_and_decode_one_sliced(
    p: pathlib.Path, codec: numcodecs.abc.Codec, out: np.ndarray, fill_value: Any
) -> None:
    try:
        buf = p.read_bytes()
    except FileNotFoundError:
        out[:] = fill_value
    else:
        codec.decode(buf, out=out)


def read_and_decode_one_noout(
    p: pathlib.Path,
    codec: numcodecs.abc.Codec,
    cp: zarr.core.indexing.ChunkProjection,
    dtype: np.dtype,
    order: MemoryOrder,
    fill_value: Any,
) -> np.ndarray:
    try:
        buf = p.read_bytes()
    except FileNotFoundError:
        return np.full(cp.shape, fill_value, dtype="b")
    else:
        decoded = codec.decode(buf)
        if decoded is not None:
            return np.frombuffer(decoded, dtype="b").view(dtype=dtype).reshape(cp.shape)

    return np.full(cp.shape, fill_value, dtype="b")


async def read_and_decode_one_sliced_async(
    store: zarr.abc.store.Store,
    key: str,
    codec: numcodecs.abc.Codec,
    out: np.ndarray,
    prototype: zarr.core.buffer.BufferPrototype,
    fill_value: Any,
) -> None:
    buf = await store.get(key, prototype=prototype)
    if buf is not None:
        await asyncio.to_thread(codec.decode, buf.as_array_like(), out=out)
    else:
        out[:] = fill_value


async def read_and_decode_one_sliced_async_no_decodeinto(
    store: zarr.abc.store.Store,
    key: str,
    codec: numcodecs.abc.Codec,
    prototype: zarr.core.buffer.BufferPrototype,
    cp: zarr.core.indexing.ChunkProjection,
) -> tuple[zarr.core.indexing.ChunkProjection, bytes | None]:
    buf = await store.get(key, prototype=prototype)
    if buf is not None:
        decoded = await asyncio.to_thread(codec.decode, buf.as_array_like())
        return (cp, decoded)
    return (cp, None)


def decode_wrapper(
    buf: zarr.core.buffer.Buffer | None,
    codec: numcodecs.abc.Codec,
    cp: zarr.core.indexing.ChunkProjection,
    dtype: np.dtype,
    order: MemoryOrder,
    fill_value: Any,
) -> np.ndarray:
    out = np.empty(cp.shape, dtype=dtype, order=order)
    if buf is not None:
        codec.decode(buf.as_array_like(), out=out)
    else:
        out[:] = fill_value

    return out


def fill(out: np.ndarray, fill_value: Any) -> None:
    out[:] = fill_value


def decode_wrapper_no_out(
    buf: zarr.core.buffer.Buffer | None,
    codec: numcodecs.abc.Codec,
    cp: zarr.core.indexing.ChunkProjection,
    dtype: np.dtype,
    order: MemoryOrder,
    fill_value: Any,
) -> np.ndarray:
    if buf is not None:
        decoded = codec.decode(buf.as_array_like())
        if decoded is not None:
            return np.frombuffer(decoded, dtype="b").view(dtype=dtype).reshape(cp.shape)
    return np.full(cp.shape, fill_value, dtype="b")
    # return out


async def read_and_decode_one_sliced_async_allocating(
    store: zarr.abc.store.Store,
    key: str,
    codec: numcodecs.abc.Codec,
    prototype: zarr.core.buffer.BufferPrototype,
    cp: zarr.core.indexing.ChunkProjection,
    dtype: np.dtype,
    order: MemoryOrder,
    fill_value: Any,
) -> np.ndarray:
    buf = await store.get(key, prototype=prototype)

    return await asyncio.to_thread(decode_wrapper, buf, codec, cp, dtype, order, fill_value)


def read_one(
    p: pathlib.Path, codec: numcodecs.abc.Codec, out: np.ndarray, out_slice: slice
) -> tuple[bytes, numcodecs.abc.Codec, np.ndarray, slice]:
    return p.read_bytes(), codec, out, out_slice


def decode_one(buf: bytes, codec: numcodecs.abc.Codec, out: np.ndarray, out_slice: slice) -> None:
    codec.decode(buf, out=out[out_slice])


def getitem_sync_direct(
    array: zarr.AsyncArray[zarr.core.metadata.v3.ArrayV3Metadata],
    selection: zarr.core.indexing.BasicSelection,
    prototype: zarr.core.buffer.BufferPrototype | None = None,
    *,
    pool: concurrent.futures.Executor | None = None,
    fast_alloc: bool = False,
) -> np.ndarray:
    pool = pool or concurrent.futures.ThreadPoolExecutor()
    prototype = prototype or zarr.core.buffer.default_buffer_prototype()
    indexer = zarr.core.indexing.BasicIndexer(
        selection, array.metadata.shape, array.metadata.chunk_grid
    )

    if not all(cp.is_complete_chunk for cp in indexer):
        # We can eventually support contiguous slices off the ends.
        # We can't (ever?) support fancy indexing.
        raise NotImplementedError("Partial chunks are not supported yet")

    if len(array.metadata.codecs) != 2:
        raise NotImplementedError("Only two-stage codecs are supported")

    if not isinstance(array.store, zarr.storage.LocalStore):
        raise NotImplementedError("Only LocalStore is supported")

    _, codec = array.metadata.codecs
    assert isinstance(codec, zarr.codecs.zstd.ZstdCodec)
    numcodecs_codec = codec._zstd_codec
    root = array.store.root

    # Get your bytes. Eventually, this will need to support zero allocation readinto, but not yet.
    keys = [array.metadata.encode_chunk_key(cp.chunk_coords) for cp in indexer]
    full_keys = [(array.store_path / key).path for key in keys]

    if is_contiguous_indexer(indexer):
        if fast_alloc:
            out = prototype.nd_buffer.from_ndarray_like(
                np.empty(  # type: ignore[call-overload]
                    shape=indexer.shape,
                    dtype=array.dtype,
                    order=array.order,
                )
            )
        else:
            out = prototype.nd_buffer.create(
                shape=indexer.shape,
                dtype=array.dtype,
                order=array.order,
                fill_value=array.metadata.fill_value,
            )

        tasks = []
        for cp, key in zip(indexer, full_keys, strict=True):
            chunk_out = out.as_ndarray_like()[cp.out_selection].view("b").ravel()  # type: ignore[assignment]
            tasks.append(
                (
                    read_and_decode_one_sliced,
                    root / key,
                    numcodecs_codec,
                    chunk_out,
                    array.metadata.fill_value,
                )
            )
        futures = [pool.submit(*task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            future.result()  # just checking for errors

        return cast(np.ndarray, out.as_ndarray_like())

    else:
        # since we don't have a contiguous indexer, we can't perform a readinto on the output
        # raise NotImplementedError("Non-contiguous slices are not supported yet")
        tasks = []
        for cp, key in zip(indexer, full_keys, strict=True):
            tasks.append(
                (
                    read_and_decode_one_noout,
                    root / key,
                    numcodecs_codec,
                    cp,
                    array.dtype,
                    array.order,
                    array.metadata.fill_value,
                )
            )
        futures = [pool.submit(*task) for task in tasks]
        arrays = [future.result() for future in concurrent.futures.as_completed(futures)]
        # I don't think concat reshape will work here?
        return np.concatenate(arrays).reshape(SHAPE)


async def getitem_async_direct(
    array: zarr.AsyncArray[zarr.core.metadata.v3.ArrayV3Metadata],
    selection: zarr.core.indexing.BasicSelection,
    prototype: zarr.core.buffer.BufferPrototype | None = None,
    *,
    fast_alloc: bool = False,
    use_decodeinto: bool = True,
    sync_decode: bool = False,
    pool: concurrent.futures.Executor | None = None,
) -> np.ndarray:
    """
    A getitem implementation that uses

    - await Store.get
    - optionally, decode into an output buffer
    - optionally, decode using a thread pool bypassing asyncio

    Parameters
    ----------
    array : zarr.AsyncArray
        The array to slice.
    selection : zarr.core.indexing.BasicSelection
        The selection to getitem from.
    prototype : zarr.core.buffer.BufferPrototype, optional
        The prototype to use for the output buffer.
    fast_alloc : bool, optional
    """
    prototype = prototype or zarr.core.buffer.default_buffer_prototype()
    indexer = zarr.core.indexing.BasicIndexer(
        selection, array.metadata.shape, array.metadata.chunk_grid
    )

    if not all(cp.is_complete_chunk for cp in indexer):
        # We can eventually support contiguous slices off the ends.
        # We can't (ever?) support fancy indexing.
        raise NotImplementedError("Partial chunks are not supported yet")

    if len(array.metadata.codecs) != 2:
        # This can be generalized pretty easily. All but the last codec
        # will output to a temporary buffer that's dynamically allocated.
        # The last codec will output to the output buffer.
        raise NotImplementedError("Only two-stage codecs are supported")

    _, codec = array.metadata.codecs
    assert isinstance(codec, zarr.codecs.zstd.ZstdCodec)
    numcodecs_codec = codec._zstd_codec

    # Get your bytes. Eventually, this will need to support zero allocation readinto, but not yet.
    keys = [array.metadata.encode_chunk_key(cp.chunk_coords) for cp in indexer]
    full_keys = [(array.store_path / key).path for key in keys]

    if sync_decode:
        # First, read the bytes asynchronously.
        coros = [array.store.get(key, prototype=prototype) for key in full_keys]
        buffers = await asyncio.gather(*coros)

        # Now, decode the bytes *synchronously* using the thread pool
        pool = pool or concurrent.futures.ThreadPoolExecutor()

        # TODO contiguous only.
        # this is mostly a copy-paste of above.
        if fast_alloc:
            out = prototype.nd_buffer.from_ndarray_like(
                np.empty(  # type: ignore[call-overload]
                    shape=indexer.shape,
                    dtype=array.dtype,
                    order=array.order,
                )
            )
        else:
            out = prototype.nd_buffer.create(
                shape=indexer.shape,
                dtype=array.dtype,
                order=array.order,
                fill_value=array.metadata.fill_value,
            )

        tasks = []
        for cp, buffer in zip(indexer, buffers, strict=True):
            chunk_out = out.as_ndarray_like()[cp.out_selection].view("b").ravel()  # type: ignore[assignment]
            if buffer is not None:
                tasks.append(
                    (
                        numcodecs_codec.decode,
                        buffer.as_array_like(),
                        chunk_out,
                    )
                )
            else:
                tasks.append(
                    (
                        fill,
                        chunk_out,
                        array.metadata.fill_value,
                    )
                )

        futures = [pool.submit(*task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            future.result()  # just checking for errors
    else:
        # this allocation on the main thread is bad-ish.
        coros = []

        if fast_alloc:
            out = prototype.nd_buffer.from_ndarray_like(
                np.empty(  # type: ignore[call-overload]
                    shape=indexer.shape,
                    dtype=array.dtype,
                    order=array.order,
                )
            )
        else:
            out = prototype.nd_buffer.create(
                shape=indexer.shape,
                dtype=array.dtype,
                order=array.order,
                fill_value=array.metadata.fill_value,
            )

        if is_contiguous_indexer(indexer) and use_decodeinto:
            for cp, key in zip(indexer, full_keys, strict=False):
                chunk_out = out.as_ndarray_like()[cp.out_selection].view("b").ravel()  # type: ignore[assignment]
                coros.append(
                    read_and_decode_one_sliced_async(
                        array.store,
                        key,
                        numcodecs_codec,
                        chunk_out,  # type: ignore[arg-type]
                        prototype=prototype,
                        fill_value=array.metadata.fill_value,
                    )
                )

            await asyncio.gather(*coros)

        else:
            # We are *not* using decodeinto. Each chunk is read into a temp
            # buffer and then copied into the output buffer.
            for cp, key in zip(indexer, full_keys, strict=False):
                # chunk_out = out.as_ndarray_like()[cp.out_selection].view("b").ravel()  # type: ignore[assignment]
                coros.append(
                    read_and_decode_one_sliced_async_no_decodeinto(
                        array.store,
                        key,
                        numcodecs_codec,
                        prototype=prototype,
                        cp=cp,
                    )
                )

            results = await asyncio.gather(*coros)

            if fast_alloc:
                out = prototype.nd_buffer.from_ndarray_like(
                    np.empty(  # type: ignore[call-overload]
                        shape=indexer.shape,
                        dtype=array.dtype,
                        order=array.order,
                    )
                )
            else:
                out = prototype.nd_buffer.create(
                    shape=indexer.shape,
                    dtype=array.dtype,
                    order=array.order,
                    fill_value=array.metadata.fill_value,
                )

            for cp, decoded in results:
                if decoded is not None:
                    out.as_ndarray_like()[cp.out_selection] = (
                        np.frombuffer(decoded, dtype="b").view(dtype=array.dtype).reshape(cp.shape)
                    )
                else:
                    out.as_ndarray_like()[cp.out_selection] = array.metadata.fill_value

    return cast(np.ndarray, out.as_ndarray_like())


async def getitem_split(
    array: zarr.AsyncArray[zarr.core.metadata.v3.ArrayV3Metadata],
    selection: zarr.core.indexing.BasicSelection,
    prototype: zarr.core.buffer.BufferPrototype | None = None,
) -> np.ndarray:
    # Split up the two (read) and (decode) steps.
    prototype = prototype or zarr.core.buffer.default_buffer_prototype()
    indexer = zarr.core.indexing.BasicIndexer(
        selection, array.metadata.shape, array.metadata.chunk_grid
    )

    if not all(cp.is_complete_chunk for cp in indexer):
        # We can eventually support contiguous slices off the ends.
        # We can't (ever?) support fancy indexing.
        raise NotImplementedError("Partial chunks are not supported yet")

    if len(array.metadata.codecs) != 2:
        raise NotImplementedError("Only two-stage codecs are supported")

    _, codec = array.metadata.codecs
    assert isinstance(codec, zarr.codecs.zstd.ZstdCodec)
    numcodecs_codec = codec._zstd_codec

    # Get your bytes. Eventually, this will need to support zero allocation readinto, but not yet.
    keys = [array.metadata.encode_chunk_key(cp.chunk_coords) for cp in indexer]
    full_keys = [(array.store_path / key).path for key in keys]

    coros = [array.store.get(key, prototype=prototype) for key in full_keys]

    buffers = await asyncio.gather(*coros)

    # TODO: handle non-contiguous slices
    # out = np.empty(indexer.shape, dtype=array.dtype, order=array.order)
    # for cp, buf in zip(indexer, buffers, strict=False):
    #     out[cp.out_selection] = buf.as_array_like()
    ndbuffers = [
        asyncio.to_thread(
            decode_wrapper_no_out,
            buf,
            numcodecs_codec,
            cp,
            array.dtype,
            array.order,
            array.metadata.fill_value,
        )
        for cp, buf in zip(indexer, buffers, strict=False)
    ]

    ndarrays = await asyncio.gather(*ndbuffers)
    return np.concatenate(ndarrays).reshape(SHAPE)

    # return out


@pytest.fixture(params=["local", "memory"])
def store(
    tmp_path: pathlib.Path, request: pytest.FixtureRequest
) -> zarr.storage.LocalStore | zarr.storage.MemoryStore:
    if request.param == "local":
        return zarr.storage.LocalStore(tmp_path / "test.zarr")
    else:
        return zarr.storage.MemoryStore()


@pytest.fixture(
    params=[
        ((100, 100), (1, 100)),
    ],
)
def shape_chunks(request: pytest.FixtureRequest) -> tuple[tuple[int, ...], tuple[int, ...]]:
    shape, chunks = request.param
    return shape, chunks


@pytest.fixture
async def array(
    store: zarr.storage.LocalStore, shape_chunks: tuple[tuple[int, ...], tuple[int, ...]]
) -> zarr.AsyncArray[zarr.core.metadata.v3.ArrayV3Metadata]:
    shape, chunks = shape_chunks
    z = await zarr.api.asynchronous.create_array(
        store=store,
        name="test",
        overwrite=True,
        shape=shape,
        chunks=chunks,
        dtype=DTYPE,
        zarr_format=3,
    )
    z = cast(zarr.AsyncArray[zarr.core.metadata.v3.ArrayV3Metadata], z)
    await z.setitem(slice(None), np.arange(math.prod(shape), dtype="int32").reshape(shape))
    return z


@pytest.mark.parametrize(
    "selection",
    [
        slice(None),
        (slice(1), slice(None)),
        (slice(None), slice(100)),
        (slice(1), slice(100)),
    ],
)
@pytest.mark.parametrize("fast_alloc", [True, False])
@pytest.mark.parametrize("sync_decode", [True, False])
async def test_getitem(
    array: zarr.AsyncArray[zarr.core.metadata.v3.ArrayV3Metadata],
    selection: zarr.core.indexing.BasicSelection,
    fast_alloc: bool,
    sync_decode: bool,
) -> None:
    if sync_decode:
        pool = concurrent.futures.ThreadPoolExecutor()
    else:
        pool = None
    result = await getitem_async_direct(
        array, selection, fast_alloc=fast_alloc, sync_decode=sync_decode, pool=pool
    )
    expected = await array.getitem(selection)
    np.testing.assert_array_equal(result, expected)


@pytest.fixture
async def array_chunked_xy(
    store: zarr.abc.store.Store,
) -> zarr.AsyncArray[zarr.core.metadata.v3.ArrayV3Metadata]:
    shape = (10, 10)
    chunks = (5, 5)
    z = await zarr.api.asynchronous.create_array(
        store=store,
        name="test",
        overwrite=True,
        shape=shape,
        chunks=chunks,
        dtype=DTYPE,
        zarr_format=3,
    )
    z = cast(zarr.AsyncArray[zarr.core.metadata.v3.ArrayV3Metadata], z)
    await z.setitem(slice(None), np.arange(math.prod(shape), dtype="int32").reshape(shape))
    return z


@pytest.mark.parametrize(
    "selection",
    [
        (slice(5), slice(5)),
        (slice(None), slice(5)),
        (slice(None), slice(None)),
        (slice(5), slice(None)),
    ],
)
@pytest.mark.parametrize("use_decodeinto", [True, False])
async def test_getitem_chunked_xy(
    array_chunked_xy: zarr.AsyncArray[zarr.core.metadata.v3.ArrayV3Metadata],
    selection: zarr.core.indexing.BasicSelection,
    use_decodeinto: bool,
) -> None:
    # some failures here when the selection is not contiguous.
    result = await getitem_async_direct(array_chunked_xy, selection, use_decodeinto=use_decodeinto)
    expected = await array_chunked_xy.getitem(selection)
    np.testing.assert_array_equal(result, expected)


async def main() -> None:
    # setup
    NUMEL = math.prod(SHAPE)
    store = zarr.storage.LocalStore("test.zarr")
    nc_array = zarr.create_array(
        store=store,
        name="custom-2-nc",
        overwrite=True,
        shape=SHAPE,
        chunks=CHUNKS[::-1],
        dtype=DTYPE,
    )
    nc_array[:] = np.arange(NUMEL, dtype="int32").reshape(SHAPE)
    # nc_z_sync = zarr.open_array(store="test.zarr", path="custom-2")
    # nc_z = await zarr.api.asynchronous.open_array(store="test.zarr", path="custom-2", zarr_format=3)

    array = zarr.create_array(
        store=store, name="custom-2", overwrite=True, shape=SHAPE, chunks=CHUNKS, dtype=DTYPE
    )
    array[:] = np.arange(NUMEL, dtype="int32").reshape(SHAPE)
    import os

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count())

    z_sync = zarr.open_array(store="test.zarr", path="custom-2")
    z = await zarr.api.asynchronous.open_array(store="test.zarr", path="custom-2", zarr_format=3)

    # collect
    rs_zarr_getitem = RecordSet.collect("Zarr sync", z_sync.__getitem__, slice(None))
    expected = rs_zarr_getitem.records[0].value

    rs_zarr_async_getitem = await RecordSet.acollect("Zarr async", z.getitem, selection=slice(None))

    rs_sync_direct = RecordSet.collect(
        "Sync direct (fast alloc)", getitem_sync_direct, z, slice(None), pool=pool, fast_alloc=True
    )
    rs_sync_direct_default = RecordSet.collect(
        "Sync direct (default alloc)",
        getitem_sync_direct,
        z,
        slice(None),
        pool=pool,
        fast_alloc=False,
    )

    rs_async_direct = await RecordSet.acollect(
        "Async direct (fast alloc)",
        getitem_async_direct,
        z,
        slice(None),
        fast_alloc=True,
        pool=pool,
    )
    rs_async_direct_default = await RecordSet.acollect(
        "Async direct (default alloc)",
        getitem_async_direct,
        z,
        slice(None),
        fast_alloc=False,
        pool=pool,
    )

    rs_async_read_sync_decorre_direct = await RecordSet.acollect(
        "async read, sync decode",
        getitem_async_direct,
        z,
        slice(None),
        fast_alloc=True,
        sync_decode=True,
        pool=pool,
    )

    rs_async_no_decodeinto = await RecordSet.acollect(
        "Async direct (no decodeinto)",
        getitem_async_direct,
        z,
        slice(None),
        use_decodeinto=False,
        fast_alloc=True,
        pool=pool,
    )
    rs_split = await RecordSet.acollect("Split", getitem_split, z, slice(None))

    # validate
    np.testing.assert_array_equal(rs_sync_direct.records[0].value, expected)
    np.testing.assert_array_equal(rs_sync_direct_default.records[0].value, expected)
    np.testing.assert_array_equal(rs_async_direct.records[0].value, expected)
    np.testing.assert_array_equal(rs_async_direct_default.records[0].value, expected)
    np.testing.assert_array_equal(rs_async_no_decodeinto.records[0].value, expected)
    np.testing.assert_array_equal(rs_split.records[0].value, expected)
    np.testing.assert_array_equal(rs_async_read_sync_decorre_direct.records[0].value, expected)

    # summarize

    results = Run(
        sets=[
            rs_zarr_getitem,
            rs_zarr_async_getitem,
            rs_sync_direct,
            rs_sync_direct_default,
            rs_async_direct,
            rs_async_direct_default,
            rs_async_read_sync_decorre_direct,
            rs_async_no_decodeinto,
            rs_split,
        ],
        nbytes=z.nbytes,
    )
    rich.print(results.summarize())


if __name__ == "__main__":
    asyncio.run(main())
