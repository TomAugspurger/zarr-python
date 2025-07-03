"""
A Zarr getitem implementation that's focused on simplicity and performance.
"""

import asyncio
import time
import concurrent.futures
import contextlib
import inspect
import math
import os
import pathlib
from collections.abc import Generator
from typing import Any, cast

import cupy
import numcodecs.abc
import numpy as np
import nvtx
import pytest

import zarr
import zarr.abc.codec
import zarr.abc.store
import zarr.api.asynchronous
import zarr.codecs.blosc
import zarr.codecs.transpose
import zarr.codecs.zstd
import zarr.core.buffer
import zarr.core.buffer.gpu
import zarr.core.chunk_grids
import zarr.core.indexing
import zarr.core.metadata.v3
import zarr.storage
from zarr.core.config import Config

_METRICS = []


def _is_contiguous_indexer(indexer: zarr.core.indexing.BasicIndexer) -> bool:
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


async def _get_wrapper(
    store: zarr.abc.store.Store,
    key: str,
    prototype: zarr.core.buffer.BufferPrototype,
) -> tuple[str, zarr.core.buffer.Buffer | None]:
    """
    A wrapper around Store.get that also returns the key used.
    """
    start = time.perf_counter()
    result = await store.get(key, prototype=prototype)
    end = time.perf_counter()
    _METRICS.append(
        {
            "op": "_get_wrapper",
            "start": start,
            "end": end,
            "nbytes": len(result) if result is not None else None,
        }
    )
    return key, result


async def _get_into_wrapper(
    store: zarr.abc.store.Store,
    key: str,
    # buffer: collections.abc.Buffer,
    host_memory_pool: cupy.cuda.pinned_memory.PinnedMemoryPool,
) -> tuple[str, np.ndarray | None]:
    """
    A wrapper around Store.get_into that also returns the key used.
    """
    # TODO: avoid this for remove file systems.
    # size = store.getsize(key)
    t1 = time.perf_counter()
    if isinstance(store, zarr.storage.LocalStore):
        size = os.path.getsize(store.root / key)  # type: ignore[attr-defined]
    else:
        size = await store.getsize(key)

    buffer = host_memory_pool.malloc(size)
    result = await store.get_into(key, buffer)

    assert result is not None

    t2 = time.perf_counter()
    _METRICS.append(
        {
            "op": "_get_into_wrapper",
            "start": t1,
            "end": t2,
            "nbytes": size,
        }
    )

    if result is None:
        return key, None
    else:
        # for some reason, malloc can return a buffer larger than what we requested.
        return key, np.asarray(buffer)[:size]


def _decode_wrapper(
    numcodecs_codecs: list[numcodecs.abc.Codec],
    buffer: zarr.core.buffer.Buffer | None,
    prototype: zarr.core.buffer.BufferPrototype,
    out: zarr.core.buffer.NDArrayLike,
    cp: zarr.core.indexing.ChunkProjection,
    use_decode_into: bool,
    fill_value: Any,
    chunk_shape: tuple[int, ...],
) -> None:
    if buffer is None:
        out[cp.out_selection] = fill_value  # type: ignore[assignment]
        return

    start = time.perf_counter()
    n_codecs = len(numcodecs_codecs)
    for i, codec in enumerate(numcodecs_codecs, 1):
        if i == n_codecs:
            # The final codec; we have our potential decode-into optimization available.
            if use_decode_into:
                # We can decode directly into the output buffer.
                codec.decode(
                    buffer.as_array_like(),
                    out[cp.out_selection].view("b").ravel(),  # type: ignore[assignment]
                )
            else:
                # We *must* use a temporary buffer here.
                tmp = codec.decode(buffer.as_array_like())
                assert tmp is not None
                # mmm this is no good.
                if isinstance(tmp, bytes):
                    # partial chunks
                    out[cp.out_selection] = (
                        prototype.buffer.from_bytes(tmp)
                        .as_array_like()
                        .view(out.dtype)
                        .reshape(chunk_shape)[cp.chunk_selection]
                    )
                else:
                    import cupy

                    out[cp.out_selection] = cupy.asarray(tmp).view(out.dtype).reshape(cp.shape)

        else:
            # We *must* use a temporary buffer here.
            tmp = codec.decode(buffer.as_array_like())
            assert tmp is not None
            buffer = prototype.buffer.from_bytes(tmp)

    stop = time.perf_counter()
    _METRICS.append(
        {
            "start": start,
            "end": stop,
            "op": "_decode_wrapper",
        }
    )


def _batch_decode_wrapper(
    numcodecs_codecs: list[numcodecs.abc.Codec],
    buffers: list[zarr.core.buffer.gpu.Buffer | None],
    out: zarr.core.buffer.NDArrayLike,
    cps: list[zarr.core.indexing.ChunkProjection],
    fill_value: Any,
) -> None:
    """
    A *batched* decoder, for high-latency, high-throughput decoders.
    """
    # TODO: consolidate buffers and cps into a single object.
    # this assumes that numcodecs_codec supports passing a list of buffers.
    # not necessarily true, but it is for nvcomp.
    # TODO: handle nones?
    # if buffer is None:
    #     out[cp.out_selection] = fill_value  # type: ignore[assignment]
    #     return

    n_codecs = len(numcodecs_codecs)
    if n_codecs > 1:
        raise NotImplementedError("batch decoding not supported for multi-stage codecs")

    codec = numcodecs_codecs[0]
    mask = [buffer is not None for buffer in buffers]
    tmp_arrays = iter(
        codec.decode([buffer.as_array_like() for buffer in buffers if buffer is not None])
    )

    assert tmp_arrays is not None
    for chunk_projection, is_valid in zip(cps, mask, strict=True):
        if is_valid:
            tmp_array = next(tmp_arrays)
            out[chunk_projection.out_selection] = (
                cupy.asarray(tmp_array).view(out.dtype).reshape(chunk_projection.shape)
            )
        else:
            out[chunk_projection.out_selection] = fill_value  # type: ignore[assignment]


async def getitem(
    array: zarr.AsyncArray[zarr.core.metadata.v3.ArrayV3Metadata],
    selection: zarr.core.indexing.BasicSelection,
    prototype: zarr.core.buffer.BufferPrototype | None = None,
    *,
    pool: concurrent.futures.Executor | None = None,
    read_timeout: float | None = None,
) -> zarr.core.buffer.NDArrayLike:
    """
    An Array.getitem implementation focused on simplicity and memory usage.

    Parameters
    ----------
    array : zarr.AsyncArray
        The array to slice.
    selection : zarr.core.indexing.BasicSelection
        The selection to getitem from.
    prototype : zarr.core.buffer.BufferPrototype, optional
        The prototype to use for the output buffer.

    Returns
    -------
    array : zarr.core.buffer.NDArrayLike
        An in-memory array.

    Notes
    -----

    This implementation

    - Reads chunks concurrently using asyncio
    - Decodes chunks in parallel using a thread pool

    I/O and decoding are overlapped. As read tasks complete, decode tasks are
    scheduled in the thread pool.

    When possible, i.e. when the chunks are contiguous in the output buffer, we
    avoid an intermediate buffer for the decoded bytes by decoding directly into
    the output buffer. Under this system, the theoretical peak memory usage is
    equal to the sum of:

    - the *compressed* chunk sizes
    - the intermediate buffers used by the decoder (if any)
    - the output buffer

    In practice, the compressed bytes can and will be freed from memory as soon
    as the decode is finished so the peak memory usage will be lower.
    """
    prototype = prototype or zarr.core.buffer.default_buffer_prototype()
    indexer = zarr.core.indexing.BasicIndexer(
        selection, array.metadata.shape, array.metadata.chunk_grid
    )
    pool = pool or concurrent.futures.ThreadPoolExecutor()

    def get_numcodecs_codec(codec: zarr.abc.codec.Codec) -> numcodecs.abc.Codec:
        match codec:
            case zarr.codecs.zstd.ZstdCodec():
                return codec._zstd_codec
            case zarr.codecs.blosc.BloscCodec():
                return codec._blosc_codec
            case _:  # pragma: no cover
                raise NotImplementedError(f"Codec {codec} not supported")

    bytes_bytes_codecs = [
        get_numcodecs_codec(x)
        for x in array.metadata.codecs
        if isinstance(x, zarr.abc.codec.BytesBytesCodec)
    ]

    # Stage 1: Read the bytes:
    keys = {array.metadata.encode_chunk_key(cp.chunk_coords): cp for cp in indexer}
    full_keys = {(array.store_path / key).path: cp for key, cp in keys.items()}
    use_decode_into = (
        _is_contiguous_indexer(indexer)
        and len(bytes_bytes_codecs) > 0
        and all(cp.is_complete_chunk for cp in indexer)
        and "out" in inspect.signature(bytes_bytes_codecs[-1].decode).parameters
    )
    try:
        use_decode_into = use_decode_into and (
            "out" in inspect.signature(bytes_bytes_codecs[-1].decode).parameters
        )
    except ValueError:
        # can't inspect PyCapsule objects from nvcomp :/
        use_decode_into = False

    out = prototype.nd_buffer.empty(
        shape=indexer.shape,
        dtype=array.dtype,
        order=array.order,
    )

    coros = [_get_wrapper(array.store, key, prototype) for key in full_keys]

    # We want to read from the store and decode (finished) chunks in parallel.
    # As soon as a read task is done, we'll schedule the decode task.
    # note: `asyncio.create_task` schedules this to run immediately.
    #
    # This is quite bad for the GPU. We really want to submit it a *batch*
    # of work to decode at once. A basic benchmark showed that decoding a
    # single buffer took 22ms, but decoding 100 buffers took 44ms.
    #
    # So how do we handle this? Another layer of indirection.
    # We'll have a Queue of decode tasks.

    read_tasks = [asyncio.create_task(coro) for coro in coros]
    decode_futures = []

    for read_future in asyncio.as_completed(read_tasks, timeout=read_timeout):
        # As soon as a read task is done, we'll schedule the decode task.
        key, maybe_buffer = await read_future
        cp = full_keys[key]

        decode_futures.append(
            pool.submit(
                _decode_wrapper,
                bytes_bytes_codecs,
                maybe_buffer,
                prototype,
                out.as_ndarray_like(),
                cp,
                use_decode_into,
                array.metadata.fill_value,
                chunk_shape=array.metadata.chunk_grid.chunk_shape,
            )
        )

    for future in concurrent.futures.as_completed(decode_futures):
        # And now we just check for errors.
        future.result()

    return out.as_ndarray_like()


async def getitem_gpu(
    array: zarr.AsyncArray[zarr.core.metadata.v3.ArrayV3Metadata],
    selection: zarr.core.indexing.BasicSelection,
    # prototype: zarr.core.buffer.BufferPrototype | None = None,
    *,
    pool: concurrent.futures.Executor | None = None,
    host_memory_pool: cupy.cuda.pinned_memory.PinnedMemoryPool | None = None,
    decode_batch_size: int = 256,
) -> cupy.ndarray:
    """
    Read a selection of chunks into a GPU-backed array.

    Notes
    -----
    This implementation ...

    -
    """
    # Differences
    # 1. Read into pinned host memory
    # 2. Decode batches
    host_memory_pool = host_memory_pool or cupy.get_default_pinned_memory_pool()
    pool = pool or concurrent.futures.ThreadPoolExecutor()
    indexer = zarr.core.indexing.BasicIndexer(
        selection, array.metadata.shape, array.metadata.chunk_grid
    )

    if len(array.metadata.codecs) != 2:
        raise NotImplementedError("Only single-stage codecs are supported")
    _, codec = array.metadata.codecs
    if not isinstance(codec, zarr.codecs.gpu.NvcompZstdCodec):
        raise NotImplementedError("Only nvcomp codecs are supported")

    assert isinstance(array.metadata.chunk_grid, zarr.core.chunk_grids.RegularChunkGrid)
    keys = {array.metadata.encode_chunk_key(cp.chunk_coords): cp for cp in indexer}
    full_keys = {(array.store_path / key).path: cp for key, cp in keys.items()}

    with nvtx.annotate("empty"):
        out = zarr.core.buffer.gpu.NDBuffer.empty(
            shape=indexer.shape,
            dtype=array.dtype,
            order=array.order,
        )

    read_timeout = None
    coros = [_get_into_wrapper(array.store, key, host_memory_pool) for key in full_keys]
    read_tasks = [asyncio.create_task(coro) for coro in coros]
    decode_futures = []
    mini_batch = []

    for read_future in asyncio.as_completed(read_tasks, timeout=read_timeout):
        key, maybe_buffer = await read_future
        cp = full_keys[key]
        mini_batch.append((key, maybe_buffer, cp))
        if len(mini_batch) == decode_batch_size:
            keys, maybe_pinned_buffers, cps = zip(*mini_batch, strict=True)

            maybe_buffers = [
                zarr.core.buffer.gpu.Buffer(maybe_buffer) if maybe_buffer is not None else None
                for maybe_buffer in maybe_pinned_buffers
            ]

            # these maybe_buffers are currently using pagable host memory.
            # We'd like them to be in pinned memory.

            decode_futures.append(
                pool.submit(
                    _batch_decode_wrapper,
                    [codec._zstd_codec],
                    maybe_buffers,
                    out.as_ndarray_like(),
                    cps,
                    array.metadata.fill_value,
                )
            )
            mini_batch = []

    if mini_batch:
        keys, maybe_pinned_buffers, cps = zip(*mini_batch, strict=True)

        # this triggers the host to device copies.
        maybe_buffers = [
            zarr.core.buffer.gpu.Buffer(cupy.asarray(maybe_buffer))
            if maybe_buffer is not None
            else None
            for maybe_buffer in maybe_pinned_buffers
        ]

        decode_futures.append(
            pool.submit(
                _batch_decode_wrapper,
                [codec._zstd_codec],
                maybe_buffers,
                out.as_ndarray_like(),
                cps,
                array.metadata.fill_value,
            )
        )

    for future in concurrent.futures.as_completed(decode_futures):
        # And now we just check for errors.
        # I wonder if we should even do this.
        # This is a barrier that prevents any downstream operations from completing.
        future.result()

    return out.as_ndarray_like()


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


@pytest.fixture(
    params=[
        {
            "array.v3_default_compressors": {
                "default": [{"name": "zstd", "configuration": {"level": 0, "checksum": False}}]
            }
        },
        {
            "array.v3_default_compressors": {
                "default": [{"name": "blosc", "configuration": {"clevel": 5}}]
            }
        },
    ]
)
def zarr_config(request: pytest.FixtureRequest) -> Generator[Config, Any, Any]:
    with zarr.config.set(request.param):
        yield zarr.config


@pytest.fixture
async def array(
    store: zarr.storage.LocalStore | zarr.storage.MemoryStore,
    shape_chunks: tuple[tuple[int, ...], tuple[int, ...]],
    zarr_config: Config,
) -> zarr.AsyncArray[zarr.core.metadata.v3.ArrayV3Metadata]:
    shape, chunks = shape_chunks
    z = await zarr.api.asynchronous.create_array(
        store=store,
        name="test",
        overwrite=True,
        shape=shape,
        chunks=chunks,
        dtype="int32",
        zarr_format=3,
    )
    z = cast(zarr.AsyncArray[zarr.core.metadata.v3.ArrayV3Metadata], z)
    await z.setitem(slice(None), np.arange(math.prod(shape), dtype="int32").reshape(shape))
    return z


@pytest.fixture
async def simple_array() -> zarr.AsyncArray[zarr.core.metadata.v3.ArrayV3Metadata]:
    """A non-parametrized array fixture."""
    store = zarr.storage.MemoryStore()
    shape, chunks = (10, 10), (5, 5)
    z = await zarr.api.asynchronous.create_array(
        store=store,
        name="test",
        overwrite=True,
        shape=shape,
        chunks=chunks,
        zarr_format=3,
        dtype="int32",
    )
    z = cast(zarr.AsyncArray[zarr.core.metadata.v3.ArrayV3Metadata], z)
    await z.setitem(slice(None), np.arange(math.prod(shape), dtype="int32").reshape(shape))
    return z


@pytest.fixture
async def simple_gpu_array() -> zarr.AsyncArray[zarr.core.metadata.v3.ArrayV3Metadata]:
    """A non-parametrized array fixture."""
    pytest.importorskip("cupy")
    store = zarr.storage.MemoryStore()
    shape, chunks = (10, 10), (5, 5)
    with zarr.config.enable_gpu():
        z = await zarr.api.asynchronous.create_array(
            store=store,
            name="test",
            overwrite=True,
            shape=shape,
            chunks=chunks,
            zarr_format=3,
            dtype="int32",
        )
        z = cast(zarr.AsyncArray[zarr.core.metadata.v3.ArrayV3Metadata], z)
        await z.setitem(slice(None), cupy.arange(math.prod(shape), dtype="int32").reshape(shape))
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
async def test_getitem(
    array: zarr.AsyncArray[zarr.core.metadata.v3.ArrayV3Metadata],
    selection: zarr.core.indexing.BasicSelection,
) -> None:
    pool = concurrent.futures.ThreadPoolExecutor()
    result = await getitem(array, selection, pool=pool)
    expected = await array.getitem(selection)
    np.testing.assert_array_equal(result, expected)


@pytest.fixture
async def array_chunked_xy(
    store: zarr.abc.store.Store,
) -> zarr.AsyncArray[zarr.core.metadata.v3.ArrayV3Metadata]:
    shape = (10, 10)
    chunks = (5, 5)
    dtype = "int32"
    z = await zarr.api.asynchronous.create_array(
        store=store,
        name="test",
        overwrite=True,
        shape=shape,
        chunks=chunks,
        dtype=dtype,
        zarr_format=3,
    )
    z = cast(zarr.AsyncArray[zarr.core.metadata.v3.ArrayV3Metadata], z)
    await z.setitem(slice(None), np.arange(math.prod(shape), dtype=dtype).reshape(shape))
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
async def test_getitem_chunked_xy(
    array_chunked_xy: zarr.AsyncArray[zarr.core.metadata.v3.ArrayV3Metadata],
    selection: zarr.core.indexing.BasicSelection,
) -> None:
    # some failures here when the selection is not contiguous.
    result = await getitem(array_chunked_xy, selection)
    expected = await array_chunked_xy.getitem(selection)
    np.testing.assert_array_equal(result, expected)


async def test_getitem_missing(store: zarr.abc.store.Store) -> None:
    z = await zarr.api.asynchronous.create_array(
        store=store,
        name="test",
        overwrite=True,
        shape=(10, 10),
        chunks=(5, 5),
        zarr_format=3,
        dtype="int32",
    )
    z = cast(zarr.AsyncArray[zarr.core.metadata.v3.ArrayV3Metadata], z)

    result = await getitem(z, slice(None))
    expected = await z.getitem(slice(None))
    np.testing.assert_array_equal(result, expected)


async def test_multistage_codec(store: zarr.abc.store.Store) -> None:
    z = await zarr.api.asynchronous.create_array(
        store=store,
        name="test",
        overwrite=True,
        shape=(10, 10),
        chunks=(5, 5),
        dtype="int32",
        zarr_format=3,
        # filters=[zarr.codecs.transpose.TransposeCodec(order=(1, 0))],
        compressors=[zarr.codecs.zstd.ZstdCodec(level=1), zarr.codecs.zstd.ZstdCodec(level=2)],
    )
    z = cast(zarr.AsyncArray[zarr.core.metadata.v3.ArrayV3Metadata], z)
    await z.setitem(slice(None), np.arange(math.prod(z.shape), dtype="int32").reshape(z.shape))
    result = await getitem(z, slice(None))
    expected = await z.getitem(slice(None))
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("decode_batch_size", [1, 2])
async def test_getitem_gpu(
    simple_gpu_array: zarr.AsyncArray[zarr.core.metadata.v3.ArrayV3Metadata],
    decode_batch_size: int,
) -> None:
    with zarr.config.enable_gpu():
        result = await getitem_gpu(
            simple_gpu_array, slice(None), decode_batch_size=decode_batch_size
        )
        expected = await simple_gpu_array.getitem(slice(None))
    np.testing.assert_array_equal(result.get(), expected.get())  # type: ignore[attr-defined]


async def main() -> None:  # pragma: no cover
    import dataclasses

    import matplotlib.pyplot as plt
    import pandas as pd
    import rich
    import seaborn as sns

    from bench import RecordSet, Run

    @dataclasses.dataclass
    class Params:
        shape: tuple[int, ...]
        chunks: tuple[int, ...]

    param_grid = [
        Params(
            shape=(10_000, 10_000),
            chunks=(100, 10_000),
        ),
        # Small chunks
        Params(
            shape=(10_000, 10_000),
            chunks=(10, 10_000),
        ),
        # Large chunks
        Params(
            shape=(10_000, 10_000),
            chunks=(10_000, 10_000),
        ),
        # non-contigous
        Params(
            shape=(10_000, 10_000),
            chunks=(10_000, 100),
        ),
        # Smaller shape
        Params(
            shape=(1_000, 1_000),
            chunks=(100, 1_000),
        ),
        Params(
            shape=(1_000, 1_000),
            chunks=(100, 100),
        ),
        Params(
            shape=(1_000, 1_000),
            chunks=(1_000, 1_000),
        ),
        Params(
            shape=(1_000, 1_000),
            chunks=(10, 10),
        ),
    ]

    records = []
    # for device in ["gpu", "cpu"]:
    for device in ["cpu", "gpu"]:
        if device == "gpu":
            ctx = zarr.config.enable_gpu()
            func = getitem_gpu
            kwargs = {"decode_batch_size": 100}
        else:
            ctx = contextlib.nullcontext()
            func = getitem
            kwargs = {}
        with ctx:
            for params in param_grid:
                # setup
                # SHAPE = (10_000, 10_000)
                # CHUNKS = (100, 10_000)
                rich.print(f"Running {device} {params.shape} {params.chunks}")
                DTYPE = "int32"
                NUMEL = math.prod(params.shape)
                store = zarr.storage.LocalStore("test.zarr")
                array = zarr.create_array(
                    store=store,
                    name="simple",
                    overwrite=True,
                    shape=params.shape,
                    chunks=params.chunks,
                    dtype=DTYPE,
                )
                if device == "gpu":
                    array[:] = cupy.arange(NUMEL, dtype="int32").reshape(params.shape)
                else:
                    array[:] = np.arange(NUMEL, dtype="int32").reshape(params.shape)
                pool = concurrent.futures.ThreadPoolExecutor(max_workers=(os.cpu_count() or 32) - 1)

                z = await zarr.api.asynchronous.open_array(
                    store="test.zarr", path="simple", zarr_format=3
                )

                rs_zarr_async_getitem = await RecordSet.acollect(
                    "Zarr async",
                    z.getitem,
                    selection=slice(None),
                )
                rs_simple = await RecordSet.acollect(
                    "simple",
                    func,
                    z,
                    slice(None),
                    pool=pool,
                    **kwargs,
                )
                # validate
                expected = rs_zarr_async_getitem.records[0].value
                if device == "gpu":
                    np.testing.assert_array_equal(rs_simple.records[0].value.get(), expected.get())
                else:
                    np.testing.assert_array_equal(rs_simple.records[0].value, expected)

                # report
                results = Run(
                    sets=[
                        rs_zarr_async_getitem,
                        rs_simple,
                    ],
                    nbytes=z.nbytes,
                )
                rich.print(results.summarize())
                # shape, chunks, throughput
                records.extend(
                    (
                        device,
                        str(params.shape),
                        str(params.chunks),
                        "simple",
                        expected.nbytes / x.duration,
                    )
                    for x in rs_simple.records
                )

                records.extend(
                    (
                        device,
                        str(params.shape),
                        str(params.chunks),
                        "zarr",
                        expected.nbytes / x.duration,
                    )
                    for x in rs_zarr_async_getitem.records
                )

    df = pd.DataFrame(
        records, columns=["device", "shape", "chunks", "implementation", "throughput"]
    )
    sns.catplot(
        data=df,
        hue="implementation",
        row="device",
        x="chunks",
        y="throughput",
        kind="bar",
        col="shape",
        # col_wrap=2,
        sharex=False,
    )
    plt.savefig("simple.png")


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(main())
