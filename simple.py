"""
┏━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
┃ Benchmark  ┃ Duration    ┃ Throughput (GB/s) ┃
┡━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
│ Zarr async │ 0.25 ± 0.02 │ 1.57              │
│ simple     │ 0.09 ± 0.02 │ 4.63              │
└────────────┴─────────────┴───────────────────┘
"""

import asyncio
import concurrent.futures
import inspect
import math
import os
import pathlib
from collections.abc import Generator
from typing import Any, cast

import numcodecs.abc
import numpy as np
import pytest

import zarr
import zarr.abc.codec
import zarr.abc.store
import zarr.api.asynchronous
import zarr.codecs.blosc
import zarr.codecs.transpose
import zarr.codecs.zstd
import zarr.core.buffer
import zarr.core.indexing
import zarr.core.metadata.v3
import zarr.storage
from zarr.core.config import Config


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
    store: zarr.abc.store.Store, key: str, prototype: zarr.core.buffer.BufferPrototype
) -> tuple[str, zarr.core.buffer.Buffer | None]:
    """
    A wrapper around Store.get that also returns the key used.
    """
    result = await store.get(key, prototype=prototype)
    return key, result


def _decode_wrapper(
    numcodecs_codecs: list[numcodecs.abc.Codec],
    buffer: zarr.core.buffer.Buffer | None,
    prototype: zarr.core.buffer.BufferPrototype,
    out: zarr.core.buffer.NDArrayLike,
    cp: zarr.core.indexing.ChunkProjection,
    use_decode_into: bool,
    fill_value: Any,
) -> None:
    if buffer is None:
        out[cp.out_selection] = fill_value  # type: ignore[assignment]
        return

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
                out[cp.out_selection] = (
                    prototype.buffer.from_bytes(tmp)
                    .as_array_like()
                    .view(out.dtype)
                    .reshape(cp.shape)
                )
        else:
            # We *must* use a temporary buffer here.
            tmp = codec.decode(buffer.as_array_like())
            assert tmp is not None
            buffer = prototype.buffer.from_bytes(tmp)


async def getitem(
    array: zarr.AsyncArray[zarr.core.metadata.v3.ArrayV3Metadata],
    selection: zarr.core.indexing.BasicSelection,
    prototype: zarr.core.buffer.BufferPrototype | None = None,
    *,
    pool: concurrent.futures.Executor | None = None,
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

    if not all(cp.is_complete_chunk for cp in indexer):
        # We can eventually support contiguous slices off the ends.
        # We can't (ever?) support fancy indexing.
        raise NotImplementedError("Partial chunks are not supported yet")

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
        and "out" in inspect.signature(bytes_bytes_codecs[-1].decode).parameters
    )

    out = prototype.nd_buffer.empty(
        shape=indexer.shape,
        dtype=array.dtype,
        order=array.order,
    )

    coros = [_get_wrapper(array.store, key, prototype) for key in full_keys]

    # We want to read from the store and decode (finished) chunks in parallel.
    # As soon as a read task is done, we'll schedule the decode task.
    # note: `asyncio.create_task` schedules this to run immediately.
    read_tasks = [asyncio.create_task(coro) for coro in coros]
    decode_futures = []

    for read_future in asyncio.as_completed(read_tasks):
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
            )
        )

    for future in concurrent.futures.as_completed(decode_futures):
        # And now we just check for errors.
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


async def test_getitem_partial_raises(
    simple_array: zarr.AsyncArray[zarr.core.metadata.v3.ArrayV3Metadata],
) -> None:
    with pytest.raises(NotImplementedError, match="Partial chunks are not supported yet"):
        await getitem(simple_array, (slice(1), slice(1)))


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
    for params in param_grid:
        # setup
        # SHAPE = (10_000, 10_000)
        # CHUNKS = (100, 10_000)
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
        array[:] = np.arange(NUMEL, dtype="int32").reshape(params.shape)
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=(os.cpu_count() or 32) - 1)

        z = await zarr.api.asynchronous.open_array(
            store="test.zarr", path="custom-2", zarr_format=3
        )

        rs_zarr_async_getitem = await RecordSet.acollect(
            "Zarr async",
            z.getitem,
            selection=slice(None),
        )
        rs_simple = await RecordSet.acollect("simple", getitem, z, slice(None), pool=pool)
        # validate
        expected = rs_zarr_async_getitem.records[0].value
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
                str(params.shape),
                str(params.chunks),
                "simple",
                expected.nbytes / x.duration,
            )
            for x in rs_simple.records
        )

        records.extend(
            (
                str(params.shape),
                str(params.chunks),
                "zarr",
                expected.nbytes / x.duration,
            )
            for x in rs_zarr_async_getitem.records
        )

    df = pd.DataFrame(records, columns=["shape", "chunks", "implementation", "throughput"])
    sns.catplot(
        data=df,
        hue="implementation",
        x="chunks",
        y="throughput",
        kind="bar",
        col="shape",
        col_wrap=2,
        sharex=False,
    )
    plt.savefig("simple.png")


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(main())
