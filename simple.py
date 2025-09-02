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


def cp_shape(cp: zarr.core.indexing.ChunkProjection) -> tuple[int, ...]:
    # TODO: add to ChunkProjection
    shape = []
    for s in cp.chunk_selection:
        if isinstance(s, slice):
            shape.append((s.stop - s.start) // (s.step or 1))
        else:
            shape.append(len(s))  # TODO: coverage

    return tuple(shape)


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


async def _get_into_wrapper(
    store: zarr.abc.store.Store, key: str, out: zarr.core.buffer.Buffer
) -> tuple[str, bool]:
    """
    A wrapper around Store.get_into that also returns the key used.
    """
    result = await store.get_into(key, out=out)
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
    if len(numcodecs_codecs) == 0:
        # we need to copy the ...
        out[cp.out_selection] = buffer.as_array_like().view(out.dtype).reshape(cp_shape(cp))
    else:
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
                        .reshape(cp_shape(cp))
                    )
            else:
                # We *must* use a temporary buffer here.
                tmp = codec.decode(buffer.as_array_like())
                assert tmp is not None
                buffer = prototype.buffer.from_bytes(tmp)


def _get_numcodecs_codec(codec: zarr.abc.codec.Codec) -> numcodecs.abc.Codec:
    # We're deliberately avoiding the zarr Codec interface, because
    # 1. We don't want to deal with batching (yet)
    # 2. We don't want to deal with async (ever). We're compute bound, so
    #    async is just overhead.
    match codec:
        case zarr.codecs.zstd.ZstdCodec():
            return codec._zstd_codec
        case zarr.codecs.blosc.BloscCodec():
            return codec._blosc_codec
        case _:  # pragma: no cover
            raise NotImplementedError(f"Codec {codec} not supported")


async def getitem(
    array: zarr.AsyncArray[zarr.core.metadata.v3.ArrayV3Metadata],
    selection: zarr.core.indexing.BasicSelection,
    prototype: zarr.core.buffer.BufferPrototype | None = None,
    *,
    pool: concurrent.futures.Executor | None = None,
    use_readinto: bool | None = None,
) -> zarr.core.buffer.NDArrayLike:
    """
    An Array.getitem focused on simplicity, memory, and performance.

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
    avoid an intermediate buffers.

    - We read directly into the output buffer if there are no codecs and the
      store supports it.
    - We decode directly into the output buffer if the codec supports it.

    Under this system, the theoretical peak memory usage is equal to the sum of:

    - the *compressed* chunk sizes
    - the intermediate buffers used by the decoder (if any)
    - the output buffer

    In practice, the compressed bytes can and will be freed from memory as soon
    as the decode is finished so the peak memory usage will be lower.

    At the moment, this implementation does *not* support:

    - Reading a subset of a chunk (i.e. you can only read all of one or more chunks)
    - Non-numcodecs codecs
    """
    # Validation
    indexer = zarr.core.indexing.BasicIndexer(
        selection, array.metadata.shape, array.metadata.chunk_grid
    )
    if not all(cp.is_complete_chunk for cp in indexer):
        # We can eventually support contiguous slices off the ends.
        # We can't (ever?) support fancy indexing.
        raise NotImplementedError("Partial chunks are not supported yet")

    # Setup
    prototype = prototype or zarr.core.buffer.default_buffer_prototype()
    pool = pool or concurrent.futures.ThreadPoolExecutor()

    bytes_bytes_codecs = [
        _get_numcodecs_codec(x)
        for x in array.metadata.codecs
        if isinstance(x, zarr.abc.codec.BytesBytesCodec)
    ]

    use_readinto2 = (
        use_readinto
        and _is_contiguous_indexer(indexer)
        and len(bytes_bytes_codecs) == 0
        and array.store.supports_get_into
    )
    if use_readinto and not use_readinto2:
        raise ValueError(
            "Required zero-copy read with 'use_readinto=True', but not possible given the indexer, codec configuration, or store."
        )
    use_readinto = use_readinto2
    use_decode_into = (
        _is_contiguous_indexer(indexer)
        and len(bytes_bytes_codecs) > 0
        and "out" in inspect.signature(bytes_bytes_codecs[-1].decode).parameters
    )

    # Stage 1: Read the bytes:
    keys = {array.metadata.encode_chunk_key(cp.chunk_coords): cp for cp in indexer}
    full_keys = {(array.store_path / key).path: cp for key, cp in keys.items()}

    out = prototype.nd_buffer.empty(
        shape=indexer.shape,
        dtype=array.dtype,
        order=array.order,
    )

    if use_readinto:
        coros = [
            _get_into_wrapper(
                array.store, key, out.as_ndarray_like()[cp.out_selection].view("b").ravel()
            )
            for key, cp in zip(full_keys, indexer, strict=True)
        ]
    else:
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

        if use_readinto:
            # maybe_buffer is a bool. If it's true, we don't need to worry about a thing
            if maybe_buffer is False:
                # we need to insert the fill value
                out[cp.out_selection] = array.metadata.fill_value  # TODO: coverage
        else:
            # we know that not use_readinto implies maybe_buffer is Buffer | None,
            # i.e. not a bool
            decode_futures.append(
                pool.submit(
                    _decode_wrapper,
                    bytes_bytes_codecs,
                    maybe_buffer,  # type: ignore[arg-type]
                    prototype,
                    out.as_ndarray_like(),
                    cp,
                    use_decode_into,
                    array.metadata.fill_value,
                )
            )

    if decode_futures:
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
        ([], False),
        ([], True),
        ([zarr.codecs.zstd.ZstdCodec(level=0)], False),
        ([zarr.codecs.blosc.BloscCodec(clevel=5)], False),
    ],
    ids=["none-noreadinto", "none-readinto", "zstd", "blosc"],
)
def compressors_use_readinto(
    request: pytest.FixtureRequest,
) -> tuple[list[zarr.abc.codec.BytesBytesCodec], bool]:
    return request.param


@pytest.fixture
async def array_use_readinto(
    store: zarr.storage.LocalStore | zarr.storage.MemoryStore,
    shape_chunks: tuple[tuple[int, ...], tuple[int, ...]],
    compressors_use_readinto: tuple[list[zarr.abc.codec.BytesBytesCodec], bool],
) -> tuple[zarr.AsyncArray[zarr.core.metadata.v3.ArrayV3Metadata], bool]:
    shape, chunks = shape_chunks
    compressors, use_readinto = compressors_use_readinto
    z = await zarr.api.asynchronous.create_array(
        store=store,
        name="test",
        overwrite=True,
        shape=shape,
        chunks=chunks,
        dtype="int32",
        zarr_format=3,
        compressors=compressors,
    )
    z = cast(zarr.AsyncArray[zarr.core.metadata.v3.ArrayV3Metadata], z)
    await z.setitem(slice(None), np.arange(math.prod(shape), dtype="int32").reshape(shape))
    return z, use_readinto


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
    array_use_readinto: tuple[zarr.AsyncArray[zarr.core.metadata.v3.ArrayV3Metadata], bool],
    selection: zarr.core.indexing.BasicSelection,
) -> None:
    array, use_readinto = array_use_readinto
    pool = concurrent.futures.ThreadPoolExecutor()
    result = await getitem(array, selection, pool=pool, use_readinto=use_readinto)
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

    # Early results show good perf for read_into *from local disk*.
    # Approximately 35% faster for Zarr (not using readinto, so just avoiding slower decompression)
    # And ~320% faster for simple (using readinto)
    # Params(shape=(10000, 10000), chunks=(100, 10000), compressors=[ZstdCodec(level=0, checksum=False)], use_readinto=False)
    # ┏━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
    # ┃ Benchmark  ┃ Duration    ┃ Throughput (GB/s) ┃
    # ┡━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
    # │ Zarr async │ 0.35 ± 0.05 │ 1.14              │
    # │ simple     │ 0.11 ± 0.02 │ 3.55              │
    # └────────────┴─────────────┴───────────────────┘
    # Params(shape=(10000, 10000), chunks=(100, 10000), compressors=[], use_readinto=True)
    # ┏━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
    # ┃ Benchmark  ┃ Duration    ┃ Throughput (GB/s) ┃
    # ┡━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
    # │ Zarr async │ 0.26 ± 0.08 │ 1.54              │
    # │ simple     │ 0.03 ± 0.00 │ 11.49             │
    # └────────────┴─────────────┴───────────────────┘

    @dataclasses.dataclass
    class Params:
        shape: tuple[int, ...]
        chunks: tuple[int, ...]
        compressors: list[zarr.abc.codec.BytesBytesCodec] = dataclasses.field(
            default_factory=lambda: [zarr.codecs.zstd.ZstdCodec(level=0)]
        )
        use_readinto: bool = False

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
        Params(
            shape=(10_000, 10_000),
            chunks=(100, 10_000),
            compressors=[],
            use_readinto=True,
        ),
        # Small chunks
        Params(
            shape=(10_000, 10_000),
            chunks=(10, 10_000),
            compressors=[],
            use_readinto=True,
        ),
        # Large chunks
        Params(
            shape=(10_000, 10_000),
            chunks=(10_000, 10_000),
            compressors=[],
            use_readinto=True,
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
            compressors=params.compressors,
        )
        array[:] = np.arange(NUMEL, dtype="int32").reshape(params.shape)
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=(os.cpu_count() or 32) - 1)

        z = await zarr.api.asynchronous.open_array(store="test.zarr", path="simple", zarr_format=3)

        rs_zarr_async_getitem = await RecordSet.acollect(
            "Zarr async",
            z.getitem,
            selection=slice(None),
        )
        rs_simple = await RecordSet.acollect(
            "simple", getitem, z, slice(None), pool=pool, use_readinto=params.use_readinto
        )
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
        rich.print(params)
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


if __name__ == "__main__":  # pragma: no cove
    asyncio.run(main())
