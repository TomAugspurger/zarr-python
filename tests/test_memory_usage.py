import pathlib
import tracemalloc

import numpy as np
import pytest

import zarr.api.asynchronous
import zarr.core.array
import zarr.core.buffer.cpu
import zarr.storage


@pytest.mark.parametrize("dtype", ["f4", "f8", "u8"])
async def test_local_store_uncompressed_read(dtype: str, tmp_path: pathlib.Path) -> None:
    store = zarr.storage.LocalStore(tmp_path)
    arr = await zarr.api.asynchronous.create_array(
        store,
        name="a",
        shape=(10000,),
        chunks=(1000,),
        dtype=dtype,
    )
    data = np.ones(1000, dtype=dtype)
    await arr.setitem(slice(len(data)), data)
    await arr.getitem(slice(len(data)))  # cache the imports

    dom_filter = tracemalloc.DomainFilter(inclusive=True, domain=np.lib.tracemalloc_domain)

    tracemalloc.start()
    before = tracemalloc.take_snapshot().filter_traces([dom_filter])

    # this is the one and only allowed allocation.j
    out = zarr.core.buffer.cpu.NDBuffer.from_ndarray_like(np.empty_like(data))
    result = await arr.getitem(slice(len(data)), out=out)
    # We filter the snapshot to only select NumPy related calls

    after = tracemalloc.take_snapshot().filter_traces([dom_filter])
    tracemalloc.stop()

    delta = after.compare_to(before, "lineno")
    assert len(delta) == 0
    assert out.as_numpy_array() is result

    np.testing.assert_array_equal(result, data)
