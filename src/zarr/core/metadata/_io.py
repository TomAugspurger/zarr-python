from __future__ import annotations

import asyncio
from typing import Literal

from zarr.core.common import ZARR_JSON, ZARRAY_JSON, ZATTRS_JSON, ZGROUP_JSON, ZarrFormat
from zarr.storage.common import StorePath


def _build_paths(
    node_type: Literal["array", "group"] | None = None,
    zarr_format: ZarrFormat | None = None,
) -> list[str]:
    state = (node_type, zarr_format)
    paths = []

    match state:
        case (_, 3):
            paths.extend([ZARR_JSON])
        case ("array", 2):
            paths.extend([ZARRAY_JSON, ZATTRS_JSON])
        case ("array", None):
            paths.extend([ZARR_JSON, ZATTRS_JSON, ZARRAY_JSON])
        case ("group", 2):
            paths.extend([ZGROUP_JSON, ZATTRS_JSON])
        case ("group", None):
            paths.extend([ZGROUP_JSON, ZATTRS_JSON, ZARR_JSON])
        case (None, 2):
            paths.extend([ZARRAY_JSON, ZATTRS_JSON, ZGROUP_JSON])
        case (None, None):
            paths.extend([ZARRAY_JSON, ZATTRS_JSON, ZGROUP_JSON, ZARR_JSON])
        case _:
            raise ValueError(f"Unhandled state: {state}")

    return paths


async def _gather_documents(
    store_path: StorePath,
    node_type: Literal["array", "group"] | None = None,
    zarr_format: ZarrFormat | None = None,
):
    # We potentially have two unknown, key pieces of information:
    # 1. node_type, which determines
    #    - the keys to read for Zarr V2
    #    - the metadata structure to parse this into for Zarr v3
    # 2. zarr_format, which determines
    #    - the keys to read for Zarr V2
    #
    # This function is purely concerned with doing the I/O to read
    # the minimum set of documents needed
    # When both node_type and zarr_format are unknown, we need to read
    #
    # 1. ZGROUP_JSON: for zarr v2 groups
    # 2. ZARRAY_JSON: for zarr v2 arrays
    # 3. ZATTRS_JSON: for zarr v2 groups and arrays
    # 4. ZARR_JSON: for v3 groups and arrays

    # if we know we're v2 (i.e. the user told us) we just need to read 1-3
    # if we know we're v3 we just need to read 4
    # if we know we're group, we just need to read 2
    paths = _build_paths(node_type, zarr_format)
    results = await asyncio.gather(*[(store_path / path).get() for path in paths])
    items = dict(zip(paths, results, strict=True))

    # Now let's look at what we have. This match statement *must* have the
    # same structure as our other one.
    state = (node_type, zarr_format)

    match state:
        case (_, 3):
            if items[ZARR_JSON] is None:
                # this is known to be missing
                raise Exception("todo")
            # figure out the node type
            node_type = ...
        case ("array", 2):
            # missing = [k for k in [ZARRAY_JSON, ZATTRS_JSON]]
            paths.extend([ZARRAY_JSON, ZATTRS_JSON])
        case ("array", None):
            paths.extend([ZARR_JSON, ZATTRS_JSON, ZARRAY_JSON])
        case ("group", 2):
            paths.extend([ZGROUP_JSON, ZATTRS_JSON])
        case ("group", None):
            paths.extend([ZGROUP_JSON, ZATTRS_JSON, ZARR_JSON])
        case (None, 2):
            paths.extend([ZARRAY_JSON, ZATTRS_JSON, ZGROUP_JSON])
        case (None, None):
            paths.extend([ZARRAY_JSON, ZATTRS_JSON, ZGROUP_JSON, ZARR_JSON])
        case _:
            raise ValueError(f"Unhandled state: {state}")
