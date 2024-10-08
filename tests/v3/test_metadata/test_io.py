from typing import Literal

import pytest

from zarr.core.common import ZARR_JSON, ZARRAY_JSON, ZATTRS_JSON, ZGROUP_JSON
from zarr.core.metadata._io import _build_paths


@pytest.mark.parametrize(
    ("node_type", "zarr_format", "expected"),
    [
        (None, None, [ZARR_JSON, ZATTRS_JSON, ZGROUP_JSON, ZARRAY_JSON]),
        (None, 2, [ZATTRS_JSON, ZGROUP_JSON, ZARRAY_JSON]),
        (None, 3, [ZARR_JSON]),
        ("array", None, [ZARR_JSON, ZATTRS_JSON, ZARRAY_JSON]),
        ("array", 2, [ZATTRS_JSON, ZARRAY_JSON]),
        ("array", 3, [ZARR_JSON]),
        ("group", None, [ZARR_JSON, ZATTRS_JSON, ZGROUP_JSON]),
        ("group", 2, [ZATTRS_JSON, ZGROUP_JSON]),
        ("group", 3, [ZARR_JSON]),
    ],
)
def test_build_paths(
    node_type: Literal["array", "group"] | None,
    zarr_format: Literal[2, 3] | None,
    expected: list[str],
) -> None:
    result = _build_paths(node_type, zarr_format)
    assert set(result) == set(expected)
