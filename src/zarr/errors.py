from typing import Any


class _BaseZarrError(ValueError):
    _msg = ""

    def __init__(self, *args: Any) -> None:
        super().__init__(self._msg.format(*args))


class ContainsGroupError(_BaseZarrError):
    _msg = "A group exists in store {0!r} at path {1!r}."


class ContainsArrayError(_BaseZarrError):
    _msg = "An array exists in store {0!r} at path {1!r}."


class ContainsArrayAndGroupError(_BaseZarrError):
    _msg = (
        "Array and group metadata documents (.zarray and .zgroup) were both found in store "
        "{0!r} at path {1!r}."
        "Only one of these files may be present in a given directory / prefix. "
        "Remove the .zarray file, or the .zgroup file, or both."
    )


class MetadataValidationError(_BaseZarrError):
    """An exception raised when the Zarr metadata is invalid in some way"""


class ZarrVersionValidationError(MetadataValidationError):
    """Raised when the zarr_version in the Zarr metadata is invalid."""

    _msg = "Invalid value for 'zarr_format'. Expected '{}'. Got '{}'."


class NodeTypeValidationError(MetadataValidationError):
    """
    Specialized exception when the node_type of the metadata document is incorrect..

    This can be raised when the value is invalid or unexpected given the context,
    for example an 'array' node when we expected a 'group'.
    """

    _msg = "Invalid value for 'node_type'. Expected '{}'. Got '{}'."


__all__ = [
    "ContainsArrayAndGroupError",
    "ContainsArrayError",
    "ContainsGroupError",
]
