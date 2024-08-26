from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Annotated, Literal, cast

from zarr.abc.metadata import Metadata
from zarr.core.common import (
    JSON,
    ChunkCoords,
    Tag,
    parse_named_configuration,
)

SeparatorLiteral = Literal[".", "/"]


def parse_separator(data: JSON) -> SeparatorLiteral:
    if data not in (".", "/"):
        raise ValueError(f"Expected an '.' or '/' separator. Got {data} instead.")
    return cast(SeparatorLiteral, data)


@dataclass
class ChunkKeyEncodingConfiguration:
    separator: SeparatorLiteral = "."


# TODO: confirm whether this is abstract?
@dataclass(frozen=True, kw_only=True)
class ChunkKeyEncoding(Metadata):
    name: Annotated[str, Tag]
    configuration: ChunkKeyEncodingConfiguration = field(
        default_factory=ChunkKeyEncodingConfiguration
    )

    def __post_init__(self) -> None: ...

    # def __post_init__(self) -> None:
    #     # separator_parsed = parse_separator(self.separator)
    #     object.__setattr__(self, "separator", separator_parsed)

    @classmethod
    def from_dict(cls, data: dict[str, JSON] | ChunkKeyEncoding) -> ChunkKeyEncoding:
        if isinstance(data, ChunkKeyEncoding):
            return data

        # configuration is optional for chunk key encodings
        name_parsed, config_parsed = parse_named_configuration(data, require_configuration=False)
        if name_parsed == "default":
            if config_parsed is None:
                # for default, normalize missing configuration to use the "/" separator.
                config_parsed = {"separator": "/"}
            return DefaultChunkKeyEncoding(**config_parsed)  # type: ignore[arg-type]
        if name_parsed == "v2":
            if config_parsed is None:
                # for v2, normalize missing configuration to use the "." separator.
                config_parsed = {"separator": "."}
            return V2ChunkKeyEncoding(**config_parsed)  # type: ignore[arg-type]
        msg = f"Unknown chunk key encoding. Got {name_parsed}, expected one of ('v2', 'default')."
        raise ValueError(msg)

    def to_dict(self) -> dict[str, JSON]:
        return {"name": self.name, "configuration": {"separator": self.configuration.separator}}

    @abstractmethod
    def decode_chunk_key(self, chunk_key: str) -> ChunkCoords:
        pass

    @abstractmethod
    def encode_chunk_key(self, chunk_coords: ChunkCoords) -> str:
        pass


@dataclass(frozen=True, kw_only=True)
class DefaultChunkKeyEncoding(ChunkKeyEncoding):
    name: Annotated[Literal["default"], Tag] = "default"

    def __post_init__(self) -> None:
        if self.name != "default":
            raise TypeError
        return super().__post_init__()

    def decode_chunk_key(self, chunk_key: str) -> ChunkCoords:
        if chunk_key == "c":
            return ()
        return tuple(map(int, chunk_key[1:].split(self.separator)))

    def encode_chunk_key(self, chunk_coords: ChunkCoords) -> str:
        return self.separator.join(map(str, ("c",) + chunk_coords))


@dataclass(frozen=True)
class V2ChunkKeyEncoding(ChunkKeyEncoding):
    name: Annotated[Literal["v2"], Tag] = "v2"

    def decode_chunk_key(self, chunk_key: str) -> ChunkCoords:
        return tuple(map(int, chunk_key.split(self.separator)))

    def encode_chunk_key(self, chunk_coords: ChunkCoords) -> str:
        chunk_identifier = self.separator.join(map(str, chunk_coords))
        return "0" if chunk_identifier == "" else chunk_identifier


CHUNK_KEY_ENCODINGS = DefaultChunkKeyEncoding | V2ChunkKeyEncoding
