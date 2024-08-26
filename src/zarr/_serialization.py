import dataclasses
import datetime
import enum
import functools
import json
import types
import typing

import numcodecs.abc
import numpy as np

from zarr.core.common import JSON, Tag

T = typing.TypeVar("T", covariant=True)


class DataclassType(typing.Protocol[T]):
    __dataclass_fields__: typing.ClassVar[dict[str, typing.Any]]


@functools.singledispatch
def encode_value(v: typing.Any) -> JSON:
    return typing.cast(JSON, v)
    # raise TypeError(f"Cannot serialize type '{type(v)}'")


@encode_value.register(np.dtype)
def _(v: np.dtype[typing.Any]) -> JSON:
    return str(v)


@encode_value.register(complex)
def _(v: complex) -> JSON:
    return [v.real, v.imag]


@encode_value.register(np.ndarray)
@encode_value.register(np.generic)
def _(v: np.ndarray[typing.Any, np.dtype[typing.Any]] | np.generic) -> JSON:
    if v.dtype.kind == "M":
        # this *must* be an int right?
        return v.view("i8").item()  # type: ignore[no-any-return]
    if v.dtype.kind == "c":
        # hmmm, NumPy complex objects *should* be
        # instances of complex, but the singledispatch
        # isn't working correctly.
        return encode_value(v.item())
    # unsure that this is actually JSON...
    return v.item()  # type: ignore[no-any-return]


@encode_value.register(enum.Enum)
def _(v: enum.Enum) -> JSON:
    # We *can't* ensure that the Enum members are JSON...
    # Kinda need to go through another encoder.
    return v.value  # type: ignore[no-any-return]


@encode_value.register(numcodecs.abc.Codec)
def _(v: numcodecs.abc.Codec) -> JSON:
    # numcodecs claims that this is JSON serializable.
    return typing.cast(dict[str, JSON], v.get_config())


@encode_value.register(datetime.datetime)
@encode_value.register(datetime.date)
def _(v: datetime.datetime | datetime.date) -> str:
    return v.isoformat()


@encode_value.register(set)
def _(v: set[T]) -> list[T]:
    return list(v)


def to_dict(obj: DataclassType[T]) -> dict[str, JSON]:
    result: dict[str, JSON] = {}
    for k, v in dataclasses.asdict(obj).items():
        if dataclasses.is_dataclass(v):
            # Seems to be an issue with mypy:
            #   Argument 1 to "to_dict" has incompatible type "type[DataclassInstance]";
            #   expected "DataclassType[Never]"
            v = to_dict(v)  # type: ignore[arg-type]
        else:
            result[k] = encode_value(v)

    return result


def encode(obj: DataclassType[T]) -> bytes:
    result: dict[str, JSON] = to_dict(obj)
    return json.dumps(result).encode("utf-8")


@functools.singledispatch
def decode_value(type: type, v: JSON) -> typing.Any:
    origin = typing.get_origin(type)
    args = typing.get_args(type)

    if origin in (tuple, list, set) and args:
        # we have to do this recursively
        if origin is tuple:
            if len(args) == 2 and args[1] is Ellipsis:
                # tuple[int, ...]
                return tuple(decode_value(args[0], x) for x in v)
            elif len(args) != len(v):
                raise TypeError(...)
            else:
                # tuple[int, str, float], etc.
                return tuple(decode_value(subtype, x) for subtype, x in zip(args, v, strict=False))
        else:
            if len(args) != 1:
                raise TypeError("todo")
            subtype = args[0]
            return type([decode_value(subtype, x) for x in v])

    # First, we try to resolve any Union types. Note that this
    # block recursively calls decode_value once it's figured out
    # which type to use.
    if origin is types.UnionType:
        variants = typing.get_args(type)
        NoneType = __builtins__["type"](None)  # type: ignore[index]

        # Match on these cases:
        if len(variants) == 2 and NoneType in variants:
            # T | None, AKA Optional[T]

            if v is None:
                return None
            else:
                t = next(x for x in variants if x is not NoneType)
                return decode_value(t, v)

        elif all(x is NoneType or dataclasses.is_dataclass(x) for x in variants):
            # This is a true union. To deserialize this, we need a "tagged union",
            # i.e. the type to resolve it to needs to be serialized with the data.
            # For example, {"name": "default", "configuration": {...}}
            # for `chunk_key_encoding` -> DefaultChunkEncoding
            #
            # To avoid the need to hardcode a key like "name", we rely on
            # Annotated[..., Tag] in the type hints to indicate the field that should
            # be used to discriminate which enum variant to select.

            if not isinstance(v, dict):
                # we need a tag here. If it's not a dict, then error
                raise TypeError("todo")

            tag_field_name: str | None = None
            tags_to_variants = {}

            for variant in variants:
                variant_types = typing.get_type_hints(variant, include_extras=True)
                tag_types = {k: t for k, t in variant_types.items() if Tag in typing.get_args(t)}
                if len(tag_types) != 1:
                    # zero or 2+ fields declared to be a tag.
                    raise TypeError("todo")

                k, t = next(iter(tag_types.items()))
                if tag_field_name is None:
                    tag_field_name = k
                elif tag_field_name != k:
                    # The 2+ variants don't agree on the tag name
                    raise TypeError("todo")

                # We have something like typing.Annotated[typing.Literal["default"]]
                # Go to typing.Literal["default"], then to "default"
                t2 = typing.get_args(t)[0]
                if typing.get_origin(t2) is typing.Literal:
                    t3 = typing.get_args(t2)
                    if len(t3) != 1:
                        raise TypeError("todo")
                    tags_to_variants[t3[0]] = variant
                else:
                    raise TypeError("todo!")

            assert tag_field_name is not None  # should be impossible
            # We have the complete mapping from tag -> variants. Now pick one,
            # and send it through the normal decoding routine.
            tag_value = v[tag_field_name]
            tag_type = tags_to_variants[tag_value]
            return decode_value(tag_type, v)

    # I don't understand this mypy warning
    if type == typing.Any:  # type: ignore[comparison-overlap]
        return v

    if origin is typing.Literal:
        if v not in typing.get_args(type):
            raise TypeError("TODO")

        # return typing.cast(type, v)
        return v

    if dataclasses.is_dataclass(type):
        return from_dict(typing.cast(dict[str, JSON], v), type)

    return type(v)


@decode_value.register(enum.EnumMeta)
def _(type: typing.Any, v: JSON) -> typing.Any:
    # can't get the type hints correct here...
    return type(v)


def from_dict(data: dict[str, JSON], type: type[DataclassType[T]]) -> T:
    fields = dataclasses.fields(type)
    hints = typing.get_type_hints(type)

    args: dict[str, typing.Any] = {}

    for field in fields:
        name = field.name
        field_t = hints[name]

        if name == "codecs":
            from zarr.core.metadata import parse_codecs

            args[field.name] = parse_codecs(data.get(field.name))
            continue

        args[field.name] = decode_value(field_t, data.get(field.name))

    return typing.cast(T, type(**args))


def decode(data: bytes, type: type[DataclassType[T]]) -> T:
    raw = json.loads(data)
    return from_dict(raw, type)
