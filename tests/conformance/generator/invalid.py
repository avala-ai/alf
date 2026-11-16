"""Files that a conforming reader MUST refuse, and the reason it must give.

A corpus of valid files can only prove a reader accepts what it should. It
cannot prove a reader REFUSES what it should — and most of this format's
specification is refusals: a payload length that contradicts the declared count,
a geometry codec that disagrees with the wire byte, a label id the editor cannot
resolve. A reader that ignores every one of them passes the entire valid corpus.

Each entry below breaks **exactly one** rule and names the refusal.

Why an identifier and not an exception type: "both readers raised an error" is
not agreement. `AlfError` covers a hundred distinct faults, so a negative test
that only checks that something was raised passes when a reader refuses for the
WRONG reason — which is precisely the failure a negative test exists to catch.
The identifier says which rule, and it is the same string in every language.

A refusal is a RESULT, not a crash: a runner prints {"refused": "<id>"} and
exits 0. Exiting non-zero would collapse "refused correctly" and "fell over"
into one outcome, and telling those apart is the whole point.
"""

from __future__ import annotations

import gzip
import json
import struct
import zlib
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class InvalidCase:
    """One rule, one file, one refusal identifier."""

    name: str
    refusal: str
    why: str
    build: Callable[[], bytes]
    extension: str


def _envelope(version: int, header: dict, payload: bytes) -> bytes:
    blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return struct.pack("B", version) + struct.pack("<I", len(blob)) + blob + payload


def _raw_envelope(version: int, header: str, payload: bytes) -> bytes:
    """Build an envelope whose JSON number spelling must survive byte-for-byte."""
    blob = header.encode("utf-8")
    return struct.pack("B", version) + struct.pack("<I", len(blob)) + blob + payload


def _oversized_valid_json_header() -> bytes:
    """Build one complete, valid header exactly one byte past the contract cap."""
    size = 10_000_001
    prefix = b'{"padding":"'
    suffix = b'"}'
    blob = prefix + b"a" * (size - len(prefix) - len(suffix)) + suffix
    assert len(blob) == size
    return struct.pack("B", 1) + struct.pack("<I", len(blob)) + blob


def _fp16(points: list) -> bytes:
    return np.array(points, dtype=np.float16).tobytes()


def _int16(values: list) -> bytes:
    return np.array(values, dtype="<i2").tobytes()


def _gzip(payload: bytes) -> bytes:
    """Return deterministic RFC 1952 bytes on every supported Python.

    Python 3.11+ may delegate ``mtime=0`` to zlib and expose the platform OS
    byte, while older versions write 255. The latter means "unknown" and keeps
    the corpus byte-identical across the Python CI matrix.
    """
    encoded = bytearray(gzip.compress(payload, mtime=0))
    encoded[9] = 255
    return bytes(encoded)


def _gzip_with_corrupt_crc(payload: bytes) -> bytes:
    """Return one complete gzip member whose CRC32 trailer is wrong."""
    encoded = bytearray(_gzip(payload))
    encoded[-8] ^= 0x01
    return bytes(encoded)


def _gzip_with_header_crc(payload: bytes, *, corrupt: bool) -> bytes:
    """Add RFC 1952 FHCRC, optionally changing only its stored value."""
    encoded = bytearray(_gzip(payload))
    encoded[3] |= 0x02
    header_crc = zlib.crc32(encoded[:10]) & 0xFFFF
    if corrupt:
        header_crc ^= 0x01
    encoded[10:10] = struct.pack("<H", header_crc)
    return bytes(encoded)


# Every identifier is declared HERE, once. A language does not invent one.
CASES: list[InvalidCase] = [
    InvalidCase(
        name="alx-count-stride-wrap",
        refusal="payload-length-mismatch",
        why="Multiplying this declared count by the four-byte stride wraps to zero in signed 64-bit arithmetic, accepting an empty sidecar as trillions of entries and breaking channel correlation.",
        extension="alx",
        build=lambda: _envelope(1, {"_total_points": 2**62, "_source_total_points": 1}, b""),
    ),
    InvalidCase(
        name="alx-count-machine-width-wrap",
        refusal="payload-length-mismatch",
        why="Narrowing the declared count to a 32-bit allocation index turns it into zero, hiding a missing sidecar payload and breaking channel correlation.",
        extension="alx",
        build=lambda: _envelope(
            1,
            {"_total_points": 2**32, "_source_total_points": 1},
            b"",
        ),
    ),
    InvalidCase(
        name="version-byte-unknown",
        refusal="version-unsupported",
        why="A version byte nobody has defined cannot be guessed at; the payload layout is exactly what it selects.",
        extension="alp",
        build=lambda: _envelope(9, {"_total_points": 1}, _fp16([[1.0, 2.0, 3.0]])),
    ),
    InvalidCase(
        name="version3-on-label-channel",
        refusal="geometry-version-on-non-geometry-channel",
        why="v3 is the .alp geometry codec. On .all it would fall through to the v1 accessors and hand back coordinates as category bytes.",
        extension="all",
        build=lambda: _envelope(3, {"_total_points": 3, "_frame_convention": "world-aligned"}, bytes([1, 2, 3])),
    ),
    InvalidCase(
        name="version2-on-non-label-channel",
        refusal="label-version-on-non-label-channel",
        why="Wire version 2 selects the interleaved .all label layout; accepting it on another channel reinterprets label bytes as unrelated measurements.",
        extension="ali",
        build=lambda: _envelope(2, {"_total_points": 3}, bytes([1, 2, 3])),
    ),
    InvalidCase(
        name="codec-int16-on-version1",
        refusal="codec-version-disagreement",
        why="A writer that set the header key and forgot the wire byte. Resolving the disagreement by picking a side decodes to plausible garbage either way.",
        extension="alp",
        build=lambda: _envelope(
            1,
            {
                # NO `_frame_convention`: this envelope is version 1, where the
                # field is itself a refusal, and this case owns one rule.
                "_total_points": 1,
                "_geom_codec": "int16",
                "_geom_scale": 0.01,
                "_geom_offset": [0.0, 0.0, 0.0],
            },
            _fp16([[1.0, 2.0, 3.0]]),
        ),
    ),
    InvalidCase(
        name="codec-unsupported",
        refusal="codec-unsupported",
        why="An unknown geometry codec has no defined payload interpretation; treating it as fp16 produces plausible but invented coordinates.",
        extension="alp",
        build=lambda: _envelope(
            1,
            {"_total_points": 1, "_geom_codec": "packed42"},
            _fp16([[1.0, 2.0, 3.0]]),
        ),
    ),
    InvalidCase(
        name="codec-fp16-on-version3",
        refusal="codec-version-disagreement",
        why="The mirror image: the byte says fixed point, the header says half-float.",
        extension="alp",
        build=lambda: _envelope(
            3, {"_total_points": 1, "_geom_codec": "fp16", "_frame_convention": "world-aligned"}, _int16([1, 0, 0])
        ),
    ),
    InvalidCase(
        name="geom-scale-missing",
        refusal="geom-scale-missing",
        why="Without the step every coordinate decodes to the wrong magnitude, and nothing about the payload reveals it.",
        extension="alp",
        build=lambda: _envelope(
            3,
            {
                "_total_points": 1,
                "_geom_codec": "int16",
                "_frame_convention": "world-aligned",
                "_geom_offset": [0.0, 0.0, 0.0],
            },
            _int16([1, 0, 0]),
        ),
    ),
    InvalidCase(
        name="geom-scale-not-positive",
        refusal="geom-scale-not-positive",
        why="A zero or negative step is not a grid. Zero collapses the frame to its offset; negative mirrors it.",
        extension="alp",
        build=lambda: _envelope(
            3,
            {
                "_total_points": 1,
                "_geom_codec": "int16",
                "_frame_convention": "world-aligned",
                "_geom_scale": 0.0,
                "_geom_offset": [0.0, 0.0, 0.0],
            },
            _int16([1, 0, 0]),
        ),
    ),
    InvalidCase(
        name="geom-offset-wrong-shape",
        refusal="geom-offset-malformed",
        why="A scalar or 2-vector offset broadcasts across axes and yields a uniformly shifted cloud rather than an error.",
        extension="alp",
        build=lambda: _envelope(
            3,
            {
                "_total_points": 1,
                "_geom_codec": "int16",
                "_frame_convention": "world-aligned",
                "_geom_scale": 0.01,
                "_geom_offset": [1.0, 2.0],
            },
            _int16([1, 0, 0]),
        ),
    ),
    InvalidCase(
        name="geom-partial-no-return",
        refusal="geom-partial-no-return",
        why="A point with some but not all words equal to the reserved invalid-return marker (-32768) cannot come from a conforming writer, which clamps real coordinates to +/-32767; reconstructing it arithmetically yields a plausible position (-327.68 here) that no later check can distinguish from measured data.",
        extension="alp",
        build=lambda: _envelope(
            3,
            {
                "_total_points": 1,
                "_geom_codec": "int16",
                "_frame_convention": "world-aligned",
                "_geom_scale": 0.01,
                "_geom_offset": [0.0, 0.0, 0.0],
            },
            _int16([-32768, 0, 0]),
        ),
    ),
    InvalidCase(
        name="geom-offset-inexact-integer",
        refusal="geom-number-inexact",
        why="Rounding an integer geometry offset outside binary64's exact range moves every point onto a grid different from the one the file declares.",
        extension="alp",
        build=lambda: _envelope(
            3,
            {
                "_total_points": 1,
                "_geom_codec": "int16",
                "_frame_convention": "world-aligned",
                "_geom_scale": 0.01,
                "_geom_offset": [9007199254740993, 0, 0],
            },
            _int16([0, 0, 0]),
        ),
    ),
    InvalidCase(
        name="geom-offset-lossy-decimal",
        refusal="geom-number-inexact",
        why="A decimal geometry offset whose stated precision is lost in binary64 shifts the whole coordinate grid even though the parsed value remains finite.",
        extension="alp",
        build=lambda: _raw_envelope(
            3,
            '{"_total_points":1,"_geom_codec":"int16","_frame_convention":"world-aligned","_geom_scale":1,'
            '"_geom_offset":[9007199254740993.5,0,0]}',
            _int16([0, 0, 0]),
        ),
    ),
    InvalidCase(
        name="geom-offset-inexact-shortest-integer",
        refusal="geom-number-inexact",
        why="An integer token can equal binary64's shortest decimal spelling while still naming a different exact integer and therefore a different coordinate grid.",
        extension="alp",
        build=lambda: _raw_envelope(
            3,
            '{"_total_points":1,"_geom_codec":"int16","_frame_convention":"world-aligned","_geom_scale":1,'
            '"_geom_offset":[18446744073709552000,0,0]}',
            _int16([0, 0, 0]),
        ),
    ),
    InvalidCase(
        name="geom-offset-inexact-scientific-integer",
        refusal="geom-number-inexact",
        why="Normalized-looking scientific notation can still state an integer different from the binary64 value; accepting it shifts the complete coordinate grid.",
        extension="alp",
        build=lambda: _raw_envelope(
            3,
            '{"_total_points":1,"_geom_codec":"int16","_frame_convention":"world-aligned","_geom_scale":1,'
            '"_geom_offset":[9.007199254740993e15,0,0]}',
            _int16([0, 0, 0]),
        ),
    ),
    InvalidCase(
        name="geom-offset-inexact-expanded-decimal",
        refusal="geom-number-inexact",
        why="An expanded integral decimal offset states an exact mathematical integer; accepting a nearby binary64 grid silently shifts every point.",
        extension="alp",
        build=lambda: _raw_envelope(
            3,
            '{"_total_points":1,"_geom_codec":"int16","_frame_convention":"world-aligned","_geom_scale":1,'
            '"_geom_offset":[1000000000000000100.0,0,0]}',
            _int16([0, 0, 0]),
        ),
    ),
    InvalidCase(
        name="geom-scale-lossy-decimal",
        refusal="geom-number-inexact",
        why="A decimal geometry scale whose stated precision is lost in binary64 reconstructs every non-zero component against a different step than the file declares.",
        extension="alp",
        build=lambda: _raw_envelope(
            3,
            '{"_total_points":1,"_geom_codec":"int16","_frame_convention":"world-aligned",'
            '"_geom_scale":9007199254740993.0,"_geom_offset":[0,0,0]}',
            _int16([1, 0, 0]),
        ),
    ),
    InvalidCase(
        name="geom-scale-inexact-expanded-exponent",
        refusal="geom-number-inexact",
        why="An expanded exponent-zero scale states an exact mathematical integer; accepting its rounded binary64 changes every non-zero decoded component.",
        extension="alp",
        build=lambda: _raw_envelope(
            3,
            '{"_total_points":1,"_geom_codec":"int16","_frame_convention":"world-aligned",'
            '"_geom_scale":1000000000000000100e0,"_geom_offset":[0,0,0]}',
            _int16([1, 0, 0]),
        ),
    ),
    InvalidCase(
        name="geom-scale-outside-binary64",
        refusal="geom-number-inexact",
        why="A JSON geometry integer outside binary64's range cannot define the same coordinate grid after decoding.",
        extension="alp",
        build=lambda: _raw_envelope(
            3,
            '{"_total_points":1,"_geom_codec":"int16","_frame_convention":"world-aligned",'
            '"_geom_scale":' + str(10**400) + ',"_geom_offset":[0,0,0]}',
            _int16([1, 0, 0]),
        ),
    ),
    InvalidCase(
        name="device-position-inexact-integer",
        refusal="geom-number-inexact",
        why="Rounding a device-position component outside binary64's exact integer range shifts every local point in the frame.",
        extension="alp",
        build=lambda: _raw_envelope(
            1,
            '{"_total_points":1,"device_position":{"x":9007199254740993,"y":0,"z":0}}',
            _fp16([[0.0, 0.0, 0.0]]),
        ),
    ),
    InvalidCase(
        name="device-position-outside-binary64",
        refusal="geom-number-inexact",
        why="A non-finite decoded anchor axis cannot place a frame; classifying its rounded shape instead hides the geometry-token fault.",
        extension="alp",
        build=lambda: _raw_envelope(
            1,
            '{"_total_points":1,"device_position":{"x":1e309,"y":0,"z":0}}',
            _fp16([[0.0, 0.0, 0.0]]),
        ),
    ),
    InvalidCase(
        name="world-centroid-lossy-decimal",
        refusal="geom-number-inexact",
        why="A world centroid whose stated decimal precision disappears in binary64 misplaces the complete aerial frame.",
        extension="alp",
        build=lambda: _raw_envelope(
            1,
            '{"_total_points":1,"world_centroid_epsg3857":{"x":9007199254740993.5,"y":0,"z":0}}',
            _fp16([[0.0, 0.0, 0.0]]),
        ),
    ),
    InvalidCase(
        name="total-points-contradicts-payload",
        refusal="payload-length-mismatch",
        why="Channels are correlated by index, so a count that disagrees with the payload misattributes every value after the gap.",
        extension="ali",
        build=lambda: _envelope(1, {"_total_points": 5}, bytes([1, 2, 3])),
    ),
    InvalidCase(
        name="total-points-fractional",
        refusal="total-points-not-an-integer",
        why="int() truncates 1.9 to 1, and a one-point payload then satisfies the very cross-check that guards against length mismatch.",
        extension="ali",
        build=lambda: _envelope(1, {"_total_points": 1.9}, bytes([7])),
    ),
    InvalidCase(
        name="total-points-present-null",
        refusal="total-points-not-an-integer",
        why="A present null count is malformed; treating it as legacy absence disables the cross-channel length check.",
        extension="ali",
        build=lambda: _envelope(1, {"_total_points": None}, bytes([7])),
    ),
    InvalidCase(
        name="total-points-outside-binary64",
        refusal="total-points-not-an-integer",
        why="A reserved count outside finite binary64 has no portable integer value and must reach the count refusal before generic metadata validation.",
        extension="ali",
        build=lambda: _raw_envelope(
            1,
            '{"_total_points":' + str(10**400) + "}",
            bytes([7]),
        ),
    ),
    InvalidCase(
        name="source-total-points-inexact-integer",
        refusal="total-points-not-an-integer",
        why="A source count that changes in binary64 makes readers disagree about the valid range for every decimation index.",
        extension="alx",
        build=lambda: _raw_envelope(
            1,
            '{"_total_points":1,"_source_total_points":9007199254740993}',
            bytes([0, 0, 0, 0]),
        ),
    ),
    InvalidCase(
        name="int16-payload-not-triples",
        refusal="payload-not-whole-points",
        why="A payload that is not a whole number of XYZ triples is truncated; decoding the prefix silently drops the rest of the frame.",
        extension="alp",
        build=lambda: _envelope(
            3,
            {
                "_geom_codec": "int16",
                "_frame_convention": "world-aligned",
                "_geom_scale": 0.01,
                "_geom_offset": [0.0, 0.0, 0.0],
            },
            _int16([1, 0, 0, 5]),
        ),
    ),
    InvalidCase(
        name="label-format-version-unknown",
        refusal="label-layout-unsupported",
        why="Treating an unknown layout as v1 returns the interleaved stream as categories and silently invents instance ids.",
        extension="all",
        build=lambda: _envelope(1, {"_total_points": 3, "_format_version": 7}, bytes([1, 2, 3])),
    ),
    InvalidCase(
        name="label-format-version-negative",
        refusal="label-layout-unsupported",
        why="A negative integral layout token is syntactically an integer but names no supported label layout.",
        extension="all",
        build=lambda: _raw_envelope(
            1,
            '{"_total_points":1,"_format_version":-1.0}',
            bytes([7]),
        ),
    ),
    InvalidCase(
        name="label-layout-disagreement-ambiguous",
        refusal="label-layout-ambiguous",
        why="Without a declared count, a three-byte payload fits both semantic and panoptic layouts; choosing either silently reinterprets labels.",
        extension="all",
        build=lambda: _envelope(1, {"_format_version": 2}, bytes([1, 0, 0])),
    ),
    InvalidCase(
        name="label-layout-disagreement-empty",
        refusal="label-layout-ambiguous",
        why="An empty counted payload fits both one-byte and three-byte label strides, so it cannot resolve disagreeing version signals.",
        extension="all",
        build=lambda: _envelope(1, {"_total_points": 0, "_format_version": 2}, b""),
    ),
    InvalidCase(
        name="instance-id-reserved",
        refusal="instance-id-reserved",
        why="65535 is reserved; the editor's registry caps valid local ids at 65534, so this file's instance identity cannot be resolved.",
        extension="all",
        build=lambda: _envelope(2, {"_total_points": 1}, bytes([7, 0xFF, 0xFF])),
    ),
    InvalidCase(
        name="instance-map-incomplete",
        refusal="instance-map-incomplete",
        why="A map that CLAIMS to record identity and misses emitted ids exports objects with a null uuid and reports success.",
        extension="all",
        build=lambda: _envelope(
            2,
            {"_total_points": 2, "instance_map": {"1": "u1"}},
            bytes([1, 0x01, 0x00, 2, 0x02, 0x00]),
        ),
    ),
    InvalidCase(
        name="instance-map-key-noncanonical",
        refusal="instance-map-malformed",
        why="Resolvers look up the canonical decimal spelling of each uint16 id, so a leading-zero key names no emitted instance.",
        extension="all",
        build=lambda: _envelope(
            2,
            {"_total_points": 1, "instance_map": {"1": "u1", "01": "unresolvable-extra"}},
            bytes([1, 0x01, 0x00]),
        ),
    ),
    InvalidCase(
        name="palette-index-out-of-range",
        refusal="palette-index-out-of-range",
        why="An index past the palette end renders as an arbitrary colour rather than failing, so the mis-colouring looks like data.",
        extension="alc",
        build=lambda: _envelope(1, {"_total_points": 2, "palette": ["ff0000"]}, bytes([0, 9])),
    ),
    InvalidCase(
        name="palette-entry-not-hex",
        refusal="palette-entry-malformed",
        why='"zzzzzz" passes a length check and then range-checks indexes against a palette no viewer can parse.',
        extension="alc",
        build=lambda: _envelope(1, {"_total_points": 1, "palette": ["zzzzzz"]}, bytes([0])),
    ),
    InvalidCase(
        name="header-not-an-object",
        refusal="header-not-an-object",
        why="Every accessor does a keyed lookup, so a list or scalar root fails later and further from the cause.",
        extension="ali",
        build=lambda: struct.pack("B", 1) + struct.pack("<I", 2) + b"[]" + bytes([1]),
    ),
    InvalidCase(
        name="alp-header-not-an-object",
        refusal="header-not-an-object",
        why="Geometry token inspection must not replace the root-object refusal merely because this otherwise valid array header belongs to .alp.",
        extension="alp",
        build=lambda: _raw_envelope(1, "[]", b""),
    ),
    InvalidCase(
        name="header-json-malformed",
        refusal="header-not-json",
        why="Malformed JSON syntax has no portable metadata interpretation and must reach the named ALF refusal rather than crash a runner.",
        extension="ali",
        build=lambda: _raw_envelope(1, "{", bytes([1])),
    ),
    InvalidCase(
        name="header-key-duplicate",
        refusal="header-key-duplicate",
        why="JSON parsers disagree on which duplicate object member wins; accepting one makes the same ALF bytes declare different metadata across languages.",
        extension="alp",
        build=lambda: _raw_envelope(
            3,
            '{"_total_points":1,"_geom_codec":"int16","_frame_convention":"world-aligned",'
            '"_geom_scale":9007199254740993,"_geom_scale":1,'
            '"_geom_offset":[0,0,0]}',
            _int16([0, 0, 0]),
        ),
    ),
    InvalidCase(
        name="header-truncated",
        refusal="header-truncated",
        why="A declared header longer than the file is how these objects go bad mid-transfer.",
        extension="ali",
        build=lambda: struct.pack("B", 1) + struct.pack("<I", 500) + b"{}" + bytes([1]),
    ),
    InvalidCase(
        name="header-too-large",
        refusal="header-too-large",
        why="A complete valid-JSON header above the 10,000,000-byte ceiling must be refused before parsers allocate another hostile copy.",
        extension="ali",
        build=_oversized_valid_json_header,
    ),
    InvalidCase(
        name="header-not-utf8",
        refusal="header-not-utf8",
        why="A reader that lets a UnicodeDecodeError escape aborts a pipeline run that would otherwise quarantine the frame.",
        extension="ali",
        build=lambda: struct.pack("B", 1) + struct.pack("<I", 4) + b'{"\xff"' + bytes([1]),
    ),
    InvalidCase(
        name="header-number-non-finite",
        refusal="header-not-json",
        why="A JSON number outside finite binary64 becomes infinity in common decoders and can silently poison unrelated metadata consumers.",
        extension="ali",
        build=lambda: _raw_envelope(1, '{"_total_points":0,"custom":1e309}', b""),
    ),
    InvalidCase(
        name="header-nesting-too-deep",
        refusal="header-nesting-too-deep",
        why="Unbounded recursive JSON decoding can overflow a reader stack before malformed input reaches its ALF quarantine path.",
        extension="ali",
        build=lambda: _raw_envelope(
            1,
            '{"_total_points":0,"custom":' + "[" * 256 + "0" + "]" * 256 + "}",
            b"",
        ),
    ),
    InvalidCase(
        name="header-integer-outside-binary64",
        refusal="header-not-json",
        why="An arbitrary-precision JSON integer outside finite binary64 is not portable to the Dart and TypeScript metadata domains.",
        extension="ali",
        build=lambda: _raw_envelope(
            1,
            '{"_total_points":1,"custom":' + str(10**400) + "}",
            bytes([7]),
        ),
    ),
    InvalidCase(
        name="anchor-extra-number-outside-binary64",
        refusal="header-not-json",
        why="Numeric members beyond an anchor object's x/y/z axes remain application metadata and must still fit the shared finite-binary64 JSON domain.",
        extension="alp",
        build=lambda: _raw_envelope(
            1,
            '{"_total_points":0,"device_position":{"x":0,"y":0,"z":0,"extra":' + str(10**400) + "}}",
            b"",
        ),
    ),
    InvalidCase(
        name="geometry-anchor-wrong-shape",
        refusal="geometry-anchor-malformed",
        why="A geometry anchor without all three axes cannot place the frame and must be refused during the envelope read rather than only when an accessor is called.",
        extension="alp",
        build=lambda: _envelope(
            1,
            {"_total_points": 1, "device_position": {"x": 1, "y": 2}},
            _fp16([[0.0, 0.0, 0.0]]),
        ),
    ),
    InvalidCase(
        name="total-points-exponent-outside-binary64",
        refusal="total-points-not-an-integer",
        why="A reserved count that overflows binary64 keeps its count-domain refusal instead of being misclassified as generic malformed metadata.",
        extension="ali",
        build=lambda: _raw_envelope(1, '{"_total_points":1e309}', b""),
    ),
    InvalidCase(
        name="total-points-hostile-positive-exponent",
        refusal="total-points-not-an-integer",
        why="A compact exponent must be bounded before a reader expands it into a hostile hundred-million-digit integer.",
        extension="ali",
        build=lambda: _raw_envelope(1, '{"_total_points":1e100000000}', b""),
    ),
    InvalidCase(
        name="total-points-hostile-negative-exponent",
        refusal="total-points-not-an-integer",
        why="A negative exponent larger than the significand's available trailing zeros must be refused before a reader constructs a hostile power-of-ten divisor.",
        extension="ali",
        build=lambda: _raw_envelope(1, '{"_total_points":1e-10000000}', b""),
    ),
    InvalidCase(
        name="v1-stale-geometry-number-outside-binary64",
        refusal="header-not-json",
        why="Version-1 geometry keys are stale format metadata, not active v3 codec parameters, so their numbers still belong to the finite-binary64 JSON domain.",
        extension="alp",
        build=lambda: _raw_envelope(
            1,
            '{"_total_points":0,"_geom_scale":' + str(10**400) + ',"_geom_offset":[0,0,0]}',
            b"",
        ),
    ),
    InvalidCase(
        name="geom-max-error-outside-binary64",
        refusal="header-not-json",
        why="A recorded geometry error outside finite binary64 is malformed metadata even when an empty payload has no reconstruction error to report.",
        extension="alp",
        build=lambda: _raw_envelope(
            3,
            '{"_total_points":0,"_geom_codec":"int16","_frame_convention":"world-aligned","_geom_scale":0.0001,'
            '"_geom_offset":[0,0,0],"_geom_max_err_m":' + str(10**400) + "}",
            b"",
        ),
    ),
    InvalidCase(
        name="gzip-concatenated-members",
        refusal="multiple-gzip-members",
        why="Concatenated members decode into one byte run, so the second file's envelope becomes the first one's payload.",
        extension="ali",
        # `mtime=0`: bare gzip.compress() stamps the CURRENT time into the
        # header, so this fixture differed between consecutive generator runs.
        # The determinism gate caught it the first time it ran, which is the
        # entire reason that gate exists.
        build=lambda: _gzip(_envelope(1, {}, bytes([1, 2]))) + _gzip(_envelope(1, {}, bytes([3, 4]))),
    ),
    InvalidCase(
        name="gzip-corrupt-crc",
        refusal="gzip-malformed",
        why="A gzip trailer CRC mismatch proves the decompressed ALF bytes are corrupt even when the deflate stream itself can be decoded.",
        extension="ali",
        build=lambda: _gzip_with_corrupt_crc(_envelope(1, {"_total_points": 2}, bytes([1, 2]))),
    ),
    InvalidCase(
        name="gzip-corrupt-header-crc",
        refusal="gzip-malformed",
        why="A member carrying FHCRC explicitly authenticates its gzip header; accepting a mismatched checksum treats marked corruption as valid input.",
        extension="ali",
        build=lambda: _gzip_with_header_crc(_envelope(1, {"_total_points": 2}, bytes([1, 2])), corrupt=True),
    ),
    InvalidCase(
        name="alx-source-total-points-missing",
        refusal="source-total-points-missing",
        why="Without the original cloud size, a sidecar index cannot be proven to address a real source point.",
        extension="alx",
        build=lambda: _envelope(
            1,
            {"_total_points": 2},
            struct.pack("<II", 1, 2),
        ),
    ),
    InvalidCase(
        name="alx-total-points-missing",
        refusal="total-points-missing",
        why="Without the kept-point count, the sidecar cannot be proven to contain one source mapping per decimated geometry point.",
        extension="alx",
        build=lambda: _envelope(
            1,
            {"_source_total_points": 10},
            struct.pack("<II", 1, 2),
        ),
    ),
    InvalidCase(
        name="alx-kept-index-duplicate",
        refusal="kept-index-duplicate",
        why="A repeated source index maps two labelled points to the same raw point while another kept point has no mapping.",
        extension="alx",
        build=lambda: _envelope(
            1,
            {"_total_points": 2, "_source_total_points": 10},
            struct.pack("<II", 4, 4),
        ),
    ),
    InvalidCase(
        name="alx-kept-index-out-of-range",
        refusal="kept-index-out-of-range",
        why="An index at or beyond the source count makes export address data outside the authoritative cloud.",
        extension="alx",
        build=lambda: _envelope(
            1,
            {"_total_points": 2, "_source_total_points": 2},
            struct.pack("<II", 0, 2),
        ),
    ),
    # --- the v3 frame convention ------------------------------------------
    #
    # A `.alp`'s points are the world points with an origin subtracted, which is
    # half of a rigid transform. Producers disagreed about the other half: some
    # translated and stopped, leaving points on world axes; some also rotated
    # into the device frame. Nothing in the bytes told them apart, and the two
    # agree exactly whenever the device heading is identity — so a consumer that
    # omitted the rotation matched a producer that omitted it, and the pair
    # looked right until the vehicle turned.
    #
    # Version 3 refuses to be ambiguous. Versions 1 and 2 are grandfathered:
    # deployed files cannot be told retroactively what they meant.
    InvalidCase(
        name="frame-convention-missing",
        refusal="frame-convention-missing",
        why="A v3 frame that does not say whether its points were rotated into the device frame cannot be placed in the world; reading it as either convention is right for one producer and silently wrong for the other.",
        extension="alp",
        build=lambda: _envelope(
            3,
            {
                "_total_points": 1,
                "_geom_codec": "int16",
                "_geom_scale": 0.01,
                "_geom_offset": [0.0, 0.0, 0.0],
            },
            _int16([1, 0, 0]),
        ),
    ),
    InvalidCase(
        name="frame-convention-unsupported",
        refusal="frame-convention-unsupported",
        why="A convention nobody has defined cannot be guessed at, and it is the field that says where these points are.",
        extension="alp",
        build=lambda: _envelope(
            3,
            {
                "_total_points": 1,
                "_geom_codec": "int16",
                "_frame_convention": "ego",
                "_geom_scale": 0.01,
                "_geom_offset": [0.0, 0.0, 0.0],
            },
            _int16([1, 0, 0]),
        ),
    ),
    InvalidCase(
        name="frame-convention-on-legacy-version",
        refusal="frame-convention-on-legacy-version",
        why="Readers apply no rotation to v1 geometry, so a v1 file declaring `device-aligned` states a promise the format does not keep; the producer believes a rotation will be undone and it will not be.",
        extension="alp",
        build=lambda: _envelope(
            1,
            {"_total_points": 1, "_frame_convention": "device-aligned"},
            _fp16([1.0, 0.0, 0.0]),
        ),
    ),
    InvalidCase(
        name="frame-rotation-missing",
        refusal="frame-rotation-missing",
        why="A device-aligned frame without its rotation cannot be returned to the world at all, and falling back to the world-aligned reading is the exact error the declaration exists to prevent.",
        extension="alp",
        build=lambda: _envelope(
            3,
            {
                "_total_points": 1,
                "_geom_codec": "int16",
                "_frame_convention": "device-aligned",
                "_geom_scale": 0.01,
                "_geom_offset": [0.0, 0.0, 0.0],
            },
            _int16([1, 0, 0]),
        ),
    ),
    InvalidCase(
        name="frame-rotation-malformed",
        refusal="frame-rotation-malformed",
        why="A quaternion missing a component is not a rotation; defaulting the absent member to zero would produce a confident rotation the producer never described.",
        extension="alp",
        build=lambda: _envelope(
            3,
            {
                "_total_points": 1,
                "_geom_codec": "int16",
                "_frame_convention": "device-aligned",
                "_frame_rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
                "_geom_scale": 0.01,
                "_geom_offset": [0.0, 0.0, 0.0],
            },
            _int16([1, 0, 0]),
        ),
    ),
    InvalidCase(
        name="frame-rotation-not-normalized",
        refusal="frame-rotation-not-normalized",
        why="Conjugation by a non-unit quaternion scales by |q|^2 as well as rotating, so every coordinate in the frame moves by a factor nothing else in the file disagrees with.",
        extension="alp",
        build=lambda: _envelope(
            3,
            {
                "_total_points": 1,
                "_geom_codec": "int16",
                "_frame_convention": "device-aligned",
                "_frame_rotation": {"w": 2.0, "x": 0.0, "y": 0.0, "z": 0.0},
                "_geom_scale": 0.01,
                "_geom_offset": [0.0, 0.0, 0.0],
            },
            _int16([1, 0, 0]),
        ),
    ),
    InvalidCase(
        name="frame-rotation-unexpected",
        refusal="frame-rotation-unexpected",
        why="`world-aligned` declares that no rotation was applied, so a rotation beside it means one of the two statements is wrong — and the reading that drops it returns plausible, wrongly-oriented geometry.",
        extension="alp",
        build=lambda: _envelope(
            3,
            {
                "_total_points": 1,
                "_geom_codec": "int16",
                "_frame_convention": "world-aligned",
                "_frame_rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
                "_geom_scale": 0.01,
                "_geom_offset": [0.0, 0.0, 0.0],
            },
            _int16([1, 0, 0]),
        ),
    ),
    # A PRESENT null is not an absent key. `metadata.get()` cannot tell them
    # apart, and the two mean opposite things: absent is the legacy state every
    # v1 file is in, while a present null is a producer that wrote the field and
    # got it wrong. The reference reader accepted this on v1 and called it
    # MISSING on v3, where the other three readers refused both — a divergence
    # no case existed to catch.
    InvalidCase(
        name="frame-convention-present-null",
        refusal="frame-convention-unsupported",
        why="A present null is something the producer wrote, not an absent field; reading it as absence accepts a v3 frame that states no convention at all.",
        extension="alp",
        build=lambda: _envelope(
            3,
            {
                "_total_points": 1,
                "_geom_codec": "int16",
                "_frame_convention": None,
                "_geom_scale": 0.01,
                "_geom_offset": [0.0, 0.0, 0.0],
            },
            _int16([1, 0, 0]),
        ),
    ),
    InvalidCase(
        name="frame-convention-present-null-on-legacy",
        refusal="frame-convention-on-legacy-version",
        why="A v1 file carrying the key at all is refused whatever its value; treating a null as absence lets a legacy file declare a convention its reader will not honour.",
        extension="alp",
        build=lambda: _envelope(
            1,
            {"_total_points": 1, "_frame_convention": None},
            _fp16([1.0, 0.0, 0.0]),
        ),
    ),
    # The rotation is under the exact-token rule, and nothing but that rule
    # catches this one: `0.99999999999999999` decodes to exactly 1.0, so the
    # quaternion is a perfect unit and the norm check passes. Two readers
    # enforced the token rule here and two did not.
    InvalidCase(
        name="frame-rotation-lossy-token",
        refusal="geom-number-inexact",
        why="A rotation component whose stated decimal changes when binary64 decodes it turns the whole frame, and it decodes to a perfect unit quaternion so no other check can see it.",
        extension="alp",
        build=lambda: _raw_envelope(
            3,
            '{"_total_points":1,"_geom_codec":"int16",'
            '"_frame_convention":"device-aligned",'
            '"_frame_rotation":{"w":0.99999999999999999,"x":0,"y":0,"z":0},'
            '"_geom_scale":0.01,"_geom_offset":[0,0,0]}',
            _int16([1, 0, 0]),
        ),
    ),
]


def identifiers() -> set:
    """Every refusal identifier this corpus can produce."""
    return {case.refusal for case in CASES}
