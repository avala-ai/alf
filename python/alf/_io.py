"""The one place ALF (Arrays in Local Frames) is read and written.

ALF's niche is narrow and worth being excellent at: **point-cloud sequences in a
LOCAL coordinate frame, streamed per frame into a browser**. That means small
files, a GPU-friendly layout, and a frame-local origin — not a general geospatial
archive (LAS/LAZ/COPC own that, and better). Within the niche, "state of the art"
has a concrete meaning: *the tightest, most honest error bound achievable per byte
delivered*, with the loss written down rather than discovered later.

Each file is one typed array plus a JSON header, and that is the whole of the
name: an `.alp` is an *array of local points*, an `.ali` an array of local
intensity, an `.alc` local color, an `.all` local labels, an `.alx` local
indexes. Channels are index-correlated — entry *i* of every channel describes
point *i* — which is the invariant most of the validation here exists to defend.

The clouds are usually lidar, and the format grew up on ego-centric automotive
sweeps — but nothing here is lidar-specific. Photogrammetry, RGBD, aerial tiles
and any other source of XYZ + per-point attributes use the same channels, and the
error analysis below is about the CODEC, not the sensor. "Local" is the load-
bearing word: a frame carries its own origin (`device_position`, or an aerial
`world_centroid_epsg3857`), so precision is spent on the frame's own span rather
than on its distance from some global datum.

Wire format, all channels::

    byte 0        version
    bytes 1-4     header length, uint32 little-endian
    bytes 5..N    UTF-8 JSON header
    rest          payload, keyed by extension AND (for .all) `_format_version`

Geometry codecs for ``.alp``
---------------------------

**v1 — float16 (legacy).** Three binary16 per point. Its error scales with each
COMPONENT's magnitude (half-ULP ~= |c| * 2^-11), so precision silently degrades
with distance from the frame origin: ~4 mm at 8 m, ~3 cm at 100 m, ~25 cm at
512 m. A typical automotive lidar is ~+/-1-2 cm, so beyond ~30 m the FORMAT, not that sensor,
is the dominant error — while near the origin it spends bits on precision no
sensor delivers. Both halves of that trade are wrong.

**v3 — int16 fixed-point (OPT-IN: pass ``codec="int16"``).** Three int16 per
point plus a per-frame ``_geom_scale`` and ``_geom_offset`` in the header::

    stored  = round((value - offset) / scale)
    decoded = stored * scale + offset

*Identical 6 bytes per point.* The error becomes ``scale/2``, UNIFORM across the
whole frame. ``offset`` is the frame's per-axis bounding-box MIDPOINT and
``scale`` comes from the centred coordinates, so the grid is built around the
data rather than around zero and the full int16 range is spent on the frame's
own span. For a +/-100 m frame that is +/-1.5 mm everywhere — about 20x better
than float16 at the same distance, for the same bytes, with the bound stated in
the header instead of inferred. Centring is what makes that true of OFF-CENTRE
frames too: without it a cloud spanning x = 100..200 m would derive its step from
200 m instead of its own 50 m half-span, and quantize 4x worse than it needed to.

This is the whole SOTA claim, and ``tests/test_io.py`` measures it rather
than asserting it.

Compatibility
-------------
**fp16/v1 is the DEFAULT, deliberately.** Porting a legacy writer to this module
must keep its output readable by every existing consumer, so `write_alp` emits
the legacy version byte, header layout and payload unless a caller explicitly
asks for `codec="int16"`.

This is payload and layout compatibility, NOT byte equality. Two header
differences are deliberate: `box` is computed from the DECODED coordinates
(a pre-encoding extremum can sit inside the decoded extent, so a legacy-style box
makes spatial filtering exclude real payload points), and the gzip envelope
forces `mtime=0` with an empty filename so the compressed bytes are a pure
function of the content. A golden comparison against legacy output will differ
in `box` and in the gzip header, by design — see `README.md`. Every consumer
must support v3 before its producer writes it. A reader that rejects unknown
versions cannot read v3; one that ignores the version and decodes float16
regardless returns wrong coordinates. See ``spec/adoption.md`` for the
migration requirements.
"""

from __future__ import annotations

import decimal
import functools
import gzip
import itertools
import json
import math
import os
import re
import stat
import struct
import uuid
import warnings
import zlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from alf._guard import check_fp16_component_budget

# Wire versions. 2 is not a geometry version: it marks the interleaved `.all`
# label layout and is carried in the header's `_format_version`, not here.
ALF_VERSION_V1 = 1
ALF_VERSION_V3 = 3
SUPPORTED_VERSIONS = (ALF_VERSION_V1, 2, ALF_VERSION_V3)

# `.all` payload layouts, keyed by the header's `_format_version`.
LABELS_V1 = 1  # 1 byte/point: category
LABELS_V2 = 2  # 3 bytes/point: category + uint16 LE instance id

GEOM_FP16 = "fp16"
GEOM_INT16 = "int16"

# How a `.alp`'s local points relate to the world frame. REQUIRED on version 3
# and forbidden before it — see `_validate_frame_convention` for why the two
# halves of that rule are not symmetric.
FRAME_WORLD_ALIGNED = "world-aligned"
FRAME_DEVICE_ALIGNED = "device-aligned"
FRAME_CONVENTIONS = (FRAME_WORLD_ALIGNED, FRAME_DEVICE_ALIGNED)

# A quaternion that is not a unit quaternion SCALES as well as rotates: the
# conjugation q*v*q⁻¹ maps a vector to |q|² times its rotation. At |q| = 1.05
# that is a 10% stretch — 10 m of error at 100 m — applied to the whole frame
# and to nothing else in the file, so nothing downstream disagrees with it.
# Readers therefore refuse rather than normalize: normalizing would silently
# repair a producer that has a bug, and the bug would stay.
#
# The bound is on the SQUARED norm because that is the factor actually applied,
# and 1e-9 is far looser than the double rounding of a genuinely normalized
# quaternion (~1e-16) and far tighter than any error worth a metre.
FRAME_ROTATION_NORM_TOLERANCE = 1e-9

# binary16 spacing stops halving with the exponent once values go subnormal
# (below 2**-14): it is fixed at 2**-24 there, so the half-ULP error bound floors
# at 2**-25 rather than continuing to shrink.
FP16_SUBNORMAL_HALF_ULP = 2.0**-25

# int16 spans [-32768, 32767]; 32767 keeps the encoding symmetric so a scale
# derived from the positive extent cannot overflow on the negative side.
_INT16_MAX = 32767

# The reserved word for "this beam produced no return", in v3 fixed-point.
#
# Free by construction: `choose_geom_scale` derives the step as
# `largest / _INT16_MAX`, so a real coordinate quantises into [-32767, +32767]
# and -32768 is unreachable. Two's complement leaves it without a positive
# counterpart, which is what makes it usable as a marker rather than a value.
#
# A whole point of these three decodes to NaN, matching what fp16 stores
# natively — so the two codecs round-trip the same frame to the same array, and
# the dense range-image grid that a lidar produces survives re-encoding.
INT16_NO_RETURN = -32768

# Floor on the fixed-point step. 0.1 mm is already an order of magnitude below
# any lidar's ranging noise; going finer would spend int16 range on precision
# nothing measures, shrinking the representable frame for no gain.
MIN_GEOM_SCALE_M = 1e-4

# int16 states its own error, so unlike float16 it is judged on the error it
# actually produces rather than on coordinate magnitude. 0.25 m is the point past
# which a frame is useless for annotation — reached only around a +/-8 km extent,
# which for an ego-centric frame means the localization failed.
INT16_ERROR_BUDGET_M = 0.25

_HEADER_STRUCT = struct.Struct("<I")

# The channels this module knows. An extension outside this set must never
# silently become "" — that disables every channel guard at once.
_KNOWN_CHANNELS = frozenset({"alp", "ali", "all", "alc", "alx"})

# Matches the long-standing limit in the legacy label-fetch path this module
# replaced. Real headers are a few KB; the largest thing they legitimately carry
# is an `instance_map` for one sequence.
_MAX_HEADER_BYTES = 10_000_000

# Count JSON objects and arrays, including the root object. This is low enough
# to stay inside every supported runtime's recursive decoder/encoder stack and
# vastly above real ALF metadata nesting.
_MAX_JSON_DEPTH = 256

# 1 GiB permits a 100-million-point geometry payload (600,000,000 bytes)
# with headroom for its envelope. Tests exercise large synthetic frames and
# measure allocation overhead. The read loop appends into a `bytearray` instead
# of collecting chunks and joining them into a second contiguous copy.
#
# Mutable on purpose: a memory-constrained worker should be able to lower it,
# and a batch job that knows it is reading tiles should be able to raise it,
# without either editing the package.
MAX_DECOMPRESSED_BYTES = 1024 * 1024 * 1024

# Reading and decoding a frame require separate allocations. Geometry expands
# fourfold from six wire bytes per point to three float64 components. `.xyz`
# checks its own configurable budget before allocating the decoded array.
MAX_DECODED_GEOMETRY_BYTES = 3 * 1024 * 1024 * 1024

# Input block for the incremental gzip decode. A module constant so the
# boundary case it creates — a member ending exactly on a block edge — is
# reachable from a test without building a 16 MB fixture.
_GZIP_CHUNK_BYTES = 8 * 1024 * 1024

# Deflate cannot expand data by more than ~0.03% plus framing, so a legitimate
# archive of a permitted envelope can sit slightly ABOVE the cap on the wire.
# The compressed input still gets a hard bound; it is just the right one.
_GZIP_OVERHEAD_ALLOWANCE = 1024 * 1024

# Input block for the HEADER-ONLY read. Deliberately far smaller than
# `_GZIP_CHUNK_BYTES`: the whole point of `read_alf_header` is that a caller
# scanning a prefix of S3 objects pays a few KB per frame instead of a full GET,
# and an 8 MB first read would have thrown that away for any real frame, whose
# envelope is a few hundred compressed bytes. Real headers fit in one block.
_HEADER_PREFIX_CHUNK_BYTES = 64 * 1024


def _limit_bytes() -> int:
    """The decompression cap, read at CALL time so the setter is live."""
    return MAX_DECOMPRESSED_BYTES


def _decoded_geometry_limit_bytes() -> int:
    """Same, for the decoded-array bound."""
    return MAX_DECODED_GEOMETRY_BYTES


def set_memory_limits(*, decompressed_bytes: int | None = None, decoded_geometry_bytes: int | None = None) -> None:
    """Adjust the two memory bounds. Pass only what you mean to change.

    A FUNCTION rather than two writable constants, because `alf` re-exports
    lazily: `alf.MAX_DECOMPRESSED_BYTES = n` would bind the name on the PACKAGE
    while every reader keeps consulting `alf._io`'s. That reads exactly like a
    working knob and does nothing — the failure mode this package keeps finding
    in other people's code, so it does not ship one of its own.

    Both bound memory, not correctness. Lower them on a constrained worker;
    raise them for a batch job that knows it is reading merged tiles. Neither
    changes which files are WELL-FORMED, only which this process will hold.
    """
    global MAX_DECOMPRESSED_BYTES, MAX_DECODED_GEOMETRY_BYTES  # noqa: PLW0603
    for name, value in (
        ("decompressed_bytes", decompressed_bytes),
        ("decoded_geometry_bytes", decoded_geometry_bytes),
    ):
        if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value <= 0):
            raise ValueError(f"{name} must be a positive int, got {value!r}")
    if decompressed_bytes is not None:
        MAX_DECOMPRESSED_BYTES = decompressed_bytes
    if decoded_geometry_bytes is not None:
        MAX_DECODED_GEOMETRY_BYTES = decoded_geometry_bytes


def memory_limits() -> dict[str, int]:
    """The two bounds currently in force, for a caller that wants to report them."""
    return {
        "decompressed_bytes": MAX_DECOMPRESSED_BYTES,
        "decoded_geometry_bytes": MAX_DECODED_GEOMETRY_BYTES,
    }


class AlfError(Exception):
    """Raised when bytes are not a well-formed ALF file."""


_HEX_COLOR = frozenset("0123456789abcdefABCDEF")


def _as_hex_color(value: Any, path: str | Path) -> str:
    """Coerce one palette entry to a six-digit `rrggbb` string.

    `str(c)` would turn the `(255, 0, 0)` tuples that the canonical palette
    generator returns into the
    literal `"(255, 0, 0)"` — structurally valid JSON that no consumer can read
    as a colour. The legacy writer converts those tuples to `ff0000`, so this
    accepts them rather than making every caller remember, and rejects anything
    that is neither.
    """
    if isinstance(value, str):
        # ONE optional leading '#'. `lstrip` strips ANY number, so "##ff0000"
        # validated clean and was written to a header no viewer can parse.
        text = value[1:] if value.startswith("#") else value
        if len(text) == 6 and all(ch in _HEX_COLOR for ch in text):
            return text.lower()
        raise ValueError(f"{path}: palette entry {value!r} is not six hex digits (rrggbb)")
    if isinstance(value, (tuple, list)) and len(value) == 3:
        rgb = []
        for component in value:
            if isinstance(component, bool) or not isinstance(component, (int, float, np.integer, np.floating)):
                raise ValueError(f"{path}: palette RGB component {component!r} is not a number")
            if not isinstance(component, (int, np.integer)):
                # FLOATS ARE REFUSED, including whole ones. A float palette is
                # ambiguous in exactly the case that matters: (1.0, 0, 0) is red
                # in a normalized [0, 1] palette and near-black in an 0..255 one,
                # and both are whole numbers, so no value check can tell them
                # apart. The viewer treats whichever we pick as authoritative,
                # so the caller has to say which scale it meant.
                raise ValueError(
                    f"{path}: palette RGB component {component!r} must be an integer 0..255. "
                    f"Floats are refused because (1.0, 0, 0) is red on a normalized [0, 1] scale "
                    f"and near-black on an 0..255 one — scale to 0..255 and pass ints."
                )
            c = int(component)
            if not 0 <= c <= 255:
                raise ValueError(f"{path}: palette RGB component {component!r} is outside 0..255")
            rgb.append(c)
        return "{:02x}{:02x}{:02x}".format(*rgb)
    raise ValueError(f"{path}: palette entry {value!r} must be an 'rrggbb' string or an (r, g, b) triple")


def _reject_json_constant(token: str) -> Any:
    """`NaN` / `Infinity` / `-Infinity` are not JSON, whatever Python accepts."""
    raise AlfError(f"Header contains the non-JSON literal {token!r}; ALF headers must be valid JSON")


class _HeaderFloat(float):
    """A header float that remembers the token it was parsed from.

    `float("1.0000000000000001")` is exactly 1.0, and `float("1e-400")` is 0.0,
    so a fractional count validated AFTER conversion looks like a clean
    integer — the evidence is destroyed before `_checked_count` or
    `format_version` ever sees it. Carrying the token lets those two ask what
    the file actually said.
    """

    __slots__ = ("token",)

    def __new__(cls, token: str):
        obj = super().__new__(cls, token)
        obj.token = token
        return obj


class _HeaderInt(int):
    """A header integer that retains the exact JSON token."""

    def __new__(cls, token: str):
        obj = super().__new__(cls, token)
        obj.token = token
        return obj


def _checked_json_float(token: str) -> float:
    # Preserve the token before classifying finiteness. Reserved counts need
    # their own `total-points-not-an-integer` refusal for an exponent such as
    # `1e309`; rejecting here misclassifies it as generic malformed metadata.
    # The iterative metadata walk below still rejects every non-reserved
    # infinity before an accessor can expose it.
    return _HeaderFloat(token)


def _checked_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build one JSON object while refusing duplicate member names.

    ``json.loads`` otherwise silently keeps the last value.  That is unsafe for
    an ALF header: two readers can inspect different occurrences of a reserved
    field, and even duplicate application metadata is not preserved faithfully.
    Apply the rule recursively by using this as ``object_pairs_hook``.
    """
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AlfError(f"Header contains duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _reject_non_integral_token(value: Any, key: str) -> None:
    """Refuse a count/layout field whose LITERAL was not an integer.

    Only meaningful for values that came from the header parser; an explicit
    Python float from a caller has no token and is judged on its value alone.
    """
    token = getattr(value, "token", None)
    if token is None:
        return
    # `Decimal`, because the question is what the LITERAL means exactly.
    # `int(token)` raises on "1.0" — which is a perfectly good integer — so
    # judging integrality by whether int() parses rejects valid headers. That
    # was my first attempt at this check and it refused `_total_points: 1.0`.
    try:
        exact = decimal.Decimal(token)
    except decimal.InvalidOperation:
        return
    if exact != exact.to_integral_value():
        raise AlfError(
            f"`{key}` must be an integer, but the header says {token!r}. It rounds to "
            f"{float(token)!r} in float64, so accepting it would silently change the value "
            f"this file declares."
        )


def _reject_inexact_count_token(value: Any, key: str) -> None:
    """Refuse an integral count token whose value changes in binary64."""
    token = getattr(value, "token", None)
    if token is None:
        return
    try:
        exact = decimal.Decimal(token)
    except decimal.InvalidOperation:
        return
    if exact != exact.to_integral_value():
        return
    # A zero significand is exactly zero regardless of exponent length. Do not
    # send `0e10000000000` through int(Decimal): no expansion is needed to prove
    # its value and the shared contract explicitly accepts this spelling.
    if exact.is_zero():
        return
    # Every finite binary64 integer has at most 309 decimal digits. `Decimal`
    # retains a huge exponent compactly, but int(Decimal("1e100000000")) would
    # materialize a hundred-million-digit bigint before float() can reject it.
    # Bound by the adjusted exponent while the representation is still compact.
    if exact.adjusted() > 308:
        raise AlfError(f"`{key}` holds an integer too large for binary64")
    integer = int(exact)
    try:
        decoded = float(integer)
    except OverflowError as exc:
        raise AlfError(f"`{key}` holds an integer too large for binary64") from exc
    if not math.isfinite(decoded) or int(decoded) != integer:
        raise AlfError(f"`{key}` states integer {integer}, which binary64 cannot represent exactly")


def _check_json_depth(text: str) -> None:
    """Bound nesting before CPython's recursive JSON decoder sees the header."""
    depth = 0
    in_string = False
    escaped = False
    for character in text:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character in "[{":
            depth += 1
            if depth > _MAX_JSON_DEPTH:
                raise AlfError(
                    "header-nesting-too-deep: Header JSON nesting is too deep; "
                    f"maximum nesting depth is {_MAX_JSON_DEPTH}"
                )
        elif character in "]}":
            depth -= 1


def _check_writer_json_depth(metadata: dict[str, Any]) -> None:
    """Enforce the reader's collection-depth ceiling without recursing."""
    stack = [(metadata, 1, False)]
    active: set[int] = set()
    while stack:
        value, depth, leaving = stack.pop()
        if leaving:
            active.remove(id(value))
            continue
        if not isinstance(value, (dict, list, tuple)):
            continue
        if depth > _MAX_JSON_DEPTH:
            raise ValueError(f"metadata exceeds the maximum JSON nesting depth of {_MAX_JSON_DEPTH}")
        identity = id(value)
        if identity in active:
            raise ValueError("metadata contains a circular JSON collection")
        active.add(identity)
        stack.append((value, depth, True))
        children = value.values() if isinstance(value, dict) else value
        stack.extend((child, depth + 1, False) for child in children)


def _as_per_point(values: Any, label: str, path: str | Path) -> np.ndarray:
    """Coerce a per-point stream to 1-D, accepting (N,) and (N, 1) only.

    A bare `ravel()` also flattens an (N, 4) point matrix or an
    intensity/ring slice into 4N values, and the writer then records
    `_total_points` for that expanded size. Every channel here is correlated
    with `.alp` BY INDEX, so the result is a structurally valid file whose
    length no longer matches the cloud — downstream attaches shifted or extra
    values to the wrong points instead of failing at the source.
    """
    if _contains_bool(values):
        raise ValueError(
            f"{path}: {label} must be numeric, not booleans — a mask would be encoded as ordinary per-point values"
        )
    # Masked and stringly-typed input, for EVERY per-point channel.
    #
    # `mask is not nomask` was too strict: an EXPLICIT all-False mask is a
    # perfectly ordinary array that exposes nothing, and refusing it rejected
    # valid data across all five channels. The question is whether anything is
    # actually masked, which is what `getmaskarray(...).any()` asks. These two
    # rules were added to geometry and to the anchors, and each time the
    # non-geometry channels kept the hole: `np.asarray` drops a MaskedArray's
    # mask and publishes the rows the caller marked invalid, and an unparsed
    # CSV column of numeric strings casts to real-looking measurements. Both
    # produce index-correlated data that is wrong rather than absent, which is
    # the whole failure mode this module exists to prevent. One place, so the
    # next channel added here inherits it.
    if isinstance(values, np.ma.MaskedArray) and np.ma.getmaskarray(values).any():
        raise ValueError(
            f"{path}: {label} is a masked array. `np.asarray` drops the mask and would publish "
            f"the masked entries as ordinary per-point values; select or fill them explicitly "
            f"(`arr.compressed()`, `arr.filled(...)`) so the intent is in the caller's code."
        )
    arr = np.asarray(values)
    # The ORIGINAL object first, then the array — the same promotion hole
    # geometry has. Every per-point channel is index-correlated with the cloud,
    # so a value that shifted during coercion is as wrong here as a coordinate.
    lossy = _lossy_before_promotion(values) or _narrows_lossily_to_float64(arr)
    if lossy:
        raise ValueError(
            f"{path}: {label} cannot narrow to float64 without merging distinct values "
            f"({lossy}). Rebase the samples, or convert at the source."
        )
    if _is_stringly_typed(arr):
        if _holds_actual_strings(arr):
            raise ValueError(
                f"{path}: {label} holds strings, not numbers (dtype {arr.dtype!r}). A numeric "
                f"string casts cleanly into a valid-looking channel and hides the parser bug that "
                f"produced it; convert at the source."
            )
        raise ValueError(
            f"{path}: {label} holds non-numeric objects (dtype {arr.dtype!r}). `Decimal`, "
            f"`Fraction` and generators all arrive as an object array NumPy cannot narrow "
            f"without deciding the precision itself; convert to float or int at the source."
        )
    _reject_non_real(arr, label, path)
    if arr.dtype == np.bool_:
        # A mask coerces to 0.0/1.0 and then encodes as ordinary bytes -- [255, 0]
        # under typical bounds -- so a mask-shaped caller bug becomes a silently
        # wrong intensity channel that drives viewer contrast.
        raise ValueError(
            f"{path}: {label} must be numeric, not booleans — a mask would be encoded as ordinary intensity values"
        )
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr.reshape(-1)
    if arr.ndim != 1:
        raise ValueError(
            f"{path}: {label} must be 1-D (or an (N, 1) column), got shape {arr.shape}. "
            f"This channel is index-correlated with the geometry, so flattening a wider array "
            f"would publish more values than there are points."
        )
    return arr


def _describe_source(source: object) -> str:
    """A short label for an ALF source, safe to put in an exception message.

    A bytes/bytearray input is a documented way to call `read_alf`, and these
    frames run to hundreds of MB — interpolating one into an error string
    allocates and then logs the entire payload while the caller is trying to
    quarantine a single bad frame. Paths and streams are already short.
    """
    if isinstance(source, (bytes, bytearray, memoryview)):
        return f"<{type(source).__name__} of {len(source)} bytes>"
    if isinstance(source, (str, Path)):
        return str(source)
    return f"<{type(source).__name__}>"


def _safe_int(k: Any) -> Any:
    try:
        return int(str(k))
    except (TypeError, ValueError):
        return object()  # never equal to str(k), so a non-numeric key is flagged


def _instance_unresolved(resolved: dict[str, Any], instance_id: int) -> bool:
    """Whether an id lacks a USABLE map entry — a present key is not enough.

    A legacy batch exporter writes `instance_lookup.get(key)`
    straight into `object_uuid`, so a null, empty or non-string value produces a
    successful export carrying missing or garbage identity. Coverage means the
    id resolves to something, not that someone wrote the key down.
    """
    key = str(instance_id)
    if key not in resolved:
        return True
    value = resolved[key]
    return not isinstance(value, str) or not value.strip()


def _present_or_absent(metadata: dict[str, Any], key: str) -> Any:
    """Return the value, or None if the key is ABSENT — rejecting a present null.

    `metadata.get(key)` collapses the two, and they mean opposite things here:
    absent is a legacy file that never carried the field (which every reader
    must still handle), while a present `null` is a header that carries it and
    says nothing — so treating them alike lets malformed metadata take the
    permissive legacy path and skip the validation that follows.
    """
    if key not in metadata:
        return None
    value = metadata[key]
    if value is None:
        raise AlfError(
            f"`{key}` is present but null. An ABSENT key is legacy and handled; a key that is "
            f"there and empty is a malformed header, and accepting it would skip the checks the "
            f"field exists for."
        )
    return value


def _checked_count(metadata: dict[str, Any], key: str) -> int | None:
    """A header point count: absent, or a non-negative JSON integer.

    Shared by `_total_points` and `_source_total_points` because both are used
    as counts before anything is allocated or range-checked against them, and
    the three languages disagree on every malformed shape — JavaScript coerces
    `"12"` and `true` through arithmetic, Dart's cast throws, Python's `int()`
    truncates `1.9`. One file, three different answers, unless the domain is
    pinned here.
    """
    if key not in metadata:
        return None
    value = metadata[key]
    if value is None:
        # PRESENT but null is not the same as absent: treating it as missing
        # disables the declared-count cross-check entirely, so a payload of any
        # whole-point length parses as valid despite a malformed header.
        raise AlfError(f"`{key}` is present but null; it must be a non-negative integer or absent")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AlfError(f"`{key}` must be a number, got {value!r}")
    _reject_non_integral_token(value, key)
    _reject_inexact_count_token(value, key)
    # Do NOT route through float(): a corrupt header declaring 10**400 raises
    # OverflowError there, escaping the quarantine contract before the AlfError
    # path can run. Ints are already whole, so only floats need the check.
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise AlfError(f"`{key}` must be a non-negative integer, got {value!r}")
        value = int(value)
    if value < 0:
        raise AlfError(f"`{key}` must be a non-negative integer, got {value!r}")
    return int(value)


def _checked_deciles(deciles: Iterable[Any]) -> list:
    """Nine whole, in-range, non-decreasing uint8 percentiles.

    Validate rather than coerce: 12.9 truncating to 12 and 300 passing through
    both publish a contrast curve that does not describe the data, and the file
    stays perfectly readable so nothing downstream reports it. Percentiles are
    also monotonic by definition, so a descending sequence describes a
    distribution that cannot exist — the viewer builds its curve from it anyway.

    Shared by the explicit-argument and inherited-metadata paths, because the
    sequence-level values normally arrive by inheritance and validating only the
    argument left the common case unguarded.
    """
    values = []
    raw = list(deciles)
    if len(raw) != 9:
        raise ValueError(f"deciles must hold 9 values (10th..90th percentile); got {len(raw)}")
    for d in raw:
        if isinstance(d, bool) or not isinstance(d, (int, float, np.integer, np.floating)):
            raise ValueError(f"deciles must be numbers, got {d!r}")
        if float(d) != int(d):
            raise ValueError(f"deciles must be whole numbers, got {d!r}")
        if not 0 <= int(d) <= 255:
            raise ValueError(f"deciles describe uint8 intensity and must be 0..255, got {d!r}")
        values.append(int(d))
    if any(b < a for a, b in itertools.pairwise(values)):
        raise ValueError(f"deciles must be non-decreasing (10th..90th percentile); got {values}")
    return values


def _reject_lossy_token(value: Any, key: str) -> None:
    """Refuse a header number whose LITERAL float64 cannot represent exactly.

    `9007199254740993.0` is a perfectly ordinary JSON token that becomes
    `9007199254740992.0` in float64 — one metre out for an anchor, a different
    grid for a codec parameter. Exact large integers such as ``2**64`` remain
    valid: magnitude is not itself precision loss.
    """
    token = getattr(value, "token", None)
    if token is None:
        return
    try:
        exact = decimal.Decimal(token)
    except decimal.InvalidOperation:
        return
    # Compared against the float's SHORTEST REPR, which is precisely what
    # float64 preserves.
    #
    # Third formulation of this check; the first two were both too strict.
    # Demanding `Decimal(token) == Decimal(float(token))` rejects every decimal
    # fraction, since almost none are exactly representable in binary. Adding
    # "integer-valued" still rejected `1e+308`, whose literal IS what float64
    # keeps. The real question is whether the header spelled more significant
    # digits than survive the read: `9007199254740993.0` comes back as
    # `...992.0` and moves an anchor a metre, while `1e+308` round-trips
    # unchanged and is nobody's precision claim.
    #
    # Restricting it to INTEGER-valued tokens was a fourth mistake, not a
    # safeguard: `9007199254740993.5` reads back as `...994.0` and shifts a
    # frame half a metre, and the repr comparison already accepts `0.1` and
    # `0.30000000000000004` on their own merits. The narrowing was defending
    # against a failure this formulation does not have.
    try:
        decoded = float(value)
    except OverflowError as exc:
        raise AlfError(f"`{key}` is too large to represent as finite float64") from exc
    if not math.isfinite(decoded):
        raise AlfError(f"`{key}` is too large to represent as finite float64")
    # A raw integer token has an exact integer meaning even when the decoded
    # float's shortest decimal spelling happens to equal the original token.
    # For example, float(18446744073709552000) is the different integer
    # 18446744073709551616, while repr() prints ``1.8446744073709552e+19``;
    # Decimal considers that spelling equal to the raw token and the generic
    # spelling fallback below therefore cannot detect the shifted grid.
    if "." not in token and "e" not in token.lower():
        if int(decoded) == int(exact):
            return
        raise AlfError(
            f"`{key}` is {token!r} in the header but float64 cannot represent exactly what that integer token states"
        )
    if exact == exact.to_integral_value():
        # Expanded `.0` and exponent-zero spellings remain exact mathematical
        # integer claims. They cannot inherit the compatibility escape for a
        # writer's normalized scientific binary64 spelling merely because both
        # forms canonicalize to the same decimal text.
        if int(decoded) == int(exact):
            return
        canonical_scientific = re.fullmatch(
            r"-?[0-9](?:\.[0-9]*[1-9])?[eE][+-]?[1-9][0-9]*",
            token,
        )
        if canonical_scientific and decimal.Decimal(repr(decoded)) == exact:
            return
        raise AlfError(
            f"`{key}` is {token!r} in the header but float64 cannot represent exactly what that integral token states"
        )
    if decimal.Decimal(repr(decoded)) != exact:
        raise AlfError(
            f"`{key}` is {token!r} in the header but {decoded!r} once read — float64 "
            f"cannot represent exactly what the token states, so this file would be interpreted "
            f"against a value it does not state"
        )


def _reject_nonfinite_number_tree(values: Iterable[Any]) -> None:
    """Reject numbers outside finite binary64 with bounded iterative walking."""
    pending = list(values)
    while pending:
        value = pending.pop()
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            try:
                decoded = float(value)
            except OverflowError as exc:
                token = getattr(value, "token", "<integer outside binary64>")
                raise AlfError(f"Header number {token!r} is not finite") from exc
            if not math.isfinite(decoded):
                token = getattr(value, "token", repr(value))
                raise AlfError(f"Header number {token!r} is not finite")
        elif isinstance(value, dict):
            pending.extend(value.values())
        elif isinstance(value, (list, tuple)):
            pending.extend(value)


def _reject_lossy_anchor_tokens(metadata: dict[str, Any]) -> None:
    """Validate exact geometry-anchor lexemes before accessors can ignore them."""
    for key in ("device_position", "world_centroid_epsg3857"):
        raw = metadata.get(key)
        if isinstance(raw, dict):
            components = [raw[axis] for axis in ("x", "y", "z") if axis in raw]
            # Extra object members are not placement axes, so they retain the
            # generic application-metadata rule rather than geometry's exact-
            # token rule. Validate their entire nested value without rejecting
            # deployed anchors merely for carrying additional metadata.
            _reject_nonfinite_number_tree(value for member, value in raw.items() if member not in {"x", "y", "z"})
        elif isinstance(raw, (list, tuple)):
            components = list(raw)
        else:
            continue
        for component in components:
            if _is_json_number(component):
                _reject_lossy_token(component, key)


def _validate_geometry_anchor_shapes(metadata: dict[str, Any]) -> None:
    """Reject unusable placement anchors while the envelope is being read."""
    for key in ("device_position", "world_centroid_epsg3857"):
        if key not in metadata:
            continue
        raw = metadata[key]
        # Deployed v1 files use an empty device_position object to mean the
        # local origin. Preserve that compatibility, but no partial anchor can
        # silently become a placement only after an accessor is called.
        if key == "device_position" and isinstance(raw, dict) and not raw:
            continue
        try:
            _as_anchor(raw, key)
        except AlfError as exc:
            raise AlfError(f"Geometry anchor `{key}` is malformed: {exc}") from exc


def _checked_frame_rotation(metadata: dict[str, Any]) -> dict[str, float]:
    """Validate `_frame_rotation` and return it as a plain w/x/y/z dict.

    An OBJECT rather than a four-element array, deliberately. Every array
    spelling of a quaternion has to choose between `[x, y, z, w]` and
    `[w, x, y, z]`, both are in wide use, and the two are indistinguishable in
    the bytes — so a reader and a writer that disagree produce a rotation that
    is wrong in a way no length check catches. Naming the members removes the
    choice.
    """
    raw = metadata["_frame_rotation"]
    if not isinstance(raw, dict):
        raise AlfError(f"`_frame_rotation` must be an object with members w, x, y and z, got {type(raw).__name__}.")
    missing = sorted({"w", "x", "y", "z"} - set(raw))
    if missing:
        raise AlfError(f"`_frame_rotation` is missing {', '.join(missing)}.")
    # EXACTLY the four members. An anchor tolerates extras because deployed
    # files carry them; this field is new, and a stray `qw` beside `w` is far
    # more likely to be a producer mixing two spellings than application data.
    extra = sorted(set(raw) - {"w", "x", "y", "z"})
    if extra:
        raise AlfError(
            f"`_frame_rotation` has unexpected members {', '.join(extra)}; exactly w, x, y "
            f"and z are allowed, so a second spelling of a component cannot sit beside the "
            f"one that is read."
        )
    components: dict[str, float] = {}
    for axis in ("w", "x", "y", "z"):
        value = raw[axis]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise AlfError(f"`_frame_rotation.{axis}` must be a JSON number, got {value!r}.")
        number = float(value)
        if not math.isfinite(number):
            raise AlfError(f"`_frame_rotation.{axis}` must be finite, got {value!r}.")
        # The same exact-token rule the grid parameters and the anchors are
        # under, and for the same reason: a component whose stated decimal
        # changes when it is decoded turns the whole frame, not one sample.
        _reject_lossy_token(value, "_frame_rotation")
        components[axis] = number
    return components


def _rotate_by_quaternion(points: np.ndarray, q: dict[str, float]) -> np.ndarray:
    """Rotate (N, 3) points by a unit quaternion, as v' = q v q*.

    Hamilton (`ij = k`) and ACTIVE, which `spec/format.md` §5.3 defines
    normatively and pins with a worked vector. It is defined there rather than
    left to each reader because naming the quaternion's members removes only
    half the ambiguity: the JPL convention agrees on every component and yields
    the transpose, so a consumer that built R from the other one would rotate
    the cloud the wrong way and get a plausible answer.

    Written out as the rotation matrix rather than as two quaternion products:
    the matrix is built once and applied with a single matmul, where the
    conjugation form allocates two intermediate (N, 4) arrays per frame.

    The caller has already established that `q` is a unit quaternion; this does
    not re-check, because a non-unit quaternion here would scale the cloud and
    that is a REFUSAL at the envelope, not something to absorb quietly.
    """
    w, x, y, z = q["w"], q["x"], q["y"], q["z"]
    matrix = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    # NaN in, NaN out: an invalid return has no position, and a rotation of "no
    # position" is still no position. The matmul carries it through per row
    # without special-casing, which is what keeps index correlation intact.
    return points @ matrix.T


def _validate_frame_convention(metadata: dict[str, Any], version: int) -> None:
    """Enforce that a v3 `.alp` STATES how its local points reach the world.

    This is the rule the format was missing, and it cost real data. A `.alp`'s
    points are the world points with an origin subtracted, which is what makes
    them small enough to be worth 6 bytes — but subtracting an origin is only
    half of a rigid transform. Producers disagreed about the other half: the
    canonical writer translated and stopped, leaving points on WORLD axes,
    while the cuboid path translated and then inverse-rotated by the device
    heading, leaving points on DEVICE axes. Both called the result "the frame".

    Nothing in the bytes tells them apart, and the two agree exactly whenever
    the device heading is identity — which, for a vehicle driving straight, is
    most frames of most sequences. So a consumer that omitted the rotation
    matched a producer that omitted it, and the pair looked correct until the
    vehicle turned, at which point geometry and annotations drifted apart by
    the accumulated heading with nothing raising.

    Version 3 refuses to be ambiguous. The field is REQUIRED rather than
    defaulted, because a default is the thing that failed: absence is exactly
    what both producers had, and reading it as either convention re-creates the
    bug for the other one.

    Versions 1 and 2 are grandfathered and keep the old behaviour, because
    deployed files cannot be retroactively told what they meant. What they may
    NOT do is carry the field: a v1 file declaring `device-aligned` would be
    read by every conforming reader as unrotated, so the declaration would be a
    promise the format does not keep.
    """
    # PRESENCE, not truthiness. `metadata.get()` cannot tell an absent key from
    # a present `null`, and the two mean opposite things here: absent is the
    # legacy state every v1 file is in, while a present null is a producer that
    # wrote the field and got it wrong. Dart, TypeScript and Rust all test
    # presence, and this reader used to accept a v1 `"_frame_convention": null`
    # that all three refused.
    convention_present = "_frame_convention" in metadata
    declared = metadata.get("_frame_convention")
    rotation_present = "_frame_rotation" in metadata

    if version != ALF_VERSION_V3:
        if convention_present or rotation_present:
            named = "_frame_convention" if convention_present else "_frame_rotation"
            raise AlfError(
                f"`{named}` is a version-3 field, but this .alp is version {version}. Readers "
                f"apply no rotation to v1/v2 geometry, so honouring the declaration is exactly "
                f"what they will not do — and a producer that set it believes otherwise. Write "
                f'this frame as version 3 (`codec="int16"`) to state a frame convention.'
            )
        return

    if not convention_present:
        raise AlfError(
            "version-3 .alp is missing `_frame_convention`. Its points are the world points "
            f"with an origin subtracted, and that is half of a rigid transform: the file must "
            f"say whether the other half — the device rotation — was applied. Use "
            f"{FRAME_WORLD_ALIGNED!r} when only the origin was subtracted, or "
            f"{FRAME_DEVICE_ALIGNED!r} with a `_frame_rotation` when the points were also "
            f"rotated into the device frame."
        )
    if not isinstance(declared, str) or declared not in FRAME_CONVENTIONS:
        raise AlfError(
            f"Unsupported `_frame_convention` {declared!r} (expected one of "
            f"{', '.join(repr(c) for c in FRAME_CONVENTIONS)}). A convention nobody has defined "
            f"cannot be guessed at; it is exactly what says where these points are."
        )

    if declared == FRAME_DEVICE_ALIGNED:
        if not rotation_present:
            raise AlfError(
                f"`_frame_convention: {FRAME_DEVICE_ALIGNED!r}` requires `_frame_rotation`. "
                f"Without the rotation the points cannot be returned to the world at all, and "
                f"treating the frame as world-aligned is the error this convention exists to "
                f"make impossible."
            )
        components = _checked_frame_rotation(metadata)
        norm_squared = sum(value * value for value in components.values())
        if abs(norm_squared - 1.0) > FRAME_ROTATION_NORM_TOLERANCE:
            raise AlfError(
                f"`_frame_rotation` is not a unit quaternion (|q|^2 = {norm_squared!r}). "
                f"Rotating by it would scale every coordinate in the frame by that factor as "
                f"well as rotating it, uniformly and silently. Normalize it in the writer; a "
                f"reader that normalized here would hide the fault rather than fix it."
            )
    elif rotation_present:
        raise AlfError(
            f"`_frame_convention: {FRAME_WORLD_ALIGNED!r}` declares that no rotation was "
            f"applied, but `_frame_rotation` is present. A reader applies neither, so the two "
            f"statements cannot both be what the producer meant — and the reading that drops "
            f"the rotation is the one that returns plausible, wrongly-oriented geometry."
        )


def _reject_nonfinite_metadata_numbers(metadata: dict[str, Any], extension: str, version: int) -> None:
    """Keep arbitrary metadata inside the shared finite-binary64 JSON domain.

    CPython retains arbitrary-precision JSON integers while Dart and JavaScript
    parse JSON numbers into binary64. A custom ``10**400`` value would
    therefore survive only in Python metadata and overflow in the other two
    implementations. Header floats also arrive here now that parsing preserves
    their token long enough to give reserved counts their specific refusal.
    Reserved numeric fields are skipped so channel-specific validators can emit
    the more precise portable refusal. Walk iteratively because the header-depth
    guard is the only recursion we should spend on attacker-controlled metadata.
    """

    reserved = {"_total_points"}
    if extension == "all":
        reserved.add("_format_version")
    elif extension == "alx":
        reserved.add("_source_total_points")
    elif extension == "alp":
        # Geometry anchors are validated from their raw tokens on every .alp
        # version. Codec parameters are special only for v3: in a v1 header
        # they are stale metadata and still belong to the generic finite-
        # binary64 JSON domain. Exempting them unconditionally let a 400-digit
        # `_geom_scale` survive on an empty v1 frame while Dart and TypeScript
        # correctly refused the same bytes.
        reserved.update(
            {
                "device_position",
                "world_centroid_epsg3857",
            }
        )
        if version == ALF_VERSION_V3:
            reserved.update({"_geom_scale", "_geom_offset"})

    _reject_nonfinite_number_tree(value for key, value in metadata.items() if key not in reserved)


def _validate_portable_writer_metadata(metadata: dict[str, Any], path: str | Path) -> None:
    """Keep writer metadata inside the string-keyed, finite-binary64 JSON domain."""
    pending: list[Any] = [metadata]
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            for key in value:
                if not isinstance(key, str):
                    raise ValueError(
                        f"{path}: metadata object key {key!r} is not a string; "
                        "JSON key coercion can create duplicate members"
                    )
            pending.extend(value.values())
            continue
        if isinstance(value, (bool, np.bool_)):
            continue
        if isinstance(value, (int, float, np.integer, np.floating)):
            try:
                decoded = float(value)
            except OverflowError as exc:
                raise ValueError(
                    f"{path}: metadata contains a non-finite number or integer outside finite binary64"
                ) from exc
            if not math.isfinite(decoded):
                raise ValueError(f"{path}: metadata contains a non-finite number or integer outside finite binary64")
        elif isinstance(value, (list, tuple)):
            pending.extend(value)


def _reject_inexact_int(value: Any, key: str) -> None:
    """Refuse a header integer whose value changes when converted to float64.

    Fourth home for this rule, after geometry, intensity and the anchors. The
    codec parameters DEFINE the coordinate system, so a `_geom_offset` of
    2**53 + 1 that converts to 2**53 decodes every point of the frame onto a
    different grid than the header describes — and nothing downstream can tell,
    because the file is self-consistent about the wrong number.
    """
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        integer = int(value)
        try:
            decoded = float(integer)
        except OverflowError as exc:
            raise AlfError(f"`{key}` holds an integer too large for float64: {integer}") from exc
        if not math.isfinite(decoded) or int(decoded) != integer:
            raise AlfError(
                f"`{key}` holds integer {integer}, which float64 cannot represent exactly — "
                f"decoding would place this frame on a different grid than its header states"
            )


def _checked_geom_offset(metadata: dict[str, Any]) -> np.ndarray:
    """The v3 fixed-point origin, validated once for every path that needs it.

    Both `.xyz` and `max_geom_error_m` depend on it: the first to reconstruct
    coordinates, the second to promise an accuracy bound for those coordinates.
    Checking only PRESENCE in the preflight let `"bad"`, `[1, 2]` and
    `[true, 0, 0]` report a confident `scale/2` for a frame the decode then
    refuses outright — the gate passing on geometry that cannot be produced.
    """
    if "_geom_offset" not in metadata:
        raise AlfError(
            "int16 .alp is missing `_geom_offset`; its coordinates are centred on that midpoint, "
            "so decoding without it silently shifts the whole frame"
        )
    raw = metadata["_geom_offset"]
    if _contains_bool(raw):
        raise AlfError(f"`_geom_offset` has boolean component(s): {raw!r}")
    if isinstance(raw, (list, tuple)) and not all(_is_json_number(v) for v in raw):
        # Same rule as the scale and the anchors: a JSON string coerces through
        # float() and would then control every decoded coordinate.
        raise AlfError(f"`_geom_offset` has non-numeric component(s): {raw!r}")
    for component in raw if isinstance(raw, (list, tuple)) else [raw]:
        _reject_lossy_token(component, "_geom_offset")
        _reject_inexact_int(component, "_geom_offset")
    try:
        offset = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AlfError(f"`_geom_offset` has a non-numeric component: {raw!r}") from exc
    if offset.shape != (3,):
        raise AlfError(
            f"`_geom_offset` must have exactly 3 components, got shape {offset.shape}. A scalar "
            f"or 1-element offset broadcasts across all axes and yields a uniformly shifted "
            f"cloud rather than an error."
        )
    if not np.isfinite(offset).all():
        raise AlfError(f"`_geom_offset` contains non-finite values: {offset.tolist()}")
    return offset


def _checked_geom_scale(metadata: dict[str, Any]) -> float:
    """The v3 fixed-point step, validated once for every path that reads it.

    Zero collapses every point onto `offset`; a negative scale mirrors the cloud
    through it; a missing key is a KeyError escaping as something other than
    `AlfError`. All three decode or report without complaint into values that
    look like data.
    """
    if "_geom_scale" not in metadata:
        raise AlfError("int16 .alp is missing `_geom_scale`, which its coordinates are meaningless without")
    raw_scale = metadata["_geom_scale"]
    if not isinstance(raw_scale, (int, float)) or isinstance(raw_scale, bool):
        # A JSON string like "0.1" coerces cleanly through float(), letting a
        # Python-only conversion define the file's semantics — while the counts
        # and `_format_version` reject strings outright. Booleans fall in here
        # too: `float(True)` is 1.0, decoding every stored int16 as one metre.
        raise AlfError(f"`_geom_scale` must be a JSON number, got {raw_scale!r}")
    # The FLOAT spelling too. The int branch below and `_geom_offset` both call
    # this; the float branch did not, so `9007199254740993.0` was accepted as
    # `...992.0` and every stored component decoded against a scale the file
    # does not declare.
    _reject_lossy_token(raw_scale, "_geom_scale")
    if isinstance(raw_scale, int):
        # An int is already exact; `float(10**400)` would raise OverflowError
        # here and escape the quarantine contract before the positive/finite
        # check could turn it into AlfError. Same reason the counts avoid float().
        #
        # "Already exact" only holds below 2**53 — past it the conversion picks
        # a NEIGHBOURING step, so every stored component decodes against a scale
        # the header does not state.
        _reject_inexact_int(raw_scale, "_geom_scale")
        if raw_scale <= 0:
            raise AlfError(f"`_geom_scale` must be a positive finite number, got {raw_scale!r}")
        try:
            return float(raw_scale)
        except OverflowError as exc:
            raise AlfError(f"`_geom_scale` is too large to represent: {raw_scale!r}") from exc
    try:
        scale = float(raw_scale)
    except (TypeError, ValueError) as exc:
        raise AlfError(f"`_geom_scale` is not a number: {metadata['_geom_scale']!r}") from exc
    if not np.isfinite(scale) or scale <= 0.0:
        raise AlfError(f"`_geom_scale` must be a positive finite number, got {scale!r}")
    return scale


def _finite_max_abs(values: np.ndarray) -> float | None:
    """Largest |component| among the FINITE entries, or None if there are none.

    Non-finite components are VALID `.alp` content (an invalid lidar return), so
    every magnitude question in this module has to be asked of the real
    coordinates only. A plain `np.max(np.abs(...))` returns `nan` the moment one
    invalid return is present, and `nan` compares False against every threshold
    — so the fp16 budget guard silently stopped guarding, and the reported error
    bound became `nan`, on exactly the files this format is mostly used for.
    """
    if values.size == 0:
        return None
    # Two REDUCTIONS, no temporaries. `np.abs(values)` allocated a full copy and
    # `magnitudes[np.isfinite(magnitudes)]` allocated a second — ~13 MB per
    # 262k-point block, which pushed the writer past the memory bound its own
    # test pins once the int16 path started calling this. `nanmin`/`nanmax`
    # answer the same question in place, and the largest magnitude is at one end
    # or the other.
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        # An all-invalid block is a legitimate frame (every beam missed), not a
        # problem to warn about; it simply has no magnitude.
        warnings.simplefilter("ignore", RuntimeWarning)
        low = np.nanmin(values)
        high = np.nanmax(values)
    if np.isfinite(low) and np.isfinite(high):
        return float(max(abs(float(low)), abs(float(high))))
    if np.isnan(low) or np.isnan(high):
        # Every entry was NaN: no magnitude exists.
        return None
    # An INFINITY reached the reduction. `nanmin`/`nanmax` skip NaN but not Inf,
    # so the fast path cannot answer this one and falls back to filtering. Rare
    # enough to be worth the copy — a writer never emits Inf, it arrives from a
    # hand-built or corrupted frame — and getting it wrong silently understates
    # the error bound, which is the number consumers trust.
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return float(np.abs(finite).max())


def _ulp_below(values: np.ndarray) -> np.ndarray:
    """Spacing to the PREVIOUS representable float, elementwise.

    `np.spacing` steps toward +inf, so at float64 max it returns Inf — for a
    value that is a perfectly ordinary finite number. Using it to bound a
    rounding error therefore made `max_geom_error_m` refuse geometry that
    decodes fine (scale = float64 max, a stored component of 1: the product is
    exact and `.xyz` succeeds). Stepping DOWN is finite for every finite input
    and is the same magnitude everywhere the two differ by less than a factor
    of two, so nothing else changes.

    This is the second place that needed it — the reach term above was fixed
    first — which is why it lives here instead of being written out twice.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        # The LARGER of the two adjacent spacings, taken from the MAGNITUDE.
        #
        # Two corrections live here. Stepping toward -inf is finite at +max but
        # overflows at -max, so working from the magnitude covers both halves of
        # the number line. And at a power of two the spacing ABOVE is twice the
        # spacing below, so charging the lower one is anti-conservative for a
        # value that rounded up onto the boundary: `3 * 0.33333333333333337` is
        # exactly 1.0 in float64 while the true product is 2**-53 above it, and
        # the lower spacing charges only 2**-54. A bound that understates is the
        # one failure mode this function exists to prevent.
        magnitude = np.abs(values)
        below = np.abs(magnitude - np.nextafter(magnitude, -np.inf))
        above = np.abs(np.nextafter(magnitude, np.inf) - magnitude)
        # `above` is inf at float64 max, where only `below` is meaningful.
        return np.where(np.isfinite(above), np.maximum(below, above), below)


def _frozen(*arrays: np.ndarray) -> tuple:
    """Mark decoded arrays read-only before they are cached.

    The payload-backed accessors became read-only when `_payload` did, but the
    DECODED copies did not — so `categories` raised on a v1 `.all` and silently
    mutated the cache on a v2 one, and `xyz` was writable for both codecs.
    Mutating a cached array changes every subsequent read from that `AlfFile`,
    and having it depend on the payload layout is the same "storage form
    decides semantics" bug the read-only payload fixed one layer down.
    """
    for array in arrays:
        array.flags.writeable = False
    return arrays


def _reconstruction_rounding(payload: bytes, scale: float, offset: Sequence[float]) -> float:
    """Largest float64 rounding error in `quantized * scale + offset`.

    The int16 decode is two floating-point operations per component, and at a
    large offset their rounding dominates the quantization step this codec
    advertises. Bounding it by the ULP at the frame's magnitude is correct but
    wildly pessimistic for the common case where the arithmetic happens to be
    exact, so the add is measured with a TwoSum rather than estimated.
    """
    if not payload or len(payload) % 6:
        # No payload (or one the decode will reject anyway) — nothing to
        # reconstruct, so the codec's own step is the whole story.
        return 0.0
    quantized = np.frombuffer(payload, dtype="<i2")
    offset_vec = np.asarray(offset, dtype=np.float64)
    worst = 0.0
    # CHUNKED. The naive form allocates ~7 full-size float64 arrays (product,
    # sum, three TwoSum intermediates, spacing, total), which measured ~195
    # bytes per point — multiple GB for a 10-million-point frame, turning the
    # documented accuracy preflight into an OOM on large geometry.
    # The result is a maximum, so it composes over blocks exactly.
    # `len(payload) % 6 == 0` above, so the component count is a multiple of 3
    # and every block starts on an x. That lets each block reshape to (-1, 3)
    # and BROADCAST the 3-element offset — no tiled copy of it per block, which
    # is two fewer full-size allocations than resizing it would cost.
    step = 3 * 65_536  # ~0.2M components: a couple of MB of scratch per block
    for begin in range(0, quantized.size, step):
        raw_block = quantized[begin : begin + step].reshape(-1, 3)
        # SENTINEL ROWS CARRY NO ERROR. `-32768` is a marker, not a coordinate,
        # so reconstructing it states the rounding error of a position that was
        # never measured. With a large `_geom_scale` the product overflows and
        # this function then declared the geometry unbounded — for a file whose
        # `.xyz` decodes perfectly to NaN, which is a self-contradiction: the
        # reader read it, and the bound said it was unreadable.
        real_rows = ~np.all(raw_block == INT16_NO_RETURN, axis=1)
        if not real_rows.any():
            continue
        block = raw_block[real_rows].astype(np.float64) if not real_rows.all() else raw_block.astype(np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            product = block * scale
            total = product + offset_vec
            # TwoSum: `total` is the rounded sum, `err_add` its EXACT error.
            near = total - product
            err_add = (product - (total - near)) + (offset_vec - near)
            # The multiply's error needs a wider type to recover, but half an
            # ULP of the product bounds it and is negligible beside `err_add`.
            block_worst = np.abs(err_add) + _ulp_below(product) / 2.0
            if not np.isfinite(block_worst).all():
                # A frame no float64 can hold. `.xyz` refuses it too, so
                # returning a number would state a bound for unreadable geometry.
                raise AlfError(
                    "this file's `_geom_scale`/`_geom_offset` describe coordinates outside "
                    "float64, so no error bound can be stated for it"
                )
            worst = max(worst, float(block_worst.max()))
    return worst


def _int_narrows_lossily(value: Any) -> bool:
    """Whether this integer comes back different after a trip through float64.

    Compare the VALUES, not the magnitude. `2**53` itself IS exactly
    representable, and so are all its multiples up to the next binade — a
    magnitude test refuses an int64 frame at exactly 2**53 that round-trips
    perfectly, while the identical float64 input is accepted. The native-dtype
    branch was corrected for this; the object branch and every later caller kept
    re-deriving the cruder test, so the comparison lives in one place now.

    A value too large to become a float at all raises instead of comparing, and
    that is the most lossy case there is.
    """
    try:
        return int(value) != float(value)
    except OverflowError:
        return True


def _exceeds_float64_exact_integers(arr: np.ndarray) -> bool:
    """Whether integer data in `arr` reaches 2**53, where float64 stops being exact.

    Object arrays are checked ELEMENTWISE. `dtype.kind` is `O` for a pandas
    nullable column or any ragged-ish input, so a kind-based test declared them
    non-integral and waved them past the guard — the same 2**53 collapse, one
    container across.
    """
    if arr.dtype.kind in "iu":
        if not arr.size:
            return False
        return any(
            _int_narrows_lossily(v)
            for v in arr.ravel().tolist()
            if isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_))
        )
    if arr.dtype == object:
        return any(
            isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_)) and _int_narrows_lossily(v)
            for v in arr.ravel().tolist()
        )
    return False


def _is_numeric_object_array(arr: np.ndarray) -> bool:
    """True when every element of an object array is a real JSON number.

    Object arrays are how ragged or mixed input arrives, so they are neither
    trustworthy enough to cast nor uniformly wrong enough to refuse outright.
    """
    return all(_is_json_number(v) for v in arr.ravel().tolist())


def _narrows_lossily_to_float64(arr: np.ndarray) -> str:
    """Why `arr` cannot be narrowed to float64 without losing distinctions, or "".

    Two ways to lose them, and every coercion in this module needs both. Wider
    floats (`np.longdouble` is 80-bit on x86) and integers at or beyond 2**53
    both collapse values that differ on input — coincident points, a merged
    intensity range, a shifted anchor. Geometry grew each check first and the
    other paths inherited neither, twice; one predicate now, called from all
    three coercions.
    """
    if arr.dtype.kind == "f" and arr.dtype.itemsize > 8:
        # Compare the VALUES, not just the dtype. Refusing every longdouble
        # array blocked `(1, 2, 3)` and `(0, 1)`, which round-trip exactly —
        # a converter or experiment carrying extended precision it never
        # actually used could not write at all. Rejecting valid input is the
        # same class of mistake as the all-False mask guard, one round later.
        if arr.size and not np.array_equal(arr, arr.astype(np.float64).astype(arr.dtype)):
            return f"dtype {arr.dtype!r} loses information narrowing to float64"
        return ""
    if arr.dtype == object and any(
        isinstance(v, np.floating) and np.dtype(type(v)).itemsize > 8 and float(v) != v for v in arr.ravel().tolist()
    ):
        # The CONTAINER is object, so the dtype test above sees nothing — but a
        # `np.longdouble` scalar inside it narrows exactly as lossily as a
        # longdouble array does.
        #
        # And the VALUE decides, not the type. The native branch gained that
        # comparison last round and this one did not, so an object array of
        # exactly-representable longdoubles was refused while the equivalent
        # native array was accepted — one rule disagreeing with itself
        # depending on the container it arrived in.
        return "object array holds floats that lose information narrowing to float64"
    if _exceeds_float64_exact_integers(arr):
        return "integer values reach 2**53, where float64 stops being exact"
    return ""


def _lossy_before_promotion(value: Any) -> str:
    """Why `np.asarray(value)` would ITSELF lose a distinction, or "".

    `_narrows_lossily_to_float64` inspects an array, so it only ever sees what
    survived the coercion that produced it. A raw Python sequence mixing an
    integer with floats promotes to float64 FIRST: `[[2**53 + 1, 0.5, 0.0]]`
    arrives with x already rounded to `2**53`, the predicate finds a clean float
    array, and the point publishes a metre out. The equivalent
    `np.array(..., dtype=np.int64)` is correctly refused — so the guard
    disagreed with itself depending on whether the caller had already built an
    array, which is not something a caller should have to know.

    `_as_anchor` closed this for the anchors by rebuilding an OBJECT array from
    the original components. The payload channels need the same rule, so it is
    written once here instead of per channel — and as a short-circuiting walk
    rather than a materialized object array, because geometry arrives with a row
    per point and rebuilding it would cost more than the write.

    The two conditions and their wording are `_narrows_lossily_to_float64`'s;
    keep them in step if that predicate grows a third.
    """
    if isinstance(value, (list, tuple)):
        for item in value:
            reason = _lossy_before_promotion(item)
            if reason:
                return reason
        return ""
    if isinstance(value, (bool, np.bool_)):
        # Booleans are somebody else's error to report: `_contains_bool` runs
        # first at every call site and says something far more useful about a
        # mask than "narrows lossily" would.
        return ""
    if isinstance(value, (int, np.integer)) and _int_narrows_lossily(value):
        return "integer values reach 2**53, where float64 stops being exact"
    if isinstance(value, np.floating) and np.dtype(type(value)).itemsize > 8 and float(value) != value:
        # A `np.longdouble` scalar inside a Python list narrows exactly as
        # lossily as a longdouble array does, and promotion hides it the same
        # way it hides the integer.
        return "object array holds floats that lose information narrowing to float64"
    return ""


def _is_stringly_typed(arr: np.ndarray) -> bool:
    """Whether `arr` carries strings where numbers are required.

    Covers the native string dtypes AND object arrays holding strings, which is
    how a pandas column of unparsed values actually arrives. Geometry and the
    per-point channels each grew their own version of this test and the
    per-point one kept only the first half, so it lives in one place now.
    """
    return arr.dtype.kind in "SU" or (arr.dtype == object and not _is_numeric_object_array(arr))


def _holds_actual_strings(arr: np.ndarray) -> bool:
    """Whether `arr`'s non-numeric content is literally strings.

    `_is_stringly_typed` is deliberately broader — it is true for ANY object
    array that is not all JSON numbers — but its callers' message is not. A
    `Decimal`, a `Fraction`, and the 0-d object array `np.asarray` builds from a
    generator all landed on "holds strings, not numbers", which sends the caller
    hunting for an unparsed CSV column they do not have. The refusal was right
    in every case; only the diagnosis was wrong, so this splits the wording
    without touching which inputs are accepted.
    """
    if arr.dtype.kind in "SU":
        return True
    if arr.dtype != object:
        return False
    return any(isinstance(v, (str, bytes)) for v in arr.ravel().tolist())


def _reject_non_real(arr: np.ndarray, what: str, path: str | Path) -> None:
    """Require a REAL NUMERIC dtype before the cast to float64.

    `astype(np.float64)` drops every imaginary component and emits only a
    `ComplexWarning`, which callers need not treat as fatal — so
    `[[1+9j, 2, 3]]` publishes a plausible point at (1, 2, 3). Same silent-
    coercion family as the boolean shapes: the data looks right and nothing
    reports the loss.

    Stated as an ALLOW-list, because refusing only complex let every other
    non-numeric dtype through: a `datetime64[D]` column casts to its epoch day
    count, so three dates published as the coordinates 19723, 19724, 19725 —
    entirely plausible, entirely wrong. `timedelta64` does the same with unit
    counts. Object arrays are inspected elementwise elsewhere and pass here.
    """
    if arr.dtype.kind == "c":
        raise ValueError(
            f"{path}: {what} must be real numbers, not complex — casting would drop every "
            f"imaginary component and publish plausible-looking values"
        )
    if arr.dtype.kind not in "iufO":
        raise ValueError(
            f"{path}: {what} has dtype {arr.dtype!r}, which is not real numeric data. Casting "
            f"it to float64 yields a plausible number that means something else entirely — a "
            f"datetime becomes its epoch count, a timedelta its unit count."
        )


def _is_json_number(v: Any) -> bool:
    """A real JSON number — not a bool, not a numeric STRING.

    `float("10")` coerces cleanly, which would let Python's parsing define the
    wire format. The counts and `_geom_scale` already refuse strings, so the
    anchors and the fixed-point offset must too, or a `{"x": "10"}` header
    places a frame from a coercion nothing in the spec sanctions.
    """
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_))


def _contains_bool(value: Any) -> bool:
    """Whether a (possibly nested) sequence carries a Python bool anywhere.

    `np.asarray([True, 2])` promotes to int64 and `np.asarray([[True, 0, 0]])`
    likewise, so a dtype check only ever catches a UNIFORMLY boolean input — a
    mixed one slips through as 1 and becomes a real coordinate, category or
    intensity byte. Every coercion site in this module needs the same test, so
    it lives in one place rather than being re-derived per accessor.
    """
    if isinstance(value, (bool, np.bool_)):
        # `isinstance(np.bool_(True), bool)` is FALSE, so a NumPy scalar — the
        # shape a pandas/NumPy mask actually arrives in — slips a plain
        # `isinstance(..., bool)` check entirely.
        return True
    if isinstance(value, np.ndarray):
        # An object-dtype array holds real Python/NumPy bools that survive to
        # `astype(float64)` as 1.0; a bool-dtype array is the uniform case.
        if value.dtype == np.bool_:
            return True
        if value.dtype == object:
            return any(_contains_bool(v) for v in value.ravel().tolist())
        return False
    if isinstance(value, (list, tuple)):
        return any(_contains_bool(v) for v in value)
    if hasattr(value, "__array__"):
        # A pandas Series (or any array-like that is not a list/tuple/ndarray)
        # never reached the branches above, so `pd.Series([True, 2])` coerced to
        # an object array and published the mask as category 1 / intensity 127 —
        # past the very invariant these writers advertise. Coerce once and
        # re-test rather than enumerating third-party container types.
        return _contains_bool(np.asarray(value))
    return False


def _as_anchor(value: Any, label: str) -> np.ndarray:
    """Coerce an origin/centroid to a finite (3,) float64 vector.

    Accepts the two shapes these anchors actually arrive in — the `{x, y, z}`
    dict ALF metadata uses, and a 3-sequence — because a caller reading the
    aerial origin out of a frame JSON has the dict and a caller writing one by
    hand has the tuple. Everything else raises, since a scalar would broadcast
    across all three axes and shift the cloud diagonally instead of failing.
    """
    if isinstance(value, dict):
        # All three axes required, no defaulting. `{"x": 1}` — a typo, or a
        # partially-populated sidecar — would otherwise be accepted as
        # (1, 0, 0) and return world coordinates shifted by however much y and
        # z really were. A missing anchor axis is exactly the silent misframing
        # `world_xyz` exists to prevent, so it cannot be defaulted away here.
        missing = [k for k in ("x", "y", "z") if k not in value]
        if missing:
            raise AlfError(f"`{label}` is missing required axis/axes {missing}: {value!r}")
        # `float(True)` is 1.0, so a boolean component would place the frame one
        # metre off an axis and return plausible coordinates -- the same reason
        # booleans are refused for the counts and the fixed-point scale.
        bad = [k for k in ("x", "y", "z") if not _is_json_number(value[k])]
        if bad:
            raise AlfError(
                f"`{label}` component(s) {bad} are not JSON numbers: {value!r}. A numeric string "
                f"coerces through float(), which would place the frame from Python's parsing "
                f"rather than from the file."
            )
        try:
            vec = np.array(
                [float(value["x"]), float(value["y"]), float(value["z"])],
                dtype=np.float64,
            )
        except (TypeError, ValueError, OverflowError) as exc:
            # A non-numeric component is a malformed FILE, not a programming
            # error, so it has to reach callers as AlfError like every other
            # corrupt-header case — they quarantine on that and keep going.
            raise AlfError(f"`{label}` has a non-numeric component: {value!r}") from exc
    else:
        # Inspect the ORIGINAL elements: `np.asarray([True, 0, 0])` promotes to
        # int64, so a dtype check only catches an ALL-boolean sequence and a
        # mixed one slips through as 1 — shifting the frame a metre off that
        # axis. Same reason the dict path checks each component.
        if isinstance(value, np.ma.MaskedArray) and np.ma.getmaskarray(value).any():
            # `np.asarray` drops the mask and hands back the BACKING value, so
            # a masked x of 100 placed the whole frame 100 m away — a component
            # the caller explicitly marked invalid deciding where the cloud
            # goes. Geometry and the per-point channels already ask this; the
            # anchors are the third home for it.
            raise AlfError(
                f"`{label}` is a masked array with masked component(s). `np.asarray` drops the "
                f"mask and would place the frame from the hidden backing value; fill or select "
                f"the anchor explicitly."
            )
        if _contains_bool(value):
            raise AlfError(f"`{label}` has boolean component(s): {value!r}")
        if isinstance(value, (list, tuple)) and not all(_is_json_number(v) for v in value):
            raise AlfError(f"`{label}` has non-numeric component(s): {value!r}")
        try:
            probe = np.asarray(value)
        except (TypeError, ValueError) as exc:
            raise AlfError(f"`{label}` is not a finite numeric 3-sequence: {value!r}") from exc
        if probe.dtype.kind not in "iuf":
            # An ndarray never reached the list/tuple check above, so a COMPLEX
            # array cast straight to float64: NumPy drops every imaginary part
            # behind a ComplexWarning and `[1+9j, 2, 3]` becomes a confident
            # (1, 2, 3). An anchor is what places the whole cloud, so an
            # upstream complex-coordinate bug would misframe the frame silently.
            # Every PAYLOAD path already refuses complex via `_reject_non_real`;
            # the anchors were the one door left open.
            # Coerced FIRST, then checked. Testing `isinstance(value, np.ndarray)`
            # covered NumPy only, so a pandas Series — or any other array-like —
            # walked straight into the float64 cast, dropping an imaginary part
            # or parsing a numeric string. One coercion, one dtype rule, every
            # container.
            raise AlfError(
                f"`{label}` has dtype {probe.dtype!r}, which is not real numeric data. "
                f"Casting it to float64 would discard information and place the frame from "
                f"whatever survived the cast."
            )
        try:
            # `np.asarray` itself raises on a RAGGED sequence such as
            # [[1], [2, 3], [4]], so it has to sit inside the guard — outside it,
            # a corrupt anchor shape aborts the pipeline with a bare ValueError
            # instead of the AlfError callers quarantine on.
            vec = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise AlfError(f"`{label}` is not a finite numeric 3-sequence: {value!r}") from exc
    # Both branches, one check. Putting this in the sequence branch alone left
    # the `{x, y, z}` dict path open — the same one-of-two miss this rule keeps
    # being written to fix. Applied to the original components, since `vec` has
    # already lost the precision by the time it exists.
    # The ORIGINAL elements, not `np.ravel(value)`. Ravelling a mixed sequence
    # promotes it to float64 first, so `(2**53 + 1, 0.5, 0)` arrives here with x
    # already rounded and the checks below see nothing wrong — the dict path was
    # covered and the sequence path was not.
    if isinstance(value, dict):
        components = [value[k] for k in ("x", "y", "z")]
    elif isinstance(value, (list, tuple)):
        components = list(value)
    else:
        components = list(np.ravel(value))
    for component in components:
        _reject_lossy_token(component, label)
    # An OBJECT array, so each component keeps its own type. `np.asarray` on a
    # mixed sequence promotes first — `{"x": 2**53 + 1, "y": 0.5, "z": 0}`
    # becomes float64 and x is already `2**53` by the time this predicate looks,
    # so the frame ships a metre out. The elementwise paths in
    # `_narrows_lossily_to_float64` exist precisely for object arrays.
    lossy = _narrows_lossily_to_float64(np.array(components, dtype=object))
    if lossy:
        # BOTH narrowing rules, at the one point both branches reach. The
        # wide-float half was first written into the sequence branch alone,
        # leaving the mapping path open — the identical one-of-two miss the
        # integer half had made one round earlier, in this same function. An
        # anchor PLACES the cloud, so an x of 2**63 + 1 becoming 2**63 shifts
        # every point in the frame by a metre.
        raise AlfError(
            f"`{label}` cannot narrow to float64 without shifting the frame's origin "
            f"({lossy}). Re-base the anchor before passing it."
        )
    if vec.shape != (3,):
        raise AlfError(f"`{label}` must be an (x, y, z) mapping or a 3-sequence, got shape {vec.shape}")
    if not np.isfinite(vec).all():
        raise AlfError(f"`{label}` contains non-finite values: {vec.tolist()}")
    return vec


def _normalized_anchor(value: Any, label: str) -> dict[str, Any]:
    """Normalize anchor axes without discarding application-owned members."""
    vec = _as_anchor(value, label)
    normalized = (
        {key: member for key, member in value.items() if key not in {"x", "y", "z"}} if isinstance(value, dict) else {}
    )
    normalized.update(
        {
            "x": float(vec[0]),
            "y": float(vec[1]),
            "z": float(vec[2]),
        }
    )
    return normalized


class _WithheldPayload:
    """Stands in for the payload of a header-only read, and refuses every use.

    `read_alf_header` deliberately never decompresses the payload, so there is
    nothing to hand an accessor. The tempting cheap substitute — `b""` — is the
    exact failure this module exists to prevent: `.intensity` would return an
    empty array, `format_version`'s stride disambiguation would "measure" a
    zero-byte payload and pick a layout, and `format_version_verified` would
    report a confident answer about bytes nobody read. All three are wrong
    ANSWERS rather than errors.

    So the substitute raises instead, and it raises from `__len__`, `__bool__`,
    `__getitem__` and `__iter__` rather than from a hand-written guard at each
    accessor. That is what makes it fail-CLOSED: an accessor added later that
    forgets `_require_payload` still cannot get a number out of this object. The
    explicit guards exist on top of it only because they word the failure better
    and because `np.frombuffer` reaches the C buffer protocol, which a Python
    object cannot intercept before 3.12, and this package supports 3.11.
    """

    __slots__ = ()

    _MESSAGE = (
        "this AlfFile came from `read_alf_header`, which reads only the envelope and never "
        "decompresses the payload — so no payload accessor can answer. Re-open the source with "
        "`read_alf` if the points, labels, intensities or indexes are actually needed."
    )

    def _refuse(self, *_args: Any) -> Any:
        raise AlfError(self._MESSAGE)

    __len__ = _refuse
    __bool__ = _refuse
    __iter__ = _refuse
    __getitem__ = _refuse
    __mod__ = _refuse


@dataclass
class AlfFile:
    """A decoded ALF file. Arrays are decoded lazily-ish: parsed once, then held."""

    version: int
    metadata: dict[str, Any]
    _payload: bytes | memoryview = field(repr=False)
    extension: str = ""
    _labels_v2_cache: tuple | None = field(default=None, repr=False, compare=False)
    _xyz_cache: np.ndarray | None = field(default=None, repr=False, compare=False)

    # -- header-derived -----------------------------------------------------

    @property
    def format_version(self) -> int:
        """`.all` payload layout. 1 = categories only, 2 = category + instance.

        Falls back to the ENVELOPE version byte when the header key is absent.
        Both signals exist in the wild and neither is universal: the Dart writer
        sets the version byte AND `_format_version`, while the legacy export
        parsers identify v2 from the version byte alone and synthesize the
        metadata key afterwards. Keying only on metadata would read those files
        as v1 — returning the interleaved `[cat, lo, hi, ...]` stream as
        categories and `None` for instances, i.e. silently corrupting labels
        that are correct on disk.

        `.all` only. This is a LABEL-layout question, and every other payload
        accessor already refuses a wrong-channel read for the same reason: a
        mixed-channel scan touching `.format_version` on a v3 int16 `.alp` got
        a confident "label layout 1" instead of the misuse being surfaced.
        """
        self._require("all")
        declared = _checked_label_layout(self.metadata)
        if declared is not None:
            byte_says = LABELS_V2 if self.version == LABELS_V2 else LABELS_V1
            if declared == byte_says:
                return declared
            if self.total_points == 0 and not self._payload:
                raise AlfError(
                    f"`.all` wire byte {self.version} and `_format_version` {declared} disagree, "
                    "and an empty counted payload fits BOTH layouts. The label layout cannot be "
                    "determined from this file."
                )
            # The two fields disagree. The PAYLOAD settles it when it can: byte 1
            # with `_format_version: 2` is a possible legacy envelope, and the stride is
            # objective evidence where the two headers are just assertions.
            resolved = self._layout_from_stride()
            if resolved is not None:
                return resolved
            if self.total_points is not None:
                if self.total_points == 0 and not self._payload:
                    raise AlfError(
                        f"`.all` wire byte {self.version} and `_format_version` {declared} "
                        "disagree, and an empty payload fits BOTH layouts. The label layout "
                        "cannot be determined from this file."
                    )
                # `_total_points` IS present and the payload matches neither N
                # nor 3N, so the file contradicts both possible layouts — it is
                # truncated or overlong, not merely ambiguous. Uncertainty is
                # the right answer only when there is nothing to check against.
                raise AlfError(
                    f"`.all` wire byte {self.version} and `_format_version` {declared} disagree, "
                    f"and the payload length matches neither the 1-byte nor the 3-byte stride for "
                    f"{self.metadata['_total_points']} points: the file is truncated or overlong."
                )
            # No `_total_points`, but the payload can still rule a layout OUT.
            # v2 is 3 bytes per point, so a length not divisible by 3 makes it
            # impossible — and v1 accepts any length, so exactly one reading
            # survives. Returning the declared v2 "unverified" there would hand
            # back a layout whose own stride rule the file already violates, and
            # leave three implementations free to reject, decode as provisional
            # v2 with no usable instance_ids, or quietly read v1.
            feasible = [v for v in (LABELS_V1, LABELS_V2) if len(self._payload) % (3 if v == LABELS_V2 else 1) == 0]
            if len(feasible) == 1:
                return feasible[0]
            # Both remain possible: same bytes, two readings, nothing to
            # separate them. FAIL CLOSED, as this module's README promises and
            # as every other ambiguous case here does — returning the declared
            # value would let `categories` and `instance_ids` silently
            # reinterpret one payload and put the burden on every consumer to
            # remember `format_version_verified` first. A counted payload can
            # resolve by stride above; an uncounted one needs unique evidence.
            raise AlfError(
                f"`.all` wire byte {self.version} and `_format_version` {declared} disagree, "
                f"there is no `_total_points` to check against, and the payload length "
                f"({len(self._payload)}) fits BOTH layouts. The label layout cannot be "
                f"determined from this file."
            )
        return LABELS_V2 if self.version == LABELS_V2 else LABELS_V1

    @property
    def format_version_verified(self) -> bool:
        """Whether `format_version` is corroborated by the PAYLOAD, not just declared.

        `False` means the two version fields disagreed and the payload could not
        settle it, so the answer came from the header alone and a caller needing
        certainty must not treat it as authoritative. `True` for a
        self-consistent file, and for a mismatch the stride resolved.
        """
        # Same channel rule as `format_version`, which this delegates to. A
        # mixed-channel scan using this documented preflight got a confident
        # True — "layout verified" — for an `.alp`/`.ali`/`.alc`/`.alx` that has
        # no label layout at all, bypassing the guard on the property below.
        self._require("all")
        if _present_or_absent(self.metadata, "_format_version") is None:
            # A version-BYTE-only v2 file has nothing corroborating it either:
            # if the payload is not a whole number of 3-byte rows the layout is
            # already contradicted, and reporting True would mark a corrupt file
            # authoritative for the caller using this as a preflight.
            if self.version == LABELS_V2:
                return len(self._payload) % 3 == 0
            return True
        # Go through `format_version`, which validates type and domain. Reading
        # the raw key with `int()` here let `true` and `1.9` truncate to a
        # "verified" v1 and a string leak a bare ValueError — and this property
        # is documented as the preflight a caller uses to decide whether the
        # layout is authoritative, so it must not be the one path that trusts a
        # corrupt header.
        raw_declared = self.metadata["_format_version"]
        byte_says = LABELS_V2 if self.version == LABELS_V2 else LABELS_V1
        if (
            not isinstance(raw_declared, bool)
            and isinstance(raw_declared, (int, float))
            and raw_declared in (LABELS_V1, LABELS_V2)
            and raw_declared != byte_says
            and self.total_points == 0
            and not self._payload
        ):
            return False
        declared = self.format_version
        if declared == byte_says:
            # AGREEING on v2 is not corroboration when there is no
            # `_total_points`: v2 rows are 3 bytes, so a 4-byte payload already
            # contradicts the layout both fields assert, and reporting it
            # authoritative misleads the preflight this property exists for.
            if declared == LABELS_V2 and self.total_points is None:
                return len(self._payload) % 3 == 0
            return True
        if not self._payload:
            # An EMPTY payload fits both a 1-byte and a 3-byte stride, so it
            # settles nothing — and this property's documented meaning is
            # exactly "the payload could not settle it". `_layout_from_stride`
            # returns the declared layout here, which reads as corroboration
            # when it is only the disagreement restated.
            return False
        if self._layout_from_stride() is not None:
            return True
        # Stride FEASIBILITY also counts as corroboration: if only one layout can
        # produce this payload length, the answer is not a guess.
        return len(self._payload) % 3 != 0

    def _layout_from_stride(self) -> int | None:
        """Which `.all` layout the PAYLOAD is consistent with, or None if neither.

        At zero points both strides hold (0 == 0 == 3*0), so the payload cannot
        settle a disagreement and this returns ``None``.
        """
        declared_count = self.total_points
        if declared_count is None:
            return None
        n = int(declared_count)
        if n == 0:
            return None
        size = len(self._payload)
        if size == n:
            return LABELS_V1
        if size == 3 * n:
            return LABELS_V2
        return None

    @property
    def geom_codec(self) -> str:
        """Geometry codec for `.alp`. Absent header means the legacy float16."""
        self._require("alp")
        codec = _present_or_absent(self.metadata, "_geom_codec")
        # Absent means the legacy float16; a present null must NOT default to it,
        # or a v3 payload with a blanked codec key decodes as fp16 garbage.
        return GEOM_FP16 if codec is None else str(codec)

    @property
    def total_points(self) -> int | None:
        """Declared point count, or None when the header omits it.

        This is the integrity check every channel-length cross-check rests on,
        so it fails closed. `int()` TRUNCATES: a header declaring 1.9 points
        would become 1, and a one-point payload would then satisfy
        `_check_payload_length` — turning the guard against channel-length
        mismatch into a rubber stamp for a malformed count.
        """
        return _checked_count(self.metadata, "_total_points")

    @property
    def device_position(self) -> np.ndarray:
        dp = _present_or_absent(self.metadata, "device_position")
        if dp is None or dp == {}:
            # Absent is legal and means the origin; malformed is not.
            return np.zeros(3, dtype=np.float64)
        # Through the same validator as every other anchor: a bare string raises
        # AttributeError here, non-numeric components ValueError and oversized
        # integers OverflowError — none of them AlfError, so a caller reading
        # the documented frame anchor to quarantine a bad frame aborts instead.
        return _as_anchor(dp, "device_position")

    @property
    def frame_convention(self) -> str | None:
        """How this `.alp`'s local points relate to world axes, or None.

        `'world-aligned'` — only the origin was subtracted, so the points are on
        world axes. `'device-aligned'` — the points were also rotated into the
        device frame, and `frame_rotation` carries the rotation that undoes it.

        None for versions 1 and 2, where the file does not say and cannot: those
        predate the field, and both conventions were written. `world_xyz` refuses
        to guess for them, which is the only honest answer available.
        """
        self._require("alp")
        declared = self.metadata.get("_frame_convention")
        return declared if isinstance(declared, str) else None

    @property
    def frame_rotation(self) -> dict[str, float] | None:
        """The device rotation as `{'w', 'x', 'y', 'z'}`, or None when absent.

        Present only on a `device-aligned` v3 frame. Applying it to a local point
        and adding the anchor returns the point to the world.
        """
        self._require("alp")
        if "_frame_rotation" not in self.metadata:
            return None
        return _checked_frame_rotation(self.metadata)

    @property
    def max_geom_error_m(self) -> float | None:
        """The frame's worst-case per-component coordinate error, in metres.

        The number a consumer needs to decide whether this file can answer its
        question. Uniform for int16; magnitude-dependent for float16, so there it
        is reported at the frame's largest component.
        """
        if self.extension != "alp":
            return None
        # Both codecs need the bytes: int16 MEASURES the reconstruction rounding
        # over the payload, and fp16's bound is reported at the frame's largest
        # decoded component. Neither is derivable from the header, so this is a
        # payload accessor wearing a header accessor's shape — refuse early
        # rather than let the sentinel surface deep inside the arithmetic.
        self._require_payload()
        if self.geom_codec == GEOM_INT16:
            # Same validation `.xyz` applies, INCLUDING the envelope check. A
            # consumer may query this FIRST to decide whether a frame meets an
            # accuracy threshold, so a v1 file carrying stale `_geom_codec:
            # int16` would otherwise report a millimetre bound for a float16
            # payload — passing the gate on a promise the decode then refuses to
            # honour. Returning -0.5 for a corrupt negative scale, or leaking
            # KeyError, are the same class of answer.
            if self.version != ALF_VERSION_V3:
                raise AlfError(
                    f"`_geom_codec: {GEOM_INT16!r}` on a version-{self.version} .alp, so this "
                    f"file's stated error bound describes an encoding it does not use."
                )
            # Callers use this as the preflight for whether the geometry meets
            # an accuracy threshold, so it must not report a usable bound for a
            # frame `.xyz` will refuse to reconstruct — same validator, not just
            # a presence check.
            _checked_geom_offset(self.metadata)
            if len(self._payload) % 6:
                # Three int16 per point. A truncated payload with otherwise
                # valid codec keys would report a usable bound for geometry
                # `.xyz` then refuses — the third way this preflight could pass
                # what the decode rejects.
                raise AlfError(
                    f"int16 .alp payload is {len(self._payload)} bytes, not a whole number of "
                    f"6-byte XYZ triples; no error bound describes a truncated frame."
                )
            # Then run the decode itself. Enumerating the ways this preflight
            # could pass what `.xyz` rejects has now failed three times (offset,
            # stride, and overflow: a positive finite scale still yields Inf at
            # a stored 32767). The decode is the authority on decodability, so
            # ask it rather than keep re-deriving its rules. Explicit checks stay
            # ABOVE this line only because they word the failure better.
            if self._payload:
                _ = self.xyz
            # Never understate. `scale / 2` is the quantization error alone; the
            # float64 spacing at the frame's own magnitude is added on top by
            # the `quantized * scale + offset` reconstruction, and for a far
            # off-origin frame that term DOMINATES. Take the worse of the
            # writer's measured value and this analytic floor, so neither a
            # legacy file lacking the key nor a stale header can report a bound
            # the geometry does not meet.
            # Validate the codec parameters FIRST, even for an empty frame.
            # The early return added last round skipped them, so a v3 file
            # declaring int16 with a missing or stringly-typed `_geom_scale`
            # reported a confident 0.0 while `.xyz` raised on the same file —
            # this property is a preflight, so certifying an unreadable frame is
            # exactly the failure it exists to prevent.
            scale = _checked_geom_scale(self.metadata)
            offset = _checked_geom_offset(self.metadata)
            if not self._payload:
                # No coordinates, no loss — the same answer the writer records
                # and that fp16 and `geom_error_report` give. Deriving a bound
                # from the grid alone made an empty frame look lossy.
                return 0.0
            # MEASURE the reconstruction rounding; do not charge a worst case.
            #
            # This bound has now been wrong in both directions. Modelling it as
            # `scale/2` alone UNDERSTATED far off-origin frames; then charging
            # one ULP at the decoded magnitude OVERSTATED them, reporting
            # ~2e292 for a constant frame at float64 max that round-trips with
            # exactly zero error, because a zero payload lands precisely on the
            # offset and no rounding occurs at all. Both directions break an
            # accuracy gate — one admits bad geometry, the other rejects good
            # geometry — so this stops estimating and measures the operation.
            #
            # `decoded = fl(fl(q*scale) + offset)`. A TwoSum recovers the add's
            # rounding EXACTLY (Knuth; no assumption about which operand is
            # larger), and the multiply's is bounded by half an ULP of the
            # product. For an all-zero payload both terms are zero, which is why
            # this returns the writer's 5e-05 for the frame above instead of
            # 2e292.
            reach = _reconstruction_rounding(self._payload, scale, offset)
            analytic = scale / 2.0 + reach
            if self._payload and 0.0 < scale < 1e-300:
                # Subnormal scales lose the half-step to rounding in EITHER
                # direction: 2.5e-323 / 2 is 1e-323 where the exact half is 2.5
                # minimum subnormals, so the reported bound sits BELOW the real
                # quantization error. Flooring only the zero case fixed one end
                # of that. `nextafter` upward is the conservative representable
                # value, and at this magnitude it costs a single subnormal.
                analytic = max(analytic, float(np.nextafter(scale / 2.0, np.inf)))
            if self._payload and analytic <= 0.0:
                # A subnormal `_geom_scale` (5e-324) underflows both terms to
                # zero, so a grid that demonstrably loses information reported
                # an EXACT bound and an accuracy gate certified it. Any nonempty
                # fixed-point grid is lossy by construction, so the floor is the
                # smallest positive float rather than zero.
                analytic = 5e-324
            recorded = self.metadata.get("_geom_max_err_m")
            if _is_json_number(recorded):
                try:
                    # A JSON integer has no width limit: `10**400` is a valid
                    # number that `float()` refuses. Raw OverflowError escaped a
                    # caller whose quarantine is `except AlfError: skip_frame()`,
                    # aborting the pipeline on a frame it meant to skip.
                    as_float = float(recorded)
                except OverflowError as exc:
                    raise AlfError(
                        "`_geom_max_err_m` is too large to represent as a float; "
                        "this file's stated error bound is malformed"
                    ) from exc
                if math.isfinite(as_float):
                    # `analytic` is finite by the check above, so this max is
                    # NaN-free — the failure that made a 0.0 bound possible.
                    return max(as_float, analytic)
            return analytic
        xyz = self.xyz
        if xyz.size == 0:
            return 0.0
        # FINITE components only. `np.max` over a frame holding one invalid
        # return is `nan`, and this property is the documented accuracy
        # preflight. A NaN bound defeats ordinary numeric threshold checks.
        largest = _finite_max_abs(xyz)
        if largest is None:
            # Every component is an invalid return. There is no coordinate whose
            # error this could describe, so state the floor rather than invent a
            # magnitude — the same answer the all-zero frame gets below.
            return FP16_SUBNORMAL_HALF_ULP
        if largest == 0.0:
            # An all-zero DECODED payload is not proof of an all-zero frame: any
            # component below half the minimum subnormal (2**-25) rounds to zero
            # on encode, so 2**-27 goes in and 0.0 comes out. The reader cannot
            # tell that apart from points genuinely at the origin, and claiming
            # an exact 0.0 bound would assert losslessness for an encode that
            # actually discarded the value. 2**-25 is the honest bound: it is
            # what "rounded to zero" can hide, and it is ~30 nanometres, so it
            # costs a caller nothing to carry.
            return FP16_SUBNORMAL_HALF_ULP
        # half-ULP of binary16 at this magnitude. Below the smallest NORMAL value
        # (2**-14) binary16 goes subnormal and the spacing stops halving with the
        # exponent — it is fixed at 2**-24, so the bound floors at 2**-25. Without
        # the clamp the exponent formula keeps shrinking and understates the error
        # of a very small frame by up to ~1000x.
        exponent = np.floor(np.log2(largest))
        return max(float(2.0 ** (exponent - 11)), FP16_SUBNORMAL_HALF_ULP)

    # -- payload ------------------------------------------------------------

    @property
    def xyz(self) -> np.ndarray:
        """(N, 3) float64 point coordinates, LOCAL to `device_position`.

        float64 so that `_geom_max_err_m` is literally true of the array handed
        back: a float32 readout would add its own rounding on top of the codec's
        bound, and a stated guarantee that the reader itself breaks is worse than
        no guarantee. Both codecs decode into it exactly (float16 and
        int16*scale are each exactly representable). Cast down at the point of
        use if a consumer wants float32.

        Cached, like the v2 labels: `max_geom_error_m` calls this for fp16 files,
        so the normal "check the accuracy bound, then use the points" flow
        decoded the payload and allocated a fresh (N, 3) float64 array twice — on
        frames that run to millions of points. `AlfFile` promises parsed-once.
        """
        self._require("alp")
        self._require_payload()
        # BEFORE allocating. Reading a frame and decoding one are different
        # allocations. Passing the read cap does not establish that the decoded
        # array fits the caller's memory budget.
        # 6 bytes on the wire per point (3 components x 2 bytes, both codecs)
        # become 24 in the returned float64 array — a 4x expansion the read cap
        # says nothing about.
        needed = (len(self._payload) // 6) * 3 * 8
        if needed > _decoded_geometry_limit_bytes():
            raise AlfError(
                f"decoded geometry too large: this .alp needs {needed // (1024 * 1024)} MB of "
                f"float64, past the {_decoded_geometry_limit_bytes() // (1024 * 1024)} MB decode "
                f"limit. Raise it with `alf.set_memory_limits(decoded_geometry_bytes=...)` if affordable, or "
                f"use `read_alf_header` if you do not need coordinates."
            )
        if self._xyz_cache is None:
            (self._xyz_cache,) = _frozen(self._decode_xyz())
        return self._xyz_cache

    def _decode_xyz(self) -> np.ndarray:
        if self.geom_codec == GEOM_INT16:
            # The envelope must agree. Metadata is routinely copied forward, so a
            # stale `_geom_codec: int16` on a v1 float16 file is a realistic
            # shape — and decoding those six bytes as three integers returns
            # coordinates that look like data: fp16 [1, 2, 3] reads back as
            # [15360, 16384, 16896]. The inverse mismatch is already refused
            # below; this is the same check from the other side.
            if self.version != ALF_VERSION_V3:
                raise AlfError(
                    f"`_geom_codec: {GEOM_INT16!r}` on a version-{self.version} .alp. Fixed-point "
                    f"geometry is version {ALF_VERSION_V3}; decoding a float16 payload as int16 "
                    f"returns plausible-looking garbage rather than an error."
                )
            scale = _checked_geom_scale(self.metadata)
            # One validator, shared with `max_geom_error_m` — the preflight
            # must refuse exactly what the decode refuses.
            offset = _checked_geom_offset(self.metadata)
            if len(self._payload) % 2:
                raise AlfError(
                    f"int16 .alp payload is {len(self._payload)} bytes, not a whole number of "
                    f"2-byte values; the file is truncated."
                )
            raw = np.frombuffer(self._payload, dtype="<i2")
            if raw.size % 3:
                raise AlfError(f"int16 .alp payload is not a multiple of 3 values (got {raw.size})")
            with np.errstate(over="ignore"):  # detected and raised on, just below
                # IN PLACE. `astype(...) * scale + offset` holds two full
                # (N, 3) float64 arrays at once — the multiplication temporary
                # and the addition result, ~240 MB each at the documented
                # 10M-point scale, on top of the still-live payload. Allocating
                # the result once and scaling into it halves the peak, and the
                # arithmetic is identical.
                words = raw.reshape(-1, 3)
                # Whole points of `INT16_NO_RETURN` are invalid returns, not
                # positions — decoded back to NaN so both codecs hand a caller
                # the same array for the same frame. Located BEFORE the finite
                # check below, which is about scale/offset overflow and would
                # otherwise refuse a frame for its own sentinel.
                #
                # Blockwise into one preallocated (N,) bool: `(words == S).all(1)`
                # builds an (N, 3) temporary, 30 MB at the documented 10M-point
                # scale, on top of the still-live payload and the result.
                no_return = np.zeros(len(words), dtype=bool)
                partial_marker = False
                for begin in range(0, len(words), 262_144):
                    chunk = words[begin : begin + 262_144]
                    hit = chunk == INT16_NO_RETURN
                    np.all(hit, axis=1, out=no_return[begin : begin + 262_144])
                    # A PARTIAL marker is corruption, and it is the one kind that
                    # cannot be caught downstream: `-32768` is reserved, so no
                    # conforming writer emits it as a coordinate (the encoder
                    # clamps to +/-32767) and a wholly non-finite point is the
                    # only thing that produces it. `[-32768, 0, 0]` therefore
                    # cannot come from a valid frame — but reconstructing it
                    # yields -327.68, a perfectly plausible position that no
                    # later check can distinguish from real data. Silence here is
                    # exactly the failure the sentinel exists to prevent.
                    if not partial_marker and bool((hit.any(axis=1) != no_return[begin : begin + 262_144]).any()):
                        partial_marker = True
                if partial_marker:
                    raise AlfError(
                        f"int16 .alp contains a point with SOME but not all components equal to "
                        f"the reserved invalid-return marker ({INT16_NO_RETURN}). No conforming "
                        f"writer can emit that word as a coordinate, so this payload is corrupt; "
                        f"decoding it would report a plausible position that was never measured."
                    )
                decoded = words.astype(np.float64)
                np.multiply(decoded, scale, out=decoded)
                np.add(decoded, offset, out=decoded)
                if no_return.any():
                    decoded[no_return] = np.nan
            # The fp16 path refuses non-finite coordinates; this one must too. A
            # positive finite scale can still overflow — `_geom_scale: 1e308`
            # times a stored 2 is Inf — and an Inf coordinate propagates through
            # world_xyz, the box and every downstream metric as a contaminant.
            # `decoded[~no_return]` COPIES every surviving row — the whole array
            # again, for a question answerable in place. A sentinel row is
            # allowed to be non-finite because that is what it means; every
            # other row must be finite.
            if decoded.size and not (np.isfinite(decoded).all(axis=1) | no_return).all():
                raise AlfError(
                    "int16 .alp decodes to non-finite coordinates; `_geom_scale`/`_geom_offset` "
                    "overflow the representable range"
                )
            return decoded

        if self.geom_codec != GEOM_FP16:
            raise AlfError(
                f"Unknown .alp geometry codec {self.geom_codec!r} (expected {GEOM_INT16!r} or {GEOM_FP16!r}). "
                f"Refusing to guess: an int16 payload is the same 6 bytes per point as float16, so decoding "
                f"it as the wrong codec returns plausible-looking coordinates rather than an error."
            )
        if self.version == ALF_VERSION_V3:
            raise AlfError(
                f"Version-3 .alp declares codec {self.geom_codec!r}; v3 is fixed-point geometry and requires "
                f"`_geom_codec: {GEOM_INT16!r}` with `_geom_scale`. A missing or misspelled key would otherwise "
                f"decode int16 as float16 and yield garbage."
            )
        # `frombuffer` raises a bare ValueError when the length is not a whole
        # number of float16s, which escapes the AlfError contract that callers
        # use to quarantine a bad frame and keep going.
        if len(self._payload) % 2:
            raise AlfError(
                f"float16 .alp payload is {len(self._payload)} bytes, not a whole number of "
                f"2-byte values; the file is truncated."
            )
        raw = np.frombuffer(self._payload, dtype="<f2")
        if raw.size % 3:
            raise AlfError(f"float16 .alp payload is not a multiple of 3 values (got {raw.size})")
        # NON-FINITE COMPONENTS ARE RETURNED AS-IS, deliberately. A NaN in
        # float16 `.alp` geometry is the invalid-return marker (spec/format.md
        # §5.1), not damage. A frame can contain any number of invalid returns.
        #
        # The points are kept IN PLACE rather than dropped, because `.ali`,
        # `.all`, `.alc` and `.alx` are correlated with geometry BY INDEX —
        # compacting the invalid returns away here would silently misalign every
        # other channel of the frame, which is the one failure this module exists
        # to prevent. Consumers filter with `pts[~np.isnan(pts).any(axis=1)]`.
        return raw.reshape(-1, 3).astype(np.float64)

    def world_xyz(self, *, origin: Any | None = None, sensor_pose: bool = False) -> np.ndarray:
        """(N, 3) float64 in the world frame.

        `xyz + device_position` is NOT universally the world frame, and assuming
        it is silently misframes whole datasets. Two conventions exist:

        * **Sensor pose**:
          `device_position` IS the world origin of the local frame.
        * **Aerial floating origin** (the aerial tile writers):
          points are local to a CENTROID in EPSG:3857, while `device_position`
          can instead be (0,0,0) or a tile-placement offset. Adding it can
          place the cloud near the origin of a
          projected CRS — kilometres from the truth.

        A legacy header may omit the anchor that distinguishes these cases.
        An application can keep `world_centroid_epsg3857` outside the ALP, where
        this reader cannot see it. `_box_frame: "local"` does not identify the
        origin either. Guessing from `device_position` would return local
        coordinates while claiming world ones.

        A NONZERO `device_position` does not resolve it either. A tile writer
        can write `tile_origin - global_centroid` there — a nonzero, entirely
        plausible-looking vector that is NOT a sensor pose, with the true centroid
        again only in the sidecar. A canonical sensor pose and an
        aerial tile offset are the same shape, the same magnitude range, and carry
        no distinguishing key. There is no test on the bytes that separates them.

        So the frame is the CALLER's to declare, and there is no default:

        1. an explicit `origin=` — the sidecar/frame-JSON centroid for aerial
           data, or `(0, 0, 0)` to assert local *is* world;
        2. `world_centroid_epsg3857` when the metadata carries it;
        3. `sensor_pose=True` to state that `device_position` is the world origin
           of the local frame;
        4. otherwise raise.

        Requiring (3) rather than defaulting to it is the whole point: a default
        can silently displace a tile by kilometres. The explicit argument makes
        callers state the convention; recording the centroid in the ALP header
        makes branch 2 sufficient without external metadata.

        A method rather than a property because it is a real conversion that takes
        arguments, and float64 throughout because at EPSG:3857 magnitudes float32
        resolves only about a metre.
        """
        # Channel FIRST: on a `.ali` the anchor check would fire before `.xyz`
        # is reached, reporting a missing `device_position` for what is really a
        # wrong-channel read — a true error with a misleading cause.
        self._require("alp")
        if origin is not None:
            anchor = _as_anchor(origin, "origin")
        # PRESENCE, not truthiness. `{}` and `[]` are falsey, so a truthiness
        # check would skip validation and fall through to the sensor-pose
        # branch — silently placing an aerial frame by the convention its
        # own metadata contradicts. A recorded-but-corrupt anchor must
        # raise, not defer: the key's presence is the claim, and a claim
        # this reader cannot honour is an error rather than a reason to
        # guess differently.
        elif "world_centroid_epsg3857" in self.metadata:
            anchor = _as_anchor(self.metadata["world_centroid_epsg3857"], "world_centroid_epsg3857")
        elif sensor_pose:
            # Through the same validator as `origin`. Asserting "this file's
            # device_position IS the world origin" and then defaulting a
            # missing axis to zero would place the frame at (100, 0, 0) for
            # a header carrying only {"x": 100} — the same silent shift the
            # explicit-anchor check exists to stop, arrived at by a
            # different door.
            # Through `device_position`, which treats an ABSENT pose as the
            # origin and still validates a malformed one. Routing `{}`
            # straight into `_as_anchor` made "missing" indistinguishable
            # from "corrupt", so an origin-local frame — exactly what
            # `write_alp` produces without the argument — could not be
            # placed at all by a caller correctly asserting the convention.
            anchor = self.device_position
        else:
            raise AlfError(
                "Cannot place this frame in the world without being told its convention. "
                "The metadata carries no `world_centroid_epsg3857`, and `device_position` "
                # RAW .get() on purpose: this is inside an error message, and
                # a validator raising here would replace the guidance the
                # caller needs with a second, less useful exception.
                f"({self.metadata.get('device_position')!r}) is ambiguous on its own: for the "
                f"canonical "
                "writers it IS the sensor pose, but the aerial writers put a "
                "(0,0,0) placeholder or a tile-placement offset there and keep the real "
                "EPSG:3857 centroid in the frame JSON beside the file. Adding the wrong one "
                "misplaces the cloud by kilometres and nothing downstream would notice. "
                "Pass `world_xyz(origin=...)` with the sidecar centroid, or "
                "`world_xyz(sensor_pose=True)` if this file's `device_position` really is "
                "the sensor pose."
            )
        local = self.xyz
        # THE ROTATION HALF. `local + anchor` alone is only correct when the
        # producer subtracted an origin and stopped; a `device-aligned` frame
        # was also rotated, and adding the anchor to it places a correctly
        # shaped cloud at the right spot facing the wrong way. Version 3 files
        # say which they are, so this is decided by the file rather than by the
        # caller. Versions 1 and 2 do not say, and `frame_convention` is None
        # for them, so they keep the historical translation-only behaviour.
        rotated = None
        if self.frame_convention == FRAME_DEVICE_ALIGNED:
            rotated = _rotate_by_quaternion(local, _checked_frame_rotation(self.metadata))
        with np.errstate(over="ignore", invalid="ignore"):
            # The overflow is DETECTED below and reported as an AlfError with a
            # cause; NumPy's RuntimeWarning on top of that is noise that would
            # also flood the logs of any pipeline scanning a bad sequence.
            if rotated is None:
                world = local + anchor
            else:
                # In place, into the array the rotation just allocated. `rotated
                # + anchor` would hold THREE full-size coordinate arrays at once
                # — the cached local decode, the rotation's output and the sum —
                # where two are needed, and this module's whole memory contract
                # is about what a frame near the decoded-geometry ceiling costs.
                rotated += anchor
                world = rotated
        # Only where a FINITE local coordinate became non-finite. An invalid
        # return is non-finite before the anchor is applied and stays non-finite
        # after it — that is the marker being carried through world placement,
        # not an overflow, and refusing it would put every such frame back to
        # unreadable through a second door.
        if world.size and bool((np.isfinite(local) & ~np.isfinite(world)).any()):
            # Finite locals plus a finite anchor can still exceed float64.
            # `_decode_xyz` and `_as_anchor` both refuse non-finite values, so
            # letting the SUM produce Inf contradicts an invariant this module
            # enforces at both inputs — and Inf in world coordinates silently
            # poisons every bound, centroid and metric computed downstream.
            raise AlfError(
                "applying the world anchor overflowed to non-finite coordinates; the frame's "
                "local geometry and its anchor are individually finite but their sum is not, "
                "so this file's world placement cannot be represented"
            )
        return world

    @property
    def header_only(self) -> bool:
        """Whether this file's payload was never read (see `read_alf_header`).

        Public because the weaker contract has to be inspectable: a caller that
        receives an `AlfFile` from a helper cannot otherwise tell whether
        `_check_payload_length` ran on it.
        """
        return isinstance(self._payload, _WithheldPayload)

    def _require_payload(self) -> None:
        """Refuse an accessor that needs bytes a header-only read never fetched.

        The sentinel would refuse anyway — that is the fail-closed backstop —
        but `np.frombuffer` reaches the buffer protocol directly and would raise
        a bare `TypeError`, which is not the `AlfError` callers quarantine on.
        Called from the accessors whose payload need is UNCONDITIONAL; the
        conditional ones (`format_version`, `format_version_verified`) are left
        to the sentinel on purpose, so they keep working for the common case
        where the header answers on its own and refuse only when they genuinely
        have to consult the bytes.
        """
        if self.header_only:
            raise AlfError(f"Cannot read this payload: {_WithheldPayload._MESSAGE}")

    def _require(self, *channels: str) -> None:
        """Refuse an accessor that does not belong to this file's channel.

        Every accessor lives on one unified `AlfFile`, so nothing stops
        `.categories` being read off an `.ali`. The payloads are raw bytes of
        compatible length, so that returns six intensity readings as six
        plausible class ids — and `.xyz` will reinterpret the same bytes as
        float16 coordinates. Both look like data and neither raises. Cheap to
        check, and the failure it prevents is silent.
        """
        if self.extension and self.extension not in channels:
            raise AlfError(
                f"This is a .{self.extension} file; that accessor belongs to "
                f"{' / '.join('.' + c for c in channels)}. Reading it here would reinterpret the "
                f"payload as a different channel and return plausible-looking wrong values."
            )

    @property
    def intensity(self) -> np.ndarray:
        """(N,) uint8 normalized intensity (`.ali`)."""
        self._require("ali")
        self._require_payload()
        return np.frombuffer(self._payload, dtype=np.uint8)

    @property
    def categories(self) -> np.ndarray:
        """(N,) uint8 per-point category (`.all`, both layouts)."""
        self._require("all")
        self._require_payload()
        if self.format_version == LABELS_V2:
            return self._labels_v2()[0]
        return np.frombuffer(self._payload, dtype=np.uint8)

    @property
    def instance_ids(self) -> np.ndarray | None:
        """(N,) uint16 per-point instance id, or None for the v1 layout.

        Raises when the ids cannot be resolved to persistent instances, because
        an unresolvable local id is not data — `write_all` requires the map to
        cover every nonzero id, and the legacy batch exporter silently emits
        `object_uuid: null` for the ones it cannot look up. Returning them
        unvalidated exports objects with no identity and reports success.

        Deliberately scoped to THIS accessor rather than the decode: `categories`
        stays readable, so a semantic-only consumer is not denied correct label
        data because the panoptic half of the header is incomplete.
        """
        self._require("all")
        self._require_payload()
        if self.format_version != LABELS_V2:
            return None
        instances = self._labels_v2()[1]
        # Reserved-id check lives HERE, not in the shared decode: 65535 makes the
        # PANOPTIC half unusable, and putting it in `_labels_v2` denied
        # `categories` to semantic-only consumers whose data is perfectly fine —
        # contradicting the accessor split this module documents.
        if instances.size and int(instances.max()) == 65535:
            raise AlfError(
                "`.all` v2 payload contains the reserved instance id 65535; valid local ids are "
                "1..65534 and 0 means unset, so this file's instance identity cannot be resolved. "
                "Read `.categories` if only the semantic labels are needed."
            )
        resolved = _present_or_absent(self.metadata, "instance_map")
        # An ABSENT map is legacy, not corruption: the legacy export parsers
        # identify v2 from the wire byte and synthesize the metadata afterwards,
        # so files exist whose identity was never recorded at all (see
        # `test_v2_labels_are_detected_from_the_envelope_byte_alone`). Refusing
        # those would deny reads of stored production data to enforce a rule
        # written after they were made. A PRESENT but incomplete map is
        # different — the file claims to record identity and silently does not
        # for some ids, which is the case worth failing on.
        if resolved is not None and instances.size and int(instances.max()) != 0:
            if not isinstance(resolved, dict):
                raise AlfError(f"`instance_map` must be a mapping of str(id) -> uuid, got {type(resolved).__name__}")

            # CANONICAL keys only. The legacy batch exporter does an
            # exact `instance_lookup.get(str(inst_id))`, so a map storing id 1
            # under "01" resolves to nothing there and exports
            # `object_uuid: null` -- normalizing keys here would call that
            # covered and hide the failure this check exists to catch.
            missing = sorted(int(v) for v in np.unique(instances) if int(v) and _instance_unresolved(resolved, int(v)))
            if missing:
                raise AlfError(
                    f"`instance_map` is missing entries for instance id(s) {missing[:10]}"
                    f"{'...' if len(missing) > 10 else ''}; those points would export with no "
                    f"object identity."
                )
        return instances

    @property
    def palette_indices(self) -> np.ndarray:
        """(N,) uint8 palette index per point (`.alc`).

        The palette itself is `metadata["palette"]` — a list of `rrggbb` hex
        strings, as the canonical palette generator writes it. Without this
        accessor a consumer has to reach into `_payload` or keep its own parser,
        which is the duplication this module exists to end; the module has
        advertised `.alc` support from its first line.
        """
        self._require("alc")
        self._require_payload()
        indices = np.frombuffer(self._payload, dtype=np.uint8)
        # The writer bounds these; external and legacy files are not guaranteed
        # to. An index past the sentinel resolves to no colour at all, so the
        # viewer either throws or paints something arbitrary — while the shared
        # reader had called the file well-formed.
        palette = self.palette
        if palette is None:
            # `.alc` bytes are ONLY offsets into the header palette, and
            # `len(palette)` defines the no-colour sentinel — without it the
            # indices cannot be resolved or even range-checked, so returning
            # them hands consumers numbers with no meaning. The writer requires
            # a non-empty palette; so does this.
            raise AlfError(
                "`.alc` has no `palette`; its indices are offsets into that list and cannot be resolved without it."
            )
        if indices.size and int(indices.max()) > len(palette):
            raise AlfError(
                f"`.alc` palette index {int(indices.max())} is out of range for a "
                f"{len(palette)}-entry palette (max {len(palette)}, the no-colour sentinel)"
            )
        return indices

    @property
    def palette(self) -> list | None:
        """`.alc` colour palette: `rrggbb` hex strings, or None if absent."""
        self._require("alc")
        palette = _present_or_absent(self.metadata, "palette")
        if palette is None:
            return None
        # A bare string passes `len()` — `"ff0000"` would read as a six-entry
        # palette, making index 5 look in range and handing the caller a string
        # where the contract promises a list of colours.
        if not isinstance(palette, list):
            raise AlfError(f"`.alc` palette must be a list of 'rrggbb' strings, got {type(palette).__name__}")
        if len(palette) > 255:
            # The writer caps at 255 and `json2color.py:74` does the same,
            # because `len(palette)` is the no-colour sentinel in a uint8
            # stream: at 256 entries that sentinel is unrepresentable, so index
            # 255 reads as a normal colour and occluded points have nowhere to
            # go. Accepting it here would let a malformed file pass as usable.
            raise AlfError(
                f"`.alc` declares {len(palette)} palette entries; the maximum is 255 because "
                f"index {len(palette)} is reserved as the no-colour sentinel."
            )
        if not palette:
            # `0 > len([])` is false, so payload byte 0 would pass the range
            # check and a colour file with no colours read as valid. On the
            # PROPERTY, so a caller reading `alf.palette` directly is refused
            # too — the writer already rejects an empty palette.
            raise AlfError("`.alc` palette is empty; its indices resolve to nothing.")
        normalized: list = []
        for entry in palette:
            # Length alone is not the contract: "zzzzzz" passes a length check
            # and then range-checks the indices against a palette the viewer
            # cannot parse, so the reader calls a file well-formed whose colours
            # do not exist. Same hex domain the writer enforces.
            # ONE optional leading '#', then exactly six hex digits. `lstrip("#")`
            # would turn "##ff0000" into a valid-looking entry, and returning the
            # raw list hands "#00FF00" to a consumer that prepends '#' or calls
            # bytes.fromhex — the accessor promises `rrggbb`, so it returns
            # `rrggbb`.
            text = entry[1:] if isinstance(entry, str) and entry.startswith("#") else entry
            if not isinstance(text, str) or len(text) != 6 or not all(c in _HEX_COLOR for c in text):
                raise AlfError(f"`.alc` palette entry {entry!r} is not six hexadecimal digits ('rrggbb')")
            normalized.append(text.lower())
        return normalized

    @property
    def kept_indices(self) -> np.ndarray:
        """(N,) uint32 original index per kept point (`.alx` decimation sidecar).

        Pairs with `write_alx`. A write-only channel would just relocate the
        hand-rolled parser to whoever consumes the mapping.
        """
        self._require("alx")
        self._require_payload()
        if len(self._payload) % 4:
            raise AlfError(
                f"`.alx` payload is {len(self._payload)} bytes, not a whole number of 4-byte "
                f"indices; the sidecar is truncated."
            )
        indices = np.frombuffer(self._payload, dtype="<u4")
        # The writer records `_source_total_points` precisely so a reader can
        # re-check the range without the writer's arguments — so it must. An
        # out-of-range index here is not caught anywhere downstream: the export
        # uses it to address the authoritative raw cloud, so it either reads the
        # wrong point or fails far from the corrupt file that caused it.
        source = _checked_count(self.metadata, "_source_total_points")
        if source is None:
            # REQUIRED, not optional. Without it the range check simply does not
            # run, so a same-length sidecar with no source bound sails through
            # and an export indexes the authoritative raw cloud with values nothing
            # ever validated. `.alx` is new in this module, so no stored file
            # predates the key — there is no compatibility case to preserve.
            raise AlfError(
                "`.alx` is missing `_source_total_points`; without it the kept indices cannot be "
                "range-checked against the cloud they address."
            )
        if indices.size and len(np.unique(indices)) != indices.size:
            # Mirrors the writer. Size and range are both satisfied by [5, 5],
            # yet two labelled points then address the same raw point while
            # another kept point has no mapping at all.
            raise AlfError(
                "`.alx` contains duplicate kept indices; each decimated point must map to a distinct source point."
            )
        if indices.size and int(indices.max()) >= source:
            raise AlfError(
                f"`.alx` kept index {int(indices.max())} is out of range for a source cloud of "
                f"{int(source)} points; the sidecar is malformed."
            )
        return indices

    @property
    def instance_map(self) -> dict[str, str] | None:
        # Same channel guard as `categories`/`instance_ids`. Without it a `.ali`
        # arriving from a mixed glob returns None (or a stale map inherited in
        # metadata) instead of surfacing the mix-up, so a wrong-channel file
        # reads as a label file that simply has no panoptic identities.
        self._require("all")
        if self.format_version != LABELS_V2:
            return None
        resolved = _present_or_absent(self.metadata, "instance_map")
        if resolved is None:
            return None
        # The annotation promises Dict[str, str], and a consumer reading this
        # property directly to resolve ids would otherwise get a raw string or
        # list — only `instance_ids` validated the map.
        if not isinstance(resolved, dict):
            raise AlfError(f"`instance_map` must be a mapping of str(id) -> uuid, got {type(resolved).__name__}")
        bad = sorted(str(k) for k, v in resolved.items() if not isinstance(v, str) or not v.strip())
        if bad:
            raise AlfError(f"`instance_map` has unusable value(s) for key(s) {bad[:10]}")
        # Keys must be CANONICAL decimal ids. The resolvers do an exact
        # `str(instance_id)` lookup, so `{"01": ...}` leaves local id 1
        # unresolved — and a caller reading this property directly to resolve
        # identities would never hit `instance_ids`' coverage check.
        noncanonical = sorted(str(k) for k in resolved if str(k) != str(_safe_int(k)))
        if noncanonical:
            raise AlfError(
                f"`instance_map` has non-canonical key(s) {noncanonical[:10]}; resolvers look up "
                f"str(id) exactly, so those entries resolve nothing"
            )
        out_of_domain = sorted(str(k) for k in resolved if not (1 <= int(str(k)) <= 65534))
        if out_of_domain:
            # DOMAIN, not just spelling — the same check the writer runs. An
            # externally produced file whose payload happens to emit only valid
            # ids still handed back sequence-level identities of "0", "65535" or
            # "99999": reserved or impossible, and accepted purely because this
            # frame did not emit them.
            raise AlfError(
                f"`instance_map` has key(s) {out_of_domain[:10]} outside the valid instance id "
                f"range 1..65534 (0 means unset, 65535 is reserved), so those entries name "
                f"identities no frame can carry"
            )
        instances = self._labels_v2()[1]
        # Reserved id here TOO. The accessor split says `categories` stays
        # readable when the panoptic half is unusable — but `instance_map` is
        # the panoptic half: a consumer resolving identities through it never
        # touches `instance_ids`, so checking only there let a mapped 65535
        # through one public door while the other refused it.
        if instances.size and int(instances.max()) == 65535:
            raise AlfError(
                "`.all` v2 payload contains the reserved instance id 65535, so this file's "
                "instance identity cannot be resolved — mapping it in `instance_map` does not "
                "make it valid. Read `.categories` if only the semantic labels are needed."
            )
        missing = sorted(int(v) for v in np.unique(instances) if int(v) and _instance_unresolved(resolved, int(v)))
        if missing:
            raise AlfError(f"`instance_map` is missing entries for emitted instance id(s) {missing[:10]}")
        return resolved

    def _labels_v2(self):
        # Cached: `categories` and `instance_ids` each need one half of this
        # decode, and both discard the other. Without the cache a consumer that
        # reads both panoptic streams — the normal case for a v2 file — pays two
        # full passes over the payload and four transient array copies, on frames
        # that run to millions of points. `AlfFile` promises parsed-once.
        if self._labels_v2_cache is None:
            self._labels_v2_cache = _frozen(*self._decode_labels_v2())
        return self._labels_v2_cache

    def _decode_labels_v2(self):
        raw = np.frombuffer(self._payload, dtype=np.uint8)
        if raw.size % 3:
            raise AlfError(
                f"Corrupt .all: v2 payload length {raw.size} is not a multiple of 3 "
                f"(1 byte category + 2 bytes instance id per point)"
            )
        rows = raw.reshape(-1, 3)
        categories = rows[:, 0].copy()
        instances = (rows[:, 1].astype(np.uint16) | (rows[:, 2].astype(np.uint16) << 8)).copy()
        return categories, instances


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def _has_nonzero(view: memoryview) -> bool:
    """Whether a byte view holds anything but NUL, without copying it.

    The tail after a gzip member can be the whole rest of a hostile input, so
    materializing it to ask this question is exactly the allocation the size cap
    exists to prevent. Scanned in blocks so the peak is the block, not the tail.
    """
    step = _GZIP_CHUNK_BYTES
    for begin in range(0, len(view), step):
        if bytes(view[begin : begin + step]).strip(b"\x00"):
            return True
    return False


def _limit_for(gzipped: bool) -> int:
    """The largest acceptable INPUT, given whether it is compressed.

    `_limit_bytes()` bounds what a frame may weigh once decoded. An
    archive of a permitted envelope can exceed that on the wire, because deflate
    expands incompressible data slightly — so measuring the archive against the
    decompressed cap rejected the writer's own output. The incremental
    decompression loop enforces the real limit; this only stops something absurd
    arriving in the first place.
    """
    return _limit_bytes() + (_GZIP_OVERHEAD_ALLOWANCE if gzipped else 0)


def _read_to_eof_capped(source: BinaryIO, describe: str) -> bytearray:
    """Drain `source` to EOF, enforcing the size cap as the bytes arrive.

    Loop to EOF. A single `read(n)` is allowed to return FEWER bytes than asked
    for without being at EOF — normal for sockets and any buffered network
    reader — so capping the read that way silently truncated the stream to its
    first chunk. A legacy file without `_total_points` then parsed as a short
    prefix and quietly lost points; one with a declared count was rejected as
    truncated.

    Shared with the path branch, which routes non-regular files here: their
    `st_size` is advisory, so the length they report cannot bound the read.
    """
    buffer = bytearray()
    while True:
        chunk = source.read(8 * 1024 * 1024)
        if not chunk:
            break
        buffer += chunk
        # Decided from the magic bytes once enough of them have arrived.
        # Applying the decompressed cap to a compressed stream rejects a
        # legitimate archive for the third time, in the third place.
        if len(buffer) > _limit_for(bytes(buffer[:2]) == b"\x1f\x8b"):
            raise AlfError(f"{describe} exceeds the {_limit_bytes() // (1024 * 1024)} MB limit.")
    # The bytearray itself. `bytes(buffer)` allocates a second contiguous copy
    # while the first is still live, so a stream near the cap needed ~2x before
    # parsing even started — the same doubling already removed from the gzip
    # branch, reintroduced in the loop written to fix the short-read truncation.
    # The payload view is made read-only downstream, so nothing mutable escapes.
    return buffer


def _open_maybe_gzip(source: str | Path | bytes | BinaryIO) -> bytes | bytearray:
    """Return the decompressed ALF bytes from a path, buffer, or stream.

    Gzip is detected by magic number rather than by filename: label diffs are
    uploaded to `_work_tracking/` UNCOMPRESSED while the merged canonical object
    is gzipped, and both arrive as `.all`. A reader that keys off the extension
    rejects live diffs.
    """
    # DECLARED, not inferred from whichever branch happens to come first. The
    # four sources legitimately produce three different buffer types, so letting
    # mypy pin `raw` to the first one it sees makes the other branches look like
    # type errors and the ordering of the `if` chain load-bearing.
    raw: bytes | bytearray | memoryview
    if isinstance(source, (str, Path)):
        # Size-checked BEFORE reading. The cap was applied to the gzip branch
        # alone, so an uncompressed multi-gigabyte `.all` — a shape this
        # function explicitly supports, since live label diffs are uploaded
        # uncompressed — sailed straight past it and OOMed the worker before
        # any header check could quarantine the frame.
        # ONE descriptor for the size, the magic and the bytes. Separately
        # calling `stat()`, opening again for the magic, and then
        # `read_bytes()` describes up to three different files if the path is
        # replaced or appended to in between — and the final read was unbounded,
        # so a file that was small at check time could grow past the cap and OOM
        # the worker the cap exists to protect.
        handle = Path(source).open("rb")
        try:
            status = os.fstat(handle.fileno())
            if not stat.S_ISREG(status.st_mode):
                # `st_size` is a real byte count only for a REGULAR file. A FIFO
                # or a character device reports 0 no matter how much it will
                # deliver, so the length rule below cannot apply to it — and it
                # never got that far anyway: the `seek(0)` that rewinds after the
                # magic-byte peek raises `io.UnsupportedOperation` on an
                # unseekable file, which is NOT the `AlfError` callers quarantine
                # on. Same contract leak as the memoryview cast, one branch over.
                # These are streams that happen to have a path, so they get the
                # stream treatment: read to EOF under the cap, magic decided from
                # the bytes as they arrive.
                raw = _read_to_eof_capped(handle, str(source))
            else:
                size = status.st_size
                gzipped = handle.read(2) == b"\x1f\x8b"
                handle.seek(0)
                # The cap is about DECOMPRESSED bytes. Applying it to a gzip file's size
                # on disk rejects a legitimate one whose payload is incompressible:
                # deflate framing makes the archive slightly LARGER than its contents,
                # so a 1024-byte envelope written at a 1024-byte cap became a 1047-byte
                # `.gz` that this refused — the writer's own output, unreadable. The
                # incremental decompression loop below enforces the real limit.
                limit = _limit_for(gzipped)
                if size > limit:
                    raise AlfError(
                        f"{source}: {size} bytes, past the "
                        f"{_limit_bytes() // (1024 * 1024)} MB read limit. Raise it with "
                        f"`alf.set_memory_limits(decompressed_bytes=...)` if this frame is legitimate; the limit "
                        f"exists to bound memory, not because ALF frames cannot be this large."
                    )
                # Bounded by the file's OWN size, not by the limit: `read(n)`
                # pre-allocates n bytes, so passing the 512 MB cap reserved 512 MB
                # to read an 8 MB frame. One byte past the stat'd size is enough to
                # notice the file grew under us.
                raw = handle.read(size + 1)
                # EXACTLY the stat'd size, not merely "no bigger". Compared with
                # the SNAPSHOT, not the global limit: the comment above always
                # said the extra byte was there to notice growth, but the check
                # asked whether the file had passed 512 MB — so a legacy channel
                # appended to between the fstat and this read was parsed as its
                # old prefix plus one byte and the rest vanished silently.
                #
                # Shrinkage is the same bug wearing the other sign, and it
                # survived the fix to growth. On a regular file `read(n)` returns
                # n bytes or hits EOF, so a SHORT read means the file lost bytes
                # between the fstat and here — mid-transfer truncation, which is
                # exactly how these objects go bad. `.ali` and `.all` without
                # `_total_points` accept any payload length, so 100 intensity
                # samples became 60 with nothing raised, misaligning every
                # index-correlated channel behind them.
                if len(raw) != size:
                    if len(raw) > size:
                        raise AlfError(
                            f"{source}: grew while being read ({size} bytes when opened, more "
                            f"now). Re-read it; parsing the prefix would silently drop the rest."
                        )
                    raise AlfError(
                        f"{source}: shrank while being read ({size} bytes when opened, "
                        f"{len(raw)} readable now). Re-read it; parsing the short payload would "
                        f"silently drop points and misalign every channel indexed against them."
                    )
        finally:
            handle.close()
    elif isinstance(source, (bytes, bytearray, memoryview)):
        # Through the SHARED normalizer, so the contiguity rule and the byte-view
        # cast cannot drift between this reader and `read_alf_header`.
        view = _as_byte_view(source)
        # Length FIRST, then no copy at all. `bytes(source)` duplicated the
        # buffer before the cap could look at it, so an oversized input OOMed on
        # the way to being rejected — and a legitimate one near the cap held ~2x.
        # A read-only view is what the rest of the reader wants anyway.
        # Same rule as the path branch. Fixing that one and leaving this made a
        # legitimate archive readable from disk and rejected from memory — the
        # cap is about DECOMPRESSED bytes either way.
        #
        # Normalizing before this check costs nothing: `cast` and `toreadonly`
        # are views, so no copy happens ahead of the cap either way.
        nbytes = view.nbytes
        if nbytes > _limit_for(bytes(view[:2]) == b"\x1f\x8b"):
            raise AlfError(f"Buffer is {nbytes} bytes, past the {_limit_bytes() // (1024 * 1024)} MB limit.")
        raw = view.toreadonly()
    else:
        # `source.read()` used to consume the whole stream, and adding the cap
        # is what broke that. Same rule as the other two branches.
        raw = _read_to_eof_capped(source, "Stream")

    if raw[:2] == b"\x1f\x8b":
        # A corrupt or truncated gzip stream is a malformed ALF file like any
        # other, and must arrive as AlfError. Leaking EOFError / BadGzipFile /
        # zlib.error aborts a pipeline run that would otherwise quarantine the
        # frame and continue -- and truncation mid-transfer is exactly how these
        # objects go bad.
        # Incremental, with a cap. `gzip.decompress` materializes the WHOLE
        # output before any header, count or payload check can look at it, and
        # these objects are untrusted: 199 KB of zeros expands to
        # 200 MB at ~1000:1, so a corrupt or hostile file OOMs the worker
        # instead of reaching the AlfError quarantine this module promises.
        #
        # The cap is deliberately far above any real frame — a 10-million-point
        # `.alp` is 60 MB of payload — so it bounds the damage without deciding
        # what a legitimate frame may weigh.
        try:
            # A bytearray, not a list of chunks joined at the end: the join
            # needed a second contiguous copy of everything while the chunks
            # were still live, so the true peak was twice the cap.
            # `zlib.decompressobj(wbits=31)`, not `GzipFile`. GzipFile decodes
            # every CONCATENATED member into one byte run, so a second ALF's
            # envelope silently becomes the first one's payload — and for a
            # legacy file with no `_total_points` any length is valid, so `.ali`
            # returned the next file's JSON header as intensity samples. This
            # form stops at the end of member one and hands back what followed,
            # which is also why `compressed.tell()` could not detect it:
            # GzipFile reads ahead, so the position is already at EOF.
            buffer = bytearray()
            engine = zlib.decompressobj(wbits=31)
            offset = 0
            while offset < len(raw) and not engine.eof:
                chunk = engine.decompress(bytes(raw[offset : offset + _GZIP_CHUNK_BYTES]), _GZIP_CHUNK_BYTES)
                offset += _GZIP_CHUNK_BYTES
                if len(buffer) + len(chunk) > _limit_bytes():
                    raise AlfError(
                        f"Compressed stream expands past the "
                        f"{_limit_bytes() // (1024 * 1024)} MB limit "
                        f"({len(raw)} compressed bytes and still going). Raise it with "
                        f"`alf.set_memory_limits(decompressed_bytes=...)` if this frame is "
                        f"legitimate; the limit bounds memory, not what ALF permits."
                    )
                buffer += chunk
                while engine.unconsumed_tail and not engine.eof:
                    chunk = engine.decompress(engine.unconsumed_tail, _GZIP_CHUNK_BYTES)
                    if len(buffer) + len(chunk) > _limit_bytes():
                        raise AlfError(
                            f"Compressed stream expands past the {_limit_bytes() // (1024 * 1024)} MB limit."
                        )
                    buffer += chunk
            if not engine.eof:
                raise AlfError("Corrupt gzip stream: ended before the member was complete")
            # BOTH leftovers. `unused_data` holds what followed the member
            # inside the last block fed to the decompressor — but when the
            # member ends exactly on a block boundary that is empty and the
            # remaining input at `raw[offset:]` was never fed at all, so a
            # second member starting right there went unseen and the
            # concatenation was accepted after all.
            # Inspected SEPARATELY, never concatenated. Building
            # `unused_data + raw[offset:]` materialized the whole remaining
            # input and then a second full-size copy of it, while the compressed
            # buffer was still live — so rejecting a hostile input near the cap
            # could need over a gigabyte, OOMing on the way to refusing.
            unread = memoryview(raw)[offset:] if offset < len(raw) else memoryview(b"")
            if bytes(engine.unused_data).strip(b"\x00") or _has_nonzero(unread):
                raise AlfError(
                    "Compressed stream holds more than one gzip member. Concatenated ALF files "
                    "decode into a single byte run, so the second file's header would be read "
                    "as the first file's payload."
                )
            # The bytearray itself, NOT `bytes(buffer)`. Every downstream use is
            # a bytes-like operation (slicing, struct.unpack_from, json.loads,
            # np.frombuffer), and the final conversion would copy the whole
            # payload a second time: measured 419 MB peak for a 200 MB frame via
            # chunk-list-and-join, 427 MB with a trailing bytes() copy, 242 MB
            # returning the buffer. Halving the peak of the SUCCESS path is
            # worth more than the type purity.
            return buffer
        except (OSError, EOFError, zlib.error) as exc:
            # gzip.BadGzipFile subclasses OSError.
            raise AlfError(f"Corrupt gzip stream: {exc}") from exc
    return raw


def _as_byte_view(source: bytes | bytearray | memoryview) -> memoryview:
    """A 1-D, C-contiguous, read-only byte view of an in-memory source.

    Shared by the full read and the header-only read so the contiguity rule
    cannot drift between them.
    """
    view = memoryview(source)
    # Contiguity FIRST — ahead of every cast, and unconditionally.
    #
    # `cast()` is documented to work only on C-contiguous views and raises a
    # bare `TypeError` otherwise, so probing the magic bytes with
    # `probe.cast("B")` before this guard let `memoryview(buf)[::2]` abort
    # with `TypeError: memoryview: casts are restricted to C-contiguous
    # views`. A pipeline whose quarantine is `except AlfError: skip_frame()`
    # then tears down the whole run instead of skipping one frame — the
    # failure shape this module exists to prevent.
    #
    # Unconditional because the guard used to sit inside the
    # `itemsize != 1 or ndim != 1` branch, which a strided BYTE view does
    # not enter: `memoryview(buf)[::2]` has itemsize 1 and ndim 1, so the
    # rejection was unreachable for exactly the views that needed it and the
    # reader went on to read every other byte as if the offsets were dense.
    #
    # `c_contiguous`, not `contiguous`. The latter is also True for a
    # FORTRAN-contiguous view, which `cast()` refuses just as loudly —
    # `memoryview(np.zeros((4, 4), np.uint8).T)` leaked the same `TypeError`
    # straight through a guard written with the looser predicate. C order is
    # what a byte offset means here anyway.
    if not view.c_contiguous:
        raise AlfError("Buffer must be a contiguous byte view; ALF offsets are byte offsets.")
    # Normalized to a BYTE view. `len()` and slicing on a `memoryview` count
    # ELEMENTS, so a view cast to `I` made every byte offset advance four
    # times too far and the size check undercount by the item size —
    # `raw[end:]` returned the tail of the payload rather than the payload.
    if view.itemsize != 1 or view.ndim != 1:
        view = view.cast("B")
    return view


def _iter_source_blocks(
    source: str | Path | bytes | bytearray | memoryview | BinaryIO,
) -> Iterable[bytes]:
    """Yield a source's raw (possibly gzipped) bytes in small bounded blocks.

    Sequential and forward-only: `.read(n)` is the only stream operation, never
    `seek` or `tell`, so a NON-SEEKABLE source — a socket, a pipe, an
    `HTTPResponse`, a boto3 `StreamingBody` — works exactly like a file. That is
    the case that motivates the header-only path at all, and the full reader
    cannot serve it cheaply because it drains to EOF before parsing anything.

    A short `read(n)` does NOT mean EOF on a network reader, so the loop stops
    only on an empty read.
    """
    if isinstance(source, (str, Path)):
        handle = Path(source).open("rb")
        try:
            while True:
                block = handle.read(_HEADER_PREFIX_CHUNK_BYTES)
                if not block:
                    return
                yield block
        finally:
            handle.close()
    elif isinstance(source, (bytes, bytearray, memoryview)):
        view = _as_byte_view(source)
        for begin in range(0, view.nbytes, _HEADER_PREFIX_CHUNK_BYTES):
            yield bytes(view[begin : begin + _HEADER_PREFIX_CHUNK_BYTES])
    else:
        while True:
            block = source.read(_HEADER_PREFIX_CHUNK_BYTES)
            if not block:
                return
            if not isinstance(block, (bytes, bytearray, memoryview)):
                # A reader that hands back something other than bytes (a mock, a
                # str, a decoded object) must fail HERE. Downstream, the envelope
                # loop grows `out` with `+=`; a non-bytes chunk turns that into
                # the chunk type's own `__radd__`, `len(out)` stops meaning
                # anything, and the loop spins forever calling `.read()` — an
                # unbounded allocation, not an error.
                raise AlfError(
                    f"stream .read() returned {type(block).__name__}, not bytes; the source must be a binary reader"
                )
            yield bytes(block) if not isinstance(block, bytes) else block


def _read_header_envelope(source: str | Path | bytes | BinaryIO, describe: str) -> bytes:
    """Read ONLY the ALF envelope: version byte, length, and the header JSON.

    Never reads the payload, and — when the source is gzipped — never inflates
    it either. The decompressor is driven with an explicit `max_length` equal to
    the bytes still needed, so the payload's compressed blocks are simply never
    fed through it. That resource property is the entire reason this function
    exists, so `tests/test_io.py::TestHeaderOnlyRead` asserts it directly rather
    than only checking that the metadata came back right.

    Two hard bounds, because a header-only read is still attacker-facing:
    the declared header length is checked against `_MAX_HEADER_BYTES` the moment
    the first five bytes are known (so a corrupt 4 GB declaration is refused
    before anything is accumulated for it), and the COMPRESSED input consumed is
    capped as well — a gzip member can be made to emit very few bytes from very
    much input, which would otherwise read the whole object to produce a header
    that never arrives.
    """
    blocks = _iter_source_blocks(source)
    out = bytearray()
    engine: Any | None = None
    gzipped: bool | None = None
    compressed_seen = 0
    needed = 5  # version + uint32 length; grows once the length is known
    try:
        while len(out) < needed:
            if engine is not None and engine.unconsumed_tail:
                # `max_length` leaves the remainder here rather than expanding
                # it. Drain it before pulling more input, or a header spanning
                # two decompressor calls stalls.
                chunk: bytes = engine.unconsumed_tail
            else:
                chunk = next(blocks, b"")
                if not chunk:
                    break
                if gzipped is None:
                    gzipped = chunk[:2] == b"\x1f\x8b"
                    if gzipped:
                        engine = zlib.decompressobj(wbits=31)
                if engine is not None:
                    compressed_seen += len(chunk)
                    if compressed_seen > _limit_for(True):
                        raise AlfError(
                            f"{describe}: read {compressed_seen} compressed bytes without "
                            f"reaching the end of the ALF header. The file is corrupt or hostile."
                        )
            if engine is None:
                out += chunk
            else:
                try:
                    out += engine.decompress(chunk, max(1, needed - len(out)))
                except zlib.error as exc:
                    # SCOPED to the inflate, not wrapped around the whole loop.
                    # A blanket `except OSError` here reported a MISSING FILE as
                    # "Corrupt gzip stream: No such file or directory" — an
                    # actively misleading diagnosis, and a divergence from
                    # `read_alf`, which lets the FileNotFoundError through. Only
                    # zlib can tell us the compressed bytes are bad.
                    raise AlfError(f"Corrupt gzip stream: {exc}") from exc
                if engine.eof and not engine.unconsumed_tail and len(out) < needed:
                    break  # the member ended before the envelope was complete
            if needed == 5 and len(out) >= 5:
                (declared,) = _HEADER_STRUCT.unpack_from(bytes(out[:5]), 1)
                if declared > _MAX_HEADER_BYTES:
                    # BEFORE accumulating anything for it. The field is a uint32,
                    # so a corrupt file can declare 4 GB of header; the full
                    # reader catches that after materializing the file, but this
                    # path would otherwise sit in the loop inflating toward it.
                    raise AlfError(
                        f"Header is too large: it declares {declared} bytes, past the "
                        f"{_MAX_HEADER_BYTES}-byte limit. "
                        f"ALF headers are small JSON objects, so this file is corrupt; decoding "
                        f"it would allocate the header twice over before anything could reject it."
                    )
                needed = 5 + declared
    finally:
        blocks.close()  # closes the file handle in the path branch
    # Truncation is reported by the SHARED envelope parser: it already words
    # "Too short to be ALF" and "Truncated header" correctly, and returning a
    # short buffer routes both through it instead of duplicating the messages.
    return bytes(out[:needed])


def _resolve_extension(source: object, extension: str) -> str:
    """The channel this read is for — explicit, or inferred from a path."""
    if extension and extension not in _KNOWN_CHANNELS:
        raise AlfError(f"Unknown ALF channel {extension!r}; expected one of {sorted(_KNOWN_CHANNELS)}.")
    if extension:
        return extension
    if not isinstance(source, (str, Path)):
        # The contract says `channel` is REQUIRED for bytes/stream input, but
        # nothing enforced it — and an empty extension disables EVERY channel
        # guard, so a raw `.all` blob could be read through `.xyz` and six label
        # bytes would decode as one plausible fp16 point. The guards are only as
        # good as knowing which channel this is.
        raise AlfError(
            "read_alf() needs an explicit `extension=` for bytes or stream input: the ALF header "
            "carries the version and metadata but NOT which channel the payload is, and without "
            "it every channel-specific accessor is unguarded."
        )
    name = Path(source).name
    if name.endswith(".gz"):
        name = name[:-3]
    inferred = Path(name).suffix.lstrip(".")
    if inferred not in _KNOWN_CHANNELS:
        # A cache file named by blob id, or any path whose suffix is not an
        # ALF channel, would otherwise leave `extension` empty and disable
        # EVERY `_require` guard -- so `.xyz` on a label payload returns one
        # plausible fp16 point instead of raising.
        raise AlfError(
            f"Cannot infer the ALF channel from {str(source)!r} (suffix {inferred!r} is not "
            f"one of {sorted(_KNOWN_CHANNELS)}); pass `extension=` explicitly."
        )
    return inferred


def _parse_envelope(raw: bytes | bytearray | memoryview, extension: str) -> tuple:
    """Validate the envelope and header JSON. Returns `(version, metadata, end)`.

    ONE implementation for both readers. Every refusal below is header-scoped —
    it is decidable from the envelope alone — which is precisely why the
    header-only path can offer the identical guarantees here and must not be
    allowed to drift into a second, laxer copy of them.
    """
    if len(raw) < 5:
        raise AlfError(f"Too short to be ALF: {len(raw)} bytes (need at least 5 for the header)")

    version = raw[0]
    if version not in SUPPORTED_VERSIONS:
        raise AlfError(f"Unsupported ALF version byte: {version} (supported: {SUPPORTED_VERSIONS})")
    # Version 2 is the interleaved LABEL layout, not a geometry version. Accepting
    # it for a `.alp` would let a mislabeled file decode as legacy float16 whenever
    # its payload happens to be a whole number of points — plausible coordinates
    # out of a file that is not geometry at all.
    if version == LABELS_V2 and extension and extension != "all":
        raise AlfError(
            f"Version byte 2 is the `.all` interleaved label layout, but this file is .{extension}. "
            f"Refusing to decode it as geometry."
        )
    # The mirror image: v3 is fixed-point GEOMETRY. On a `.ali`/`.all`/`.alc` it
    # would otherwise fall through to the v1 accessors — a version-3 `.all`
    # without `_format_version` would hand back category bytes as if nothing
    # were wrong.
    if version == ALF_VERSION_V3 and extension and extension != "alp":
        raise AlfError(
            f"Version byte 3 is fixed-point `.alp` geometry, but this file is .{extension}. "
            f"Refusing to decode it as a v1 payload."
        )

    (metadata_size,) = _HEADER_STRUCT.unpack_from(raw, 1)
    if metadata_size > _MAX_HEADER_BYTES:
        # The field is a uint32, so a corrupt or hostile file can declare up to
        # 4 GB of header — which this would then UTF-8-decode and JSON-parse
        # into strings and objects on top of the already-materialized file,
        # OOMing the worker BEFORE the frame could reach the AlfError
        # quarantine that exists to skip it. Reject the oversized declaration
        # before allocating or parsing the header.
        raise AlfError(
            f"Header is too large: it declares {metadata_size} bytes, past the "
            f"{_MAX_HEADER_BYTES}-byte limit. "
            f"ALF headers are small JSON objects, so this file is corrupt; decoding it would "
            f"allocate the header twice over before anything could reject it."
        )
    end = 5 + metadata_size
    if len(raw) < end:
        raise AlfError(f"Truncated header: declared {metadata_size} bytes, file holds {len(raw) - 5}")

    if metadata_size:
        # Corrupt bytes must arrive as AlfError like every other malformed-file
        # case. Callers quarantine bad frames by catching AlfError, so leaking a
        # UnicodeDecodeError or JSONDecodeError aborts a pipeline run instead of
        # skipping one file.
        try:
            # Python's json ACCEPTS the non-standard `NaN`/`Infinity` literals and
            # silently overflows `1e309` to inf. JSON has no such values, so every
            # other implementation rejects those headers — and here they surface
            # later as a ValueError from int(nan), or sit unnoticed in metadata a
            # consumer trusts. Refuse them at the parse, where the cause is known.
            header_text = bytes(raw[5:end]).decode("utf-8")
            _check_json_depth(header_text)
            metadata = json.loads(
                # `bytes(...)` because a memoryview has no `.decode`, and the
                # buffer branch now hands one through. This copies only the
                # HEADER, which is capped at 10 MB, not the payload.
                header_text,
                parse_constant=_reject_json_constant,
                parse_float=_checked_json_float,
                parse_int=_HeaderInt,
                object_pairs_hook=_checked_json_object,
            )
        except RecursionError as exc:
            # Deeply nested but syntactically valid JSON. Not a ValueError, so it
            # escaped as itself and aborted a pipeline whose whole quarantine
            # strategy is `except AlfError: skip_frame()`. A malformed header is
            # a malformed header regardless of which exception type the decoder
            # reaches for.
            raise AlfError(
                f"Header nesting is too deep to decode ({exc.__class__.__name__}); the frame is malformed"
            ) from exc
        except UnicodeDecodeError as exc:
            raise AlfError(f"Header is not valid UTF-8: {exc}") from exc
        except ValueError as exc:
            # Covers JSONDecodeError AND the plain ValueError CPython raises for
            # an integer literal past its digit limit — a 5000-digit value in a
            # corrupt header would otherwise escape the AlfError contract before
            # any accessor runs.
            raise AlfError(f"Header is not valid JSON: {exc}") from exc
        if not isinstance(metadata, dict):
            raise AlfError(
                f"Header must be a JSON object, got {type(metadata).__name__}. Every accessor "
                f"here does a keyed lookup, so a list or scalar root would fail later and "
                f"further from the cause."
            )
        _reject_nonfinite_metadata_numbers(metadata, extension, version)
        if extension == "alp":
            _reject_lossy_anchor_tokens(metadata)
            _validate_geometry_anchor_shapes(metadata)
            # Header-scoped, so `read_alf_header` enforces it too: which frame a
            # file's points are in is decidable from the envelope alone, and a
            # scan that reads placement metadata needs the same answer.
            _validate_frame_convention(metadata, version)
    else:
        metadata = {}
    return version, metadata, end


def _checked_label_layout(metadata: dict[str, Any]) -> int | None:
    """The declared `.all` layout, or None when the header omits it.

    HEADER-SCOPED, all of it — which is why it is a function rather than the
    first half of `format_version`. The rest of that property picks between the
    wire byte and this field using the payload's length, and a header-only read
    has no payload to pick with; this part needs none.
    """
    declared = _present_or_absent(metadata, "_format_version")
    if declared is None:
        return None
    # `int()` TRUNCATES, so a header carrying 1.5 -- a corruption, or a future
    # layout id -- would become 1 and sail through the guard below, decoding an
    # unknown payload as v1 categories. Require an exactly-integral value before
    # narrowing.
    if isinstance(declared, bool) or not isinstance(declared, (int, float)):
        raise AlfError(f"`.all` _format_version must be a number, got {declared!r}")
    _reject_non_integral_token(declared, "`.all` _format_version")
    _reject_inexact_count_token(declared, "`.all` _format_version")
    # Same reason as `_checked_count`: float() on 10**400 raises OverflowError
    # before the unsupported-layout AlfError can.
    if isinstance(declared, float):
        if not math.isfinite(declared) or not declared.is_integer():
            raise AlfError(f"`.all` _format_version must be an integer, got {declared!r}")
    declared = int(declared)
    if declared not in (LABELS_V1, LABELS_V2):
        # Fail closed. Treating an unknown layout as v1 would return the payload
        # as categories and silently drop whatever instance-like stream a newer
        # (or mistyped) version carries.
        raise AlfError(
            f"Unsupported `.all` _format_version {declared} (known: {LABELS_V1}, {LABELS_V2}). "
            f"Refusing to decode it as v1 categories."
        )
    return declared


def _check_header_scoped_channel_rules(metadata: dict[str, Any], extension: str) -> None:
    """Channel rules a header-only read must ALSO enforce.

    The envelope rules apply to every channel; these depend on which one you
    opened, and they were reachable only through the channel decoders — so
    `read_alf_header`, which reaches no decoder, accepted files `read_alf`
    refuses. That is the "validates less" drift the header-only contract exists
    to rule out, and it was present in every implementation.

    Only rules the full reader enforces EAGERLY belong here. `.alx` missing
    `_source_total_points` is refused at the accessor rather than at the read,
    so adding it would make a header scan stricter than the read it stands in
    for.
    """
    if extension == "alx" and "_total_points" not in metadata:
        raise AlfError(
            "`.alx` requires `_total_points`: a decimation sidecar indexes a "
            "different cloud, so its own payload cannot state the count."
        )
    if extension == "all":
        _checked_label_layout(metadata)


def read_alf_header(source: str | Path | bytes | BinaryIO, *, extension: str = "") -> AlfFile:
    """Parse ONLY an ALF file's envelope, without reading or inflating the payload.

    For the scan-many-frames case: a batched S3 prefetch that needs
    `_total_points`, `device_position`, `_geom_scale` or a `palette` for a
    thousand frames should not pay a full GET and a full decompression for each.
    `read_alf` drains the source to EOF and inflates everything before it returns
    anything; this reads a bounded prefix and stops, avoiding payload allocation
    when only envelope metadata is needed.

    Sources: paths, `bytes`/`bytearray`/`memoryview`, and any file-like object
    with `.read(n)`. Streams need only be FORWARD-readable — nothing here seeks
    or tells — so sockets, pipes and `botocore` streaming bodies are supported.

    A SEPARATE FUNCTION rather than `read_alf(..., header_only=True)` on purpose.
    The returned file carries a genuinely weaker guarantee: the payload-scoped
    refusals (`_check_payload_length`'s count-vs-payload cross-check, the
    concatenated-gzip-member refusal, the decompressed-size cap over the whole
    frame) cannot run, because their evidence is the bytes this deliberately did
    not read. A boolean flag would let that weaker contract be selected by a
    variable at a call site that reads as if it were the strong one; a distinct
    name makes it say so. Every HEADER-scoped refusal is unchanged and shared
    verbatim — `_parse_envelope` is the same code both readers call.

    Payload accessors on the result raise `AlfError` rather than returning
    plausible empties; `header_only` reports which kind of file this is.
    """
    extension = _resolve_extension(source, extension)
    raw = _read_header_envelope(source, _describe_source(source))
    version, metadata, _end = _parse_envelope(raw, extension)
    # EAGERLY, here rather than when someone asks. `_total_points` is decidable
    # from the header alone, so refusing it is a header-scoped rule and this
    # function claims to enforce every one of those. Leaving it to the property
    # made that claim false in a way only a caller would find: a malformed count
    # sailed through the scan and raised later, from an accessor, on whichever
    # frame of the sequence someone happened to look at closely. TypeScript's
    # reader refuses it here and this did not, which is a conformance split
    # between two implementations the support matrix calls equal.
    _checked_count(metadata, "_total_points")
    _check_header_scoped_channel_rules(metadata, extension)
    return AlfFile(
        version=version,
        metadata=metadata,
        _payload=_WithheldPayload(),
        extension=extension,
    )


def read_alf(source: str | Path | bytes | BinaryIO, *, extension: str = "") -> AlfFile:
    """Parse any ALF channel from a path, bytes, or stream.

    `extension` ("alp" / "ali" / "all" / "alc") is only needed for accessors that
    depend on it; it is inferred from a path when not given.

    Reads the WHOLE source. `read_alf_header` is the cheap alternative when only
    the envelope is wanted, and documents which refusals it cannot make.
    """
    extension = _resolve_extension(source, extension)
    raw = _open_maybe_gzip(source)
    version, metadata, end = _parse_envelope(raw, extension)
    # A memoryview, NOT a slice copy. `raw[end:]` duplicates almost the whole
    # buffer while `raw` is still live, so a frame near the decompression cap
    # transiently needed ~2x the payload on top of the compressed input — the
    # bounded path OOMing a constrained worker on its way to succeeding. Every
    # consumer of `_payload` is a bytes-like operation (len, %, slicing,
    # np.frombuffer), so a view costs nothing and copies nothing.
    # READ-ONLY, and the same for compressed and uncompressed input. Switching
    # the gzip branch to a bytearray made its memoryview WRITABLE, so every
    # zero-copy accessor built on `np.frombuffer` — `intensity`, v1
    # `categories`, `palette_indices`, `kept_indices` — handed back mutable
    # storage: `alf.intensity[0] = 123` silently rewrote the payload and
    # changed later reads. Worse, the same statement RAISED on an uncompressed
    # file, so whether a decoded array could be corrupted depended on how the
    # object happened to be stored.
    payload_view = memoryview(raw)[end:].toreadonly()
    alf = AlfFile(version=version, metadata=metadata, _payload=payload_view, extension=extension)
    _check_payload_length(alf, source)
    return alf


# Bytes per point, by channel and layout. `.alc` is one palette index per point.
_BYTES_PER_POINT = {"ali": 1, "alc": 1, "alx": 4}


def _check_payload_length(alf: AlfFile, source: object) -> None:
    """Cross-check the payload against the header's own point count.

    Channels are separate objects that are only correlated BY INDEX: point i in
    `.alp` is described by byte i of `.ali` and `.all`. A file truncated (or
    extended) by whole points still decodes into a valid-looking array of the
    wrong length, and the misalignment then shows up as labels or intensities
    attached to the wrong geometry — with nothing raised anywhere. The header
    already states the answer, so checking it is nearly free.
    """
    declared = alf.total_points
    if declared is None and alf.extension == "alx":
        # `.alx` is new here, so there is no legacy no-count case to preserve —
        # and without the count the ONLY check that the sidecar has one mapping
        # per point in its decimated `.alp` silently does not run, leaving a
        # one-entry sidecar pairable with a full cloud.
        raise AlfError(
            "`.alx` is missing `_total_points`; without it the mapping length cannot be checked "
            "against the geometry it accompanies."
        )
    if declared is None or not alf.extension:
        return
    if alf.extension == "alp":
        per_point = 6  # 3 components x 2 bytes, both codecs
    elif alf.extension == "all":
        per_point = 3 if alf.format_version == LABELS_V2 else 1
    else:
        per_point = _BYTES_PER_POINT.get(alf.extension, 0)
    if not per_point:
        return
    actual = len(alf._payload)
    if declared > actual:
        # Bound the count against the payload BEFORE any arithmetic reaches a
        # message. A JSON integer has no width limit, so a 4300-digit
        # `_total_points` multiplies into a 4301-digit `expected` and
        # interpolating it trips CPython's integer-to-string limit — a bare
        # ValueError escaping a caller whose quarantine is
        # `except AlfError: skip_frame()`. No count larger than the payload can
        # be valid anyway, so this is the honest check as well as the safe one.
        raise AlfError(
            f"`_total_points` declares more points than the {actual}-byte payload can hold, "
            f"so this file's count and payload contradict each other"
        )
    expected = declared * per_point
    if actual != expected:
        raise AlfError(
            f"{_describe_source(source)}: header declares {declared} points ({expected} payload "
            f"bytes for .{alf.extension}), "
            f"but the payload holds {actual} bytes."
            + (
                # LEGACY NAMING: two legacy writers store semantic segmentation
                # LABELS as `.alp.gz` -- their own headers say so outright --
                # so path inference calls them
                # geometry and they land here with a length error that names the
                # wrong cause. The reader CANNOT distinguish that from a
                # truncated `.alp`: a 2-point frame missing one point is 6 bytes,
                # exactly the v2 label stride. So name both possibilities rather
                # than guess, and never infer the channel from the payload.
                f" NOTE: {actual} bytes is also what a `.all` LABEL file of "
                f"{declared} points looks like, and some legacy writers store segmentation "
                f'labels as `.alp.gz` — if this is one of those, pass `extension="all"` '
                f"explicitly."
                if alf.extension == "alp" and actual in (declared, declared * 3)
                else ""
            )
            + " Channels are correlated by index, so a length "
            "mismatch misaligns geometry against intensity and labels."
        )


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def _write_channel(path: str | Path, version: int, metadata: dict[str, Any], payload: bytes) -> None:
    """Serialize one ALF channel. Single place the envelope is produced."""
    _check_writer_json_depth(metadata)
    _validate_portable_writer_metadata(metadata, path)
    try:
        # `allow_nan=False`: Python would otherwise emit bare NaN/Infinity tokens,
        # which are NOT valid JSON. Python reads them back happily, so the file
        # looks fine here and fails only in the browser and TypeScript readers —
        # the worst place to discover it. `write_ali` records raw intensity
        # bounds straight from the caller, which is one way an infinity arrives.
        header = json.dumps(metadata, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except ValueError as exc:
        raise ValueError(
            f"{path}: header contains a non-finite value (NaN/Infinity), which is not valid JSON and is "
            f"rejected by the browser and TypeScript readers: {exc}"
        ) from exc
    envelope = 5 + len(header) + len(payload)
    if envelope > _limit_bytes():
        # `read_alf` refuses anything past this during decompression, so
        # publishing it produces a file the paired reader cannot consume — the
        # writer already honours `_MAX_HEADER_BYTES` for exactly this reason and
        # this is the same rule for the whole envelope.
        raise ValueError(
            f"{path}: the {envelope}-byte envelope is past the "
            f"{_limit_bytes() // (1024 * 1024)} MB limit `read_alf` enforces. Writing "
            f"it would produce a file this module cannot read back; split the frame."
        )
    if len(header) > _MAX_HEADER_BYTES:
        # `read_alf` refuses a header past this limit unconditionally, so
        # publishing one here emits a file the paired reader classifies as
        # corrupt — `write_ali(metadata={"blob": "x" * 10_000_000})` gzipped
        # down to something small and unremarkable and then failed to load.
        raise ValueError(
            f"{path}: serialized header is {len(header)} bytes, past the "
            f"{_MAX_HEADER_BYTES}-byte limit `read_alf` enforces. Writing it would produce a "
            f"file this module cannot read back; move the bulk out of the header."
        )
    path = Path(path)
    # NO eager unlink here. `os.replace` below overwrites atomically, so nothing
    # stale can survive a successful write — and deleting first opened a window
    # where a CONCURRENT writer's freshly published file was destroyed and the
    # path sat empty for the whole compression, or permanently if this call then
    # failed. Clearing stale output on REJECTION is handled at entry to each
    # channel writer, which is where a rejection can actually happen.
    # Publish ATOMICALLY. These channels run to hundreds of MB, and a kill, a full
    # disk or any mid-stream I/O error would otherwise leave a truncated `.gz` at
    # the final path — which every consumer here finds by globbing a sequence
    # directory, so a half-written frame reads as current output rather than as a
    # failure. The temp file is a sibling so `os.replace` stays on one filesystem
    # (and therefore atomic), and it is cleaned up if anything goes wrong.
    #
    # `filename=""` and `mtime=0` keep the COMPRESSED bytes a pure function of the
    # ALF content. Passing a path to GzipFile stamps its basename into the gzip
    # header, so the same frame written to two names would differ byte-for-byte —
    # which would quietly defeat the golden-file corpus this writer exists to make
    # checkable. The temp name must therefore never reach GzipFile either.
    # Unique per call, not per process. A PID-only name is shared by every
    # THREAD in one process: two concurrent writes to the same channel path
    # opened the same temp file with truncation, and the first `os.replace`
    # unlinked that name — so the loser either raised FileNotFoundError at its
    # own replace or kept writing through a live descriptor into the file that
    # had already been published, exposing a half-written gzip stream to
    # readers. `uuid4` costs nothing next to the payload write.
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}.{uuid.uuid4().hex}")
    try:
        with open(tmp, "wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as f:
                f.write(struct.pack("B", version))
                f.write(_HEADER_STRUCT.pack(len(header)))
                f.write(header)
                f.write(payload)
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _clear_stale_output(path: str | Path) -> tuple | None:
    """Fingerprint any existing file at `path` so a later failure can clear it.

    A rejected re-run must not leave the previous frame behind for downstream
    globbing to pick up as current — but unlinking EAGERLY destroys a
    concurrent writer's freshly published output and empties the path for the
    duration of this call. So nothing is removed up front; the caller records
    the fingerprint and removes the file only if a failure occurs AND the file
    is still byte-identically the one that was there when it started.
    """
    try:
        st = Path(path).stat()
    except (OSError, ValueError):
        return None
    return (st.st_ino, st.st_mtime_ns, st.st_size)


def _drop_if_unchanged(path: str | Path, fingerprint: tuple | None) -> None:
    """Remove `path` only if it still matches `fingerprint`.

    A mismatch means someone else published while we were working, and their
    valid output is not ours to delete.

    NOT ATOMIC, and knowingly so. A writer that publishes between the stat here
    and the unlink below still loses its file. POSIX has no compare-and-unlink —
    `unlink` takes a path, not a descriptor, and Linux has no `funlinkat` — so
    no amount of tightening removes that window; only serializing same-path
    writes does, which is a locking decision for the maintainers (see PR #2805).
    This narrows the race from "always" to "a scheduling coincidence"; it does
    not close it.
    """
    if fingerprint is None:
        return
    if _clear_stale_output(path) == fingerprint:
        Path(path).unlink(missing_ok=True)


def _clears_stale_output_on_failure(fn):
    """Remove a REJECTED write's prior output, without racing a concurrent one.

    Every channel writer used to `unlink` at entry, which satisfied "a rejected
    re-run leaves nothing stale" by also destroying whatever a parallel writer
    had just published. Deferring it to the failure path keeps the guarantee and
    closes the window, and `_drop_if_unchanged` declines to delete a file that
    someone else replaced while we were working.
    """

    channel = fn.__name__.replace("write_", "")

    @functools.wraps(fn)
    def wrapper(path, *args, **kwargs):
        _require_destination(path, channel)
        fingerprint = _clear_stale_output(path)
        try:
            return fn(path, *args, **kwargs)
        except BaseException:
            _drop_if_unchanged(path, fingerprint)
            raise

    return wrapper


def _require_destination(path: str | Path, channel: str) -> None:
    """Refuse a destination whose suffix names a DIFFERENT ALF channel.

    `read_alf` infers the channel from the filename, so `write_ali(path,
    ...)` with a `.all.gz` destination publishes intensity bytes that read back
    as category ids. Both are version 1, one byte per point, with the same
    `_total_points` — so every integrity check in this module passes and the
    mix-up surfaces as plausible labels. The inverse mistake is equally silent.

    An UNRECOGNISED suffix is left alone: cache files named by blob id are a
    real pattern here, and `read_alf` already handles them via `extension=`.
    """
    name = Path(path).name
    if name.endswith(".gz"):
        name = name[:-3]
    suffix = Path(name).suffix.lstrip(".")
    if suffix in _KNOWN_CHANNELS and suffix != channel:
        raise ValueError(
            f"{path}: this is the .{channel} writer but the destination names a .{suffix} "
            f"file. `read_alf` infers the channel from the filename, so this would publish "
            f"{channel} data that reads back as {suffix} data."
        )


def _as_xyz(local_xyz: np.ndarray, path: str | Path) -> np.ndarray:
    """Coerce input to (N, 3), refusing shapes that would silently reinterpret.

    A bare `.reshape(-1, 3)` accepts an (N, 4) array — the common "XYZ plus
    intensity" or "XYZ plus ring" matrix — and folds it into (4N/3, 3), turning
    intensity readings into coordinates and inventing a point count that no
    longer matches the source. A ported writer that hands over its full point
    array would corrupt every frame with nothing raised, so the shape is checked
    rather than assumed.
    """
    if _contains_bool(local_xyz):
        raise ValueError(
            f"{path}: coordinates must be numeric, not booleans — a mask or comparison result "
            f"would be written as 0.0/1.0 points"
        )
    if isinstance(local_xyz, np.ma.MaskedArray) and np.ma.getmaskarray(local_xyz).any():
        # `np.asarray` DISCARDS the mask and exposes the underlying values, so a
        # row the caller explicitly marked invalid is written as an ordinary
        # point — counted in `_total_points`, folded into the box, and shifting
        # every index-correlated label and intensity stream after it. The mask
        # is the caller's statement that those points are not data.
        raise ValueError(
            f"{path}: coordinates are a masked array. `np.asarray` drops the mask and would "
            f"publish the masked rows as real points; select or fill them explicitly "
            f"(`arr.compressed()`, `arr.filled(...)`) so the intent is in the caller's code."
        )
    raw = np.asarray(local_xyz)
    if _is_stringly_typed(raw):
        # Ahead of the general dtype gate, which would also refuse these but
        # with a message about datetimes. The parser-bug wording below is the
        # one a caller staring at an unparsed CSV column needs.
        if _holds_actual_strings(raw):
            raise ValueError(
                f"{path}: coordinates are strings, not numbers (dtype {raw.dtype!r}). A numeric "
                f"string parses cleanly into a valid-looking point and hides the parser bug that "
                f"produced it; convert with an explicit float cast at the source."
            )
        raise ValueError(
            f"{path}: coordinates are non-numeric objects (dtype {raw.dtype!r}). `Decimal`, "
            f"`Fraction` and generators all arrive as an object array NumPy cannot narrow "
            f"without deciding the precision itself; convert with an explicit float cast at "
            f"the source."
        )
    _reject_non_real(raw, "coordinates", path)
    # The ORIGINAL object first, then the array. `raw` has already been through
    # `np.asarray`, which promotes a mixed Python sequence to float64 and
    # rounds away the very integer this is looking for.
    lossy = _lossy_before_promotion(local_xyz) or _narrows_lossily_to_float64(raw)
    if lossy:
        raise ValueError(
            f"{path}: coordinates cannot narrow to float64 without merging distinct points "
            f"({lossy}). Re-center the cloud, or convert at the source."
        )
    if raw.dtype == np.bool_:
        # `astype(float64)` turns a mask or a comparison result into 0.0/1.0
        # COORDINATES, and geometry is the channel every label and intensity
        # stream is indexed against — so a shape-correct mask would publish a
        # valid-looking cloud that silently defines the frame everything else
        # aligns to.
        raise ValueError(
            f"{path}: coordinates must be numeric, not booleans — a mask or comparison result "
            f"would be written as 0.0/1.0 points"
        )
    if _is_stringly_typed(raw):
        # `np.asarray([["1","2","3"]]).astype(float64)` succeeds and publishes a
        # point at (1, 2, 3). A CSV column or an object array that never got
        # parsed is a PARSER bug, and geometry is the index base every label and
        # intensity channel aligns to — so the frame that hides it corrupts the
        # whole sequence. The header validators already refuse numeric strings
        # (`_is_json_number`); the payload gets the same rule.
        if _holds_actual_strings(raw):
            raise ValueError(
                f"{path}: coordinates are strings, not numbers (dtype {raw.dtype!r}). A numeric "
                f"string parses cleanly into a valid-looking point and hides the parser bug that "
                f"produced it; convert with an explicit float cast at the source."
            )
        raise ValueError(
            f"{path}: coordinates are non-numeric objects (dtype {raw.dtype!r}). `Decimal`, "
            f"`Fraction` and generators all arrive as an object array NumPy cannot narrow "
            f"without deciding the precision itself; convert with an explicit float cast at "
            f"the source."
        )
    xyz = raw.astype(np.float64)
    if xyz.size == 0:
        # Genuinely-empty shapes only. `np.empty((5, 0))` is a parser that
        # selected NO coordinate columns from five records: it has size 0, so a
        # bare size check reshapes it to (0, 3) and `write_alp` reports
        # `_total_points: 0` — five records discarded, exit status success. An
        # empty frame is legitimate; five records becoming an empty frame is the
        # missing-XYZ-columns bug this must surface.
        if xyz.ndim == 2 and xyz.shape[0] and xyz.shape[1] != 3:
            raise ValueError(
                f"{path}: coordinate array has shape {raw.shape} — {xyz.shape[0]} record(s) with "
                f"{xyz.shape[1]} coordinate column(s), not 3. Writing this as an empty frame would "
                f"discard every record; check the column selection that produced it."
            )
        return xyz.reshape(0, 3)
    if xyz.ndim == 1:
        if xyz.size % 3:
            raise ValueError(f"{path}: flat coordinate array of length {xyz.size} is not a whole number of XYZ points")
        return xyz.reshape(-1, 3)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(
            f"{path}: expected (N, 3) XYZ, got {xyz.shape}. Pass only coordinates — an (N, 4) array "
            f"(XYZ plus intensity or ring) would be silently reinterpreted as {xyz.size // 3} points."
        )
    return xyz


def _require_finite_geometry(local_xyz: np.ndarray, label: str) -> np.ndarray:
    """Coerce to (N, 3) and refuse non-finite coordinates.

    The public geometry helpers are entry points, so they answer for their own
    input. `write_alp` refuses NaN/Inf separately, which left these two
    returning a `nan` offset and an `inf` scale to a direct caller — numbers
    that then poison every coordinate derived from them without raising.
    """
    coords = _as_xyz(local_xyz, label)
    if coords.size and not np.isfinite(coords).all():
        raise ValueError(
            f"{label}: coordinates contain NaN or Inf, so the value derived from them would "
            f"be non-finite too; mask the invalid rows first. These helpers SIZE the int16 "
            f"fixed-point grid, and a grid derived from a slot that holds no position is a "
            f"wrong grid. The v3 payload itself can carry an invalid return — `write_alp` "
            f"encodes it as the reserved marker, masking exactly this way internally "
            f"(spec/format.md §5.2)."
        )
    return coords


def choose_geom_offset(local_xyz: np.ndarray) -> np.ndarray:
    """Per-axis bounding-box midpoint — the origin the int16 grid is built around.

    Without this the grid is centred on zero and the scale comes from the frame's
    largest ABSOLUTE coordinate, which silently wastes range on any frame that is
    not symmetric about the origin. A frame spanning x = 100..200 m would derive
    its step from 200 m rather than from its own 50 m half-span, quadrupling the
    quantization error for geometry that fits comfortably in the format.

    Ego-centric lidar is roughly origin-centred already, so this changes little
    for the common case — but the aerial tile writers are exactly the offset,
    one-sided frames it matters for, and `_geom_offset` exists in the wire format
    precisely so the encoder can use it. Hard-coding it to zero left the header
    field present and permanently useless.

    Returned as float64 and written verbatim into the header, so the decode
    (`stored * scale + offset`) reproduces it exactly.

    Shape-validated because this is a PUBLIC entry point now. Reducing over
    axis 0 turns the flat `[1, 2, 3]` single point that `write_alp` accepts
    into the SCALAR 2, and an (N, 4) XYZ-plus-intensity matrix into four
    components — both of which then broadcast through centring and produce
    wrong geometry without raising.
    """
    local_xyz = _as_xyz(local_xyz, "choose_geom_offset")
    if local_xyz.size == 0:
        return np.zeros(3, dtype=np.float64)
    # REAL returns only. An invalid return is valid content that v3 stores as a
    # reserved word, so it must not size the grid.
    #
    # `nanmin`/`nanmax` were the obvious way to skip them in place, but they
    # skip only NaN — an INFINITE invalid return survived the reduction and made
    # `hi` infinite, which then took the "no position to centre on" branch and
    # returned a zero origin for a frame that had plenty of real positions. A
    # frame near 1e9 with one Inf return was refused outright ("step 30518 m
    # gives 15259 m error"), while the same frame with a NaN return encoded
    # fine. Two spellings of "no return", opposite outcomes.
    #
    # Reducing in CHUNKS rather than over `local_xyz[valid]`, because that
    # filter copies the whole frame in the common all-valid case and the
    # writer's memory bound measures exactly that. Extra memory here is one
    # chunk, not one frame.
    lo = np.full(3, np.inf, dtype=np.float64)
    hi = np.full(3, -np.inf, dtype=np.float64)
    saw_real_return = False
    for begin in range(0, len(local_xyz), 262_144):
        chunk = local_xyz[begin : begin + 262_144]
        keep = np.isfinite(chunk).all(axis=1)
        if not keep.any():
            continue
        real = chunk[keep]
        saw_real_return = True
        np.minimum(lo, real.min(axis=0), out=lo)
        np.maximum(hi, real.max(axis=0), out=hi)
    if not saw_real_return:
        # Every return in the frame was invalid: there is no position to centre
        # on, and zero is as good an origin as any.
        return np.zeros(3, dtype=np.float64)
    # `(lo + hi) / 2` overflows when both ends sit near the same float64
    # extreme, giving an infinite offset that then rejects a frame which is
    # exactly representable — a CONSTANT frame at float64 max needs that very
    # coordinate as its offset and an all-zero payload. `lo + (hi - lo) / 2`
    # is exact whenever the span is representable, which covers the same-sign
    # case; the halved form covers the opposite-sign case where the span is
    # what overflows.
    with np.errstate(over="ignore", invalid="ignore"):
        # The overflow IS the branch condition — `np.isfinite(span)` selects the
        # halved form for exactly the frames where this subtraction cannot be
        # represented, so NumPy's warning is describing the check working.
        span = hi - lo
        mid = np.where(np.isfinite(span), lo + span / 2.0, lo / 2.0 + hi / 2.0)
    return np.asarray(mid, dtype=np.float64)


def choose_geom_scale(local_xyz: np.ndarray) -> float:
    """Pick the int16 step that uses the full range for this frame's extent.

    Expects coordinates ALREADY CENTRED on `choose_geom_offset` — it derives the
    step from the largest absolute value it is handed, so passing raw one-sided
    coordinates reproduces the very waste the offset exists to remove.

    Deliberately per-frame: a fixed global scale would either waste range on
    small frames or clip large ones. The resulting error (`scale/2`) is recorded
    in the header, so a consumer never has to infer it.
    """
    local_xyz = _as_xyz(local_xyz, "choose_geom_scale")
    widest = _finite_max_abs(local_xyz)
    if widest is None:
        return MIN_GEOM_SCALE_M
    return _scale_from_extent(widest)


def _scale_from_extent(largest: float) -> float:
    """The step for a known centred extent — the rule, without the coercion.

    `choose_geom_scale` validates its input because it is a public entry point;
    the writer and the error report have already produced a validated extent
    blockwise and would otherwise have to rebuild an (N, 3) array to ask this
    question. One rule, two doors: re-deriving `largest / 32767` at either call
    site is what made the error bound wrong four times.
    """
    if largest == 0.0:
        return MIN_GEOM_SCALE_M
    return max(largest / _INT16_MAX, MIN_GEOM_SCALE_M)


def _writer_frame_fields(
    codec: str,
    frame_convention: str | None,
    frame_rotation: Any | None,
    path: str | Path,
) -> dict[str, Any]:
    """Validate the caller's frame declaration and return the header it becomes.

    NORMALIZATION LIVES HERE, and only here. A pose quaternion that has been
    through a few compositions is a unit quaternion to about 1e-7, not to the
    1e-9 the reader requires, so a writer that passed it through untouched would
    emit files its own reader refuses. The writer knows what the caller meant and
    can normalize; a reader knows only what it was handed, and normalizing there
    would turn a producer's bug into a silent 10% scale on every coordinate.

    What the writer does NOT absorb is a quaternion that is not nearly unit
    already. Drift is a small number; |q| = 2 is a units or ordering mistake, and
    normalizing it would produce a confident rotation from a value that never
    described one.
    """
    if codec != GEOM_INT16:
        if frame_convention is not None or frame_rotation is not None:
            raise ValueError(
                f"{path}: frame_convention/frame_rotation are version-3 fields, and this write "
                f"is codec={codec!r} (version 1). Readers apply no rotation to v1 geometry, so "
                f"recording one would be a promise the format does not keep. Pass "
                f'codec="int16" to state a frame convention.'
            )
        return {}

    if frame_convention is None:
        raise ValueError(
            f"{path}: codec={GEOM_INT16!r} requires frame_convention. `local_xyz` is the world "
            f"points with an origin subtracted, and that is half of a rigid transform — the "
            f"file has to say whether you also rotated them into the device frame. Pass "
            f"{FRAME_WORLD_ALIGNED!r} if you only subtracted the origin, or "
            f"{FRAME_DEVICE_ALIGNED!r} with frame_rotation if you rotated as well."
        )
    if frame_convention not in FRAME_CONVENTIONS:
        raise ValueError(
            f"{path}: unknown frame_convention {frame_convention!r} (expected one of "
            f"{', '.join(repr(c) for c in FRAME_CONVENTIONS)})."
        )

    if frame_convention == FRAME_WORLD_ALIGNED:
        if frame_rotation is not None:
            raise ValueError(
                f"{path}: frame_convention={FRAME_WORLD_ALIGNED!r} means no rotation was "
                f"applied, so a frame_rotation cannot also be recorded. If you did rotate the "
                f"points, the convention is {FRAME_DEVICE_ALIGNED!r}."
            )
        return {"_frame_convention": FRAME_WORLD_ALIGNED}

    if frame_rotation is None:
        raise ValueError(
            f"{path}: frame_convention={FRAME_DEVICE_ALIGNED!r} requires frame_rotation — the "
            f"rotation that returns these points to world axes. Without it the frame cannot be "
            f"placed at all."
        )
    if not isinstance(frame_rotation, dict):
        raise ValueError(
            f"{path}: frame_rotation must be a mapping with w, x, y and z, got "
            f"{type(frame_rotation).__name__}. Naming the members is deliberate: every array "
            f"spelling has to choose between [x, y, z, w] and [w, x, y, z], and the two are "
            f"indistinguishable once written."
        )
    missing = sorted({"w", "x", "y", "z"} - set(frame_rotation))
    if missing:
        raise ValueError(f"{path}: frame_rotation is missing {', '.join(missing)}.")
    extra = sorted(set(frame_rotation) - {"w", "x", "y", "z"})
    if extra:
        raise ValueError(
            f"{path}: frame_rotation has unexpected members {', '.join(extra)}; exactly w, x, y and z are allowed."
        )

    values = {}
    for axis in ("w", "x", "y", "z"):
        component = frame_rotation[axis]
        # `np.integer` alongside the floats, because this is a NumPy writer and
        # the identity quaternion is most naturally spelled from an integer
        # array: `dict(zip("wxyz", np.array([1, 0, 0, 0])))` hands over
        # `np.int64` components, which are numbers by every other validator in
        # this module. Booleans stay out — `np.bool_` is an instance of neither
        # `bool` nor `np.integer`, so it falls through the type check rather
        # than needing its own clause, and `True` would otherwise normalize into
        # a rotation.
        if isinstance(component, bool) or not isinstance(component, (int, float, np.integer, np.floating)):
            raise ValueError(f"{path}: frame_rotation.{axis} must be a number, got {component!r}.")
        number = float(component)
        if not math.isfinite(number):
            raise ValueError(f"{path}: frame_rotation.{axis} must be finite, got {component!r}.")
        values[axis] = number

    norm = math.sqrt(sum(value * value for value in values.values()))
    if norm == 0.0:
        raise ValueError(
            f"{path}: frame_rotation is the zero quaternion, which describes no rotation and "
            f"cannot be normalized into one."
        )
    # Drift is small. Anything else is a different number than the caller thinks
    # they have, and scaling it to length 1 would manufacture a plausible
    # rotation out of it.
    if not 0.9 <= norm <= 1.1:
        raise ValueError(
            f"{path}: frame_rotation has |q| = {norm:.6g}, too far from 1 to be accumulated "
            f"drift. A quaternion this far off is usually a component ordering or a units "
            f"mistake, and normalizing it would turn that into a confident wrong rotation."
        )
    return {
        "_frame_convention": FRAME_DEVICE_ALIGNED,
        "_frame_rotation": {axis: value / norm for axis, value in values.items()},
    }


@_clears_stale_output_on_failure
def write_alp(
    path: str | Path,
    local_xyz: np.ndarray,
    metadata: dict[str, Any] | None = None,
    *,
    device_position: Any | None = None,
    codec: str = GEOM_FP16,
    check_budget: bool = True,
    frame_convention: str | None = None,
    frame_rotation: Any | None = None,
) -> dict[str, Any]:
    """Write a `.alp.gz`. `local_xyz` MUST already be local to `device_position`.

    Returns the header actually written, so callers can log or assert on the
    recorded error bound.

    The `box` header is computed HERE, from the same array that gets encoded —
    the bug class where bounds described world coordinates while the payload held
    local ones becomes unrepresentable rather than fixed once per writer.
    """
    # Drop any prior output BEFORE any validation — including the shape check
    # below. A rejected re-run must not leave the previous frame at this path
    # for downstream globbing to pick up as if the write had succeeded, and the
    # very first thing that can reject is the coercion.

    # Validated here, before the encode, so a frame declaration that cannot be
    # honoured costs nothing beyond the decorator's unlink. It is deliberately
    # INSIDE the decorated body rather than in front of it: everything that can
    # reject has to sit where the stale-output guard can see it fail.
    frame_fields = _writer_frame_fields(codec, frame_convention, frame_rotation, path)

    xyz = _as_xyz(local_xyz, path)
    header: dict[str, Any] = dict(metadata or {})
    header["_total_points"] = len(xyz)
    # A caller passing metadata read back from another file would otherwise
    # inherit a stale declaration and publish it over a payload it does not
    # describe. The explicit arguments own these two keys.
    header.pop("_frame_convention", None)
    header.pop("_frame_rotation", None)
    if device_position is not None:
        # The EXPLICIT argument replaces inherited anchor AXES while preserving
        # non-axis application metadata in that object. A rewrite that
        # re-localises a payload while passing the previous header through would
        # otherwise translate the new coordinates by the OLD origin if merge
        # order happened to favour the inherited value — `world_xyz()` would
        # then place the cloud wherever the last frame was.
        anchor = _as_anchor(device_position, "device_position")
        # An inherited `world_centroid_epsg3857` OUTRANKS device_position in
        # `world_xyz`, so carrying it through would leave the frame translated by
        # the old aerial origin despite the explicit replacement — the cloud
        # lands kilometres away and the argument that was supposed to fix it is
        # silently ignored. Replacing the device axes also removes that
        # competing anchor; it does not discard application-owned members of
        # the device object.
        header.pop("world_centroid_epsg3857", None)
        inherited_anchor = header.get("device_position")
        preserved_members = (
            {key: member for key, member in inherited_anchor.items() if key not in {"x", "y", "z"}}
            if isinstance(inherited_anchor, dict)
            else {}
        )
        if isinstance(device_position, dict):
            # The explicit public argument may itself be the application-owned
            # x/y/z object. Preserve its non-axis members too; requiring the
            # caller to duplicate them under ``metadata`` made provenance such
            # as GPS quality silently disappear on an otherwise valid write.
            preserved_members.update(
                {key: member for key, member in device_position.items() if key not in {"x", "y", "z"}}
            )
        preserved_members.update(
            {
                "x": float(anchor[0]),
                "y": float(anchor[1]),
                "z": float(anchor[2]),
            }
        )
        header["device_position"] = preserved_members

    else:
        # No explicit anchor: whatever the pass-through header carries becomes
        # this file's anchor, so it has to survive the SAME validator the
        # accessors will run. An inherited `device_position: null` or
        # `world_centroid_epsg3857: {"x": 1}` from an external header otherwise
        # writes clean and raises on the first `.device_position`/`.world_xyz`
        # of the file just produced. Validate in place rather than dropping:
        # silently discarding an anchor a caller believed they were preserving
        # would relocate the cloud to the origin, which is worse than refusing.
        for key, label in (
            ("device_position", "device_position"),
            ("world_centroid_epsg3857", "world_centroid_epsg3857"),
        ):
            if key in header:
                if key == "device_position" and isinstance(header[key], dict) and not header[key]:
                    # `{}` is the legacy/default shape real producers emit, and
                    # `AlfFile.device_position` reads it as a zero origin —
                    # refusing it means `write_alp(..., metadata=alf.metadata)`
                    # cannot rewrite a file the shared reader accepts.
                    #
                    # `device_position` ONLY. This exemption was first written
                    # for both loop entries, but `{}` means "zero origin" for a
                    # sensor pose and NOTHING for an aerial centroid: `world_xyz`
                    # requires all three axes of `world_centroid_epsg3857`, so
                    # exempting it there just moved the writer/reader
                    # disagreement one key across.
                    continue
                try:
                    # NORMALIZE, not just validate. A NumPy `(3,)` anchor passes
                    # `_as_anchor` and then dies in `json.dumps` as
                    # "Object of type ndarray is not JSON serializable" — so
                    # merely fixing the ambiguous-truth-value crash moved the
                    # failure one line down. The explicit-argument path already
                    # writes the `{x, y, z}` mapping; the inherited path now
                    # produces the same thing from whatever shape it was given.
                    header[key] = _normalized_anchor(header[key], label)
                except AlfError as exc:
                    raise ValueError(
                        f"{path}: inherited `{key}` is unusable ({exc}). Pass an explicit "
                        f"device_position, or drop the key from `metadata` — writing it through "
                        f"produces a file whose own `.{label}` accessor raises."
                    ) from exc

    if len(xyz) and codec == GEOM_INT16 and not np.isfinite(xyz).all():
        # Fixed-point CAN represent an invalid return, through a reserved word.
        #
        # A scanning lidar can encode a DENSE range image with A azimuth samples
        # per beam: point `i` is beam `i // A`, azimuth `i % A`, and every
        # companion channel is indexed the same way. Dropping invalid returns would
        # destroy the beam/azimuth mapping AND desynchronise `.ali` and `.all`
        # against the geometry, which is the one invariant this format has.
        #
        # `INT16_NO_RETURN` (-32768) costs nothing to reserve: `choose_geom_scale`
        # derives the step as `largest / 32767`, so the encoding is deliberately
        # SYMMETRIC and -32768 was already unreachable. Two's complement gives it
        # no positive counterpart, which is exactly what a sentinel wants.
        invalid = ~np.isfinite(xyz)
        whole_point = invalid.all(axis=1)
        if not np.array_equal(invalid.any(axis=1), whole_point):
            # A partly-invalid point is not a shape any measured frame has, and
            # it has no meaning: half a coordinate is not a position. Refusing is
            # better than inventing a rule for data that does not exist.
            raise ValueError(
                f"{path}: codec={GEOM_INT16!r} found points with SOME but not all components "
                f"non-finite. An invalid return is a whole point — a partial one has no "
                f"interpretation, and encoding it would have to guess. Make each such point "
                f"entirely non-finite, or filter it out."
            )

    if len(xyz) and check_budget and codec == GEOM_FP16:
        # Only float16 has a magnitude-dependent budget to blow; int16 states
        # its own error in the header regardless of extent.
        # FINITE components only: `np.max` over a frame with one invalid return
        # is `nan`, and every comparison inside the guard is False against `nan`
        # — so the guard accepted anything at all on the frames it most needed
        # to check. `None` means the frame is entirely invalid returns, which has
        # no magnitude to judge.
        widest = _finite_max_abs(xyz)
        if widest is not None:
            check_fp16_component_budget(widest, str(path))

    if codec == GEOM_INT16:
        offset_vec = choose_geom_offset(xyz)
        # The centred EXTENT, without materializing the centred array. `centered`
        # plus `choose_geom_scale`'s own `np.abs` temporary were two more
        # full-size float64 arrays on top of the caller's input and the
        # `_as_xyz` copy — ~1 GB at the documented 10M-point scale, before
        # quantization had started. `geom_error_report` got this same treatment
        # two rounds ago; the writer did not.
        #
        # `choose_geom_scale` derives the step from `max(abs(...))` alone, so it
        # is handed that one value. Inlining the arithmetic instead would be
        # another independent derivation of a rule that already has one home.
        # FINITE components only, for the same reason the fp16 budget check
        # above says so: `np.max` over a block holding an invalid return is
        # `nan`, and `max(0.0, nan)` is `0.0` — so a frame full of real points
        # sized itself from nothing and fell back to `MIN_GEOM_SCALE_M`, which
        # then clipped every genuine coordinate to +/-32767. The fp16 path had
        # already been bitten by this; the int16 path only avoided it by
        # refusing invalid returns outright, which it no longer does.
        largest_centered = 0.0
        for begin in range(0, len(xyz), 262_144):
            block = xyz[begin : begin + 262_144] - offset_vec
            block_max = _finite_max_abs(block)
            if block_max is not None:
                largest_centered = max(largest_centered, block_max)
        scale = _scale_from_extent(largest_centered)
        if check_budget and scale / 2.0 >= INT16_ERROR_BUDGET_M:
            raise ValueError(
                f"{path}: int16 step {scale:.4f} m gives {scale / 2.0:.3f} m error, at or past the "
                f"{INT16_ERROR_BUDGET_M} m budget. The frame spans ~{largest_centered * 2:.0f} m, "
                f"which for an ego-centric frame means the localization failed; tile large scenes instead."
            )
        offset = [float(v) for v in offset_vec]
        # CHUNKED, like the reader's error bound. Holding `xyz`, `centered`, the
        # rint/clip temporaries, `decoded` and `decoded - xyz` simultaneously
        # requires many times the payload size. Quantization is elementwise and
        # the error is a maximum, so both compose exactly over bounded blocks.
        quantized = np.empty(xyz.shape, dtype="<i2")
        measured_error = 0.0
        non_finite = False
        dec_lo = np.full(3, np.inf)
        dec_hi = np.full(3, -np.inf)
        step = 262_144
        for begin in range(0, len(xyz), step):
            block = xyz[begin : begin + step]
            # `rint(nan/scale)` is nan and `nan.astype(int16)` is
            # platform-defined, so the invalid returns are masked out and the
            # sentinel written explicitly rather than left to the cast.
            # ONE component, not all three: the partial-invalid case is refused
            # above, so within this branch a point is either wholly finite or
            # wholly not. Testing `block[:, 0]` reads a strided view and yields
            # an (N,) bool; `~np.isfinite(block).all(axis=1)` built an (N, 3)
            # temporary first, which is pure overhead for a question already
            # answered.
            invalid_block = ~np.isfinite(block[:, 0])
            # No `nan_to_num` copy: that allocated a second full-size block and
            # pushed peak memory past the bound the test below pins. The cast of
            # a NaN to int16 IS platform-defined, but every row it lands in is
            # overwritten with the sentinel on the very next line, so no
            # undefined value survives into the payload.
            with np.errstate(invalid="ignore"):
                block_q = np.clip(np.rint((block - offset_vec) / scale), -_INT16_MAX, _INT16_MAX).astype("<i2")
            block_q[invalid_block] = INT16_NO_RETURN
            quantized[begin : begin + step] = block_q
            with np.errstate(over="ignore", invalid="ignore"):
                block_decoded = block_q.astype(np.float64) * scale + offset_vec
                # Sentinel rows are not coordinates; judging the encoding by
                # what `-32768 * scale + offset` happens to produce would fail a
                # frame for a word that was never a position.
                # Sentinel rows are not coordinates. Judging the encoding by
                # what `-32768 * scale + offset` produces would fail a frame for
                # a word that was never a position, and including them in the
                # box would put a phantom point at the offset.
                #
                # Both the error and the extent are therefore measured over the
                # REAL rows only. The NaN case previously passed here by
                # accident — `|0 - nan|` is `nan` and `max(0.0, nan)` is `0.0`,
                # so the error silently read as zero — while the otherwise
                # identical Inf case failed loudly. An accident that works for
                # one non-finite value and not the other is not a rule.
                real = ~invalid_block
                if not real.any():
                    continue
                block_decoded = block_decoded[real]
                block_real = block[real]
                if not np.isfinite(block_decoded).all():
                    non_finite = True
                    break
                measured_error = max(measured_error, float(np.max(np.abs(block_decoded - block_real))))
                # `box` is computed from the DECODED extent (see below), so the
                # extrema are accumulated here rather than from a full-size
                # decoded array that no longer exists. min/max compose exactly.
                dec_lo = np.minimum(dec_lo, block_decoded.min(axis=0))
                dec_hi = np.maximum(dec_hi, block_decoded.max(axis=0))
        header["_geom_codec"] = GEOM_INT16
        header["_geom_scale"] = scale
        header["_geom_offset"] = offset
        # Validated at the TOP of the write (below), so a rejected declaration
        # never reaches the encode; recorded here because these are the fields
        # that make a v3 frame placeable.
        header.update(frame_fields)
        payload = quantized.tobytes()
        version = ALF_VERSION_V3
        if non_finite:
            # The writer generating exactly the shape the reader refuses. With
            # `check_budget=False` and coordinates near float64 max, the derived
            # scale reconstructs to Inf: `write_alp` recorded an `-inf:...:inf`
            # box and `read_alf(path).xyz` raised on the file it had just
            # written. The fp16 branch already checks its encoded coordinates;
            # this one did not.
            raise ValueError(
                f"{path}: int16 reconstruction overflows to non-finite coordinates at scale "
                f"{scale:.6g}. The frame's extent cannot be represented in fixed point; tile "
                f"it or write fp16 instead of publishing a file this module cannot read back."
            )
        # The MEASURED error, not the ideal `scale / 2`. Reconstruction is
        # `quantized * scale + offset` in float64, and for an off-origin frame
        # the float64 spacing AT THE OFFSET can exceed the quantization step: a
        # frame around -4.488e11 spanning 8 mm picks scale 1e-4 and would record
        # 5e-5, while the file actually decodes with 6.1035e-5 error — exactly
        # one ULP at that magnitude. An accuracy-gated caller comparing against
        # its threshold would accept geometry outside it.
        #
        # Measured per block above, so the bound costs no extra full-size array.
        # An EMPTY frame has no coordinates to lose, so it has no error — which
        # is what the fp16 branch and `geom_error_report` both say. Recording
        # the scale floor here made a strict accuracy filter reject empty int16
        # frames for a loss that cannot have occurred.
        measured = max(scale / 2.0, measured_error) if len(xyz) else 0.0
        if check_budget and measured >= INT16_ERROR_BUDGET_M:
            # The budget is a promise about the FILE, and the check above tested
            # only the ideal quantization term. At a large offset the float64
            # rounding in `quantized * scale + offset` lands on top of it: a
            # frame around 2**50 spanning ~30 km picks an ideal 0.228 m and
            # reconstructs at 0.25 m, so the branch that promises to reject the
            # threshold wrote it anyway.
            raise ValueError(
                f"{path}: int16 reconstruction achieves {measured:.4f} m error, at or past the "
                f"{INT16_ERROR_BUDGET_M} m budget — the ideal step is {scale / 2.0:.4f} m but "
                f"float64 rounding at this frame's offset adds to it. Tile the scene or "
                f"re-localize it closer to the origin."
            )
        header["_geom_max_err_m"] = measured
    elif codec == GEOM_FP16:
        # The legacy layout: no codec keys, v1 version byte. (Not byte-equal —
        # `box` comes from the decoded coordinates; see the module docstring.)
        # Strip any inherited v3 keys — metadata is often copied forward from an
        # existing frame, and a surviving `_geom_codec: int16` would make
        # `read_alf` decode this float16 payload as fixed-point: garbage
        # coordinates from a file this module itself produced.
        for reserved in ("_geom_codec", "_geom_scale", "_geom_offset", "_geom_max_err_m"):
            header.pop(reserved, None)
        # The overflow is DETECTED below and raised on, so numpy's RuntimeWarning
        # about it is noise on a path that already fails loudly.
        with np.errstate(over="ignore"):
            encoded = xyz.astype("<f2")
        # Representability is NOT part of the quality budget, so this check is
        # unconditional. A finite input beyond binary16's range (70000.0, say)
        # overflows to Inf, and with `check_budget=False` the file would publish
        # successfully with `inf` in its own `box` — a file this module's reader
        # then rejects as corrupt. Refusing to write it is the only consistent
        # answer; `check_budget=False` permits a LARGE frame, not an unwritable
        # coordinate.
        # Only where a FINITE input overflowed. A non-finite INPUT component is
        # an invalid return being stored faithfully — float16 has NaN and Inf, so
        # it round-trips — and this check exists for a representability failure,
        # not for the marker.
        if len(xyz) and bool((np.isfinite(xyz) & ~np.isfinite(encoded)).any()):
            worst = _finite_max_abs(xyz)
            raise ValueError(
                f"{path}: coordinate magnitude {worst:g} m overflows float16 (max ~65504) and "
                f"would be written as Inf. This is a representability limit, not the quality "
                f"budget `check_budget` waives — use codec={GEOM_INT16!r} for frames this large."
            )
        payload = encoded.tobytes()
        version = ALF_VERSION_V1
        # Blocked, like the int16 branch. This full `(N, 3)` float64 array
        # existed only to reduce its own extrema, and at the documented
        # 10-million-point scale it is ~240 MB on top of the caller's input, the
        # `_as_xyz` copy, the encoded array and the payload — enough to OOM a
        # constrained worker in the DEFAULT codec path.
        if len(xyz):
            dec_lo = np.full(3, np.inf)
            dec_hi = np.full(3, -np.inf)
            for begin in range(0, len(encoded), 262_144):
                block = encoded[begin : begin + 262_144].astype(np.float64)
                # NaN-AWARE. `block.min` propagates a single invalid return into
                # every axis, so the box became "nan:nan:nan:nan:nan:nan" — and
                # the server turns this header straight into item-metadata
                # bounds, so a NaN box takes the frame out of every spatial
                # query rather than merely looking odd. The box describes where
                # the REAL points are; an invalid return has no position.
                # NON-FINITE MASKED OUT, Inf as well as NaN. `block.min`
                # propagates a single invalid return into every axis, so the box
                # became "nan:nan:nan:nan:nan:nan" — and the server turns this
                # header straight into item-metadata bounds, so that takes the
                # frame out of every spatial query rather than merely looking
                # odd. An Inf component is excluded for the same reason from the
                # other end: the box is a string of six numbers, and "inf" in it
                # is unparseable to every consumer. The box describes where the
                # REAL points are; a component with no position is not one.
                finite_block = np.where(np.isfinite(block), block, np.nan)
                with warnings.catch_warnings():
                    # A block with no finite component on some axis legitimately
                    # has no extent there; the all-NaN-slice warning is noise on
                    # a case `np.fmin`/`np.fmax` already handle by skipping.
                    warnings.simplefilter("ignore", RuntimeWarning)
                    dec_lo = np.fmin(dec_lo, np.nanmin(finite_block, axis=0))
                    dec_hi = np.fmax(dec_hi, np.nanmax(finite_block, axis=0))
    else:
        raise ValueError(f"Unknown geometry codec {codec!r} (expected {GEOM_INT16!r} or {GEOM_FP16!r})")

    # `box` describes what the FILE contains, so it is computed from the DECODED
    # coordinates, never from the input. Both codecs are lossy and round in
    # either direction, so a pre-encoding extremum can sit inside the decoded
    # extent: a frame whose y spans ±0.005 m decodes at ±0.006104 m under int16,
    # and the server turns this header straight into item-metadata bounds — so
    # spatial filtering and culling would exclude real payload points that fall
    # outside a box the writer itself published.
    # A frame with no finite component on some axis has no extent to publish
    # there, so it publishes NO box rather than one containing "nan" or "inf".
    if len(xyz) and np.isfinite(dec_lo).all() and np.isfinite(dec_hi).all():
        lo, hi = dec_lo, dec_hi
        header["box"] = f"{lo[0]}:{lo[1]}:{lo[2]}:{hi[0]}:{hi[1]}:{hi[2]}"
        header["_box_frame"] = "local"
    else:
        # Metadata is often copied forward from a previous frame. If filtering
        # left this one empty, an inherited `box` would publish the OLD spatial
        # extent for a frame that now has no geometry.
        header.pop("box", None)
        header.pop("_box_frame", None)

    _write_channel(path, version, header, payload)
    return header


def _check_integer_ids(values: np.ndarray, kind: str, wire_type: str, lo: int, hi: int) -> None:
    """Reject ids that cannot survive the cast to their wire type.

    Range alone is not enough: a float array like `[1.9]` passes a min/max check
    and then TRUNCATES to 1 on `astype`, relabelling the point silently — the
    same failure the range guard exists to prevent, arriving by a different door.
    """
    if not values.size:
        return
    # bool FIRST: `np.issubdtype(bool, np.integer)` is False but `np.mod` on it
    # is 0 and its range is [0, 1], so a mask passed where labels belong would
    # clear both checks and be written as valid-looking data. These streams are
    # index-correlated with the cloud, so a mask-shaped caller bug becomes
    # silently wrong labels rather than an error.
    if values.dtype == np.bool_:
        raise ValueError(
            f"{kind} ids must be integers, not booleans — a mask was passed where per-point "
            f"values belong, and it would be written as {values.size} valid-looking 0/1 ids"
        )
    if not np.issubdtype(values.dtype, np.integer):
        if not np.all(np.equal(np.mod(values, 1), 0)):
            raise ValueError(
                f"{kind} ids must be whole numbers; got fractional values such as {values[np.mod(values, 1) != 0][0]}"
            )
    actual_lo, actual_hi = int(np.min(values)), int(np.max(values))
    if actual_lo < lo or actual_hi > hi:
        raise ValueError(f"{kind} ids must fit in {wire_type} [{lo}, {hi}]; got range [{actual_lo}, {actual_hi}]")


def _is_integral_intensity(values: Any, lo: Any, hi: Any) -> bool:
    """Whether this channel's samples and bounds are integers.

    Only integer inputs carry a consecutive-value promise for float64 to break;
    a float range spanning the whole number line is legitimate and normalizes
    fine (see the span branch below).
    """
    if not (isinstance(lo, (int, np.integer)) and isinstance(hi, (int, np.integer))):
        return False
    if isinstance(lo, bool) or isinstance(hi, bool):
        return False
    arr = getattr(values, "dtype", None)
    return arr is None or arr.kind in "iu"


def map_intensity(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Normalize raw intensity onto uint8. Float division, matching the canonical
    generator — two aerial writers used integer division and came out a
    consistent one low."""
    # Validated HERE, not only by `write_ali`. This is a package-level public
    # API now, and a direct caller was getting none of the mask, complex,
    # string or lossy-narrowing checks the writer performs before calling it:
    # `np.longdouble` endpoints one apart collapsed to [0, 0], a masked sample
    # published its hidden backing value, and a complex component was dropped
    # behind a ComplexWarning. Second helper this has happened to after
    # `choose_geom_offset` — exporting a function is an API decision, and the
    # validation has to come with it.
    #
    # `write_ali` calls `_as_per_point` and then this, so the double pass is
    # idempotent and costs one coercion on an array that is already clean.
    raw_values = values
    values = _as_per_point(values, "raw_intensity", "map_intensity")
    values = np.asarray(values, dtype=np.float64)
    if values.size and not np.isfinite(values).all():
        # `clip(...).astype(uint8)` would turn NaN/Inf into endpoint bytes, making
        # a broken sensor or parser indistinguishable from a genuine minimum or
        # maximum reading. Geometry already fails fast on non-finite input; so
        # does intensity.
        raise ValueError("intensity values contain NaN or Inf; filter non-finite samples before writing")
    for _name, _v in (("lo", lo), ("hi", hi)):
        # A WHITELIST, not a bool blacklist. The blacklist form missed
        # `np.bool_` — the shape a bound propagated from a NumPy/pandas mask
        # actually has — because `isinstance(np.bool_(True), bool)` is False
        # while `math.isfinite` accepts it, so `lo=np.bool_(False)` sailed
        # through and defined the whole channel's normalization. Every other
        # numeric check in this module uses a whitelist for that reason; this
        # was the one that did not.
        if isinstance(_v, bool) or not isinstance(_v, (int, float, np.integer, np.floating)):
            raise ValueError(f"intensity bound `{_name}` must be a real number, got {_v!r} ({type(_v).__name__})")
    if not math.isfinite(lo) or not math.isfinite(hi):
        # Previously caught only because `hi` landed in the header and the
        # non-finite header check rejected it. With the raw bounds no longer
        # written, an infinite `hi` would make `(v - lo) * 255 / (hi - lo)`
        # evaluate to 0 for every sample and silently publish a black channel.
        raise ValueError(f"intensity bounds must be finite; got lo={lo}, hi={hi}")
    if hi < lo:
        # Reversed bounds are a caller bug, not a degenerate frame. Folding them
        # into the `hi == lo` branch writes an all-zero `.ali.gz` that records the
        # reversed range in its header and looks entirely valid — the original
        # intensities are unrecoverable and nothing downstream can tell the file
        # apart from a genuinely constant one.
        raise ValueError(f"intensity bounds are reversed: lo={lo} > hi={hi}")
    if hi == lo:
        # Genuinely constant intensity: every sample maps to 0, the legacy
        # behaviour and the only defensible answer for a zero-width range. But
        # the bounds must actually DESCRIBE the samples: `lo=hi=0` over [0, 255]
        # would otherwise zero real contrast and publish a header claiming a
        # zero-width raw range — readings unrecoverable, file looking correct.
        if values.size and not np.all(values == lo):
            raise ValueError(
                f"intensity bounds lo == hi == {lo} but the samples are not constant "
                f"(range {float(values.min())}..{float(values.max())}); zero-filling would "
                f"discard real contrast irreversibly"
            )
        return np.zeros(values.shape, dtype=np.uint8)
    # Normalize on whichever formulation the bounds can actually represent.
    #
    # `(values - lo) * 255 / (hi - lo)` overflows when the bounds are large.
    # Halving both operands fixes that — and BREAKS subnormal ranges, where
    # `hi / 2` underflows to zero and the denominator collapses: `lo=0.0,
    # hi=nextafter(0, 1)` published [0, 0] where it had published [0, 255].
    # Trading an overflow for an underflow is not fixing the class, so this
    # branches on the actual condition instead of transforming unconditionally.
    # INTEGER inputs only. Beyond 2**53 float64 cannot represent consecutive
    # integers, so the eager cast COLLAPSES distinct raw values before
    # normalization: `[2**53, 2**53 + 1]` over those same bounds became equal
    # and wrote [0, 0] — no contrast, from inputs that differ. int64/uint64
    # sensor counts reach this range. Floats make no consecutive-integer
    # promise, so a huge FLOAT range stays legal (and is covered above).
    #
    # `>=`, not `>`: `float(2**53 + 1)` rounds back to `2**53`, so a strict
    # comparison never fires on the exact boundary this exists to catch.
    if (_is_integral_intensity(raw_values, lo, hi) or _exceeds_float64_exact_integers(np.asarray(raw_values))) and (
        abs(int(lo)) >= 2**53 or abs(int(hi)) >= 2**53 or (values.size and float(np.max(np.abs(values))) >= 2.0**53)
    ):
        raise ValueError(
            "integer intensity values or bounds reach 2**53, where float64 can no longer "
            "distinguish consecutive integers — normalizing here would collapse distinct raw "
            "values into the same byte. Rebase the samples (subtract a common origin) first."
        )
    if values.size:
        # `lo`/`hi` are documented as THE raw range, so a sample outside them is
        # a range-calculation bug — stale bounds, or bounds taken from a
        # different chunk. Clipping silently turns that into a valid-looking
        # channel with its contrast destroyed: `[0, 100]` with `hi=1` published
        # `[0, 255]`. The canonical intensity mapper
        # path fails here too, via `struct.pack("B", ...)` refusing the
        # out-of-range integer; this one just did it quietly.
        below = float(np.min(values))
        above = float(np.max(values))
        if below < lo or above > hi:
            raise ValueError(
                f"intensity samples span [{below:g}, {above:g}], outside the declared raw range "
                f"[{lo:g}, {hi:g}]. Clipping would publish a channel whose contrast is silently "
                f"wrong; recompute lo/hi from the same data these samples came from."
            )
    span = hi - lo
    if math.isfinite(span) and span != 0.0:
        scaled = (values - lo) / span * 255.0
    else:
        # The span itself overflowed. Halving is safe HERE precisely because
        # the operands are enormous: `hi/2 - lo/2` is representable for any
        # finite lo <= hi, and nothing near a subnormal reaches this branch.
        scaled = (values / 2.0 - lo / 2.0) / (hi / 2.0 - lo / 2.0) * 255.0
    return np.clip(scaled, 0, 255).astype(np.uint8)


@_clears_stale_output_on_failure
def write_ali(
    path: str | Path,
    raw_intensity: np.ndarray,
    *,
    lo: float,
    hi: float,
    deciles: Iterable[int] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a `.ali.gz`.

    `lo`/`hi` are the RAW range the normalization used. They are NOT recorded in
    the header: no legacy writer emits raw bounds, so adding them would change
    every ported `.ali` envelope. The legacy writers hardcoded `_min_intensity: 0, _max_intensity: 255`
    "by mapping definition", which threw away the only information that made the
    mapping reversible.

    `_intensities_every_10_percent` is a SEQUENCE-level distribution in every
    existing writer: the canonical frame writer and the streaming converter both
    accumulate intensities across all frames and stamp the same deciles onto
    every `.ali`. The viewer uses it to pick contrast, so a per-frame histogram
    would make the same surface change shade between frames, and a sparse or
    low-variance frame would publish a distribution that describes nothing.

    So the precedence here is: an explicit `deciles` argument, else whatever the
    caller already has in `metadata`, and only as a last resort this frame's own
    distribution — which is correct only for a single-frame dataset.
    """
    # Flatten first. An (N, 1) column vector is the shape a caller naturally has
    # after slicing one field out of a structured array, and it survives
    # `map_intensity` unchanged — then `np.sort` stays 2-D and `int(ordered[i])`
    # raises TypeError on an array, while a scalar raises AxisError. `write_all`
    # already ravels its labels; this is the same normalization.
    normalized = map_intensity(_as_per_point(raw_intensity, "raw_intensity", path), lo, hi)
    header: dict[str, Any] = dict(metadata or {})
    header["_total_points"] = int(normalized.size)
    header["_min_intensity"] = 0
    header["_max_intensity"] = 255
    # `lo`/`hi` select the encoding but are not recorded as raw-bounds keys.
    # Adding `_raw_intensity_min`/`_raw_intensity_max` would change the legacy
    # header contract; callers can supply application provenance in metadata.
    if deciles is not None:
        values = _checked_deciles(deciles)
        header["_intensities_every_10_percent"] = values
    elif "_intensities_every_10_percent" in header:
        # INHERITED deciles get the same checks. Sequence-level distributions
        # normally arrive this way — copied forward through `metadata` rather
        # than passed explicitly — so validating only the argument left the
        # common path unguarded, and a malformed list was serialized unchanged
        # for the viewer to build a contrast curve from.
        header["_intensities_every_10_percent"] = _checked_deciles(header["_intensities_every_10_percent"])
    elif "_intensities_every_10_percent" not in header:
        if normalized.size:
            ordered = np.sort(normalized)
            idx = [min(int(ordered.size / 10 * i), ordered.size - 1) for i in range(1, 10)]
            header["_intensities_every_10_percent"] = [int(ordered[i]) for i in idx]
        else:
            header["_intensities_every_10_percent"] = [0] * 9
    _write_channel(path, ALF_VERSION_V1, header, normalized.tobytes())
    return header


@_clears_stale_output_on_failure
def write_all(
    path: str | Path,
    categories: np.ndarray,
    *,
    instance_ids: np.ndarray | None = None,
    instance_map: dict[str, str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a `.all.gz` label channel, v1 (categories) or v2 (with instances)."""
    # Unlink BEFORE validating, same as the .alp path: a rejected rerun must not
    # leave the previous label file on disk for downstream globbing to treat as
    # the result of this run.

    categories = _as_per_point(categories, "categories", path)
    # Validate BEFORE casting: `asarray(300, dtype=uint8)` wraps to 44, so an
    # out-of-range class would be written as a different, valid-looking label
    # with nothing raised anywhere. Labels are the product here — a silent
    # relabel is the worst failure this module could have.
    _check_integer_ids(categories, "category", "uint8", 0, 255)
    categories = categories.astype(np.uint8)
    header: dict[str, Any] = dict(metadata or {})
    # Writer-OWNED keys are always recomputed, never inherited. Metadata is
    # routinely copied forward from a previous frame, so a stale `_total_points`
    # or `_format_version` would describe the old payload while the arrays
    # describe this one — and the reader's own cross-checks would then fail on a
    # file this writer produced. `instance_map` is the exception: it is inherited
    # ON PURPOSE (below), because a rewrite that passes `read_alf(...).metadata`
    # back through would otherwise blank the object identities.
    header.pop("_format_version", None)
    header["_total_points"] = int(categories.size)

    if instance_ids is None:
        if instance_map is not None:
            # EXPLICIT map, no ids. Dropping it here discarded the stable
            # identities the caller asked for and reported success, hiding the
            # omitted panoptic stream — the argument was passed for a reason.
            # An INHERITED map (via `metadata`) is different: that is pass-through
            # of someone else's header, and dropping it below is correct because
            # a v1 file has no ids for it to describe.
            raise ValueError(
                f"{path}: instance_map was passed without instance_ids. A semantic-only "
                f"`.all` has no instance stream for the map to identify; pass instance_ids "
                f"too, or drop the map to write a v1 file."
            )
        # Drop inherited v2 keys. `format_version` gives metadata precedence, so a
        # surviving `_format_version: 2` on a one-byte-per-point payload makes the
        # reader either raise on a non-multiple-of-3 length or read every third
        # byte as a category with invented instance ids.
        header.pop("_format_version", None)
        header.pop("instance_map", None)
        payload = categories.tobytes()
        version = ALF_VERSION_V1
    else:
        instances = _as_per_point(instance_ids, "instance_ids", path)
        # 65535 is RESERVED, not merely representable. The editor's registry caps
        # valid local ids at 65534 (`persistent_instance_registry.dart:25`,
        # documented range 1-65534), so a file carrying 65535 clears every uint16
        # check here and then silently loses that instance's identity when the
        # viewer loads it. 0 stays legal and means "unset".
        _check_integer_ids(instances, "instance", "uint16", 0, 65534)
        instances = instances.astype(np.uint16)
        if instances.size != categories.size:
            raise ValueError(f"categories ({categories.size}) and instance_ids ({instances.size}) must match")
        rows = np.empty((categories.size, 3), dtype=np.uint8)
        rows[:, 0] = categories
        rows[:, 1] = (instances & 0xFF).astype(np.uint8)
        rows[:, 2] = (instances >> 8).astype(np.uint8)
        payload = rows.tobytes()
        header["_format_version"] = LABELS_V2
        # Preserve an inherited map when the caller did not supply one: a
        # rewrite that passes `read_alf(...).metadata` back through would
        # otherwise blank it, and the legacy export resolves panoptic instances to
        # stable UUIDs through exactly this key. The numeric
        # ids would survive and the object identity would not.
        if instance_map is not None:
            # A list or string here would serialize fine and then fail every
            # `str(id) in map` lookup, so the coverage check below would report
            # every id as unmapped for a caller whose real mistake was the type.
            if not isinstance(instance_map, dict):
                raise ValueError(
                    f"{path}: instance_map must be a mapping of str(id) -> uuid, got {type(instance_map).__name__}"
                )
            # The EXPLICIT argument wins over anything inherited from metadata.
            # A rewrite that passes `read_alf(...).metadata` back through while
            # supplying a new map would otherwise publish the old identities
            # alongside the new labels, depending on merge order.
            header["instance_map"] = instance_map
        else:
            header.setdefault("instance_map", {})
        # The INHERITED map needs the same type check as an explicit one. A
        # string sails through `setdefault` and then through the coverage check
        # by accident -- `str(1) in "1"` is True -- so a v2 `.all` gets written
        # with a string where a map belongs, and `read_alf(...).instance_ids`
        # rejects the file this writer just produced.
        if not isinstance(header["instance_map"], dict):
            raise ValueError(
                f"{path}: inherited instance_map must be a mapping of str(id) -> uuid, got "
                f"{type(header['instance_map']).__name__}"
            )
        # Presence is not coverage. A map that merely EXISTS satisfies "an id
        # needs a map" while still leaving specific ids unresolvable — id 5 with
        # a map holding only id 6 writes labels whose object identity is gone,
        # and nothing downstream can tell that apart from a correct file. Zero
        # means "unset" and is never mapped.
        resolved = header["instance_map"]
        # EVERY entry must be usable, not just the ones this frame emits. The
        # map is sequence-level, so a pass-through rewrite or a frame subset
        # legitimately carries ids that are absent here — but the reader's
        # `instance_map` property validates the whole mapping, so leaving a null
        # value for an unemitted id writes a file the shared API cannot read
        # back. Entries for absent ids are KEPT (that is what inheritance is
        # for); they just have to resolve to something.
        unusable_entries = sorted(k for k, v in resolved.items() if not isinstance(v, str) or not v.strip())
        # Keys, not just values. `{"1": "u1", "01": "stale"}` passes the value
        # check and the coverage check below, and `read_alf(...).instance_map`
        # then raises on the file this writer just produced — the shared writer
        # emitting files the shared reader refuses is the one outcome this
        # module exists to prevent. Pass-through rewrites and frame subsets
        # carry sequence-level maps, so this is reachable without hand-editing.
        non_string = sorted(repr(k) for k in resolved if not isinstance(k, str))
        if non_string:
            # JSON members are strings, so `{1: "a", "1": "b"}` serializes BOTH
            # to `"1"` and `json.loads` keeps only the last — `write_all`
            # succeeded while silently discarding a persistent object identity.
            # A non-string key alone already fails the coverage check below
            # (resolvers look up `str(id)`), so requiring strings here costs no
            # working caller and turns a confusing "missing entries" into the
            # actual problem.
            raise ValueError(
                f"{path}: instance_map has non-string key(s) {non_string[:10]}. JSON members are "
                f"strings, so an int key either collides with its string twin and drops one "
                f"identity, or resolves nothing; use string keys throughout."
            )
        noncanonical = sorted(str(k) for k in resolved if str(k) != str(_safe_int(k)))
        if noncanonical:
            raise ValueError(
                f"{path}: instance_map has non-canonical key(s) {noncanonical[:10]}"
                f"{'...' if len(noncanonical) > 10 else ''}. Resolvers look up str(id) exactly, "
                f"so those entries resolve nothing and the reader rejects the map."
            )
        out_of_domain = sorted(str(k) for k in resolved if not (1 <= int(str(k)) <= 65534))
        if out_of_domain:
            # Spelling was checked; the DOMAIN was not. The payload validator
            # reserves 65535, caps ids at 65534 and treats 0 as unset, so a
            # sequence-level map carrying "65535", "0" or "99999" describes
            # identities no frame can ever emit — and `read_alf` hands them
            # straight back.
            raise ValueError(
                f"{path}: instance_map has key(s) {out_of_domain[:10]} outside the valid "
                f"instance id range 1..65534 (0 means unset, 65535 is reserved)."
            )
        if unusable_entries:
            raise ValueError(
                f"{path}: instance_map has unusable value(s) for key(s) {unusable_entries[:10]}"
                f"{'...' if len(unusable_entries) > 10 else ''}. Every entry must resolve to a "
                f"persistent instance, including ids this frame does not emit — the reader "
                f"validates the whole map."
            )
        emitted = {int(v) for v in np.unique(instances) if int(v) != 0}
        unmapped = sorted(i for i in emitted if _instance_unresolved(resolved, i))
        if unmapped:
            raise ValueError(
                f"{path}: instance_map is missing entries for instance id(s) {unmapped[:10]}"
                f"{'...' if len(unmapped) > 10 else ''}. Every nonzero id in the payload must "
                f"resolve to a persistent instance; a partially-populated map writes labels whose "
                f"object identity is silently lost."
            )
        version = 2

    _write_channel(path, version, header, payload)
    return header


@_clears_stale_output_on_failure
def write_alc(
    path: str | Path,
    palette_indices: np.ndarray,
    palette: Iterable[str],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a `.alc.gz` — one uint8 palette index per point, plus the palette.

    `palette` is a list of `rrggbb` hex strings, matching
    the canonical palette generator. Indices are validated against its length,
    with one deliberate exception: **`len(palette)` is the no-colour sentinel**
    (as the canonical colour writer does), written for points that matched no camera or were
    out of view. Anything beyond that is refused, since it renders as whatever
    the viewer's bounds handling does with it — a colour bug nobody traces back
    to the file.
    """
    # Unlink FIRST, like every other writer here: validation below can reject the
    # call, and a rejected re-run must not leave the previous frame behind for
    # downstream globbing to pick up as current output.
    indices = _as_per_point(palette_indices, "palette_indices", path)
    colors = [_as_hex_color(c, path) for c in palette]
    _check_integer_ids(indices, "palette index", "uint8", 0, 255)
    indices = indices.astype(np.uint8)
    if not colors:
        raise ValueError(f"{path}: .alc needs a non-empty palette")
    # 255 entries, not 256. The payload is uint8 and `len(palette)` is the
    # no-colour sentinel, so a 256-entry palette leaves no representable value
    # for it — every index 0..255 would be a real colour and occluded points
    # would have nowhere to go. The canonical colour writer asserts the same bound
    # ("ALC files support up to 255 colors"); stating it here keeps the shared
    # writer from quietly accepting what the generator refuses.
    if len(colors) > 255:
        raise ValueError(
            f"{path}: .alc supports at most 255 palette entries (got {len(colors)}); "
            f"index 255 is reserved as the no-colour sentinel"
        )
    # `len(colors)` is the NO-COLOUR SENTINEL, not an error. The canonical colour writer
    # writes it deliberately for points that matched no camera or fell out of
    # view, and it is out of the normal 0..n-1 range by design. A naive bounds
    # check would reject every real frame that has an occluded point, and a
    # "helpful" clamp would recolour those points with the last palette entry.
    # So one past the end is legal; two past is not.
    if indices.size and int(indices.max()) > len(colors):
        raise ValueError(
            f"{path}: palette index {int(indices.max())} is out of range for a "
            f"{len(colors)}-entry palette (max {len(colors)}, where {len(colors)} "
            f"is the no-colour sentinel)"
        )
    header: dict[str, Any] = dict(metadata or {})
    header["_total_points"] = int(indices.size)
    header["palette"] = colors
    _write_channel(path, ALF_VERSION_V1, header, indices.tobytes())
    return header


@_clears_stale_output_on_failure
def write_alx(
    path: str | Path,
    kept_indices: np.ndarray,
    *,
    alp_total_points: int,
    source_total_points: int,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a `.alx.gz` — the decimation sidecar, uint32 original index per point.

    Written alongside a decimated `.alp` so a later export can map a labelled
    point back to its index in the source cloud. Strictly increasing is not
    required (a caller may keep a reordered subset), but negative or non-integer
    indices are refused: they would silently wrap through uint32 into indices
    that address real, wrong points.

    Both counts are REQUIRED, because an optional invariant is not an invariant:
    a caller that forgets the argument disables the only check standing between
    a decimator bug and silently mislabelled data.

    `alp_total_points` is the point count of the `.alp` this sidecar accompanies.
    The two are correlated BY INDEX — entry i describes point i of that file — so
    a decimator that emits K points but K-1 indices writes an `.alx` that is
    independently valid and misassociates every point after the gap.

    `source_total_points` is the size of the ORIGINAL cloud the indices point
    into, and it catches the other half of the same failure: a sidecar can have
    exactly the right length and still hold an index past the end of the raw
    data, at which point an export reads the authoritative source out of
    range or lands on the wrong point. Length agreement says the mapping is
    complete; range agreement says it is meaningful. It is recorded in the
    header so a reader can re-check it without the writer's arguments.
    """
    indices = _as_per_point(kept_indices, "kept_indices", path)
    _check_integer_ids(indices, "kept index", "uint32", 0, 2**32 - 1)
    # `np.integer` too. These counts come from `np.sum(mask)` or a pandas
    # count in every NumPy-based pipeline that would write a sidecar, and the
    # int/float whitelist rejected `np.int64(2)` as "not a number" — refusing a
    # perfectly valid count for its container type. Booleans stay out: `True`
    # IS an int in Python and a mask that leaked in here would read as 1 point.
    if isinstance(alp_total_points, (bool, np.bool_)) or not isinstance(
        alp_total_points, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{path}: alp_total_points must be a number")
    if isinstance(alp_total_points, (float, np.floating)) and not float(alp_total_points).is_integer():
        raise ValueError(f"{path}: alp_total_points must be whole, got {alp_total_points!r}")
    if int(alp_total_points) != int(indices.size):
        raise ValueError(
            f"{path}: {indices.size} kept indices for an .alp of {alp_total_points} points. "
            f"The sidecar is index-correlated with the geometry, so a mismatch silently "
            f"misassociates every point after the gap."
        )
    if isinstance(source_total_points, (bool, np.bool_)) or not isinstance(
        source_total_points, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{path}: source_total_points must be a number")
    if isinstance(source_total_points, (float, np.floating)) and not float(source_total_points).is_integer():
        # int() TRUNCATES, so 1.9 would be written as 1 and the reader's own
        # domain check would never see the invalid value the writer normalized
        # away — the sidecar range invariant quietly weakened at its source.
        raise ValueError(f"{path}: source_total_points must be whole, got {source_total_points!r}")
    source_total_points = int(source_total_points)
    try:
        narrowed_source_total = float(source_total_points)
    except OverflowError as exc:
        raise ValueError(f"{path}: source_total_points must be exactly representable as finite binary64") from exc
    if not math.isfinite(narrowed_source_total) or int(narrowed_source_total) != source_total_points:
        raise ValueError(f"{path}: source_total_points must be exactly representable as finite binary64")
    if source_total_points < 0:
        raise ValueError(f"{path}: source_total_points must be non-negative")
    if indices.size and len(np.unique(indices)) != indices.size:
        dup = np.unique(indices)[np.bincount(np.searchsorted(np.unique(indices), indices)) > 1]
        raise ValueError(
            f"{path}: kept indices contain duplicates (e.g. {int(dup[0])}). The sidecar maps each "
            f"decimated point back to a distinct source point, so a repeat means two labelled "
            f"points address the same raw point while another kept point has no mapping at all. "
            f"Reordered subsets are fine; repeats are not."
        )
    if indices.size and int(indices.max()) >= int(source_total_points):
        raise ValueError(
            f"{path}: kept index {int(indices.max())} is out of range for a source cloud of "
            f"{source_total_points} points. An export would read the authoritative raw "
            f"data out of range or land on the wrong point."
        )
    header: dict[str, Any] = dict(metadata or {})
    header["_total_points"] = int(indices.size)
    header["_source_total_points"] = int(source_total_points)
    _write_channel(path, ALF_VERSION_V1, header, indices.astype("<u4").tobytes())
    return header


def geom_error_report(local_xyz: np.ndarray) -> dict[str, float]:
    """Worst-case per-component error each codec would give this frame, in metres.

    Used by the tests to MEASURE the v3 claim rather than assert it, and useful
    for a converter that wants to log what a dataset is giving up.

    Shape-validated exactly as `write_alp` is, and for a worse reason: a bare
    `reshape(-1, 3)` folds the common (N, 4) "XYZ plus intensity/ring" matrix
    into (4N/3, 3) whenever the element count happens to divide by three, so
    intensity readings become coordinates. Here that does not just corrupt a
    frame — it inflates the extent, which inflates `choose_geom_scale`, which
    makes the reported int16 error larger than the codec would really produce.
    The number this function exists to provide would be wrong in the direction
    that makes v3 look worse than it is.
    """
    xyz = _as_xyz(local_xyz, "geom_error_report")
    if xyz.size == 0:
        return {GEOM_FP16: 0.0, GEOM_INT16: 0.0}
    if not np.isfinite(xyz).all():
        # `write_alp` refuses non-finite geometry, so a report that quietly
        # returns {"fp16": nan, "int16": nan} describes a frame that could never
        # be written — and a single NaN poisons any aggregate built from these
        # numbers without anything looking wrong.
        raise ValueError("geom_error_report: coordinates contain NaN or Inf")
    # Both codecs measured in BLOCKS. The writer and the reader's preflight
    # already do; this was the third full-frame pass over the same arithmetic
    # and would OOM the 10-million-point scale it exists to report on. Both
    # results are maxima, so blocking changes nothing but the peak.
    fp16 = 0.0
    # Mirror the writer exactly — including ADDING THE OFFSET BACK. Measuring
    # in centered coordinates omits the float64 rounding that reconstruction
    # introduces at the frame's true magnitude, and for a far off-origin frame
    # that term dominates: the -4.488e11 case reported 3.27e-05 while
    # `write_alp` round-tripped at 6.10e-05. This is the THIRD place that
    # derived this bound independently (writer, accessor, report) and the third
    # to get it wrong, so it now reconstructs rather than models.
    offset_vec = choose_geom_offset(xyz)
    step = 262_144
    # The centred extent, without materializing the centred array — that was
    # the last full-size temporary here. `choose_geom_scale` derives the step
    # from `max(abs(...))` alone, so handing it that one value is exact. Doing
    # the arithmetic inline instead would be a fourth independent derivation of
    # a rule that already has one home.
    largest = 0.0
    for begin in range(0, len(xyz), step):
        largest = max(largest, float(np.max(np.abs(xyz[begin : begin + step] - offset_vec))))
    scale = _scale_from_extent(largest)
    int16 = 0.0
    for begin in range(0, len(xyz), step):
        block = xyz[begin : begin + step]
        with np.errstate(over="ignore", invalid="ignore"):
            # `inf` is the CORRECT answer for a frame fp16 cannot represent at
            # all — it is how this report says "do not pick fp16 here" — so
            # NumPy's warning is noise for a value returned on purpose.
            fp16 = max(
                fp16,
                float(np.abs(block.astype(np.float16).astype(np.float64) - block).max()),
            )
            quantized = np.clip(np.rint((block - offset_vec) / scale), -_INT16_MAX, _INT16_MAX)
            int16 = max(int16, float(np.abs(quantized * scale + offset_vec - block).max()))
    return {GEOM_FP16: float(fp16), GEOM_INT16: float(int16)}


def iter_frames(paths: Iterable[str | Path]) -> Iterable[AlfFile]:
    """Convenience for the many scripts that walk a sequence directory."""
    for p in paths:
        yield read_alf(p)
