#!/usr/bin/env python3
"""Build the conformance corpus deterministically without rewriting its oracle.

The corpus files are NOT committed — `.gitignore` excludes them and this script
rebuilds them before every run. What IS committed is the expectations plus
`data/CHECKSUMS.txt`.

The default reconstructs only ignored binary fixtures. ``--update`` is the
explicit maintainer operation that replaces expectations and checksums.
`--verify` is the gate. It regenerates everything in scratch space, asserts every checksum still
matches, and asserts that TWO CONSECUTIVE RUNS PRODUCE BYTE-IDENTICAL OUTPUT.
That second check is the one that catches accidental nondeterminism in a writer
— dict iteration order, a timestamp in a gzip header, a hash seed — which is
otherwise invisible until someone else's CI fails on a diff they did not cause.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import shutil
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "generator"))

import invalid as invalid_cases
import numpy as np
from alf import read_alf, write_alc, write_ali, write_all, write_alp, write_alx

DATA = pathlib.Path(__file__).resolve().parent / "data"


def _frame() -> np.ndarray:
    """A frame whose values are NOT exactly representable in float16.

    Coordinates like 1.5 or 100.0 round-trip through fp16 exactly, so a corpus
    built from them cannot show any difference between the codecs — the
    comparison it exists to make passes trivially.
    """
    return np.array([[1.37, -2.11, 0.43], [87.62, 41.19, -3.07], [-0.31, 12.73, 5.88], [63.45, -28.06, 1.92]])


def _with_gzip_zero_padding(write):
    """Wrap a writer and append RFC 1952-permitted zero padding."""

    def padded(path: pathlib.Path) -> None:
        write(path)
        path.write_bytes(path.read_bytes() + b"\0\0\0\0")

    return padded


def _valid_variants() -> dict:
    """name -> (writer callable, canonical expectation)."""
    xyz = _frame()
    return {
        "alp-fp16": (lambda p: write_alp(p, xyz, codec="fp16", check_budget=False), "alp"),
        # Stepping above the largest finite half reaches infinity, but its
        # quantization error is still bounded by half the finite spacing.
        "alp-fp16-largest-finite": (
            lambda p: write_alp(p, np.array([[65504.0, -65504.0, 0.0]]), check_budget=False),
            "alp",
        ),
        "alp-fp16-all-invalid": (
            lambda p: write_alp(p, np.full((1, 3), np.nan), check_budget=False),
            "alp",
        ),
        "alp-int16": (
            lambda p: write_alp(p, xyz, codec="int16", frame_convention="world-aligned", check_budget=False),
            "alp",
        ),
        # A real device rotation: a quarter turn about +Z, which is what an ego
        # frame differs from world axes by after the vehicle turns a corner.
        # The stored points are IDENTICAL to `alp-int16` — the whole difference
        # between the two files is the declaration, which is the point: without
        # it the bytes cannot tell a reader which of these two frames it holds.
        "alp-int16-device-aligned": (
            lambda p: write_alp(
                p,
                xyz,
                codec="int16",
                frame_convention="device-aligned",
                frame_rotation={
                    "w": 0.7071067811865476,
                    "x": 0.0,
                    "y": 0.0,
                    "z": 0.7071067811865476,
                },
                check_budget=False,
            ),
            "alp",
        ),
        # Identity is a legal device rotation, not an absent one. A vehicle
        # pointing along +x has this heading, and it is exactly the case where
        # the two conventions agree — so a reader that treats identity as "no
        # declaration" passes every straight-line frame and fails at the corner.
        "alp-int16-device-aligned-identity": (
            lambda p: write_alp(
                p,
                xyz,
                codec="int16",
                frame_convention="device-aligned",
                frame_rotation={"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
                check_budget=False,
            ),
            "alp",
        ),
        # Non-finite fp16 geometry is VALID CONTENT, and this is the shape the
        # corpus must preserve. The previous corpus asserted such a file must
        # be REFUSED. A reader that drops or compacts
        # the invalid returns fails here on the point count as well as the
        # values, which is the property that keeps every index-correlated
        # channel aligned.
        "alp-fp16-invalid-returns": (
            lambda p: write_alp(
                p,
                np.array(
                    [
                        [1.5, -2.5, 3.75],
                        [float("nan"), float("nan"), float("nan")],
                        [4.0, 5.0, 6.0],
                        [float("nan"), 0.5, -0.25],
                    ]
                ),
                codec="fp16",
                check_budget=False,
            ),
            "alp",
        ),
        # The two infinities kept DISTINCT from NaN and from each other. A
        # reader that collapses them to one non-finite marker, or to null, has
        # changed what the file says.
        "alp-fp16-infinities": (
            lambda p: p.write_bytes(
                invalid_cases._gzip(
                    invalid_cases._envelope(
                        1,
                        {"_total_points": 2},
                        invalid_cases._fp16(
                            [
                                [float("inf"), float("-inf"), 0.0],
                                [1.0, float("nan"), -1.0],
                            ]
                        ),
                    )
                )
            ),
            "alp",
        ),
        # The v3 sentinel. This is the case that decides whether version 3 can
        # carry invalid returns: a scanning sensor can emit a DENSE range
        # image with slots that hold no return. Compacting them would destroy the
        # beam/azimuth mapping and desynchronise every index-correlated channel,
        # so v3 encodes an invalid return as three `-32768` words — a value the
        # encoder cannot otherwise emit, because the scale is derived as
        # `extent / 32767`.
        #
        # An implementation that decodes the sentinel arithmetically instead of
        # recognising it produces a coordinate at roughly `-32768 * scale +
        # offset`: finite, plausible, and wrong. That is what this case catches.
        "alp-int16-invalid-returns": (
            lambda p: write_alp(
                p,
                np.array(
                    [
                        [1.5, -2.5, 3.75],
                        [float("nan"), float("nan"), float("nan")],
                        [4.0, 5.0, 6.0],
                    ]
                ),
                codec="int16",
                frame_convention="world-aligned",
                check_budget=False,
            ),
            "alp",
        ),
        "alp-int16-off-centre": (
            lambda p: write_alp(
                p,
                xyz + np.array([1000.0, 0.0, 0.0]),
                codec="int16",
                frame_convention="world-aligned",
                check_budget=False,
            ),
            "alp",
        ),
        "alp-int16-exact-wide-integer": (
            lambda p: p.write_bytes(
                invalid_cases._gzip(
                    invalid_cases._raw_envelope(
                        3,
                        '{"_total_points":1,"_geom_codec":"int16","_frame_convention":"world-aligned","_geom_scale":0.0001,'
                        '"_geom_offset":[18446744073709551616,0,0]}',
                        invalid_cases._int16([0, 0, 0]),
                    )
                )
            ),
            "alp",
        ),
        "alp-int16-exact-wide-integral-spellings": (
            lambda p: p.write_bytes(
                invalid_cases._gzip(
                    invalid_cases._raw_envelope(
                        3,
                        '{"_total_points":1,"_geom_codec":"int16","_frame_convention":"world-aligned",'
                        '"_geom_scale":18446744073709551616.0,'
                        '"_geom_offset":[18446744073709551616e0,0,0]}',
                        invalid_cases._int16([0, 0, 0]),
                    )
                )
            ),
            "alp",
        ),
        "alp-int16-cancelled-geometry-exponent": (
            lambda p: p.write_bytes(
                invalid_cases._gzip(
                    invalid_cases._raw_envelope(
                        3,
                        '{"_total_points":1,"_geom_codec":"int16","_frame_convention":"world-aligned","_geom_scale":0.'
                        + ("0" * 9_980)
                        + '18446744073709551616e10000,"_geom_offset":[0,0,0]}',
                        invalid_cases._int16([1, 0, 0]),
                    )
                )
            ),
            "alp",
        ),
        "alp-int16-leading-zero-exponent": (
            lambda p: p.write_bytes(
                invalid_cases._gzip(
                    invalid_cases._raw_envelope(
                        3,
                        '{"_total_points":2,"_geom_codec":"int16","_frame_convention":"world-aligned",'
                        '"_geom_scale":0.25e00000000000,"_geom_offset":[0,0,0],'
                        '"_geom_max_err_m":0.125}',
                        invalid_cases._int16([-32767, 0, 0, 32767, 0, 0]),
                    )
                )
            ),
            "alp",
        ),
        "alp-int16-zero-hostile-exponent": (
            lambda p: p.write_bytes(
                invalid_cases._gzip(
                    invalid_cases._raw_envelope(
                        3,
                        '{"_total_points":1,"_geom_codec":"int16","_frame_convention":"world-aligned",'
                        '"_geom_scale":0.0001,'
                        '"_geom_offset":[0e10000000000,0,0],'
                        '"_geom_max_err_m":0.00005}',
                        invalid_cases._int16([0, 0, 0]),
                    )
                )
            ),
            "alp",
        ),
        "alp-int16-positive-subnormal-scale": (
            lambda p: p.write_bytes(
                invalid_cases._gzip(
                    invalid_cases._raw_envelope(
                        3,
                        '{"_total_points":1,"_geom_codec":"int16","_frame_convention":"world-aligned",'
                        '"_geom_scale":5e-324,"_geom_offset":[0,0,0],'
                        '"_geom_max_err_m":5e-324}',
                        invalid_cases._int16([0, 0, 0]),
                    )
                )
            ),
            "alp",
        ),
        "alp-int16-canonical-rounding-boundary": (
            lambda p: p.write_bytes(
                invalid_cases._gzip(
                    invalid_cases._raw_envelope(
                        3,
                        '{"_total_points":1,"_geom_codec":"int16","_frame_convention":"world-aligned",'
                        '"_geom_scale":0.0000025,"_geom_offset":[0,0,0]}',
                        invalid_cases._int16([1, 0, 0]),
                    )
                )
            ),
            "alp",
        ),
        "alp-int16-exact-large-offset": (
            lambda p: write_alp(
                p,
                np.array([[float(2**53), 0.0, 0.0]]),
                codec="int16",
                frame_convention="world-aligned",
                check_budget=False,
            ),
            "alp",
        ),
        "alp-int16-largest-finite-product": (
            lambda p: p.write_bytes(
                invalid_cases._gzip(
                    invalid_cases._envelope(
                        3,
                        {
                            "_total_points": 1,
                            "_geom_codec": "int16",
                            "_frame_convention": "world-aligned",
                            "_geom_scale": float(np.finfo(np.float64).max),
                            "_geom_offset": [0.0, 0.0, 0.0],
                        },
                        invalid_cases._int16([1, 0, 0]),
                    )
                )
            ),
            "alp",
        ),
        "alp-int16-empty-recorded-error": (
            lambda p: p.write_bytes(
                invalid_cases._gzip(
                    invalid_cases._envelope(
                        3,
                        {
                            "_total_points": 0,
                            "_geom_codec": "int16",
                            "_frame_convention": "world-aligned",
                            "_geom_scale": 0.0001,
                            "_geom_offset": [0.0, 0.0, 0.0],
                            "_geom_max_err_m": 7.0,
                        },
                        b"",
                    )
                )
            ),
            "alp",
        ),
        "alp-empty": (lambda p: write_alp(p, np.zeros((0, 3))), "alp"),
        "alp-legacy-empty-device-anchor": (
            lambda p: p.write_bytes(
                invalid_cases._gzip(
                    invalid_cases._raw_envelope(
                        1,
                        '{"_total_points":1,"device_position":{}}',
                        np.array([[0.0, 0.0, 0.0]], dtype=np.float16).tobytes(),
                    )
                )
            ),
            "alp",
        ),
        "alp-permuted-anchor-object": (
            lambda p: p.write_bytes(
                invalid_cases._gzip(
                    invalid_cases._raw_envelope(
                        1,
                        '{"_total_points":1,"device_position":{"z":3,"y":2,"x":1}}',
                        np.array([[0.0, 0.0, 0.0]], dtype=np.float16).tobytes(),
                    )
                )
            ),
            "alp",
        ),
        "alp-anchor-extra-members": (
            lambda p: p.write_bytes(
                invalid_cases._gzip(
                    invalid_cases._raw_envelope(
                        1,
                        '{"_total_points":1,"device_position":{"z":3,"note":'
                        '{"source":"gps","flags":["rtk",{"fixed":true}]},'
                        '"x":1,"quality":0.875,"y":2}}',
                        np.array([[0.0, 0.0, 0.0]], dtype=np.float16).tobytes(),
                    )
                )
            ),
            "alp",
        ),
        "alp-fp16-wide-anchor": (
            lambda p: p.write_bytes(
                invalid_cases._gzip(
                    invalid_cases._raw_envelope(
                        1,
                        '{"_total_points":1,"device_position":{"x":18446744073709551616,"y":0,"z":0,"note":"gps"}}',
                        np.array([[0.0, 0.0, 0.0]], dtype=np.float16).tobytes(),
                    )
                )
            ),
            "alp",
        ),
        "alp-int16-wide-anchor": (
            lambda p: p.write_bytes(
                invalid_cases._gzip(
                    invalid_cases._raw_envelope(
                        3,
                        '{"_total_points":1,"device_position":'
                        '{"x":18446744073709551616,"y":0,"z":0,"note":"gps"},'
                        '"_geom_codec":"int16","_frame_convention":"world-aligned","_geom_scale":0.0001,'
                        '"_geom_offset":[0,0,0],"_geom_max_err_m":0.00005}',
                        invalid_cases._int16([0, 0, 0]),
                    )
                )
            ),
            "alp",
        ),
        "ali-basic": (lambda p: write_ali(p, np.array([0.0, 5.0, 10.0]), lo=0.0, hi=10.0), "ali"),
        "ali-zero-length-header": (
            lambda p: p.write_bytes(invalid_cases._gzip(invalid_cases._raw_envelope(1, "", bytes([7])))),
            "ali",
        ),
        "ali-integral-count-exponent": (
            lambda p: p.write_bytes(
                invalid_cases._gzip(invalid_cases._raw_envelope(1, '{"_total_points":1e0}', bytes([7])))
            ),
            "ali",
        ),
        "ali-integral-count-padded-exponent": (
            lambda p: p.write_bytes(
                invalid_cases._gzip(
                    invalid_cases._raw_envelope(
                        1,
                        '{"_total_points":1e00000000000}',
                        bytes([7]),
                    )
                )
            ),
            "ali",
        ),
        "ali-integral-count-cancelled-exponent": (
            lambda p: p.write_bytes(
                invalid_cases._gzip(
                    invalid_cases._raw_envelope(
                        1,
                        '{"_total_points":1' + ("0" * 10_000) + "e-10000}",
                        bytes([7]),
                    )
                )
            ),
            "ali",
        ),
        "ali-zero-count-hostile-exponent": (
            lambda p: p.write_bytes(
                invalid_cases._gzip(
                    invalid_cases._raw_envelope(
                        1,
                        '{"_total_points":0e10000000000}',
                        b"",
                    )
                )
            ),
            "ali",
        ),
        "ali-nongeometry-anchor-metadata": (
            lambda p: p.write_bytes(
                invalid_cases._gzip(
                    invalid_cases._raw_envelope(
                        1,
                        '{"_total_points":1,"device_position":{"x":9007199254740993,"y":0,"z":0}}',
                        bytes([7]),
                    )
                )
            ),
            "ali",
        ),
        "ali-gzip-zero-padded": (
            _with_gzip_zero_padding(lambda p: write_ali(p, np.array([0.0, 5.0, 10.0]), lo=0.0, hi=10.0)),
            "ali",
        ),
        "alc-basic": (
            lambda p: write_alc(p, np.array([0, 1], dtype=np.uint8), ["ff0000", "00ff00"]),
            "alc",
        ),
        "alc-leading-hash": (
            lambda p: p.write_bytes(
                invalid_cases._gzip(
                    invalid_cases._envelope(
                        1,
                        {"_total_points": 2, "palette": ["#FF0000", "00ff00"]},
                        bytes([0, 1]),
                    )
                )
            ),
            "alc",
        ),
        "all-v1": (lambda p: write_all(p, np.array([1, 2, 3], dtype=np.uint8)), "all"),
        "all-v2": (
            lambda p: write_all(
                p,
                np.array([1, 2], dtype=np.uint8),
                instance_ids=np.array([1, 2], dtype=np.uint16),
                instance_map={"1": "uuid-a", "2": "uuid-b"},
            ),
            "all",
        ),
        "all-v2-integral-format-exponent": (
            lambda p: p.write_bytes(
                invalid_cases._gzip(
                    invalid_cases._raw_envelope(
                        2,
                        '{"_total_points":1,"_format_version":2e0,"instance_map":{"1":"u1"}}',
                        bytes([7, 1, 0]),
                    )
                )
            ),
            "all",
        ),
        "alx-basic": (
            lambda p: write_alx(
                p,
                np.array([7, 2, 19], dtype=np.uint32),
                alp_total_points=3,
                source_total_points=20,
            ),
            "alx",
        ),
        "alx-exact-wide-source-count": (
            lambda p: p.write_bytes(
                invalid_cases._gzip(
                    invalid_cases._raw_envelope(
                        1,
                        '{"_total_points":1.0,"_source_total_points":18014398509481984}',
                        bytes([0, 0, 0, 0]),
                    )
                )
            ),
            "alx",
        ),
        "alx-source-count-above-uint64": (
            lambda p: write_alx(p, np.array([0], dtype=np.uint32), alp_total_points=1, source_total_points=2**64),
            "alx",
        ),
        "alx-source-count-wide-decimal": (
            lambda p: p.write_bytes(
                invalid_cases._gzip(
                    invalid_cases._raw_envelope(
                        1, '{"_total_points":1,"_source_total_points":18446744073709551616.0}', bytes(4)
                    )
                )
            ),
            "alx",
        ),
        "alx-source-count-wide-exponent": (
            lambda p: p.write_bytes(
                invalid_cases._gzip(
                    invalid_cases._raw_envelope(
                        1, '{"_total_points":1,"_source_total_points":18446744073709551616e0}', bytes(4)
                    )
                )
            ),
            "alx",
        ),
        "alx-source-count-large-binary64": (
            lambda p: write_alx(p, np.array([0], dtype=np.uint32), alp_total_points=1, source_total_points=2**1023),
            "alx",
        ),
    }


def _canonical_component(value) -> object:
    """One coordinate, in the spelling every language must produce.

    JSON HAS NO NaN. `json.dumps(float("nan"))` emits the bare token `NaN`,
    which is not JSON and which this format refuses inside its own headers;
    Dart and JavaScript `JSON.stringify` turn it into `null` instead. So three
    conforming readers would disagree on the text while agreeing on the bytes.

    Non-finite components are VALID `.alp` content (spec/format.md §5.1), so the
    corpus has to be able to say so. They are spelled as STRINGS, and the three
    are kept distinct rather than collapsed to one marker: a reader that turns
    +Infinity into NaN has changed the file's meaning, and `null` could not tell
    us it had.
    """
    number = float(value)
    if math.isnan(number):
        return "NaN"
    if math.isinf(number):
        return "Infinity" if number > 0 else "-Infinity"
    return round(number, 6)


def canonical(path: pathlib.Path, extension: str = "") -> dict:
    """What a correct reader must produce. The same shape in every language.

    `extension` is needed for the invalid corpus, whose files are named `.bin`
    so nothing infers a channel from the suffix — the point of those files is
    the bytes, not the name.
    """
    alf = read_alf(path, extension=extension) if extension else read_alf(path)
    out: dict = {"version": alf.version, "extension": alf.extension}
    if alf.extension == "alp":
        out["codec"] = alf.geom_codec
        out["points"] = [[_canonical_component(c) for c in point] for point in alf.xyz]
        bound = alf.max_geom_error_m
        out["max_geom_error_m"] = None if bound is None else round(float(bound), 12)
        # SURFACED, not merely validated. The refusals below prove a reader
        # parses the frame declaration; this proves it can answer with it. The
        # bug this field exists for was a field that was present and ignored, so
        # "the reader accepted the file" is not the property worth pinning.
        # `null` for v1/v2, which do not say and cannot.
        out["frame_convention"] = alf.frame_convention
        rotation = alf.frame_rotation
        if rotation is not None:
            out["frame_rotation"] = {axis: _canonical_component(rotation[axis]) for axis in ("w", "x", "y", "z")}
        anchors = {
            key: alf.metadata[key] for key in ("device_position", "world_centroid_epsg3857") if key in alf.metadata
        }
        if anchors:
            out["anchors"] = anchors
    elif alf.extension == "ali":
        out["intensity"] = [int(v) for v in alf.intensity]
    elif alf.extension == "alc":
        out["palette"] = alf.palette
        out["palette_indices"] = [int(v) for v in alf.palette_indices]
    elif alf.extension == "all":
        out["format_version"] = alf.format_version
        out["categories"] = [int(v) for v in alf.categories]
        ids = alf.instance_ids
        out["instance_ids"] = None if ids is None else [int(v) for v in ids]
        out["instance_map"] = alf.instance_map
    elif alf.extension == "alx":
        out["kept_indices"] = [int(v) for v in alf.kept_indices]
        out["source_total_points"] = alf.metadata["_source_total_points"]
    return out


def build(output_root: pathlib.Path = DATA, *, write_expectations: bool = True) -> dict:
    """Write every corpus file and return name -> sha256."""
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "invalid").mkdir(exist_ok=True)
    sums: dict = {}

    for name, (write, ext) in _valid_variants().items():
        path = output_root / f"{name}.{ext}.gz"
        write(path)
        expectation = output_root / f"{name}.json"
        expectation_bytes = (json.dumps(canonical(path), indent=2, sort_keys=True) + "\n").encode()
        if write_expectations:
            expectation.write_bytes(expectation_bytes)
        sums[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        sums[expectation.name] = hashlib.sha256(expectation_bytes).hexdigest()

    for case in invalid_cases.CASES:
        path = output_root / "invalid" / f"{case.name}.{case.extension}.bin"
        path.write_bytes(case.build())
        expectation = output_root / "invalid" / f"{case.name}.json"
        expectation_bytes = (
            json.dumps({"refused": case.refusal, "why": case.why}, indent=2, sort_keys=True) + "\n"
        ).encode()
        if write_expectations:
            expectation.write_bytes(expectation_bytes)
        sums[f"invalid/{path.name}"] = hashlib.sha256(path.read_bytes()).hexdigest()
        sums[f"invalid/{expectation.name}"] = hashlib.sha256(expectation_bytes).hexdigest()

    # PRUNE anything the case lists no longer produce. Removing a case used to
    # leave its committed expectation and its fixture on disk, and `run.py`
    # discovers cases by GLOBBING the data directory — so the deleted case kept
    # executing, against a rule the corpus no longer states. `--verify` could
    # not see it either, because the checksum manifest is rebuilt from the same
    # case lists and simply stops mentioning the orphan. Found while flipping
    # `fp16-coordinate-non-finite` from invalid to valid: the stale file made
    # the suite report 94 discovered against 93 advertised, and the only reason
    # it was noticed at all is that the total is separately asserted.
    expected_names = set(sums) | {"CHECKSUMS.txt"}
    for pattern in ("*.json", "*.gz", "invalid/*.json", "invalid/*.bin"):
        for existing in output_root.glob(pattern):
            if existing.relative_to(output_root).as_posix() not in expected_names:
                existing.unlink()

    return sums


def write_checksums(sums: dict, output_root: pathlib.Path = DATA) -> None:
    lines = [f"{digest}  {name}\n" for name, digest in sorted(sums.items())]
    (output_root / "CHECKSUMS.txt").write_text("".join(lines))


def materialize_fixtures(output_root: pathlib.Path | None = None) -> int:
    """Build ignored binaries without touching the committed JSON oracle."""
    destination_root = DATA if output_root is None else output_root
    with tempfile.TemporaryDirectory(prefix="alf-corpus-build-") as directory:
        scratch = pathlib.Path(directory)
        generated = build(scratch, write_expectations=False)
        fixtures = [name for name in generated if not name.endswith(".json")]
        for name in fixtures:
            destination = destination_root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(scratch / name, destination)
    return len(fixtures)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="regenerate and gate on determinism")
    parser.add_argument(
        "--update",
        action="store_true",
        help="explicitly replace committed expectations and CHECKSUMS.txt",
    )
    args = parser.parse_args()
    if args.verify and args.update:
        parser.error("--verify and --update are mutually exclusive")

    if args.update:
        first = build()
        write_checksums(first)
        print(f"updated {len(first)} artefacts in {DATA}")
        return 0

    if not args.verify:
        # CI and ordinary local runs need the ignored binary fixtures, but must
        # not rewrite the committed oracle before the verification gate reads
        # it. Baseline changes require the explicit ``--update`` mode.
        count = materialize_fixtures()
        print(f"generated {count} fixtures in {DATA}; committed oracle unchanged")
        return 0

    # Verification is read-only with respect to the committed baseline.  Build
    # twice in scratch directories so a failed gate cannot first rewrite the
    # expectation it was meant to check.
    with tempfile.TemporaryDirectory(prefix="alf-corpus-first-") as first_dir:
        first = build(pathlib.Path(first_dir))
    with tempfile.TemporaryDirectory(prefix="alf-corpus-second-") as second_dir:
        second = build(pathlib.Path(second_dir))
    if first != second:
        drifted = sorted(k for k in first if first.get(k) != second.get(k))
        print("NONDETERMINISTIC generator — two runs differ:", file=sys.stderr)
        for name in drifted:
            print(f"  {name}", file=sys.stderr)
        return 1

    recorded = DATA / "CHECKSUMS.txt"
    if not recorded.exists():
        print("CHECKSUMS.txt missing; run generate.py --update first", file=sys.stderr)
        return 1
    expected = {
        line.split("  ", 1)[1].strip(): line.split("  ", 1)[0]
        for line in recorded.read_text().splitlines()
        if line.strip()
    }
    if expected != first:
        print("corpus drifted from CHECKSUMS.txt", file=sys.stderr)
        for name in sorted(set(expected) | set(first)):
            if expected.get(name) != first.get(name):
                print(
                    f"  {name}: recorded={expected.get(name)} now={first.get(name)}",
                    file=sys.stderr,
                )
        return 1

    # JSON expectations are the committed human-reviewable oracle. Ensure the
    # files themselves still match CHECKSUMS; regenerated binary fixtures are
    # deliberately ignored by git and are checked above through ``first``.
    for name, digest in expected.items():
        if not name.endswith(".json"):
            continue
        baseline = DATA / name
        actual = hashlib.sha256(baseline.read_bytes()).hexdigest() if baseline.is_file() else None
        if actual != digest:
            print(
                f"committed expectation differs from CHECKSUMS: {name}: recorded={digest} actual={actual}",
                file=sys.stderr,
            )
            return 1

    fixture_count = materialize_fixtures()
    print(f"verified {len(first)} artefacts: deterministic and unchanged; materialized {fixture_count} fixtures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
