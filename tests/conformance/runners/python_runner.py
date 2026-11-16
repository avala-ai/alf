#!/usr/bin/env python3
"""Reference runner: read one ALF file, print the canonical answer, exit 0.

The contract every language's runner implements.

A REFUSAL IS A RESULT. When the file is invalid this prints
`{"refused": "<identifier>"}` and still exits 0, so the harness diffs it like
any other answer. Exiting non-zero would collapse "refused correctly" and "fell
over" into one outcome, and telling those apart is what the invalid corpus is
for.
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "python"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from alf import AlfError
from generate import canonical

# Maps this implementation's messages onto the identifiers declared in
# generator/invalid.py. A language does not invent identifiers; it proves it can
# tell the rules apart by naming the one it hit.
_REFUSALS = [
    ("more than one gzip member", "multiple-gzip-members"),
    ("Corrupt gzip stream", "gzip-malformed"),
    ("not valid UTF-8", "header-not-utf8"),
    ("Truncated header", "header-truncated"),
    ("Header is too large", "header-too-large"),
    ("Header is not valid JSON", "header-not-json"),
    ("Header must be a JSON object", "header-not-an-object"),
    ("header-nesting-too-deep", "header-nesting-too-deep"),
    ("is not finite", "header-not-json"),
    ("fixed-point `.alp` geometry, but this file is", "geometry-version-on-non-geometry-channel"),
    ("`.all` interleaved label layout, but this file is", "label-version-on-non-label-channel"),
    ("Unsupported ALF version byte", "version-unsupported"),
    ("Unknown .alp geometry codec", "codec-unsupported"),
    ("requires `_geom_codec", "codec-version-disagreement"),
    ("Fixed-point geometry is version 3", "codec-version-disagreement"),
    ("Unknown .alp geometry codec", "codec-unsupported"),
    ("missing `_geom_scale`", "geom-scale-missing"),
    ("must be a positive finite number", "geom-scale-not-positive"),
    ("`_geom_offset` must have exactly 3 components", "geom-offset-malformed"),
    ("missing `_geom_offset`", "geom-offset-malformed"),
    ("binary64 cannot represent exactly", "total-points-not-an-integer"),
    ("integer too large for binary64", "total-points-not-an-integer"),
    ("Geometry anchor", "geometry-anchor-malformed"),
    # The v3 frame convention. Each needle names ONE rule: a broad
    # "`_frame_rotation`" catch-all sat here first and swallowed the exact-token
    # refusal, reporting a lossy quaternion component as a malformed shape.
    ("is missing `_frame_convention`", "frame-convention-missing"),
    ("Unsupported `_frame_convention`", "frame-convention-unsupported"),
    ("is a version-3 field", "frame-convention-on-legacy-version"),
    ("requires `_frame_rotation`", "frame-rotation-missing"),
    ("is not a unit quaternion", "frame-rotation-not-normalized"),
    ("declares that no rotation was applied", "frame-rotation-unexpected"),
    ("`_frame_rotation` must be an object", "frame-rotation-malformed"),
    ("`_frame_rotation` is missing", "frame-rotation-malformed"),
    ("`_frame_rotation` has unexpected members", "frame-rotation-malformed"),
    ("must be a JSON number", "frame-rotation-malformed"),
    ("must be finite, got", "frame-rotation-malformed"),
    ("where float64 stops being exact", "geom-number-inexact"),
    ("once read", "geom-number-inexact"),
    ("too large to represent", "geom-number-inexact"),
    ("float64 cannot represent exactly", "geom-number-inexact"),
    ("float64 cannot represent that integer exactly", "geom-number-inexact"),
    ("Header number", "geom-number-inexact"),
    ("integer too large for float64", "geom-number-inexact"),
    ("duplicate JSON object key", "header-key-duplicate"),
    ("not a multiple of 3 values", "payload-not-whole-points"),
    ("not a whole number of", "payload-not-whole-points"),
    ("must be an integer", "total-points-not-an-integer"),
    ("must be a non-negative integer", "total-points-not-an-integer"),
    ("present but null", "total-points-not-an-integer"),
    ("missing `_total_points`", "total-points-missing"),
    ("but the payload holds", "payload-length-mismatch"),
    ("declares more points than", "payload-length-mismatch"),
    # BEFORE the generic non-finite entries: a partial marker is a distinct
    # wire violation, not a coordinate that failed to be finite.
    ("invalid-return marker", "geom-partial-no-return"),
    ("non-finite coordinates", "coordinate-non-finite"),
    ("contains NaN or Inf", "coordinate-non-finite"),
    ("Unsupported `.all` _format_version", "label-layout-unsupported"),
    ("fits BOTH layouts", "label-layout-ambiguous"),
    ("reserved instance id 65535", "instance-id-reserved"),
    ("missing entries for", "instance-map-incomplete"),
    ("non-canonical key(s)", "instance-map-malformed"),
    ("missing `_source_total_points`", "source-total-points-missing"),
    ("contains duplicate kept indices", "kept-index-duplicate"),
    ("out of range for a source cloud", "kept-index-out-of-range"),
    ("is out of range for a", "palette-index-out-of-range"),
    ("hexadecimal digits", "palette-entry-malformed"),
]


def identify(error: Exception) -> str:
    text = str(error)
    for needle, identifier in _REFUSALS:
        if needle in text:
            return identifier
    # An unnamed refusal is a GAP, not a pass: it matches no expectation and
    # fails with a readable diff rather than quietly counting as success.
    return ""


def main() -> int:
    path = pathlib.Path(sys.argv[1])
    extension = sys.argv[2] if len(sys.argv) > 2 else ""
    try:
        answer = canonical(path, extension)
    except (AlfError, ValueError) as exc:
        print(json.dumps({"refused": identify(exc)}, sort_keys=True))
        return 0
    print(json.dumps(answer, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
