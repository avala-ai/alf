#!/usr/bin/env python3
"""Private Python adapter for the shared decoded writer-request protocol."""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))

from alf import (
    write_alc,
    write_ali,
    write_all,
    write_alp,
    write_alx,
)

_NON_FINITE = {"NaN": float("nan"), "Infinity": float("inf"), "-Infinity": float("-inf")}


def _component(value):
    """Parse one coordinate from the request's canonical spelling.

    JSON has no NaN, so a non-finite component travels as a string. A candidate
    writer that ignores this and casts the string straight to a number produces
    garbage rather than an invalid return, which is exactly what the gate is for.
    """
    return _NON_FINITE[value] if isinstance(value, str) else float(value)


def main() -> int:
    if len(sys.argv) != 4:
        print(
            "usage: python_encode_roundtrip.py <request.json> <channel> <output>",
            file=sys.stderr,
        )
        return 64
    source, channel, output = sys.argv[1:]
    request = json.loads(pathlib.Path(source).read_text())
    if request.get("channel") != channel:
        print("request channel disagrees with command channel", file=sys.stderr)
        return 64
    metadata = dict(request["metadata"])
    if channel == "alp":
        write_alp(
            output,
            np.asarray([_component(c) for c in request["points"]], dtype=np.float64).reshape((-1, 3)),
            metadata,
            codec=request["codec"],
            frame_convention=request.get("frame_convention"),
            frame_rotation=request.get("frame_rotation"),
            check_budget=False,
        )
    elif channel == "ali":
        write_ali(output, np.asarray(request["intensity"]), lo=0, hi=255, metadata=metadata)
    elif channel == "all":
        instance_ids = request["instance_ids"]
        write_all(
            output,
            np.asarray(request["categories"]),
            instance_ids=None if instance_ids is None else np.asarray(instance_ids),
            instance_map=request["instance_map"],
            metadata=metadata,
        )
    elif channel == "alc":
        write_alc(
            output,
            np.asarray(request["palette_indices"]),
            request["palette"],
            metadata,
        )
    elif channel == "alx":
        write_alx(
            output,
            np.asarray(request["kept_indices"], dtype=np.uint32),
            alp_total_points=request["alp_total_points"],
            source_total_points=request["source_total_points"],
            metadata=metadata,
        )
    else:
        print(f"unknown channel: {channel}", file=sys.stderr)
        return 64
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
