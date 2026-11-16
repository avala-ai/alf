#!/usr/bin/env python3
"""Verify a language writer through Python and optional additional readers.

The candidate command receives ``<request.json> <channel> <output>``. The
request contains decoded typed values and metadata, never an ALF wire image it
could copy as its answer. The candidate must write an equivalent ALF file.
This gate then decodes that output with Python and compares canonical meaning
with the committed expectation. A package reading its own output is useful
unit coverage; this is the independent implementation boundary that proves
interoperability.

Repeat ``--reader`` to pass the candidate's actual bytes through other language
readers. Each must agree with Python's decoding of that candidate, including
its measured geometry error, rather than just accept the original corpus.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
import shlex
import sys
import tempfile

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE / "data"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "python"))

from alf import read_alf
from alf._io import _open_maybe_gzip
from bounded_process import MAX_CAPTURE_BYTES, run_bounded
from generate import _canonical_component, canonical
from run import _run_isolated

MAX_CANDIDATE_BYTES = 513 * 1024 * 1024

# Fields that a writer derives from its typed payload/API arguments. Their
# semantics are compared by ``canonical`` or, for grid-dependent geometry
# error, independently against the candidate bytes by ``_writer_wire_problem``;
# comparing their inherited source spelling would
# incorrectly require a writer to preserve stale format state. Everything else
# is application metadata and must survive a rewrite.
_REGENERATED_METADATA_FIELDS: dict[str, frozenset[str]] = {
    "alp": frozenset(
        {
            "_total_points",
            "_geom_codec",
            "_geom_scale",
            "_geom_offset",
            "_geom_max_err_m",
            "_frame_convention",
            "_frame_rotation",
            "box",
            "_box_frame",
        }
    ),
    "ali": frozenset(
        {
            "_total_points",
            "_min_intensity",
            "_max_intensity",
            "_intensities_every_10_percent",
        }
    ),
    "all": frozenset({"_total_points", "_format_version", "instance_map"}),
    "alc": frozenset({"_total_points", "palette"}),
    "alx": frozenset({"_total_points", "_source_total_points"}),
}


def _ceil_positive_half(value: float) -> float:
    """Return the least binary64 value no smaller than the exact half."""
    half = value / 2.0
    if half * 2.0 < value:
        return math.nextafter(half, math.inf)
    return half


def _normalize_metadata_value(value: object) -> object:
    """Normalize the portable JSON value a binary64 runtime can preserve."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (int, float)):
        # Python retains arbitrary integer tokens while Dart and JavaScript
        # expose the shared binary64 value. Readers have already rejected
        # non-finite values, so this conversion expresses the portable domain
        # and intentionally treats 9007199254740993 like 9007199254740992.
        return float(value)
    if isinstance(value, (list, tuple)):
        return [_normalize_metadata_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _normalize_metadata_value(item) for key, item in value.items()}
    raise TypeError(f"metadata contains non-JSON value {type(value).__name__}")


def _normalized_application_metadata(path: pathlib.Path, channel: str) -> object:
    """Return metadata a conforming writer must preserve across a rewrite."""
    metadata = dict(read_alf(path, extension=channel).metadata)
    for key in _REGENERATED_METADATA_FIELDS[channel]:
        metadata.pop(key, None)

    # ALP writers canonicalize either supported anchor shape to an x/y/z
    # object. Normalize both forms to components before comparison. On every
    # other channel these names are ordinary metadata and their structure must
    # remain untouched (the ali fixture exists specifically to prove that).
    if channel == "alp":
        for key in ("device_position", "world_centroid_epsg3857"):
            anchor = metadata.get(key)
            if isinstance(anchor, dict) and set(anchor) == {"x", "y", "z"}:
                metadata[key] = [anchor[axis] for axis in ("x", "y", "z")]

    return _normalize_metadata_value(metadata)


def _source_geometry_snapshot(path: pathlib.Path, channel: str) -> tuple[np.ndarray, str] | None:
    """Decode immutable source geometry before candidate code can mutate its input."""
    if channel != "alp":
        return None
    source = read_alf(path, extension=channel)
    xyz = np.array(source.xyz, copy=True)
    xyz.setflags(write=False)
    return xyz, source.geom_codec


def _writer_request_bytes(path: pathlib.Path, channel: str) -> bytes:
    """Serialize decoded writer inputs without exposing a reusable ALF wire."""
    source = read_alf(path, extension=channel)
    request: dict[str, object] = {
        "channel": channel,
        "metadata": dict(source.metadata),
    }
    if channel == "alp":
        # Non-finite components go over the wire in the CANONICAL SPELLING, not
        # as bare `NaN`. `allow_nan=False` below is deliberate — the request must
        # be valid JSON, because three languages parse it — and JSON has no NaN,
        # so a frame carrying invalid returns could not be handed to a candidate
        # writer at all, leaving invalid-return encoding untested.
        request.update(
            codec=source.geom_codec,
            points=[_canonical_component(c) for c in source.xyz.reshape(-1).tolist()],
            # A writer INPUT, not something an encoder can derive: the
            # coordinates of a world-aligned and a device-aligned frame are
            # indistinguishable, which is the whole reason the field exists.
            frame_convention=source.frame_convention,
            frame_rotation=source.frame_rotation,
        )
    elif channel == "ali":
        request["intensity"] = source.intensity.tolist()
    elif channel == "alc":
        request.update(
            palette_indices=source.palette_indices.tolist(),
            palette=list(source.palette),
        )
    elif channel == "all":
        instance_ids = source.instance_ids
        request.update(
            categories=source.categories.tolist(),
            instance_ids=None if instance_ids is None else instance_ids.tolist(),
            instance_map=source.instance_map,
        )
    elif channel == "alx":
        request.update(
            kept_indices=source.kept_indices.tolist(),
            alp_total_points=len(source.kept_indices),
            source_total_points=source.metadata["_source_total_points"],
        )
    else:  # pragma: no cover - variants derive channels from load-bearing suffixes
        raise ValueError(f"unsupported writer request channel {channel}")
    return json.dumps(
        request,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _run_bounded(
    argv: list[str],
    timeout: float = 120,
    *,
    output_path: pathlib.Path | None = None,
    max_file_bytes: int = MAX_CANDIDATE_BYTES,
) -> tuple[int, str, str, set[str], bool]:
    """Run an untrusted encoder without buffering unlimited diagnostics."""
    preexec_fn = None
    try:
        import resource

        def limit_file_size() -> None:
            _soft, hard = resource.getrlimit(resource.RLIMIT_FSIZE)
            cap = max_file_bytes if hard == resource.RLIM_INFINITY else min(max_file_bytes, hard)
            resource.setrlimit(resource.RLIMIT_FSIZE, (cap, cap))

        preexec_fn = limit_file_size
    except ImportError:  # pragma: no cover - CI/release platforms are POSIX
        pass
    return run_bounded(
        argv,
        timeout,
        output_path=output_path,
        max_file_bytes=max_file_bytes,
        preexec_fn=preexec_fn,
    )


def _writer_wire_problem(
    name: str,
    output: pathlib.Path,
    *,
    source_geometry: tuple[np.ndarray, str] | None,
    channel: str,
) -> str | None:
    """Check canonical writer-owned bytes before a reader normalizes them."""
    raw = _open_maybe_gzip(output)
    if len(raw) < 5:
        return "candidate output is too short to contain an ALF header"
    header_length = int.from_bytes(raw[1:5], "little")
    if 5 + header_length > len(raw):
        return "candidate output declares a truncated ALF header"
    try:
        metadata = json.loads(bytes(raw[5 : 5 + header_length]).decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as error:
        return f"candidate header cannot be inspected: {error}"
    if not isinstance(metadata, dict):
        return "candidate header must be a JSON object"

    measured: object = None
    raw_scale: object = None
    if channel == "alp" and raw[0] == 3:
        measured = metadata.get("_geom_max_err_m")
        try:
            valid_measured = (
                not isinstance(measured, bool)
                and isinstance(measured, (int, float))
                and measured >= 0
                and math.isfinite(float(measured))
            )
        except OverflowError:
            valid_measured = False
        if not valid_measured:
            return "v3 writer must emit a finite non-negative _geom_max_err_m"
        raw_scale = metadata.get("_geom_scale")
        try:
            valid_scale = (
                not isinstance(raw_scale, bool)
                and isinstance(raw_scale, (int, float))
                and raw_scale > 0
                and math.isfinite(float(raw_scale))
            )
        except OverflowError:
            valid_scale = False
        if not valid_scale:
            return "v3 writer must emit a finite positive _geom_scale"

    source_xyz: np.ndarray | None = None
    candidate_xyz: np.ndarray | None = None
    if channel == "alp":
        if source_geometry is None:
            return "source geometry was not snapshotted before invoking the candidate"
        source_xyz, source_geom_codec = source_geometry
        candidate_alf = read_alf(output, extension=channel)
        candidate_xyz = candidate_alf.xyz
        if source_xyz.shape != candidate_xyz.shape:
            return "candidate geometry shape differs from its source"
        # Every decoded fp16 value is exactly representable by fp16 again. A
        # changed component therefore means the writer changed payload meaning,
        # even when the human-readable six-decimal projection hides it.
        #
        # `equal_nan=True`, because NaN IS the payload here: an invalid return is
        # valid `.alp` content (spec/format.md §5.1) and must survive the round
        # trip like any other component. Plain `array_equal` reports every
        # invalid return as a difference — NaN != NaN — so it would fail a writer
        # that preserved them perfectly, while still failing one that replaced a
        # NaN with 0.0. This comparison must distinguish those two, not conflate
        # them. The infinities are ordinary values to `array_equal` and stay
        # exact, which is what keeps +Inf from silently becoming NaN.
        if source_geom_codec == "fp16" and not np.array_equal(source_xyz, candidate_xyz, equal_nan=True):
            return "candidate decoded geometry differs from its source"

        # The box describes the FINITE extent. An invalid return has no
        # position, so it contributes nothing — and a frame whose every value on
        # some axis is an invalid return has no extent to publish there at all,
        # which is the one case where a non-empty frame legitimately emits no
        # box. Requiring one unconditionally would force a writer to invent
        # "nan" or "inf" bounds, and the server turns this header straight into
        # item-metadata spatial bounds.
        finite_components = np.isfinite(candidate_xyz)
        has_extent = bool(candidate_xyz.size and finite_components.any(axis=0).all())
        if candidate_xyz.size and not has_extent:
            if "box" in metadata or "_box_frame" in metadata:
                return "ALP candidate with no finite extent must not emit a box"
        elif candidate_xyz.size:
            if metadata.get("_box_frame") != "local":
                return "non-empty ALP candidate must emit _box_frame local"
            raw_box = metadata.get("box")
            if not isinstance(raw_box, str):
                return "non-empty ALP candidate must emit a six-component box"
            parts = raw_box.split(":")
            try:
                box = np.asarray([float(part) for part in parts], dtype=np.float64)
            except (OverflowError, ValueError):
                return "non-empty ALP candidate must emit a finite six-component box"
            if len(parts) != 6 or not np.isfinite(box).all():
                return "non-empty ALP candidate must emit a finite six-component box"
            finite_only = np.where(finite_components, candidate_xyz, np.nan)
            decoded_box = np.concatenate((np.nanmin(finite_only, axis=0), np.nanmax(finite_only, axis=0)))
            if not np.array_equal(box, decoded_box):
                return "candidate box differs from its decoded geometry"
        elif "box" in metadata or "_box_frame" in metadata:
            return "empty ALP candidate must not emit box metadata"

    if channel == "alp" and raw[0] == 3:
        assert source_xyz is not None and candidate_xyz is not None
        assert isinstance(measured, (int, float)) and isinstance(raw_scale, (int, float))
        # An invalid return carries no position, so it has no reconstruction
        # error to measure — and `nan - nan` is NaN, which would poison the max
        # over the whole frame. Excluding those rows is only sound if the writer
        # is separately held to preserving WHICH rows are invalid; otherwise a
        # writer could dump a hard-to-quantize real point into the sentinel and
        # exempt itself from the very bound this gate exists to enforce.
        #
        # The predicate is "not finite", not "is NaN": v3 has no encoding for an
        # infinity, so an infinite source component is deliberately stored as an
        # invalid return (spec/format.md §5.2) and decodes back as NaN. That is
        # a permitted lossy mapping, not a lost point.
        source_invalid = ~np.isfinite(source_xyz).all(axis=1)
        candidate_invalid = ~np.isfinite(candidate_xyz).all(axis=1)
        if not np.array_equal(source_invalid, candidate_invalid):
            return "candidate did not preserve which returns are invalid"
        if source_xyz.size:
            real = ~source_invalid
            actual_error = float(abs(candidate_xyz[real] - source_xyz[real]).max()) if real.any() else 0.0
            if not math.isfinite(actual_error):
                return "candidate reconstruction error is not finite"
            half_scale_floor = _ceil_positive_half(float(raw_scale))
            if float(measured) < half_scale_floor:
                return "v3 writer _geom_max_err_m understates the positive half-scale floor"
            required_bound = max(actual_error, half_scale_floor)
        else:
            required_bound = 0.0
        tolerance = 8 * math.ulp(required_bound)
        if float(measured) + tolerance < required_bound:
            return "v3 writer _geom_max_err_m understates independently measured reconstruction error"
    if name == "alc-leading-hash":
        palette = metadata.get("palette")
        if not isinstance(palette, list) or not all(
            isinstance(entry, str) and re.fullmatch(r"[0-9A-Fa-f]{6}", entry) for entry in palette
        ):
            return "writer must emit hash-free rrggbb palette entries"
    return None


def score(encoder: str, *, readers: list[str] | None = None) -> list[tuple[str, str]]:
    """Return ``(variant, reason)`` for every failed writer comparison."""
    failures: list[tuple[str, str]] = []
    expectation_paths = sorted(DATA.glob("*.json"))
    if not expectation_paths:
        return [("corpus", "no valid variant expectations were found")]

    # Candidate code must never choose the oracle it is scored against. Read
    # every expectation and source meaning before invoking the first encoder,
    # then hand the process only a decoded, read-only request. Expectation bytes
    # are retained too so an attempted mutation of the committed oracle is a
    # visible failure even when the candidate later restores plausible output.
    expectations: list[tuple[pathlib.Path, bytes, object]] = []
    for expectation_path in expectation_paths:
        expectation_bytes = expectation_path.read_bytes()
        expectations.append((expectation_path, expectation_bytes, json.loads(expectation_bytes)))

    variants: list[tuple[str, object, object, tuple[np.ndarray, str] | None, str, str, bytes]] = []
    for expectation_path, _expectation_bytes, expected in expectations:
        name = expectation_path.stem
        candidates = sorted(DATA.glob(f"{name}.*.gz"))
        if not candidates:
            failures.append((name, "committed expectation has no generated input fixture"))
            continue
        source = candidates[0]
        channel = source.name.removesuffix(".gz").rsplit(".", 1)[-1]
        variants.append(
            (
                name,
                expected,
                _normalized_application_metadata(source, channel),
                _source_geometry_snapshot(source, channel),
                f"{name}.request.json",
                channel,
                _writer_request_bytes(source, channel),
            )
        )

    with tempfile.TemporaryDirectory(prefix="alf-encode-") as directory:
        output_root = pathlib.Path(directory)
        input_root = output_root / "inputs"
        candidate_root = output_root / "outputs"
        input_root.mkdir()
        candidate_root.mkdir()
        for (
            _name,
            _expected,
            _source_metadata,
            _source_geometry,
            request_name,
            _channel,
            request_bytes,
        ) in variants:
            request = input_root / request_name
            request.write_bytes(request_bytes)
            request.chmod(0o444)
        input_root.chmod(0o555)
        try:
            for (
                name,
                expected,
                source_metadata,
                source_geometry,
                request_name,
                channel,
                _request_bytes,
            ) in variants:
                request_path = input_root / request_name
                output = candidate_root / f"{name}.{channel}.gz"
                argv = [*shlex.split(encoder), str(request_path), channel, str(output)]
                try:
                    returncode, _stdout, stderr, overflow, timed_out = _run_bounded(
                        argv,
                        output_path=output,
                    )
                except OSError as error:
                    failures.append((name, f"encoder could not run: {error}"))
                    continue
                if overflow:
                    if "output" in overflow:
                        failures.append(
                            (
                                name,
                                f"encoder output exceeded the {MAX_CANDIDATE_BYTES}-byte file limit",
                            )
                        )
                    else:
                        streams = " and ".join(sorted(overflow))
                        failures.append(
                            (
                                name,
                                f"encoder {streams} exceeded the {MAX_CAPTURE_BYTES}-byte capture limit",
                            )
                        )
                    continue
                if timed_out:
                    failures.append((name, "encoder exceeded the 120-second timeout"))
                    continue
                if returncode != 0:
                    detail = stderr.strip()[-500:] or f"exit {returncode}"
                    failures.append((name, f"encoder failed: {detail}"))
                    continue
                if not output.is_file():
                    failures.append((name, "encoder exited zero without writing its output"))
                    continue
                try:
                    wire_problem = _writer_wire_problem(
                        name,
                        output,
                        source_geometry=source_geometry,
                        channel=channel,
                    )
                except Exception as error:  # candidate output is untrusted bytes
                    failures.append((name, f"candidate wire fields cannot be inspected: {error}"))
                    continue
                if wire_problem is not None:
                    failures.append((name, wire_problem))
                    continue
                try:
                    actual = canonical(output)
                except Exception as error:  # candidate output is untrusted bytes
                    failures.append((name, f"Python refused candidate output: {error}"))
                    continue
                comparable_actual = dict(actual)
                comparable_expected = dict(expected)
                if channel == "alp":
                    # A conforming writer may derive a different fixed-point
                    # grid, so its correct error bound need not equal the
                    # source grid's bound. The raw gate above checks the value
                    # against this candidate's decoded geometry and scale.
                    comparable_actual.pop("max_geom_error_m", None)
                    comparable_expected.pop("max_geom_error_m", None)
                if comparable_actual != comparable_expected:
                    failures.append(
                        (
                            name,
                            "canonical mismatch: "
                            "expected="
                            f"{json.dumps(comparable_expected, sort_keys=True)[:300]} "
                            f"actual={json.dumps(comparable_actual, sort_keys=True)[:300]}",
                        )
                    )
                    continue
                try:
                    candidate_metadata = _normalized_application_metadata(output, channel)
                except Exception as error:  # candidate output is untrusted bytes
                    failures.append((name, f"candidate metadata cannot be inspected: {error}"))
                    continue
                if candidate_metadata != source_metadata:
                    failures.append((name, "application metadata differs from source"))
                    continue
                if readers:
                    # Freeze the exact writer output before external readers
                    # run. One reader cannot rewrite another reader's input or
                    # move the Python oracle to match its own answer.
                    candidate_bytes = output.read_bytes()
                    reader_problems = []
                    for reader in readers:
                        decoded = _run_isolated(reader, f"{name}.{channel}.gz", candidate_bytes, channel)
                        if decoded != actual:
                            reader_problems.append(
                                f"reader {reader!r} canonical mismatch: "
                                f"expected={json.dumps(actual, sort_keys=True)[:300]} "
                                f"actual={json.dumps(decoded, sort_keys=True)[:300]}"
                            )
                    if reader_problems:
                        failures.append((name, "; ".join(reader_problems)))
        finally:
            try:
                input_root.chmod(0o755)
            except OSError:
                # Candidate code is untrusted and may remove or replace its
                # isolated input directory. Cleanup must not turn that into a
                # harness crash; the candidate's own comparison already fails.
                pass

    mutated = [
        expectation_path.name
        for expectation_path, expectation_bytes, _expected in expectations
        if not expectation_path.is_file() or expectation_path.read_bytes() != expectation_bytes
    ]
    if mutated:
        failures.append(("corpus", f"encoder mutated committed expectation(s): {', '.join(mutated)}"))
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--encoder",
        required=True,
        help="candidate command; request, channel, and output paths are appended",
    )
    parser.add_argument(
        "--reader",
        action="append",
        default=[],
        help="additional reader command; candidate path and channel are appended (repeatable)",
    )
    args = parser.parse_args()

    failures = score(args.encoder, readers=args.reader)
    for name, reason in failures:
        print(f"FAIL {name}: {reason}")
    checked = len(list(DATA.glob("*.json")))
    failed_variants = sum(name != "corpus" for name, _ in failures)
    print(f"\n{max(checked - failed_variants, 0)}/{checked} writer variants passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
