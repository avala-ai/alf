from __future__ import annotations

import gzip
import importlib.util
import json
import os
import pathlib
import re
import shlex
import shutil
import struct
import sys
import time


def _load_encode_roundtrip():
    path = pathlib.Path(__file__).parent / "conformance" / "encode_roundtrip.py"
    spec = importlib.util.spec_from_file_location("alf_encode_roundtrip", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_tool(name: str):
    path = pathlib.Path(__file__).parent / "conformance" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"alf_conformance_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_writer_score_refuses_a_committed_expectation_without_input(tmp_path, monkeypatch) -> None:
    roundtrip = _load_encode_roundtrip()
    (tmp_path / "missing-variant.json").write_text("{}\n")
    monkeypatch.setattr(roundtrip, "DATA", tmp_path)

    assert roundtrip.score("encoder-is-never-run") == [
        ("missing-variant", "committed expectation has no generated input fixture")
    ]


def test_writer_score_checks_other_readers_against_candidate_bytes(tmp_path, monkeypatch) -> None:
    roundtrip = _load_encode_roundtrip()
    header = b'{"_total_points":1}'
    raw = struct.pack("B", 1) + struct.pack("<I", len(header)) + header + bytes([7])
    source = tmp_path / "interop.ali.gz"
    source.write_bytes(gzip.compress(raw, mtime=0))
    expected = roundtrip.canonical(source)
    (tmp_path / "interop.json").write_text(json.dumps(expected))
    # Different wire bytes with identical meaning: the additional reader must
    # see what the candidate wrote, not the original conformance fixture.
    candidate = gzip.compress(raw, mtime=1)
    encoder = tmp_path / "encoder.py"
    encoder.write_text(f"import pathlib, sys\npathlib.Path(sys.argv[3]).write_bytes({candidate!r})\n")
    reader = tmp_path / "reader.py"
    reader.write_text(
        "import json, pathlib, sys\n"
        "path = pathlib.Path(sys.argv[1])\n"
        f"assert path.read_bytes() == {candidate!r}\n"
        "assert sys.argv[2] == 'ali'\n"
        # Each invocation must get its own pristine copy, even when another
        # reader overwrites the input it was handed.
        "path.chmod(0o644)\n"
        "path.write_bytes(b'corrupted')\n"
        f"print(json.dumps({expected!r}))\n"
    )
    monkeypatch.setattr(roundtrip, "DATA", tmp_path)
    reader_command = shlex.join([sys.executable, str(reader)])

    assert (
        roundtrip.score(
            shlex.join([sys.executable, str(encoder)]),
            readers=[reader_command, reader_command],
        )
        == []
    )
    assert source.read_bytes() == gzip.compress(raw, mtime=0)


def test_writer_score_fails_when_another_reader_disagrees(tmp_path, monkeypatch) -> None:
    roundtrip = _load_encode_roundtrip()
    header = b'{"_total_points":1}'
    raw = struct.pack("B", 1) + struct.pack("<I", len(header)) + header + bytes([7])
    source = tmp_path / "interop.ali.gz"
    source.write_bytes(gzip.compress(raw, mtime=0))
    (tmp_path / "interop.json").write_text(json.dumps(roundtrip.canonical(source)))
    encoder = tmp_path / "encoder.py"
    encoder.write_text(f"import pathlib, sys\npathlib.Path(sys.argv[3]).write_bytes({source.read_bytes()!r})\n")
    reader = tmp_path / "reader.py"
    reader.write_text('print(\'{"version": 1, "extension": "ali", "intensity": [99]}\')\n')
    monkeypatch.setattr(roundtrip, "DATA", tmp_path)

    failures = roundtrip.score(
        shlex.join([sys.executable, str(encoder)]),
        readers=[shlex.join([sys.executable, str(reader)])],
    )

    assert len(failures) == 1
    assert failures[0][0] == "interop"
    assert "reader" in failures[0][1] and "mismatch" in failures[0][1]


def test_writer_score_snapshots_oracle_and_isolates_encoder_input(tmp_path, monkeypatch) -> None:
    roundtrip = _load_encode_roundtrip()
    header = b'{"_total_points":1}'
    source_bytes = gzip.compress(
        struct.pack("B", 1) + struct.pack("<I", len(header)) + header + bytes([7]),
        mtime=0,
    )
    source = tmp_path / "hostile.ali.gz"
    source.write_bytes(source_bytes)
    expectation = tmp_path / "hostile.json"
    expectation.write_text(json.dumps({"version": 1, "extension": "ali", "intensity": [99]}) + "\n")
    replacement = json.dumps({"version": 1, "extension": "ali", "intensity": [7]}) + "\n"
    encoder = tmp_path / "hostile_encoder.py"
    encoder.write_text(
        "import pathlib, shutil, sys\n"
        "source, _channel, output = sys.argv[1:]\n"
        "shutil.copyfile(source, output)\n"
        "pathlib.Path(source).chmod(0o644)\n"
        "pathlib.Path(source).write_bytes(b'corrupted')\n"
        f"pathlib.Path({str(expectation)!r}).write_text({replacement!r})\n"
    )
    monkeypatch.setattr(roundtrip, "DATA", tmp_path)

    failures = roundtrip.score(shlex.join([sys.executable, str(encoder)]))

    assert failures and failures[0][0] == "hostile"
    assert "candidate" in failures[0][1]
    assert any(name == "corpus" and "mutated committed expectation" in reason for name, reason in failures)
    assert source.read_bytes() == source_bytes


def test_writer_score_refuses_candidate_that_copies_input_or_sibling(tmp_path, monkeypatch) -> None:
    """The writer gate must supply data, not an already acceptable answer."""
    roundtrip = _load_encode_roundtrip()
    header = b'{"_total_points":1}'
    source_bytes = gzip.compress(
        struct.pack("B", 1) + struct.pack("<I", len(header)) + header + bytes([7]),
        mtime=0,
    )
    for name in ("copy-input", "copy-sibling"):
        source = tmp_path / f"{name}.ali.gz"
        source.write_bytes(source_bytes)
        (tmp_path / f"{name}.json").write_text(json.dumps(roundtrip.canonical(source), sort_keys=True) + "\n")
    encoder = tmp_path / "copy_answer.py"
    encoder.write_text(
        "import pathlib, shutil, sys\n"
        "source, output = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[3])\n"
        "candidate = source\n"
        "if source.name.startswith('copy-sibling'):\n"
        "    candidate = next(path for path in source.parent.iterdir() if path != source)\n"
        "shutil.copyfile(candidate, output)\n"
    )
    monkeypatch.setattr(roundtrip, "DATA", tmp_path)

    failures = roundtrip.score(shlex.join([sys.executable, str(encoder)]))

    assert [name for name, _reason in failures] == ["copy-input", "copy-sibling"]
    assert all("candidate" in reason for _name, reason in failures)


def test_writer_score_snapshots_decoded_source_geometry(tmp_path, monkeypatch) -> None:
    roundtrip = _load_encode_roundtrip()
    python_root = pathlib.Path(__file__).parents[1] / "python"
    source = tmp_path / "geometry.alp.gz"
    header = b'{"_total_points":1,"box":"0:0:0:0:0:0","_box_frame":"local"}'
    source.write_bytes(
        gzip.compress(
            struct.pack("B", 1) + struct.pack("<I", len(header)) + header + bytes(6),
            mtime=0,
        )
    )
    (tmp_path / "geometry.json").write_text(json.dumps(roundtrip.canonical(source), sort_keys=True) + "\n")
    encoder = tmp_path / "overwrite_geometry.py"
    encoder.write_text(
        "import json, pathlib, sys\n"
        "import numpy as np\n"
        f"sys.path.insert(0, {str(python_root)!r})\n"
        "from alf import write_alp\n"
        "source, _channel, output = map(pathlib.Path, sys.argv[1:])\n"
        "request = json.loads(source.read_text())\n"
        "request['points'] = [5.960464477539063e-08] * 3\n"
        "source.chmod(0o644)\n"
        "source.write_text(json.dumps(request))\n"
        "write_alp(output, np.asarray(request['points']).reshape((-1, 3)), "
        "request['metadata'], codec=request['codec'], check_budget=False)\n"
    )
    monkeypatch.setattr(roundtrip, "DATA", tmp_path)

    failures = roundtrip.score(shlex.join([sys.executable, str(encoder)]))

    assert failures == [("geometry", "candidate decoded geometry differs from its source")]


def test_writer_score_refuses_dropped_application_metadata(tmp_path, monkeypatch) -> None:
    roundtrip = _load_encode_roundtrip()

    def encoded(header: dict[str, object]) -> bytes:
        blob = json.dumps(header, separators=(",", ":")).encode()
        raw = struct.pack("B", 1) + struct.pack("<I", len(blob)) + blob + bytes([7])
        return gzip.compress(raw, mtime=0)

    source = tmp_path / "metadata.ali.gz"
    source.write_bytes(
        encoded(
            {
                "_total_points": 1,
                "device_position": {"x": 9007199254740993, "y": 0, "z": 0},
            }
        )
    )
    (tmp_path / "metadata.json").write_text(json.dumps({"version": 1, "extension": "ali", "intensity": [7]}) + "\n")
    candidate = tmp_path / "candidate.bin"
    candidate.write_bytes(encoded({"_total_points": 1}))
    encoder = tmp_path / "drop_metadata.py"
    encoder.write_text(f"import pathlib, shutil, sys\nshutil.copyfile({str(candidate)!r}, pathlib.Path(sys.argv[3]))\n")
    monkeypatch.setattr(roundtrip, "DATA", tmp_path)

    failures = roundtrip.score(shlex.join([sys.executable, str(encoder)]))

    assert failures == [("metadata", "application metadata differs from source")]


def test_writer_score_recomputes_candidate_alp_box(tmp_path, monkeypatch) -> None:
    roundtrip = _load_encode_roundtrip()
    python_root = pathlib.Path(__file__).parents[1] / "python"
    source = tmp_path / "bounds.alp.gz"
    header = b'{"_total_points":2,"box":"1:2:3:4:5:6","_box_frame":"local"}'
    source.write_bytes(
        gzip.compress(
            struct.pack("B", 1) + struct.pack("<I", len(header)) + header + struct.pack("<eeeeee", 1, 2, 3, 4, 5, 6),
            mtime=0,
        )
    )
    (tmp_path / "bounds.json").write_text(json.dumps(roundtrip.canonical(source), sort_keys=True) + "\n")
    encoder = tmp_path / "stale_box.py"
    encoder.write_text(
        "import gzip, json, pathlib, struct, sys\n"
        "import numpy as np\n"
        f"sys.path.insert(0, {str(python_root)!r})\n"
        "from alf import write_alp\n"
        "source, _channel, output = map(pathlib.Path, sys.argv[1:])\n"
        "request = json.loads(source.read_text())\n"
        "write_alp(output, np.asarray(request['points']).reshape((-1, 3)), "
        "request['metadata'], codec=request['codec'], check_budget=False)\n"
        "raw = gzip.decompress(output.read_bytes())\n"
        "size = int.from_bytes(raw[1:5], 'little')\n"
        "header = json.loads(raw[5:5+size])\n"
        "header['box'] = '0:0:0:0:0:0'\n"
        "blob = json.dumps(header, separators=(',', ':')).encode()\n"
        "output.write_bytes(gzip.compress(raw[:1] + struct.pack('<I', len(blob)) + blob + raw[5+size:], mtime=0))\n"
    )
    monkeypatch.setattr(roundtrip, "DATA", tmp_path)

    failures = roundtrip.score(shlex.join([sys.executable, str(encoder)]))

    assert failures == [("bounds", "candidate box differs from its decoded geometry")]


def test_writer_score_requires_local_alp_box_frame(tmp_path, monkeypatch) -> None:
    roundtrip = _load_encode_roundtrip()
    python_root = pathlib.Path(__file__).parents[1] / "python"
    source = tmp_path / "frame.alp.gz"
    header = b'{"_total_points":1,"box":"1:2:3:1:2:3","_box_frame":"local"}'
    source.write_bytes(
        gzip.compress(
            struct.pack("B", 1) + struct.pack("<I", len(header)) + header + struct.pack("<eee", 1, 2, 3),
            mtime=0,
        )
    )
    (tmp_path / "frame.json").write_text(json.dumps(roundtrip.canonical(source), sort_keys=True) + "\n")
    encoder = tmp_path / "wrong_box_frame.py"
    encoder.write_text(
        "import gzip, json, pathlib, struct, sys\n"
        "import numpy as np\n"
        f"sys.path.insert(0, {str(python_root)!r})\n"
        "from alf import write_alp\n"
        "source, _channel, output = map(pathlib.Path, sys.argv[1:])\n"
        "request = json.loads(source.read_text())\n"
        "write_alp(output, np.asarray(request['points']).reshape((-1, 3)), "
        "request['metadata'], codec=request['codec'], check_budget=False)\n"
        "raw = gzip.decompress(output.read_bytes())\n"
        "size = int.from_bytes(raw[1:5], 'little')\n"
        "header = json.loads(raw[5:5+size])\n"
        "header['_box_frame'] = 'world'\n"
        "blob = json.dumps(header, separators=(',', ':')).encode()\n"
        "output.write_bytes(gzip.compress(raw[:1] + struct.pack('<I', len(blob)) + blob + raw[5+size:], mtime=0))\n"
    )
    monkeypatch.setattr(roundtrip, "DATA", tmp_path)

    failures = roundtrip.score(shlex.join([sys.executable, str(encoder)]))

    assert failures == [("frame", "non-empty ALP candidate must emit _box_frame local")]


def test_writer_metadata_comparison_normalizes_only_portable_values_and_alp_anchors(
    tmp_path,
) -> None:
    roundtrip = _load_encode_roundtrip()

    def write(path: pathlib.Path, channel: str, header: dict[str, object]) -> None:
        blob = json.dumps(header, separators=(",", ":")).encode()
        payload = bytes(6) if channel == "alp" else bytes([7])
        raw = struct.pack("B", 1) + struct.pack("<I", len(blob)) + blob + payload
        path.write_bytes(gzip.compress(raw, mtime=0))

    ali_exact = tmp_path / "exact.ali.gz"
    ali_rounded = tmp_path / "rounded.ali.gz"
    ali_changed_shape = tmp_path / "changed-shape.ali.gz"
    write(
        ali_exact,
        "ali",
        {
            "_total_points": 1,
            "device_position": {"x": 9007199254740993, "y": 0, "z": 0},
        },
    )
    write(
        ali_rounded,
        "ali",
        {
            "_total_points": 1,
            "device_position": {"x": 9007199254740992, "y": 0, "z": 0},
        },
    )
    write(
        ali_changed_shape,
        "ali",
        {"_total_points": 1, "device_position": [9007199254740992, 0, 0]},
    )
    assert roundtrip._normalized_application_metadata(ali_exact, "ali") == roundtrip._normalized_application_metadata(
        ali_rounded, "ali"
    )
    assert roundtrip._normalized_application_metadata(ali_exact, "ali") != roundtrip._normalized_application_metadata(
        ali_changed_shape, "ali"
    )

    alp_array = tmp_path / "array.alp.gz"
    alp_object = tmp_path / "object.alp.gz"
    write(alp_array, "alp", {"_total_points": 1, "device_position": [1, 2, 3]})
    write(
        alp_object,
        "alp",
        {"_total_points": 1, "device_position": {"x": 1, "y": 2, "z": 3}},
    )
    assert roundtrip._normalized_application_metadata(alp_array, "alp") == roundtrip._normalized_application_metadata(
        alp_object, "alp"
    )


def test_writer_runner_bounds_untrusted_output() -> None:
    roundtrip = _load_encode_roundtrip()
    command = [
        sys.executable,
        "-c",
        f"import sys;sys.stdout.write('x'*{roundtrip.MAX_CAPTURE_BYTES + 1})",
    ]

    returncode, stdout, stderr, overflow, timed_out = roundtrip._run_bounded(command, timeout=5)

    assert isinstance(returncode, int)
    assert len(stdout.encode()) == roundtrip.MAX_CAPTURE_BYTES
    assert stderr == ""
    assert overflow == {"stdout"}
    assert timed_out is False


def test_writer_runner_caps_candidate_file_growth(tmp_path) -> None:
    roundtrip = _load_encode_roundtrip()
    output = tmp_path / "hostile.ali.gz"
    command = [
        sys.executable,
        "-c",
        "import pathlib,sys;pathlib.Path(sys.argv[1]).write_bytes(b'x'*4096)",
        str(output),
    ]

    _returncode, _stdout, _stderr, overflow, timed_out = roundtrip._run_bounded(
        command,
        timeout=5,
        output_path=output,
        max_file_bytes=1024,
    )

    assert output.stat().st_size <= 1024
    assert overflow == {"output"}
    assert timed_out is False


def test_writer_wire_gate_sees_palette_before_reader_normalization(tmp_path) -> None:
    roundtrip = _load_encode_roundtrip()

    def write(palette: list[str], name: str) -> pathlib.Path:
        header = json.dumps({"_total_points": 1, "palette": palette}).encode()
        raw = struct.pack("B", 1) + struct.pack("<I", len(header)) + header + bytes([0])
        path = tmp_path / name
        path.write_bytes(gzip.compress(raw, mtime=0))
        return path

    assert (
        roundtrip._writer_wire_problem(
            "alc-leading-hash",
            write(["#FF0000"], "bad.alc.gz"),
            source_geometry=None,
            channel="alc",
        )
        == "writer must emit hash-free rrggbb palette entries"
    )
    assert (
        roundtrip._writer_wire_problem(
            "alc-leading-hash",
            write(["FF0000"], "good.alc.gz"),
            source_geometry=None,
            channel="alc",
        )
        is None
    )


def test_writer_wire_gate_requires_valid_v3_error_metadata(tmp_path) -> None:
    roundtrip = _load_encode_roundtrip()

    def write(metadata: dict[str, object], name: str) -> pathlib.Path:
        header = json.dumps(metadata, separators=(",", ":")).encode()
        raw = struct.pack("B", 3) + struct.pack("<I", len(header)) + header + bytes(6)
        path = tmp_path / name
        path.write_bytes(gzip.compress(raw, mtime=0))
        return path

    base: dict[str, object] = {
        "_total_points": 1,
        "_geom_codec": "int16",
        "_frame_convention": "world-aligned",
        "_geom_scale": 1,
        "_geom_offset": [0, 0, 0],
        "box": "0:0:0:0:0:0",
        "_box_frame": "local",
    }
    source = write({**base, "_geom_max_err_m": 0.5}, "source.alp.gz")
    assert (
        roundtrip._writer_wire_problem(
            "alp-int16",
            write(base, "missing.alp.gz"),
            source_geometry=roundtrip._source_geometry_snapshot(source, "alp"),
            channel="alp",
        )
        == "v3 writer must emit a finite non-negative _geom_max_err_m"
    )
    for index, invalid in enumerate((True, -1, float("nan"), 10**400)):
        metadata = {**base, "_geom_max_err_m": invalid}
        assert (
            roundtrip._writer_wire_problem(
                "alp-int16",
                write(metadata, f"bad-{index}.alp.gz"),
                source_geometry=roundtrip._source_geometry_snapshot(source, "alp"),
                channel="alp",
            )
            == "v3 writer must emit a finite non-negative _geom_max_err_m"
        )
    assert (
        roundtrip._writer_wire_problem(
            "alp-int16",
            write({**base, "_geom_max_err_m": 0.5}, "good.alp.gz"),
            source_geometry=roundtrip._source_geometry_snapshot(source, "alp"),
            channel="alp",
        )
        is None
    )


def test_writer_wire_gate_checks_measured_v3_error_value(tmp_path) -> None:
    roundtrip = _load_encode_roundtrip()

    def write(path: pathlib.Path, offset: float, measured: float, stored: int) -> None:
        decoded = stored * 0.1 + offset
        header = json.dumps(
            {
                "_total_points": 1,
                "_geom_codec": "int16",
                "_frame_convention": "world-aligned",
                "_geom_scale": 0.1,
                "_geom_offset": [offset, 0, 0],
                "_geom_max_err_m": measured,
                "box": f"{decoded}:0:0:{decoded}:0:0",
                "_box_frame": "local",
            },
            separators=(",", ":"),
        ).encode()
        raw = struct.pack("B", 3) + struct.pack("<I", len(header)) + header + struct.pack("<hhh", stored, 0, 0)
        path.write_bytes(gzip.compress(raw, mtime=0))

    source = tmp_path / "source.alp.gz"
    candidate = tmp_path / "candidate.alp.gz"
    write(source, 0.0, 0.05, 0)
    write(candidate, 0.0, 0.05, 1)

    assert (
        roundtrip._writer_wire_problem(
            "alp-measured-error",
            candidate,
            source_geometry=roundtrip._source_geometry_snapshot(source, "alp"),
            channel="alp",
        )
        == "v3 writer _geom_max_err_m understates independently measured reconstruction error"
    )


def test_writer_wire_gate_uses_only_ulp_scaled_error_allowance(tmp_path) -> None:
    """A fixed absolute tolerance erases the complete bound at tiny scales."""
    roundtrip = _load_encode_roundtrip()

    def write(path: pathlib.Path, offset: float, measured: float) -> None:
        header = json.dumps(
            {
                "_total_points": 1,
                "_geom_codec": "int16",
                "_frame_convention": "world-aligned",
                "_geom_scale": 5e-324,
                "_geom_offset": [offset, 0, 0],
                "_geom_max_err_m": measured,
                "box": f"{offset}:0:0:{offset}:0:0",
                "_box_frame": "local",
            },
            separators=(",", ":"),
        ).encode()
        raw = struct.pack("B", 3) + struct.pack("<I", len(header)) + header + bytes(6)
        path.write_bytes(gzip.compress(raw, mtime=0))

    source = tmp_path / "source.alp.gz"
    candidate = tmp_path / "candidate.alp.gz"
    write(source, 0.0, 5e-324)
    write(candidate, 5e-16, 5e-324)

    assert (
        roundtrip._writer_wire_problem(
            "alp-sub-femtometre-error",
            candidate,
            source_geometry=roundtrip._source_geometry_snapshot(source, "alp"),
            channel="alp",
        )
        == "v3 writer _geom_max_err_m understates independently measured reconstruction error"
    )


def test_writer_wire_gate_compares_geometry_before_canonical_rounding(tmp_path) -> None:
    roundtrip = _load_encode_roundtrip()

    def write(path: pathlib.Path, stored: int) -> None:
        decoded = struct.unpack("<e", struct.pack("<H", stored))[0]
        header = json.dumps(
            {
                "_total_points": 1,
                "box": f"{decoded}:{decoded}:{decoded}:{decoded}:{decoded}:{decoded}",
                "_box_frame": "local",
            },
            separators=(",", ":"),
        ).encode()
        raw = (
            struct.pack("B", 1) + struct.pack("<I", len(header)) + header + struct.pack("<HHH", stored, stored, stored)
        )
        path.write_bytes(gzip.compress(raw, mtime=0))

    source = tmp_path / "source.alp.gz"
    candidate = tmp_path / "candidate.alp.gz"
    write(source, 0x0000)
    write(candidate, 0x0001)

    assert (
        roundtrip._writer_wire_problem(
            "alp-fp16-wide-anchor",
            candidate,
            source_geometry=roundtrip._source_geometry_snapshot(source, "alp"),
            channel="alp",
        )
        == "candidate decoded geometry differs from its source"
    )


def test_writer_wire_gate_requires_positive_subnormal_half_scale_floor(tmp_path) -> None:
    roundtrip = _load_encode_roundtrip()

    def write(path: pathlib.Path, measured: float) -> None:
        header = json.dumps(
            {
                "_total_points": 1,
                "_geom_codec": "int16",
                "_frame_convention": "world-aligned",
                "_geom_scale": 5e-324,
                "_geom_offset": [0, 0, 0],
                "_geom_max_err_m": measured,
                "box": "0:0:0:0:0:0",
                "_box_frame": "local",
            },
            separators=(",", ":"),
        ).encode()
        raw = struct.pack("B", 3) + struct.pack("<I", len(header)) + header + bytes(6)
        path.write_bytes(gzip.compress(raw, mtime=0))

    source = tmp_path / "source.alp.gz"
    candidate = tmp_path / "candidate.alp.gz"
    write(source, 5e-324)
    write(candidate, 0.0)

    assert (
        roundtrip._writer_wire_problem(
            "alp-int16-positive-subnormal-scale",
            candidate,
            source_geometry=roundtrip._source_geometry_snapshot(source, "alp"),
            channel="alp",
        )
        == "v3 writer _geom_max_err_m understates the positive half-scale floor"
    )


def test_reader_gate_fails_missing_fixture_and_wrong_advertised_total(tmp_path, monkeypatch, capsys) -> None:
    runner = _load_tool("run")
    (tmp_path / "CHECKSUMS.txt").write_text("")
    (tmp_path / "missing.json").write_text("{}\n")
    monkeypatch.setattr(runner, "DATA", tmp_path)
    monkeypatch.setattr(runner, "ADVERTISED_VALID_CASES", 2)
    monkeypatch.setattr(sys, "argv", ["run.py", "--valid-only"])

    assert runner.main() == 1
    output = capsys.readouterr().out
    assert "FAIL missing" in output
    assert '"_fixture_missing": true' in output
    assert "FAIL corpus-total" in output
    assert "0/2 passed" in output


def test_reader_gate_snapshots_corpus_and_isolates_runner_input(tmp_path, monkeypatch, capsys) -> None:
    runner = _load_tool("run")
    (tmp_path / "CHECKSUMS.txt").write_text("snapshot\n")
    first_fixture = tmp_path / "a-first.ali.gz"
    first_fixture.write_bytes(b"first")
    (tmp_path / "a-first.json").write_text('{"case":"first"}\n')
    second_fixture = tmp_path / "b-second.ali.gz"
    second_fixture.write_bytes(b"original")
    second_expectation = tmp_path / "b-second.json"
    second_expectation.write_text('{"case":"original"}\n')
    observed_paths = tmp_path / "observed-paths.txt"
    hostile = tmp_path / "hostile_runner.py"
    hostile.write_text(
        "import json, pathlib, sys\n"
        "path = pathlib.Path(sys.argv[1])\n"
        f"with pathlib.Path({str(observed_paths)!r}).open('a') as stream: stream.write(str(path) + '\\n')\n"
        "if path.name.startswith('a-first'):\n"
        f"    pathlib.Path({str(second_fixture)!r}).write_bytes(b'forged')\n"
        f'    pathlib.Path({str(second_expectation)!r}).write_text(\'{{"case":"forged"}}\\n\')\n'
        "    print(json.dumps({'case': 'first'}))\n"
        "else:\n"
        "    print(json.dumps({'case': path.read_bytes().decode()}))\n"
    )
    monkeypatch.setattr(runner, "DATA", tmp_path)
    monkeypatch.setattr(runner, "ADVERTISED_VALID_CASES", 2)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run.py", "--valid-only", "--runner", shlex.join([sys.executable, str(hostile)])],
    )

    assert runner.main() == 1
    output = capsys.readouterr().out
    assert "runner mutated committed corpus" in output
    assert "2/2 passed" in output
    assert all(tmp_path not in pathlib.Path(line).parents for line in observed_paths.read_text().splitlines())


def test_reader_gate_bounds_both_runner_diagnostic_streams(tmp_path) -> None:
    runner = _load_tool("run")
    flood = 64 * 1024 + 1
    command = shlex.join(
        [
            sys.executable,
            "-c",
            f"import sys;sys.stdout.write('o'*{flood});sys.stderr.write('e'*{flood})",
        ]
    )

    actual = runner._run(command, tmp_path / "unused.ali.gz")

    assert actual == {"_runner_output_too_large": "runner diagnostics exceeded the 65536-byte capture limit"}


def test_bounded_runner_terminates_forked_pipe_descendants(tmp_path) -> None:
    """A successful parent must not leave a pipe-owning child behind."""
    bounded = _load_tool("bounded_process")
    child_pid_path = tmp_path / "child.pid"
    hostile = tmp_path / "forked_runner.py"
    hostile.write_text(
        "import os, pathlib, sys, time\n"
        "child_pid_path = pathlib.Path(sys.argv[1])\n"
        "child = os.fork()\n"
        "if child == 0:\n"
        "    sys.stderr.close()\n"
        "    child_pid_path.write_text(str(os.getpid()))\n"
        "    time.sleep(30)\n"
        "    os._exit(0)\n"
        "while not child_pid_path.exists():\n"
        "    time.sleep(0.001)\n"
        "os._exit(0)\n"
    )

    def child_is_active(pid: int) -> bool:
        try:
            # A killed orphan may remain briefly as a zombie until PID 1 reaps
            # it. It no longer owns a pipe or consumes candidate resources.
            # Also identify the hostile command: busy CI can reuse a dead PID
            # before this assertion observes it.
            state = pathlib.Path(f"/proc/{pid}/stat").read_text().split()[2]
            command = pathlib.Path(f"/proc/{pid}/cmdline").read_bytes()
            return state != "Z" and b"forked_runner.py" in command
        except (FileNotFoundError, ProcessLookupError):
            if pathlib.Path("/proc").is_dir():
                return False
            try:  # pragma: no cover - the conformance CI hosts are Linux
                os.kill(pid, 0)
                return True
            except ProcessLookupError:
                return False

    started = time.monotonic()
    child_pid: int | None = None
    try:
        returncode, _stdout, _stderr, overflow, timed_out = bounded.run_bounded(
            [sys.executable, str(hostile), str(child_pid_path)],
            timeout=2,
        )
        elapsed = time.monotonic() - started
        child_pid = int(child_pid_path.read_text())

        assert returncode == 0
        assert overflow == set()
        assert timed_out is False
        assert elapsed < 2
        deadline = time.monotonic() + 1
        while child_is_active(child_pid) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not child_is_active(child_pid)
    finally:
        if child_pid is not None and child_is_active(child_pid):
            os.kill(child_pid, 9)


def test_generate_verify_does_not_rewrite_a_drifted_baseline(tmp_path, monkeypatch) -> None:
    generator = _load_tool("generate")
    source = pathlib.Path(__file__).parent / "conformance" / "data"
    for path in source.rglob("*.json"):
        destination = tmp_path / path.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    shutil.copyfile(source / "CHECKSUMS.txt", tmp_path / "CHECKSUMS.txt")
    expectation = tmp_path / "alp-fp16.json"
    expectation.write_text('{"deliberately":"drifted"}\n')
    before = expectation.read_bytes()
    monkeypatch.setattr(generator, "DATA", tmp_path)
    monkeypatch.setattr(sys, "argv", ["generate.py", "--verify"])

    assert generator.main() == 1
    assert expectation.read_bytes() == before


def test_generate_missing_checksum_points_to_update(tmp_path, monkeypatch, capsys) -> None:
    generator = _load_tool("generate")
    monkeypatch.setattr(generator, "DATA", tmp_path)
    monkeypatch.setattr(generator, "build", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(sys, "argv", ["generate.py", "--verify"])

    assert generator.main() == 1
    assert "CHECKSUMS.txt missing; run generate.py --update first" in capsys.readouterr().err


def test_generate_prunes_corpus_files_whose_case_no_longer_exists(tmp_path, monkeypatch) -> None:
    """Removing a case used to leave its committed expectation and fixture behind.

    `run.py` discovers cases by GLOBBING the data directory, so the deleted rule
    kept being enforced against every implementation — and `--verify` could not
    see it either, because the checksum manifest is rebuilt from the same case
    lists and simply stops mentioning the orphan. Found while moving
    `fp16-coordinate-non-finite` out of the invalid corpus: the stale pair made
    the runner report 94 discovered against 93 advertised, and the only reason
    that surfaced at all is that the total is separately asserted.
    """
    generator = _load_tool("generate")
    monkeypatch.setattr(generator, "DATA", tmp_path)
    (tmp_path / "invalid").mkdir(parents=True, exist_ok=True)
    orphans = [
        tmp_path / "retired-valid-case.json",
        tmp_path / "retired-valid-case.alp.gz",
        tmp_path / "invalid" / "retired-invalid-case.json",
        tmp_path / "invalid" / "retired-invalid-case.alp.bin",
    ]
    for orphan in orphans:
        orphan.write_bytes(b"stale")

    generator.build(tmp_path)

    for orphan in orphans:
        assert not orphan.exists(), f"{orphan.name} survived a case removal"
    # And the live corpus is untouched by the pruning.
    assert (tmp_path / "alp-fp16.json").is_file()
    assert len(list(tmp_path.glob("*.gz"))) == 45


def test_generate_default_builds_fixtures_without_rewriting_oracle(tmp_path, monkeypatch) -> None:
    generator = _load_tool("generate")
    expectation = tmp_path / "alp-fp16.json"
    expectation.write_text('{"committed":"oracle"}\n')
    checksums = tmp_path / "CHECKSUMS.txt"
    checksums.write_text("committed checksums\n")
    before = (expectation.read_bytes(), checksums.read_bytes())
    monkeypatch.setattr(generator, "DATA", tmp_path)
    monkeypatch.setattr(sys, "argv", ["generate.py"])

    assert generator.main() == 0
    assert (expectation.read_bytes(), checksums.read_bytes()) == before
    assert len(list(tmp_path.glob("*.gz"))) == 45


def test_agents_regeneration_command_updates_the_committed_oracle() -> None:
    """The mandatory contributor workflow must name the intentional update mode."""
    agents = pathlib.Path(__file__).resolve().parents[1] / "AGENTS.md"
    assert "Regenerate: `python3 tests/conformance/generate.py --update`." in agents.read_text()


def test_generate_verify_materializes_fixtures_after_proving_oracle(tmp_path, monkeypatch) -> None:
    generator = _load_tool("generate")
    source = pathlib.Path(__file__).parent / "conformance" / "data"
    for path in source.rglob("*.json"):
        destination = tmp_path / path.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    shutil.copyfile(source / "CHECKSUMS.txt", tmp_path / "CHECKSUMS.txt")
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*.json")}
    monkeypatch.setattr(generator, "DATA", tmp_path)
    monkeypatch.setattr(sys, "argv", ["generate.py", "--verify"])

    assert generator.main() == 0
    assert {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*.json")} == before
    assert len(list(tmp_path.glob("*.gz"))) == 45
    assert len(list((tmp_path / "invalid").glob("*.bin"))) == 72


def test_published_evidence_totals_match_the_generated_contract() -> None:
    generator = _load_tool("generate")
    runner = _load_tool("run")
    valid = len(generator._valid_variants())
    invalid = len(generator.invalid_cases.CASES)
    total = valid + invalid
    root = pathlib.Path(__file__).parents[1]

    assert runner.ADVERTISED_VALID_CASES == valid
    assert runner.ADVERTISED_TOTAL_CASES == total
    # Derived from CI, not hardcoded. These counts used to be the literal 3, which
    # is the same staleness the numbers themselves are guarded against: adding the
    # Rust reader made README carry four "95/95" claims and this assertion failed
    # for being right. Tie them to what CI actually executes instead — a reader
    # count from `--runner`, a writer count from `--encoder` — so a fifth language
    # updates them by being wired up, and a doc that forgets it still fails.
    workflow = (root / ".github" / "workflows" / "ci.yml").read_text()
    readers = len({m for m in re.findall(r'--runner\s+"([^"]+)"', workflow)}) + 1  # +1: implicit Python
    # A loop's shell-variable invocation repeats the declared writers; it does
    # not advertise a fourth implementation named "$encoder".
    writers = len({m for m in re.findall(r'--encoder\s+"([^"]+)"', workflow) if not m.startswith("$")})

    readme = (root / "README.md").read_text()
    assert readme.count(f"{total}/{total} conformance cases") == readers
    badge_claim = f"{total}%2F{total}%20%C3%97%20{readers}%20languages"
    assert f"conformance-{badge_claim}-brightgreen" in readme
    support = (root / "spec" / "support.md").read_text()
    assert support.count(f"{total}/{total}") == readers
    # Rust is a reader with no writer, so the writer count trails the reader count.
    # That asymmetry is the fact worth pinning, not a number to keep in step.
    assert support.count(f"{valid}/{valid}") == writers
    conformance = (root / "tests" / "conformance" / "README.md").read_text()
    assert f"current corpus has {total} cases" in conformance
    assert f"and {invalid} one-rule invalid variants" in conformance
    normalized_conformance = " ".join(conformance.split())
    assert normalized_conformance.count(f"all {valid} valid variants") == 2

    sdk_template = (root / ".github" / "ISSUE_TEMPLATE" / "new-language-sdk.md").read_text()
    assert f'--runner "<their runner>"` passes {total}/{total}' in sdk_template
    assert f'--encoder "<theirs>"` passes {valid}/{valid}' in sdk_template
    security = (root / "SECURITY.md").read_text()
    assert f"invalid corpus in `tests/conformance/` is {invalid} cases" in security

    # The website is the public face of this evidence. It originally kept
    # advertising 95 cases after the corpus grew to 107 because only the root
    # README and specification were tied to the generated contract.
    website = root / "website"
    reference = (website / "docs" / "reference" / "conformance.md").read_text()
    assert f"**{total} reader cases**: {valid} valid" in reference
    assert f"and {invalid} invalid variants" in reference
    assert f"**{valid} writer variants**" in reference
    assert f"writer for all {valid} valid variants" in reference
    reference_index = (website / "docs" / "reference" / "index.md").read_text()
    assert f"through that gate at {total}/{total}" in reference_index
    homepage = (website / "src" / "pages" / "index.js").read_text()
    assert f"shared corpus of {total} cases — {valid} valid variants and {invalid}" in homepage


def test_normative_legacy_anchor_and_label_version_policy_match_the_corpus() -> None:
    generator = _load_tool("generate")
    assert "alp-legacy-empty-device-anchor" in generator._valid_variants()

    negative = next(case for case in generator.invalid_cases.CASES if case.name == "label-format-version-negative")
    assert negative.refusal == "label-layout-unsupported"
    assert b'"_format_version":-1.0' in negative.build()

    format_spec = " ".join((pathlib.Path(__file__).parents[1] / "spec" / "format.md").read_text().split())
    assert "legacy `device_position: {}`" in format_spec
    assert "`_format_version` shares only the exact-integral binary64 spelling rule" in format_spec
    assert "negative values such as `-1.0` are `label-layout-unsupported`" in format_spec


def test_normative_geometry_integral_spelling_policy_matches_the_corpus() -> None:
    generator = _load_tool("generate")
    invalid = {case.name: case.refusal for case in generator.invalid_cases.CASES}
    assert invalid["geom-offset-inexact-expanded-decimal"] == "geom-number-inexact"
    assert invalid["geom-scale-inexact-expanded-exponent"] == "geom-number-inexact"
    assert "alp-int16-exact-wide-integral-spellings" in generator._valid_variants()

    format_spec = " ".join((pathlib.Path(__file__).parents[1] / "spec" / "format.md").read_text().split())
    assert "Expanded integral decimal and exponent-zero tokens are exact integer claims" in format_spec
    assert "Normalized scientific notation is instead a binary64-preserving writer spelling" in format_spec


def test_hand_built_gzip_fixtures_pin_the_cross_python_os_byte() -> None:
    generator = _load_tool("generate")

    encoded = generator.invalid_cases._gzip(b"ALF")

    assert encoded[:3] == b"\x1f\x8b\x08"
    assert encoded[9] == 255
