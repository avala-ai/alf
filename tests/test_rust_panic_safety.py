"""A malformed file must REFUSE, never crash.

The conformance corpus proves the reader gets the RULES right. This proves it
survives the bytes between them.

The distinction is the runner contract's whole point: a refusal is
`{"refused": id}` on exit 0, so the harness can diff it like any other answer.
A Rust panic exits 101 with a backtrace on stderr, which reads as "fell over" —
the outcome the invalid corpus exists to tell apart from "refused correctly".
Rust makes that failure mode specifically easy to reach: a slice index, an
integer cast or an arithmetic overflow on hostile input panics where the other
three languages raise.

Skipped when the release binary is absent, and the reason prints (`-ra`),
because a suite that skipped its evidence must not read like one that passed.
"""

from __future__ import annotations

import gzip
import json
import pathlib
import random
import struct
import subprocess
import tempfile

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_BINARY = _ROOT / "rust" / "alf" / "target" / "release" / "alf-conformance"
_CORPUS = _ROOT / "tests" / "conformance" / "data" / "alp-fp16.alp.gz"

pytestmark = pytest.mark.skipif(
    not _BINARY.is_file(),
    reason=f"no release binary at {_BINARY}; run `cargo build --release` in rust/alf",
)

_CHANNELS = ("alp", "ali", "alc", "all", "alx")


def _run(data: bytes, extension: str) -> tuple[int, str]:
    with tempfile.NamedTemporaryFile(suffix=f".{extension}.bin", delete=False) as handle:
        handle.write(data)
        path = handle.name
    try:
        done = subprocess.run([str(_BINARY), path, extension], capture_output=True, timeout=60, check=False)
        return done.returncode, done.stderr.decode(errors="replace")
    finally:
        pathlib.Path(path).unlink(missing_ok=True)


def _assert_refuses(data: bytes, extension: str, label: str) -> None:
    code, stderr = _run(data, extension)
    assert code == 0, (
        f"{label}: exited {code} instead of refusing. A non-zero exit is 'fell over', "
        f"which the harness cannot tell from a crash:\n{stderr[:400]}"
    )
    assert "panicked" not in stderr, f"{label}: panicked instead of refusing:\n{stderr[:400]}"


def _envelope(version: int, header: bytes, payload: bytes) -> bytes:
    raw = struct.pack("B", version) + struct.pack("<I", len(header)) + header + payload
    return gzip.compress(raw)


@pytest.mark.parametrize("cut", range(0, 120, 3))
def test_a_truncation_at_any_boundary_refuses(cut: int) -> None:
    """Every prefix of a real file, including mid-header and mid-payload."""
    real = _CORPUS.read_bytes()
    _assert_refuses(real[:cut], "alp", f"gzip truncated to {cut}")
    _assert_refuses(gzip.decompress(real)[:cut], "alp", f"raw truncated to {cut}")


@pytest.mark.parametrize("version", [0, 1, 2, 3, 4, 255])
@pytest.mark.parametrize("declared", [0, 1, 0x7FFFFFFF, 0xFFFFFFFF, 10_000_001])
def test_a_hostile_declared_header_length_refuses(version: int, declared: int) -> None:
    """The length is a CLAIM. Believing it is how a reader allocates 4 GiB."""
    raw = struct.pack("B", version) + struct.pack("<I", declared) + b'{"_total_points":1}' + b"\x00\x00"
    _assert_refuses(gzip.compress(raw), "alp", f"version={version} declared={declared}")


# Each of these targets a specific way a hand-written parser dies: unterminated
# input, a lone surrogate, a truncated escape, an exponent with no digits, and
# nesting deep enough to blow a recursive parser's stack.
_HOSTILE_HEADERS = [
    b"",
    b"{",
    b"}",
    b"[]",
    b"null",
    b'{"a"',
    b'{"a":}',
    b'{"a":1,}',
    b"\xff\xfe\xfd",
    b'{"a":"\xed\xa0\x80"}',
    b'{"a":"\\ud800"}',
    b'{"a":"\\u"}',
    b'{"a":"\\',
    b'{"a":1e}',
    b'{"a":-}',
    b'{"a":1.2.3}',
    b'{"_total_points":1e999999999999999999999}',
    b'{"_total_points":1e-999999999}',
    b"{" * 300 + b"}" * 300,
    b"[" * 300,
    b'{"_geom_offset":[1,2]}',
    b'{"device_position":[]}',
    b'{"palette":[1,2]}',
    b'{"instance_map":{"x":1}}',
]


@pytest.mark.parametrize("header", _HOSTILE_HEADERS, ids=range(len(_HOSTILE_HEADERS)))
@pytest.mark.parametrize("channel", _CHANNELS)
def test_a_hostile_header_refuses(header: bytes, channel: str) -> None:
    _assert_refuses(_envelope(1, header, b"\x00" * 6), channel, f"header={header[:40]!r}")


def test_random_bytes_and_mutations_refuse() -> None:
    """Byte soup, and single-bit damage to a file that IS valid.

    Seeded, so a failure is reproducible rather than a story about a build that
    once went red.
    """
    rng = random.Random(20260816)
    real = _CORPUS.read_bytes()

    for _ in range(120):
        size = rng.randint(0, 300)
        data = bytes(rng.getrandbits(8) for _ in range(size))
        _assert_refuses(data, rng.choice(_CHANNELS), f"random {size} bytes")

    for _ in range(120):
        mutated = bytearray(real)
        for _ in range(rng.randint(1, 12)):
            mutated[rng.randrange(len(mutated))] = rng.getrandbits(8)
        _assert_refuses(bytes(mutated), "alp", "mutated corpus file")


def test_a_valid_file_still_decodes() -> None:
    """The control. Without it, a reader that refused EVERYTHING would pass."""
    code, stderr = _run(_CORPUS.read_bytes(), "alp")
    assert code == 0, stderr[:400]
    done = subprocess.run([str(_BINARY), str(_CORPUS), "alp"], capture_output=True, timeout=60, check=False)
    answer = json.loads(done.stdout)
    assert answer["extension"] == "alp"
    assert answer["points"], "the control file decoded to no points"
