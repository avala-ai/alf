"""Contract tests for the shared ALF reader/writer.

Three things are pinned here:

1. **v1 output matches the legacy writer's version byte, header layout and
   payload**, so stored files stay readable by every existing consumer. NOT
   byte equality: `box` comes from the decoded coordinates rather than the
   originals (deliberately — see `README.md`), and the gzip envelope is
   forced to `mtime=0`. The box difference is pinned explicitly below, because
   the earlier version of this test fed the writer's OWN header back into the
   legacy reproduction and so compared that field against itself.
2. **v3 int16 beats float16 by the margin claimed**, measured on realistic
   geometry rather than asserted in a comment.
3. The reader survives the shapes real files actually take: uncompressed label
   diffs, empty metadata, empty payloads, v2 interleaved labels, truncation.
"""

import gzip
import io
import json
import math
import os
import struct
import zlib

import numpy as np
import pytest

# Where `alf` lives, for subprocess probes that start a clean interpreter.
_PACKAGE_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python")

import alf
import alf._io as alf_io_module
from alf._io import (
    FRAME_WORLD_ALIGNED,
    GEOM_FP16,
    GEOM_INT16,
    INT16_NO_RETURN,
    MIN_GEOM_SCALE_M,
    AlfError,
    choose_geom_offset,
    choose_geom_scale,
    geom_error_report,
    map_intensity,
    read_alf,
    read_alf_header,
    write_alc,
    write_ali,
    write_all,
    write_alp,
    write_alx,
)


def _legacy_alp_bytes(local_xyz, metadata):
    """Byte-for-byte reproduction of the canonical v1 writer.

    Deliberately written the legacy way — per-point `struct.pack("<e")`, compact
    JSON separators, version byte 1 — so it is an INDEPENDENT expectation rather
    than a second call into the code under test.
    """
    header = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
    out = bytearray()
    out += struct.pack("B", 1)
    out += struct.pack("<I", len(header))
    out += header
    for x, y, z in local_xyz:
        out += struct.pack("<e", x)
        out += struct.pack("<e", y)
        out += struct.pack("<e", z)
    return bytes(out)


def _rewrite_metadata(path, mutate):
    """Re-emit an ALF file with its JSON header mutated, payload untouched.

    Corrupt-header cases have to be built by hand: the writer refuses to emit
    these shapes, which is the point of the reader-side check.
    """
    raw = gzip.decompress(path.read_bytes())
    version = raw[0]
    (mlen,) = struct.unpack("<I", raw[1:5])
    meta = json.loads(raw[5 : 5 + mlen].decode("utf-8"))
    payload = raw[5 + mlen :]
    mutate(meta)
    header = json.dumps(meta, separators=(",", ":")).encode("utf-8")
    out = struct.pack("B", version) + struct.pack("<I", len(header)) + header + payload
    path.write_bytes(gzip.compress(out))


def _road_scene(rng, n=4000):
    """A frame shaped like the data ALF actually carries.

    Ego-centric driving: wide in XY (out to ~100 m), shallow in Z (the sensor sits
    ~2 m up, the road is below it). That asymmetry is exactly why float16's
    per-COMPONENT error matters — Z stays precise while XY degrades with range.
    """
    xy = rng.uniform(-100.0, 100.0, size=(n, 2))
    z = rng.uniform(-2.5, 4.0, size=(n, 1))
    return np.hstack([xy, z])


class TestV1ByteCompatibility:
    def test_fp16_output_matches_the_legacy_writer(self, tmp_path):
        """Version byte, header layout and payload — NOT byte equality.

        The previous version of this test passed the writer's own returned
        header into `_legacy_alp_bytes`, so the `box` field was compared
        against itself and no discrepancy in it could ever fail. It also used
        coordinates that are all exactly representable in fp16, which hides
        the difference a second way.
        """
        xyz = np.array([[-2.0, 1.0, -0.5], [30.0, -12.0, 0.25], [0.5, 0.5, 4.0]])
        path = tmp_path / "000000.alp.gz"

        header = write_alp(path, xyz, {"timestamp": 1700000000000}, codec=GEOM_FP16)

        assert gzip.decompress(path.read_bytes()) == _legacy_alp_bytes(xyz, header)

    def test_box_comes_from_decoded_coordinates_not_the_input(self, tmp_path):
        """The one header field that deliberately differs from the legacy
        writers, which compute it from the ORIGINAL local coordinates.

        Both codecs round in either direction, so a pre-encoding extremum can
        sit inside the decoded extent, and the server turns this header
        straight into item-metadata bounds — a legacy-style box makes spatial
        filtering exclude real payload points. Pinned because `README.md`
        documents the difference and a migration comparing goldens will see it.
        """
        xyz = np.array([[0.1, 0.2, 0.3]])
        path = tmp_path / "000000.alp.gz"
        write_alp(path, xyz, codec=GEOM_FP16)

        box = read_alf(path).metadata["box"]
        assert box != "0.1:0.2:0.3:0.1:0.2:0.3", "legacy formula — see README"
        assert box.startswith("0.0999755859375:")
        # and it really does contain the payload it describes
        decoded = read_alf(path).xyz
        lo = [float(v) for v in box.split(":")[:3]]
        hi = [float(v) for v in box.split(":")[3:]]
        assert np.all(decoded >= lo) and np.all(decoded <= hi)

    def test_fp16_writes_version_byte_1_and_no_codec_keys(self, tmp_path):
        path = tmp_path / "000000.alp.gz"
        write_alp(path, np.zeros((2, 3)), codec=GEOM_FP16)

        alf = read_alf(path)
        assert alf.version == 1
        assert alf.geom_codec == GEOM_FP16
        assert "_geom_scale" not in alf.metadata


class TestGeometryPrecision:
    """The SOTA claim, measured."""

    def test_int16_is_far_more_accurate_than_fp16_at_the_same_byte_cost(self, tmp_path):
        rng = np.random.default_rng(7)
        xyz = _road_scene(rng)

        fp16_path = tmp_path / "a.alp.gz"
        int16_path = tmp_path / "b.alp.gz"
        write_alp(fp16_path, xyz, codec=GEOM_FP16)
        write_alp(int16_path, xyz, codec=GEOM_INT16, frame_convention=FRAME_WORLD_ALIGNED)

        fp16_err = np.abs(read_alf(fp16_path).xyz.astype(np.float64) - xyz).max()
        int16_err = np.abs(read_alf(int16_path).xyz.astype(np.float64) - xyz).max()

        # Same bytes per point — this is the whole point of the trade.
        assert len(gzip.decompress(fp16_path.read_bytes())) - len(
            json.dumps(read_alf(fp16_path).metadata)
        ) == pytest.approx(len(xyz) * 6, abs=64)

        # A ±100 m frame: int16 lands ~1.5 mm, fp16 ~3 cm at the far edge.
        assert int16_err < 0.002, f"int16 error {int16_err:.4f} m"
        assert fp16_err > 0.015, f"fp16 error {fp16_err:.4f} m — scene too small to be representative"
        assert fp16_err / int16_err > 10, f"expected >10x improvement, got {fp16_err / int16_err:.1f}x"

    def test_int16_error_is_uniform_while_fp16_degrades_with_distance(self, tmp_path):
        """The qualitative difference, not just the headline number.

        float16's error tracks each component's magnitude, so a distant point is
        far worse than a near one. int16's step is constant across the frame.
        """
        # Deliberately NOT round numbers: 1.0 and 100.0 are both exactly
        # representable in binary16, so a tidy-looking test would measure zero
        # error and prove nothing.
        near = np.array([[1.234, 1.234, 1.234]])
        far = np.array([[100.234, 100.234, 1.234]])
        both = np.vstack([near, far])

        for codec, expect_uniform in ((GEOM_INT16, True), (GEOM_FP16, False)):
            path = tmp_path / f"{codec}.alp.gz"
            write_alp(path, both, **_frame_kwargs(codec))
            decoded = read_alf(path).xyz.astype(np.float64)
            near_err = np.abs(decoded[0] - both[0]).max()
            far_err = np.abs(decoded[1] - both[1]).max()
            if expect_uniform:
                # Both bounded by the same step; near-zero error is allowed.
                scale = read_alf(path).metadata["_geom_scale"]
                assert near_err <= scale / 2 + 1e-12
                assert far_err <= scale / 2 + 1e-12
            else:
                assert far_err > near_err * 10

    def test_header_states_the_error_bound(self, tmp_path):
        """A consumer must be able to read the guarantee, not infer it."""
        rng = np.random.default_rng(11)
        xyz = _road_scene(rng)
        path = tmp_path / "000000.alp.gz"
        write_alp(path, xyz, codec=GEOM_INT16, frame_convention=FRAME_WORLD_ALIGNED)

        alf = read_alf(path)
        actual = np.abs(alf.xyz.astype(np.float64) - xyz).max()
        assert actual <= alf.metadata["_geom_max_err_m"] + 1e-12
        assert alf.max_geom_error_m == pytest.approx(alf.metadata["_geom_max_err_m"])

    def test_fp16_files_report_their_magnitude_dependent_bound(self, tmp_path):
        path = tmp_path / "000000.alp.gz"
        write_alp(path, np.array([[100.0, 0.0, 0.0]]), codec=GEOM_FP16)
        # half-ULP of binary16 in the [64, 128) band is 3.125 cm
        assert read_alf(path).max_geom_error_m == pytest.approx(0.03125)

    def test_error_report_matches_what_the_codecs_actually_do(self, tmp_path):
        rng = np.random.default_rng(3)
        xyz = _road_scene(rng, n=500)
        report = geom_error_report(xyz)

        for codec in (GEOM_FP16, GEOM_INT16):
            path = tmp_path / f"{codec}.alp.gz"
            write_alp(path, xyz, **_frame_kwargs(codec))
            measured = np.abs(read_alf(path).xyz.astype(np.float64) - xyz).max()
            assert measured == pytest.approx(report[codec], abs=1e-9)

    def test_scale_floor_stops_pointless_precision_on_tiny_frames(self):
        # A 1 cm frame would otherwise get a sub-nanometre step, spending int16
        # range on precision no sensor delivers.
        assert choose_geom_scale(np.array([[0.01, 0.0, 0.0]])) == 1e-4
        assert choose_geom_scale(np.zeros((0, 3))) == 1e-4


class TestBoxHeader:
    def test_box_is_local_frame_and_flagged(self, tmp_path):
        xyz = np.array([[-2.0, 1.0, -0.5], [3.0, -4.0, 2.0]])
        path = tmp_path / "000000.alp.gz"
        write_alp(path, xyz, {"device_position": {"x": 100.0, "y": -250.0, "z": 3.5}})

        alf = read_alf(path)
        assert alf.metadata["_box_frame"] == "local"
        assert [float(v) for v in alf.metadata["box"].split(":")] == [
            -2.0,
            -4.0,
            -0.5,
            3.0,
            1.0,
            2.0,
        ]

    def test_box_describes_the_payload_that_follows_it(self, tmp_path):
        """The E1 bug class: bounds must match the encoded bytes, not the input
        the caller happened to hold."""
        rng = np.random.default_rng(5)
        xyz = _road_scene(rng, n=200)
        path = tmp_path / "000000.alp.gz"
        write_alp(path, xyz, codec=GEOM_INT16, frame_convention=FRAME_WORLD_ALIGNED)

        alf = read_alf(path)
        lo = [float(v) for v in alf.metadata["box"].split(":")][:3]
        hi = [float(v) for v in alf.metadata["box"].split(":")][3:]
        decoded = alf.xyz.astype(np.float64)
        tol = alf.metadata["_geom_max_err_m"]
        assert (decoded.min(axis=0) >= np.array(lo) - tol).all()
        assert (decoded.max(axis=0) <= np.array(hi) + tol).all()

    def test_uncentered_coordinates_are_refused_for_fp16(self, tmp_path):
        with pytest.raises(ValueError, match="localization"):
            write_alp(tmp_path / "000000.alp.gz", np.array([[5000.0, 1.0, 2.0]]), codec=GEOM_FP16)
        assert not (tmp_path / "000000.alp.gz").exists()


class TestWorldFrame:
    def test_world_xyz_adds_device_position_in_float64(self, tmp_path):
        xyz = np.array([[1.0, 2.0, 3.0]])
        dp = {"x": 1_000_000.5, "y": -250.25, "z": 3.5}
        path = tmp_path / "000000.alp.gz"
        write_alp(path, xyz, {"device_position": dp})

        world = read_alf(path).world_xyz(sensor_pose=True)
        assert world.dtype == np.float64
        # float32 would lose the .5 at 1e6; float64 keeps it.
        assert world[0][0] == pytest.approx(1_000_001.5, abs=1e-3)


class TestIntensity:
    def test_raw_bounds_are_not_written_to_the_header(self, tmp_path):
        """No legacy writer emits raw bounds, so recording them would add two
        keys to every ported .ali and break header compatibility. `lo`/`hi`
        select the encoding without adding wire fields."""
        path = tmp_path / "000000.ali.gz"
        write_ali(path, np.array([10.0, 200.0]), lo=10.0, hi=200.0)

        alf = read_alf(path)
        assert list(alf.intensity) == [0, 255]
        assert "_raw_intensity_min" not in alf.metadata
        assert "_raw_intensity_max" not in alf.metadata
        assert alf.metadata["_min_intensity"] == 0
        assert alf.metadata["_max_intensity"] == 255
        assert len(alf.metadata["_intensities_every_10_percent"]) == 9

    def test_float_division_matches_the_canonical_generator(self, tmp_path):
        """Two aerial writers used integer division and came out one low."""
        path = tmp_path / "000000.ali.gz"
        write_ali(path, np.array([128.0]), lo=0.0, hi=255.0)
        assert read_alf(path).intensity[0] == 128

    def test_degenerate_range_does_not_divide_by_zero(self, tmp_path):
        path = tmp_path / "000000.ali.gz"
        write_ali(path, np.array([5.0, 5.0]), lo=5.0, hi=5.0)
        assert list(read_alf(path).intensity) == [0, 0]


class TestLabels:
    def test_v1_roundtrip(self, tmp_path):
        path = tmp_path / "000000.all.gz"
        write_all(path, np.array([0, 3, 255], dtype=np.uint8))

        alf = read_alf(path)
        assert alf.format_version == 1
        assert list(alf.categories) == [0, 3, 255]
        assert alf.instance_ids is None

    def test_v2_interleaves_category_and_little_endian_instance(self, tmp_path):
        path = tmp_path / "000000.all.gz"
        write_all(
            path,
            np.array([1, 2], dtype=np.uint8),
            instance_ids=np.array([1, 65534], dtype=np.uint16),
            instance_map={"1": "uuid-a", "65534": "uuid-b"},
        )

        alf = read_alf(path)
        assert alf.format_version == 2
        assert list(alf.categories) == [1, 2]
        assert list(alf.instance_ids) == [1, 65534]
        assert alf.instance_map["65534"] == "uuid-b"

    def test_v2_payload_truncated_is_rejected_against_the_declared_count(self, tmp_path):
        """A byte-level truncation is caught by the header cross-check, which is
        stricter than the multiple-of-3 shape test and names the real problem."""
        path = tmp_path / "000000.all.gz"
        write_all(
            path,
            np.array([1, 2], dtype=np.uint8),
            instance_ids=np.array([1, 2], dtype=np.uint16),
            instance_map={"1": "uuid-a", "2": "uuid-b"},
        )
        raw = bytearray(gzip.decompress(path.read_bytes()))
        path.write_bytes(gzip.compress(bytes(raw[:-1])))

        with pytest.raises(AlfError, match="payload holds"):
            read_alf(path)

    def test_v2_shape_is_still_checked_without_a_declared_count(self, tmp_path):
        """With no `_total_points` there is nothing to cross-check against, so the
        multiple-of-3 guard is the only thing standing between a malformed
        payload and invented instance ids."""
        header = json.dumps({"_format_version": 2}, separators=(",", ":")).encode()
        payload = bytes([1, 0, 0, 2])  # 4 bytes: not a whole number of points
        path = tmp_path / "000000.all"
        path.write_bytes(struct.pack("B", 2) + struct.pack("<I", len(header)) + header + payload)

        with pytest.raises(AlfError, match="multiple of 3"):
            _ = read_alf(path).categories

    def test_mismatched_instance_length_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="must match"):
            write_all(
                tmp_path / "000000.all.gz",
                np.array([1, 2], dtype=np.uint8),
                instance_ids=np.array([1], dtype=np.uint16),
            )


class TestReaderRobustness:
    def test_reads_uncompressed_files(self, tmp_path):
        """Label diffs land in `_work_tracking/` UNCOMPRESSED while the merged
        canonical object is gzipped — both arrive as `.all`."""
        path = tmp_path / "000000.all.gz"
        write_all(path, np.array([7], dtype=np.uint8))
        plain = tmp_path / "diff.all"
        plain.write_bytes(gzip.decompress(path.read_bytes()))

        assert list(read_alf(plain).categories) == [7]

    def test_empty_metadata_block(self, tmp_path):
        path = tmp_path / "000000.alp"
        path.write_bytes(struct.pack("B", 1) + struct.pack("<I", 0))
        alf = read_alf(path)
        assert alf.metadata == {}
        assert alf.xyz.shape == (0, 3)

    def test_empty_payload_roundtrips(self, tmp_path):
        path = tmp_path / "000000.alp.gz"
        write_alp(path, np.zeros((0, 3)))
        alf = read_alf(path)
        assert alf.total_points == 0
        assert alf.xyz.shape == (0, 3)
        assert "box" not in alf.metadata

    def test_unsupported_version_is_named(self, tmp_path):
        path = tmp_path / "000000.alp"
        path.write_bytes(struct.pack("B", 31) + struct.pack("<I", 0))
        with pytest.raises(AlfError, match="Unsupported ALF version byte: 31"):
            read_alf(path)

    def test_truncated_header_is_refused(self, tmp_path):
        path = tmp_path / "000000.alp"
        path.write_bytes(struct.pack("B", 1) + struct.pack("<I", 999))
        with pytest.raises(AlfError, match="Truncated header"):
            read_alf(path)

    def test_too_short_to_be_alf(self, tmp_path):
        path = tmp_path / "000000.alp"
        path.write_bytes(b"\x01\x00")
        with pytest.raises(AlfError, match="Too short"):
            read_alf(path)

    def test_non_ascii_metadata_survives_the_roundtrip(self, tmp_path):
        """The wire format says UTF-8. A TypeScript consumer decodes with
        `String.fromCharCode`, which is latin-1 — this pins the correct behaviour
        the shared corpus will hold it to."""
        path = tmp_path / "000000.alp.gz"
        write_alp(path, np.zeros((1, 3)), {"note": "señal — 测试"})
        assert read_alf(path).metadata["note"] == "señal — 测试"

    def test_stale_output_is_removed_before_writing(self, tmp_path):
        path = tmp_path / "000000.alp.gz"
        path.write_bytes(b"stale-not-even-gzip")
        write_alp(path, np.array([[1.0, 2.0, 3.0]]))
        assert read_alf(path).total_points == 1


def _written(path, xyz, **kwargs):
    """Write `xyz` as int16 geometry and hand back the path, for one-liners."""
    write_alp(path, xyz, codec=GEOM_INT16, frame_convention=FRAME_WORLD_ALIGNED, check_budget=False, **kwargs)
    return path


def _frame_kwargs(codec):
    """`write_alp` kwargs for a codec, for tests that loop over both.

    v3 must state its frame convention and v1 must not carry one, so a loop
    over the two codecs cannot pass the same arguments to both.
    """
    if codec == GEOM_INT16:
        return {"codec": codec, "frame_convention": FRAME_WORLD_ALIGNED}
    return {"codec": codec}


def _v3_envelope(words, *, points, scale=0.01, offset=(0.0, 0.0, 0.0)):
    """A raw v3 `.alp` from literal int16 words.

    Hand-built precisely because a conforming WRITER cannot produce the payloads
    these tests are about — a partial `INT16_NO_RETURN` is the shape no encoder
    may emit, so it has to be forged to test that the reader rejects it.
    """
    header = json.dumps(
        {
            "_total_points": points,
            "_geom_codec": "int16",
            "_geom_scale": scale,
            "_geom_offset": list(offset),
            # Required on v3. These fixtures are about geometry rules, so they
            # take the convention that changes nothing about the coordinates.
            "_frame_convention": "world-aligned",
        }
    ).encode()
    payload = np.asarray(words, dtype="<i2").tobytes()
    return struct.pack("B", 3) + struct.pack("<I", len(header)) + header + payload


class TestProductionSafety:
    """The properties that keep existing v1/v2 datasets and readers working.

    Every case here is a way this module could silently break something that is
    correct on disk today. They matter more than the precision work: a wrong
    number is worse than a coarse one.
    """

    def test_default_codec_is_v1_so_a_ported_writer_changes_nothing(self, tmp_path):
        """Safe by DEFAULT.

        Porting a legacy writer is mechanical, and a migration that forgets
        `codec=` must not start emitting a version a legacy reader cannot read.
        A reader that ignores the version and assumes float16 returns wrong
        coordinates without necessarily raising an error.
        """
        path = tmp_path / "000000.alp.gz"
        write_alp(path, np.array([[1.0, 2.0, 3.0]]))

        alf = read_alf(path)
        assert alf.version == 1, "default must stay v1 until every reader is migrated"
        assert alf.geom_codec == GEOM_FP16
        assert "_geom_codec" not in alf.metadata

    def test_v2_labels_are_detected_from_the_envelope_byte_alone(self, tmp_path):
        """A v2 `.all` whose header omits `_format_version`.

        The legacy export parsers identify v2 from the version BYTE and synthesize
        the metadata key afterwards, so such files exist. Reading them as v1
        would hand back the interleaved `[cat, lo, hi, ...]` stream as categories
        and `None` for instances — corrupting labels that are fine on disk.
        """
        payload = bytes([7, 0x01, 0x00, 9, 0xFE, 0xFF])  # 2 points: (7, 1), (9, 65534)
        raw = struct.pack("B", 2) + struct.pack("<I", 0) + payload
        path = tmp_path / "000000.all"
        path.write_bytes(raw)

        alf = read_alf(path)
        assert alf.format_version == 2
        assert list(alf.categories) == [7, 9]
        assert list(alf.instance_ids) == [1, 65534]

    def test_unresolvable_layout_mismatch_fails_closed(self, tmp_path):
        """Byte 2 + `_format_version: 1`, no `_total_points`, and a 3-byte
        payload that fits BOTH layouts. Nothing separates them, so the reader
        refuses rather than letting `categories` and `instance_ids` reinterpret
        one payload two ways."""
        header = json.dumps({"_format_version": 1}, separators=(",", ":")).encode()
        path = tmp_path / "000000.all"
        path.write_bytes(struct.pack("B", 2) + struct.pack("<I", len(header)) + header + bytes([1, 2, 3]))

        with pytest.raises(AlfError, match="fits BOTH layouts"):
            read_alf(path).categories

    def test_a_partly_invalid_point_is_refused_by_int16(self, tmp_path):
        """A whole invalid point is content; half of one is not.

        This class previously asserted that int16 refused ANY non-finite
        coordinate, on the reasoning that fixed-point has no encoding for an
        invalid return. It has one now — `INT16_NO_RETURN`, a word the encoder
        could never emit for a real coordinate. A dense range image's empty
        slots preserve its beam/azimuth mapping and companion channel indices.

        A partly non-finite point stays refused: the sentinel describes a whole
        missing position, so half a coordinate has no encoding.
        """
        path = tmp_path / "000000.alp.gz"
        with pytest.raises(ValueError, match="SOME but not all"):
            write_alp(path, np.array([[1.0, 2.0, np.nan]]), codec=GEOM_INT16, frame_convention=FRAME_WORLD_ALIGNED)
        assert not path.exists()

    def test_both_codecs_round_trip_an_invalid_return_identically(self, tmp_path):
        """The point of the sentinel: same frame in, same array out."""
        for bad in (np.nan, np.inf, -np.inf):
            xyz = np.array([[1.0, 2.0, 3.0], [bad, bad, bad], [4.0, 5.0, 6.0]])
            fp16_path = tmp_path / "a.alp.gz"
            int16_path = tmp_path / "b.alp.gz"
            write_alp(fp16_path, xyz)
            write_alp(int16_path, xyz, codec=GEOM_INT16, frame_convention=FRAME_WORLD_ALIGNED)

            from_fp16 = read_alf(fp16_path).xyz
            from_int16 = read_alf(int16_path).xyz
            # Both mark slot 1 invalid, and both keep slots 0 and 2 where they
            # were — index alignment is the whole reason the slot is kept.
            # fp16 stores +/-Inf AS Inf; int16's sentinel decodes to NaN. Both
            # say "no return", which is the property that matters — asserting
            # NaN specifically would be asserting a storage detail.
            assert not np.isfinite(from_fp16[1]).any()
            assert not np.isfinite(from_int16[1]).any()
            assert np.isfinite(from_int16[[0, 2]]).all()
            np.testing.assert_allclose(from_int16[[0, 2]], xyz[[0, 2]], atol=1e-3)
            fp16_path.unlink()
            int16_path.unlink()

    def test_an_infinite_invalid_return_sizes_the_grid_like_a_nan_one(self, tmp_path):
        """Two spellings of "no return" must not have opposite outcomes.

        `nanmin`/`nanmax` skip NaN but NOT infinities, so an infinite invalid
        return survived the reduction, made `hi` infinite, and sent
        `choose_geom_offset` down its "every return was invalid" branch — a zero
        origin for a frame full of real positions near 1e9. The writer then
        refused the frame outright ("step 30518 m gives 15259 m error") while
        the identical frame spelled with NaN encoded fine.
        """
        real = np.array([[1e9, 1e9, 1e9], [1e9 + 10, 1e9 + 10, 1e9 + 10]])
        baseline = read_alf(_written(tmp_path / "clean.alp.gz", real)).xyz

        for marker in (np.nan, np.inf, -np.inf):
            frame = np.vstack([real, [marker, marker, marker]])
            decoded = read_alf(_written(tmp_path / f"m{marker}.alp.gz", frame)).xyz

            assert not np.isfinite(decoded[2]).any(), "the invalid return stays invalid"
            # And the REAL points are unmoved by which marker sat beside them.
            np.testing.assert_allclose(decoded[:2], baseline, rtol=0, atol=1e-6)

    def test_a_partial_marker_is_refused_rather_than_decoded_as_a_position(self, tmp_path):
        """The corruption nothing downstream can catch.

        `-32768` is reserved and the encoder clamps to +/-32767, so no
        conforming writer emits it as a coordinate. Reconstructing it anyway
        gives `-327.68` at `_geom_scale: 0.01` — finite, plausible, and never
        measured. Pinned as conformance case `invalid/geom-partial-no-return`.
        """
        for words in ([-32768, 0, 0], [0, -32768, 0], [0, 0, -32768]):
            path = tmp_path / "partial.alp"
            path.write_bytes(_v3_envelope(words, points=1))
            with pytest.raises(AlfError, match="SOME but not all"):
                read_alf(path).xyz

    def test_the_error_bound_ignores_sentinel_rows(self, tmp_path):
        """A file whose `.xyz` decodes cannot be "too large to bound".

        The bound reconstructed every row including the markers, so a large but
        finite `_geom_scale` overflowed on `-32768 * scale` and the reader
        declared the geometry unbounded — for a file it had just decoded
        perfectly to NaN. Two answers about one file, one of them wrong.
        """
        mixed = tmp_path / "mixed.alp"
        mixed.write_bytes(_v3_envelope([100, 200, 300, -32768, -32768, -32768], points=2))
        real_only = tmp_path / "real.alp"
        real_only.write_bytes(_v3_envelope([100, 200, 300], points=1))

        # The marker row contributes no error, so the bound is what the frame
        # without it reports — not larger, and not an exception.
        assert read_alf(mixed).max_geom_error_m == read_alf(real_only).max_geom_error_m

        huge = tmp_path / "huge.alp"
        huge.write_bytes(_v3_envelope([-32768] * 3, points=1, scale=1e308))
        assert np.isnan(read_alf(huge).xyz).all()
        read_alf(huge).max_geom_error_m  # must not raise

    def test_a_frame_of_nothing_but_invalid_returns_is_writable(self, tmp_path):
        """A fully-occluded sweep is legal content, not an error.

        The degenerate end of the range, and the one that breaks a writer whose
        origin comes from `min`/`max` seeds: with no finite component anywhere,
        the seeds stay at +/-inf and their midpoint is NaN, which becomes the
        header origin. Every word here is the sentinel, so any finite origin
        decodes identically — but a NaN one poisons the header and can make the
        writer reject its own output. Pinned in Dart and TypeScript too; both
        had exactly this bug while Python did not.
        """
        path = tmp_path / "occluded.alp.gz"
        write_alp(
            path, np.full((4, 3), np.nan), codec=GEOM_INT16, frame_convention=FRAME_WORLD_ALIGNED, check_budget=False
        )

        decoded = read_alf(path).xyz
        assert decoded.shape == (4, 3), "invalid returns keep their index"
        assert np.isnan(decoded).all()

    def test_the_sentinel_is_a_word_the_encoder_cannot_otherwise_emit(self, tmp_path):
        """Free by construction, and this is what makes it safe.

        `choose_geom_scale` derives the step as `largest / 32767`, so a real
        coordinate quantises into [-32767, +32767]. If that symmetry ever broke,
        a genuine coordinate could collide with the marker and decode as an
        invalid return — silently losing a point rather than raising.
        """
        rng = np.random.default_rng(0)
        xyz = rng.uniform(-500.0, 500.0, size=(20_000, 3))
        path = tmp_path / "dense.alp.gz"
        write_alp(path, xyz, codec=GEOM_INT16, frame_convention=FRAME_WORLD_ALIGNED)
        raw = gzip.decompress(path.read_bytes())
        header_len = struct.unpack("<I", raw[1:5])[0]
        words = np.frombuffer(raw[5 + header_len :], dtype="<i2")
        assert words.min() > INT16_NO_RETURN, "a real coordinate reached the sentinel word"

    def test_out_of_range_categories_are_refused_not_wrapped(self, tmp_path):
        """`asarray(300, dtype=uint8)` is 44 — a different, valid-looking class."""
        with pytest.raises(ValueError, match="uint8"):
            write_all(tmp_path / "a.all.gz", np.array([1, 300]))
        with pytest.raises(ValueError, match="uint8"):
            write_all(tmp_path / "b.all.gz", np.array([-1, 2]))

    def test_out_of_range_instance_ids_are_refused(self, tmp_path):
        with pytest.raises(ValueError, match="uint16"):
            write_all(
                tmp_path / "a.all.gz",
                np.array([1, 2], dtype=np.uint8),
                instance_ids=np.array([1, 70000]),
            )

    def test_int16_refuses_a_frame_whose_error_would_be_unusable(self, tmp_path):
        """int16 is judged on the error it produces, not on magnitude — the
        honest metric, since it states that error in the header.

        Since the grid is centred on the frame, the budget tracks the frame's
        SPAN rather than its distance from the origin, which is the quantity
        quantization error actually depends on."""
        path = tmp_path / "000000.alp.gz"
        with pytest.raises(ValueError, match="localization failed"):
            write_alp(
                path,
                np.array([[-20_000.0, 0.0, 0.0], [20_000.0, 0.0, 0.0]]),
                codec=GEOM_INT16,
                frame_convention=FRAME_WORLD_ALIGNED,
            )
        assert not path.exists()

    def test_a_distant_but_compact_frame_is_accepted(self, tmp_path):
        """The mirror of the case above: far from the origin but only metres
        across. Before centring this was refused for its magnitude alone, even
        though int16 represents it to well under a millimetre."""
        path = tmp_path / "000000.alp.gz"
        xyz = np.array([[20_000.0, 0.0, 0.0], [20_010.0, 5.0, 2.0]])
        write_alp(path, xyz, codec=GEOM_INT16, frame_convention=FRAME_WORLD_ALIGNED)
        alf = read_alf(path)
        assert np.abs(alf.xyz - xyz).max() < 1e-3

    def test_a_rejected_rerun_does_not_leave_the_previous_frame_behind(self, tmp_path):
        """Validation happens AFTER the unlink, so a failed re-run cannot leave a
        stale file that downstream globbing treats as current."""
        path = tmp_path / "000000.alp.gz"
        write_alp(path, np.array([[1.0, 2.0, 3.0]]), codec=GEOM_FP16)
        assert path.exists()

        with pytest.raises(ValueError):
            write_alp(path, np.array([[5000.0, 1.0, 2.0]]), codec=GEOM_FP16)
        assert not path.exists(), "stale output survived a rejected rerun"

    def test_v1_and_v2_label_files_written_today_still_read_back(self, tmp_path):
        """Round-trip both label layouts through the writer and reader together,
        so a change to either side that breaks the pair is caught here."""
        v1 = tmp_path / "v1.all.gz"
        write_all(v1, np.array([0, 128, 255], dtype=np.uint8))
        assert list(read_alf(v1).categories) == [0, 128, 255]
        assert read_alf(v1).instance_ids is None

        v2 = tmp_path / "v2.all.gz"
        write_all(
            v2,
            np.array([1, 2], dtype=np.uint8),
            instance_ids=np.array([0, 65534], dtype=np.uint16),
            instance_map={"65534": "uuid-b"},
        )
        assert read_alf(v2).format_version == 2
        assert list(read_alf(v2).instance_ids) == [0, 65534]

    def test_int16_roundtrip_is_deterministic(self, tmp_path):
        """Same input, same bytes — required before any golden-file corpus can
        mean anything."""
        rng = np.random.default_rng(19)
        xyz = _road_scene(rng, n=1000)
        a, b = tmp_path / "a.alp.gz", tmp_path / "b.alp.gz"
        write_alp(a, xyz, codec=GEOM_INT16, frame_convention=FRAME_WORLD_ALIGNED)
        write_alp(b, xyz, codec=GEOM_INT16, frame_convention=FRAME_WORLD_ALIGNED)
        assert gzip.decompress(a.read_bytes()) == gzip.decompress(b.read_bytes())


class TestWriterHygiene:
    """Ways a writer could quietly emit a file that its own reader misreads."""

    def test_gz_bytes_do_not_depend_on_the_filename(self, tmp_path):
        """Passing a path to GzipFile stamps the basename into the gzip header,
        so the same frame written to two names would differ byte-for-byte —
        quietly defeating a golden-file corpus that compares `.gz` artifacts."""
        rng = np.random.default_rng(23)
        xyz = _road_scene(rng, n=200)
        a, b = tmp_path / "000000.alp.gz", tmp_path / "totally-different-name.alp.gz"
        write_alp(a, xyz, codec=GEOM_INT16, frame_convention=FRAME_WORLD_ALIGNED)
        write_alp(b, xyz, codec=GEOM_INT16, frame_convention=FRAME_WORLD_ALIGNED)
        assert a.read_bytes() == b.read_bytes()

    def test_fp16_write_strips_inherited_v3_geometry_keys(self, tmp_path):
        """Metadata is routinely copied forward from an existing frame. A
        surviving `_geom_codec: int16` would make the reader decode a float16
        payload as fixed-point — garbage from a file this module just wrote."""
        v3 = tmp_path / "v3.alp.gz"
        write_alp(v3, np.array([[1.0, 2.0, 3.0]]), codec=GEOM_INT16, frame_convention=FRAME_WORLD_ALIGNED)
        inherited = dict(read_alf(v3).metadata)
        assert inherited["_geom_codec"] == GEOM_INT16

        v1 = tmp_path / "v1.alp.gz"
        write_alp(v1, np.array([[1.0, 2.0, 3.0]]), inherited, codec=GEOM_FP16)

        alf = read_alf(v1)
        assert alf.geom_codec == GEOM_FP16
        assert "_geom_scale" not in alf.metadata
        assert alf.xyz[0] == pytest.approx([1.0, 2.0, 3.0], abs=0.01)

    def test_v1_label_write_strips_inherited_v2_keys(self, tmp_path):
        """`format_version` gives metadata precedence, so a leftover
        `_format_version: 2` over a one-byte-per-point payload makes the reader
        invent instance ids from neighbouring categories."""
        v2 = tmp_path / "v2.all.gz"
        write_all(
            v2,
            np.array([1, 2], dtype=np.uint8),
            instance_ids=np.array([5, 6], dtype=np.uint16),
            instance_map={"5": "uuid-a", "6": "uuid-b"},
        )
        inherited = dict(read_alf(v2).metadata)

        v1 = tmp_path / "v1.all.gz"
        write_all(v1, np.array([3, 4], dtype=np.uint8), metadata=inherited)

        alf = read_alf(v1)
        assert alf.format_version == 1
        assert list(alf.categories) == [3, 4]
        assert alf.instance_ids is None
        assert "instance_map" not in alf.metadata

    def test_fractional_label_ids_are_refused(self, tmp_path):
        """`[1.9]` passes a min/max range check and then TRUNCATES to 1 on the
        cast — the same silent relabel the range guard exists to stop."""
        with pytest.raises(ValueError, match="whole numbers"):
            write_all(tmp_path / "a.all.gz", np.array([1.0, 1.9]))
        with pytest.raises(ValueError, match="whole numbers"):
            write_all(
                tmp_path / "b.all.gz",
                np.array([1, 2], dtype=np.uint8),
                instance_ids=np.array([1.0, 2.5]),
            )

    def test_a_rejected_label_rerun_removes_the_previous_file(self, tmp_path):
        path = tmp_path / "000000.all.gz"
        write_all(path, np.array([1, 2], dtype=np.uint8))
        assert path.exists()
        with pytest.raises(ValueError):
            write_all(path, np.array([1, 300]))
        assert not path.exists()

    def test_truncated_alp_is_rejected_against_the_declared_count(self, tmp_path):
        path = tmp_path / "000000.alp.gz"
        write_alp(path, np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        raw = gzip.decompress(path.read_bytes())
        path.write_bytes(gzip.compress(raw[:-6]))  # drop one whole point
        with pytest.raises(AlfError, match="payload holds"):
            read_alf(path)


class TestFailClosed:
    """Where guessing would return plausible-looking wrong data, raise instead."""

    def test_xyz_plus_intensity_matrix_is_refused(self, tmp_path):
        """(N, 4) folded into (4N/3, 3) turns intensity into coordinates."""
        with pytest.raises(ValueError, match=r"expected \(N, 3\)"):
            write_alp(tmp_path / "a.alp.gz", np.array([[1.0, 2.0, 3.0, 99.0], [4.0, 5.0, 6.0, 98.0]]))

    def test_flat_xyz_is_accepted_but_a_ragged_one_is_not(self, tmp_path):
        write_alp(tmp_path / "ok.alp.gz", np.array([1.0, 2.0, 3.0]))
        assert read_alf(tmp_path / "ok.alp.gz").total_points == 1
        with pytest.raises(ValueError, match="whole number of XYZ"):
            write_alp(tmp_path / "bad.alp.gz", np.array([1.0, 2.0, 3.0, 4.0]))

    def test_v3_envelope_without_the_codec_key_is_refused(self, tmp_path):
        """int16 and float16 are both 6 bytes/point, so the length check passes
        and a wrong-codec decode returns plausible garbage."""
        good = tmp_path / "v3.alp.gz"
        write_alp(good, np.array([[1.0, 2.0, 3.0]]), codec=GEOM_INT16, frame_convention=FRAME_WORLD_ALIGNED)
        raw = gzip.decompress(good.read_bytes())
        size = struct.unpack("<I", raw[1:5])[0]
        meta = json.loads(raw[5 : 5 + size])
        meta.pop("_geom_codec")
        header = json.dumps(meta, separators=(",", ":")).encode()
        broken = tmp_path / "broken.alp"
        broken.write_bytes(bytes([3]) + struct.pack("<I", len(header)) + header + raw[5 + size :])

        with pytest.raises(AlfError, match="fixed-point geometry"):
            _ = read_alf(broken).xyz

    def test_unknown_geometry_codec_is_refused(self, tmp_path):
        path = tmp_path / "a.alp.gz"
        write_alp(
            path,
            np.array([[1.0, 2.0, 3.0]]),
            {"_geom_codec": "int24"},
            codec=GEOM_INT16,
            frame_convention=FRAME_WORLD_ALIGNED,
        )
        raw = bytearray(gzip.decompress(path.read_bytes()))
        size = struct.unpack("<I", bytes(raw[1:5]))[0]
        meta = json.loads(bytes(raw[5 : 5 + size]))
        meta["_geom_codec"] = "int24"
        # Rebuilt as version 1 below, and `_frame_convention` is a v3 field:
        # leaving it would break a second rule and this case owns exactly one.
        meta.pop("_frame_convention", None)
        header = json.dumps(meta, separators=(",", ":")).encode()
        broken = tmp_path / "broken.alp"
        broken.write_bytes(bytes([1]) + struct.pack("<I", len(header)) + header + bytes(raw[5 + size :]))

        with pytest.raises(AlfError, match=r"Unknown \.alp geometry codec"):
            _ = read_alf(broken).xyz

    def test_unsupported_label_format_version_is_refused(self, tmp_path):
        header = json.dumps({"_format_version": 3}, separators=(",", ":")).encode()
        path = tmp_path / "000000.all"
        path.write_bytes(bytes([1]) + struct.pack("<I", len(header)) + header + bytes([1, 2, 3]))

        with pytest.raises(AlfError, match=r"Unsupported `\.all` _format_version 3"):
            _ = read_alf(path).categories

    def test_inherited_instance_map_survives_a_rewrite(self, tmp_path):
        """The legacy batch exporter resolves panoptic instances to stable UUIDs through this
        key; a rewrite that omits the argument must not blank it."""
        first = tmp_path / "a.all.gz"
        write_all(
            first,
            np.array([1, 2], dtype=np.uint8),
            instance_ids=np.array([5, 6], dtype=np.uint16),
            instance_map={"5": "uuid-a", "6": "uuid-b"},
        )
        inherited = dict(read_alf(first).metadata)

        second = tmp_path / "b.all.gz"
        write_all(
            second,
            np.array([1, 2], dtype=np.uint8),
            instance_ids=np.array([5, 6], dtype=np.uint16),
            metadata=inherited,
        )
        assert read_alf(second).instance_map == {"5": "uuid-a", "6": "uuid-b"}


class TestStaleAndInheritedState:
    """Rewrites that carry forward metadata, and reruns that fail partway."""

    def test_a_rejected_rerun_removes_the_file_even_on_a_shape_error(self, tmp_path):
        """The shape check is the FIRST thing that can reject, so the unlink has
        to precede it or a bad rerun leaves the previous frame in place."""
        path = tmp_path / "000000.alp.gz"
        write_alp(path, np.array([[1.0, 2.0, 3.0]]))
        assert path.exists()

        with pytest.raises(ValueError, match=r"expected \(N, 3\)"):
            write_alp(path, np.array([[1.0, 2.0, 3.0, 99.0]]))
        assert not path.exists(), "stale frame survived a rejected rerun"

    def test_an_emptied_frame_does_not_inherit_the_previous_extent(self, tmp_path):
        """Filtering can empty a frame while its metadata is copied forward. An
        inherited `box` would publish the OLD extent for a frame with no
        geometry — the server turns that header straight into item bounds."""
        full = tmp_path / "full.alp.gz"
        write_alp(full, np.array([[-5.0, -5.0, -1.0], [5.0, 5.0, 1.0]]))
        inherited = dict(read_alf(full).metadata)
        assert "box" in inherited

        emptied = tmp_path / "emptied.alp.gz"
        write_alp(emptied, np.zeros((0, 3)), inherited)

        alf = read_alf(emptied)
        assert alf.total_points == 0
        assert "box" not in alf.metadata
        assert "_box_frame" not in alf.metadata

    def test_label_version_byte_is_refused_on_a_geometry_file(self, tmp_path):
        """Version 2 is the interleaved label layout. On a `.alp` whose payload
        happens to be a whole number of points it would otherwise decode as
        float16 and return plausible coordinates from a non-geometry file."""
        path = tmp_path / "000000.alp"
        payload = bytes(12)  # 2 points' worth of bytes
        header = json.dumps({"_total_points": 2}, separators=(",", ":")).encode()
        path.write_bytes(bytes([2]) + struct.pack("<I", len(header)) + header + payload)

        with pytest.raises(AlfError, match="interleaved label layout"):
            read_alf(path)

    def test_non_finite_header_values_are_refused(self, tmp_path):
        """Python emits bare NaN/Infinity, which is not valid JSON — the file
        reads back fine here and fails only in the browser/TS readers."""
        with pytest.raises(ValueError, match="non-finite"):
            write_alp(tmp_path / "a.alp.gz", np.array([[1.0, 2.0, 3.0]]), {"bad": float("nan")})
        with pytest.raises(ValueError, match="must be finite"):
            write_ali(tmp_path / "b.ali.gz", np.array([1.0]), lo=0.0, hi=float("inf"))


class TestSequenceLevelIntensity:
    """`_intensities_every_10_percent` is a SEQUENCE distribution, not a frame one."""

    def test_explicit_deciles_win(self, tmp_path):
        path = tmp_path / "000000.ali.gz"
        write_ali(path, np.array([0.0, 255.0]), lo=0.0, hi=255.0, deciles=list(range(10, 100, 10)))
        assert read_alf(path).metadata["_intensities_every_10_percent"] == list(range(10, 100, 10))

    def test_inherited_deciles_are_not_overwritten_by_this_frame(self, tmp_path):
        """A sparse frame must not replace the sequence-wide distribution the
        caller already computed — the viewer picks contrast from it, so a
        per-frame histogram makes the same surface shade differently per frame."""
        sequence_deciles = [5, 10, 20, 40, 80, 120, 160, 200, 250]
        path = tmp_path / "000000.ali.gz"
        write_ali(
            path,
            np.array([200.0, 200.0]),  # a low-variance frame
            lo=0.0,
            hi=255.0,
            metadata={"_intensities_every_10_percent": sequence_deciles},
        )
        assert read_alf(path).metadata["_intensities_every_10_percent"] == sequence_deciles

    def test_falls_back_to_this_frame_when_nothing_is_supplied(self, tmp_path):
        path = tmp_path / "000000.ali.gz"
        write_ali(path, np.array([0.0, 255.0]), lo=0.0, hi=255.0)
        assert len(read_alf(path).metadata["_intensities_every_10_percent"]) == 9

    def test_wrong_decile_count_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="9 values"):
            write_ali(tmp_path / "a.ali.gz", np.array([1.0]), lo=0.0, hi=2.0, deciles=[1, 2, 3])

    def test_non_finite_intensity_is_refused(self, tmp_path):
        """clip+astype would turn NaN/Inf into endpoint bytes, indistinguishable
        from a real minimum or maximum reading."""
        for bad in (np.nan, np.inf):
            with pytest.raises(ValueError, match="NaN or Inf"):
                write_ali(tmp_path / "a.ali.gz", np.array([1.0, bad]), lo=0.0, hi=2.0)

    def test_v3_version_byte_is_refused_on_non_geometry_channels(self, tmp_path):
        header = json.dumps({"_total_points": 3}, separators=(",", ":")).encode()
        path = tmp_path / "000000.all"
        path.write_bytes(bytes([3]) + struct.pack("<I", len(header)) + header + bytes([1, 2, 3]))
        with pytest.raises(AlfError, match=r"fixed-point `\.alp` geometry"):
            read_alf(path)


class TestWorldFrameConventions:
    """`device_position` is not always the world origin."""

    def test_sensor_pose_convention(self, tmp_path):
        path = tmp_path / "a.alp.gz"
        write_alp(
            path,
            np.array([[1.0, 2.0, 3.0]]),
            {"device_position": {"x": 100.0, "y": 200.0, "z": 5.0}},
        )
        assert read_alf(path).world_xyz(sensor_pose=True)[0] == pytest.approx([101.0, 202.0, 8.0])

    def test_aerial_floating_origin_wins_over_device_position(self, tmp_path):
        """The aerial LAS/tile writers keep the true origin in
        `world_centroid_epsg3857` and use `device_position` for tile placement.
        Adding device_position would put the cloud kilometres from the truth."""
        path = tmp_path / "a.alp.gz"
        write_alp(
            path,
            np.array([[1.0, 2.0, 3.0]]),
            {
                "device_position": {"x": 0.0, "y": 0.0, "z": 0.0},
                "world_centroid_epsg3857": {"x": -13_600_000.0, "y": 4_500_000.0, "z": 12.0},
            },
        )
        assert read_alf(path).world_xyz()[0] == pytest.approx([-13_599_999.0, 4_500_002.0, 15.0])

    def test_tile_placement_offset_is_not_treated_as_the_origin(self, tmp_path):
        """The tile writer sets device_position = tile_centroid - global_centroid for
        mosaicking, while the tile's own centroid is the real origin."""
        path = tmp_path / "a.alp.gz"
        write_alp(
            path,
            np.array([[0.0, 0.0, 0.0]]),
            {
                "device_position": {"x": 1200.0, "y": -800.0, "z": 0.0},
                "world_centroid_epsg3857": {"x": -13_601_200.0, "y": 4_500_800.0, "z": 10.0},
            },
        )
        assert read_alf(path).world_xyz()[0] == pytest.approx([-13_601_200.0, 4_500_800.0, 10.0])

    def test_no_declared_frame_refuses(self, tmp_path):
        """Both aerial writers keep `world_centroid_epsg3857` in the frame
        JSON, never in the ALP header. Neither a zero nor a nonzero
        device_position identifies the convention, so the caller must say."""
        for dp in ({"x": 0.0, "y": 0.0, "z": 0.0}, {"x": 1200.0, "y": -800.0, "z": 0.0}):
            path = tmp_path / "a.alp.gz"
            write_alp(path, np.array([[1.0, 2.0, 3.0]]), {"device_position": dp})
            with pytest.raises(AlfError, match="without being told its convention"):
                read_alf(path).world_xyz()

    def test_tile_placement_offset_is_not_mistaken_for_a_sensor_pose(self, tmp_path):
        """The tile writer writes `tile_origin - global_centroid` into
        device_position — nonzero, plausible, and NOT a sensor pose. A nonzero
        fallback would return global-centroid-relative coordinates, kilometres
        from EPSG:3857."""
        path = tmp_path / "a.alp.gz"
        write_alp(
            path,
            np.array([[1.0, 2.0, 3.0]]),
            {"device_position": {"x": 1200.0, "y": -800.0, "z": 0.0}},
        )
        alf = read_alf(path)
        with pytest.raises(AlfError):
            alf.world_xyz()
        # the sidecar centroid is the right anchor, and it is not device_position
        got = alf.world_xyz(origin={"x": -13_601_200.0, "y": 4_500_800.0, "z": 10.0})
        assert got[0] == pytest.approx([-13_601_199.0, 4_500_802.0, 13.0])

    def test_explicit_origin_supplies_the_missing_anchor(self, tmp_path):
        path = tmp_path / "a.alp.gz"
        write_alp(path, np.array([[1.0, 2.0, 3.0]]), {"device_position": {"x": 0.0, "y": 0.0, "z": 0.0}})
        alf = read_alf(path)
        # the sidecar shape (a dict) and the hand-written shape (a tuple) both work
        assert alf.world_xyz(origin={"x": -13_600_000.0, "y": 4_500_000.0, "z": 12.0})[0] == pytest.approx(
            [-13_599_999.0, 4_500_002.0, 15.0]
        )
        assert alf.world_xyz(origin=(0.0, 0.0, 0.0))[0] == pytest.approx([1.0, 2.0, 3.0])

    def test_explicit_origin_outranks_recorded_centroid(self, tmp_path):
        path = tmp_path / "a.alp.gz"
        write_alp(
            path,
            np.array([[1.0, 2.0, 3.0]]),
            {
                "device_position": {"x": 0.0, "y": 0.0, "z": 0.0},
                "world_centroid_epsg3857": {"x": 5.0, "y": 5.0, "z": 5.0},
            },
        )
        assert read_alf(path).world_xyz(origin=(10.0, 0.0, 0.0))[0] == pytest.approx([11.0, 2.0, 3.0])

    def test_scalar_origin_is_rejected_not_broadcast(self, tmp_path):
        path = tmp_path / "a.alp.gz"
        write_alp(path, np.array([[1.0, 2.0, 3.0]]), {"device_position": {"x": 0.0, "y": 0.0, "z": 0.0}})
        with pytest.raises(AlfError, match="3-sequence"):
            read_alf(path).world_xyz(origin=7.0)

    def test_non_finite_origin_is_rejected(self, tmp_path):
        path = tmp_path / "a.alp.gz"
        write_alp(path, np.array([[1.0, 2.0, 3.0]]), {"device_position": {"x": 0.0, "y": 0.0, "z": 0.0}})
        with pytest.raises(AlfError, match="non-finite"):
            read_alf(path).world_xyz(origin=(float("nan"), 0.0, 0.0))


class TestFrameRotationConvention:
    """`R(_frame_rotation)` is Hamilton and active, and nothing had pinned it.

    Naming the quaternion's members settled which number is which. It did not
    settle what the four of them MEAN: a reader using the passive/JPL reading
    agrees on every component and rotates the other way, returning a plausible
    cloud that is wrong by the device heading — the same failure the convention
    field exists to end, arrived at through the formula instead of the bytes.
    `spec/format.md` §5.3 now defines it, and these pin what it defines.
    """

    @staticmethod
    def _device_aligned(rotation, local_words, *, anchor=(0.0, 0.0, 0.0), points=1):
        """A v3 `.alp` built by hand at `_geom_scale: 1.0`.

        Hand-built rather than written through `write_alp` so the int16 grid is
        the identity: word `n` decodes to exactly `n`. Quantisation error would
        otherwise force a tolerance onto an assertion whose whole point is that
        the arithmetic is exact.
        """
        header = json.dumps(
            {
                "_total_points": points,
                "_geom_codec": "int16",
                "_geom_scale": 1.0,
                "_geom_offset": [0, 0, 0],
                "_frame_convention": "device-aligned",
                "_frame_rotation": rotation,
                "device_position": {"x": anchor[0], "y": anchor[1], "z": anchor[2]},
            },
            separators=(",", ":"),
        ).encode()
        payload = np.array(local_words, dtype="<i2").tobytes()
        return struct.pack("B", 3) + struct.pack("<I", len(header)) + header + payload

    def test_the_pinned_vector_from_the_specification(self):
        """The one worked example §5.3 publishes, reproduced exactly.

        q = (0.5, 0.5, 0.5, 0.5) is 120° about (1,1,1)/√3 — it cycles the axes
        rather than reflecting them, so the transposed reading lands somewhere
        visibly different instead of somewhere nearly right. Its components are
        exact in binary64 and its norm is exactly 1, so a failure here is a
        convention error and never a tolerance one.
        """
        raw = self._device_aligned({"w": 0.5, "x": 0.5, "y": 0.5, "z": 0.5}, [1, 2, 3])
        world = read_alf(raw, extension="alp").world_xyz(sensor_pose=True)
        assert world.tolist() == [[3.0, 1.0, 2.0]]
        # The failure this pins is not "no rotation" — it is the OTHER rotation,
        # which is what a reader that fed Hamilton components to a JPL formula
        # would return. Asserting the right answer alone would pass for an
        # implementation that transposed twice as readily as one that did not.
        assert world.tolist() != [[2.0, 3.0, 1.0]]

    def test_a_quarter_turn_about_z_is_not_its_own_inverse(self):
        """The corpus rotation, on a point that separates the two readings.

        `alp-int16-device-aligned` carries this quaternion and asserts only the
        LOCAL points, so the corpus never saw which way it turned.
        """
        half = math.sqrt(0.5)
        raw = self._device_aligned({"w": half, "x": 0.0, "y": 0.0, "z": half}, [10, 0, 0])
        world = read_alf(raw, extension="alp").world_xyz(sensor_pose=True)
        assert world[0] == pytest.approx([0.0, 10.0, 0.0], abs=1e-12)

    def test_negating_the_quaternion_is_the_same_rotation(self):
        """`q` and `-q` name one rotation — every term of R is even-degree.

        A producer normalising a pose may emit either sign, so a reader that
        read the sign of `w` as information would disagree with itself across
        two frames of one sequence.
        """
        local = [1, 2, 3]
        positive = read_alf(
            self._device_aligned({"w": 0.5, "x": 0.5, "y": 0.5, "z": 0.5}, local), extension="alp"
        ).world_xyz(sensor_pose=True)
        negative = read_alf(
            self._device_aligned({"w": -0.5, "x": -0.5, "y": -0.5, "z": -0.5}, local),
            extension="alp",
        ).world_xyz(sensor_pose=True)
        assert positive.tolist() == negative.tolist() == [[3.0, 1.0, 2.0]]

    def test_the_rotation_precedes_the_anchor(self):
        """`world = R·local + anchor`, never `R·(local + anchor)`.

        The anchor is a world-frame translation; rotating it too would swing the
        whole frame around the world origin. With a metre of local offset and a
        kilometre of anchor, the wrong order is off by the kilometre.
        """
        raw = self._device_aligned({"w": 0.5, "x": 0.5, "y": 0.5, "z": 0.5}, [1, 2, 3], anchor=(1000.0, 0.0, 0.0))
        world = read_alf(raw, extension="alp").world_xyz(sensor_pose=True)
        assert world.tolist() == [[1003.0, 1.0, 2.0]]

    def test_identity_rotates_nothing_and_is_still_a_declaration(self):
        """Identity is a legal `device-aligned` rotation, and it is exactly the
        case in which the two conventions agree — a straight-line frame. It must
        place like `world-aligned` without BEING it."""
        raw = self._device_aligned({"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}, [1, 2, 3])
        alf = read_alf(raw, extension="alp")
        assert alf.frame_convention == "device-aligned"
        assert alf.world_xyz(sensor_pose=True).tolist() == [[1.0, 2.0, 3.0]]

    def test_an_invalid_return_survives_the_rotation_in_place(self):
        """A rotation of "no position" is still no position, and it keeps its
        index — the sentinel row must not be dropped, compacted, or turned into
        a coordinate by the matmul."""
        raw = self._device_aligned(
            {"w": 0.5, "x": 0.5, "y": 0.5, "z": 0.5},
            [1, 2, 3, INT16_NO_RETURN, INT16_NO_RETURN, INT16_NO_RETURN, 4, 5, 6],
            points=3,
        )
        world = read_alf(raw, extension="alp").world_xyz(sensor_pose=True)
        assert world.shape == (3, 3)
        assert world[0].tolist() == [3.0, 1.0, 2.0]
        assert np.isnan(world[1]).all()
        assert world[2].tolist() == [6.0, 4.0, 5.0]

    def test_the_writer_takes_numpy_integer_components(self, tmp_path):
        """`dict(zip("wxyz", np.array([1, 0, 0, 0])))` is how the identity
        quaternion gets spelled in a NumPy pipeline, and the components are
        `np.int64`. Refusing them made a valid device-aligned write impossible
        without converting every component by hand, in the writer of a library
        whose only runtime dependency is NumPy."""
        path = tmp_path / "a.alp.gz"
        write_alp(
            path,
            np.array([[1.0, 2.0, 3.0]]),
            codec=GEOM_INT16,
            frame_convention="device-aligned",
            frame_rotation=dict(zip("wxyz", np.array([1, 0, 0, 0]), strict=True)),
        )
        assert read_alf(path).frame_rotation == {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}

    def test_booleans_are_still_not_components(self, tmp_path):
        """Widening to `np.integer` must not widen to booleans by the back door.
        `True` normalizes into the identity rotation and `np.bool_` is an
        instance of neither `bool` nor `np.integer`, so both have to be refused
        — one by the explicit guard, one by falling through the type check."""
        for bad in (True, np.bool_(True)):
            with pytest.raises(ValueError, match="must be a number"):
                write_alp(
                    tmp_path / "b.alp.gz",
                    np.array([[1.0, 2.0, 3.0]]),
                    codec=GEOM_INT16,
                    frame_convention="device-aligned",
                    frame_rotation={"w": bad, "x": 0.0, "y": 0.0, "z": 0.0},
                )


class TestReviewRoundFixes:
    """Regressions for defects found in review that all fail SILENTLY."""

    def test_fp16_error_bound_floors_at_the_subnormal_spacing(self, tmp_path):
        """Below 2**-14 binary16 goes subnormal and the spacing stops halving.
        The exponent formula alone keeps shrinking and understates the real
        rounding bound of a tiny frame by up to ~1000x."""
        path = tmp_path / "a.alp.gz"
        tiny = float(np.nextafter(np.float16(0), np.float16(1)))  # min positive subnormal
        write_alp(path, np.array([[tiny, 0.0, 0.0]]), check_budget=False)
        assert read_alf(path).max_geom_error_m == pytest.approx(2.0**-25)

    def test_fp16_error_bound_unchanged_in_the_normal_range(self, tmp_path):
        path = tmp_path / "a.alp.gz"
        write_alp(path, np.array([[100.0, 0.0, 0.0]]))
        # 2**6 <= 100 < 2**7, so half-ULP is 2**(6-11)
        assert read_alf(path).max_geom_error_m == pytest.approx(2.0**-5)

    def test_zeroed_fp16_frame_reports_the_subnormal_bound_not_zero(self, tmp_path):
        """A component below half the minimum subnormal rounds to zero on encode.
        Reporting an exact 0.0 would assert losslessness for an encode that
        actually discarded the value."""
        path = tmp_path / "a.alp.gz"
        write_alp(path, np.array([[2.0**-27, 0.0, 0.0]]), check_budget=False)
        alf = read_alf(path)
        assert alf.xyz.tolist() == [[0.0, 0.0, 0.0]]  # the value really is gone
        assert alf.max_geom_error_m == pytest.approx(2.0**-25)

    def test_empty_frame_still_reports_zero_error(self, tmp_path):
        """No points, nothing lost — distinct from a frame that rounded to zero."""
        path = tmp_path / "a.alp.gz"
        write_alp(path, np.zeros((0, 3)))
        assert read_alf(path).max_geom_error_m == 0.0

    def test_geom_error_report_refuses_an_n_by_4_matrix(self):
        """(N, 4) whose element count divides by 3 folds intensity into
        coordinates, inflating the extent and the int16 scale derived from it —
        making v3 look worse than it is in the function that measures the claim."""
        xyz_plus_intensity = np.array([[1.0, 2.0, 3.0, 99.0], [4.0, 5.0, 6.0, 98.0], [7.0, 8.0, 9.0, 97.0]])
        assert xyz_plus_intensity.size % 3 == 0  # the reshape would have succeeded
        with pytest.raises(ValueError, match=r"expected \(N, 3\)"):
            geom_error_report(xyz_plus_intensity)

    def test_offset_frame_uses_the_full_int16_range(self, tmp_path):
        """A one-sided frame (x = 100..200 m) has a 50 m half-span. Centring on
        the bounding-box midpoint derives the step from 50 m instead of 200 m."""
        rng = np.random.default_rng(5)
        xyz = np.hstack(
            [
                rng.uniform(100.0, 200.0, size=(2000, 1)),
                rng.uniform(-10.0, 10.0, size=(2000, 1)),
                rng.uniform(0.0, 5.0, size=(2000, 1)),
            ]
        )
        path = tmp_path / "a.alp.gz"
        write_alp(path, xyz, codec=GEOM_INT16, frame_convention=FRAME_WORLD_ALIGNED)
        alf = read_alf(path)

        # offset is the per-axis midpoint, so the grid is built around the frame
        assert alf.metadata["_geom_offset"][0] == pytest.approx(150.0, abs=0.5)
        # step from the 50 m half-span, not the 200 m absolute extent
        assert alf.metadata["_geom_scale"] < (200.0 / 32767) / 3
        err = np.abs(alf.xyz - xyz).max()
        assert err <= alf.metadata["_geom_max_err_m"] + 1e-12
        assert err < 0.002

    def test_centring_does_not_regress_an_origin_centred_frame(self, tmp_path):
        """Ego-centric lidar is already symmetric about zero — centring must not
        make the common case worse."""
        rng = np.random.default_rng(9)
        xyz = _road_scene(rng)
        path = tmp_path / "a.alp.gz"
        write_alp(path, xyz, codec=GEOM_INT16, frame_convention=FRAME_WORLD_ALIGNED)
        alf = read_alf(path)
        assert np.abs(alf.xyz - xyz).max() < 0.002

    def test_error_report_models_the_centring_the_writer_does(self, tmp_path):
        """The report is the measurement of the v3 claim; if it skipped centring
        it would understate v3 on exactly the frames centring improves."""
        rng = np.random.default_rng(13)
        xyz = np.hstack(
            [
                rng.uniform(100.0, 200.0, size=(500, 1)),
                rng.uniform(-10.0, 10.0, size=(500, 1)),
                rng.uniform(0.0, 5.0, size=(500, 1)),
            ]
        )
        report = geom_error_report(xyz)
        path = tmp_path / "a.alp.gz"
        write_alp(path, xyz, codec=GEOM_INT16, frame_convention=FRAME_WORLD_ALIGNED)
        measured = np.abs(read_alf(path).xyz - xyz).max()
        assert measured == pytest.approx(report[GEOM_INT16], abs=1e-9)

    def test_geom_error_report_refuses_non_finite(self):
        with pytest.raises(ValueError, match="NaN or Inf"):
            geom_error_report(np.array([[1.0, 2.0, float("nan")]]))

    def test_incomplete_anchor_mapping_is_rejected(self, tmp_path):
        """`{"x": 1}` would otherwise be read as (1, 0, 0) and shift the cloud by
        whatever y and z really were."""
        path = tmp_path / "a.alp.gz"
        write_alp(path, np.array([[1.0, 2.0, 3.0]]), {"device_position": {"x": 0.0, "y": 0.0, "z": 0.0}})
        with pytest.raises(AlfError, match="missing required axis"):
            read_alf(path).world_xyz(origin={"x": 1.0})

    def test_incomplete_file_anchor_is_rejected_during_read(self, tmp_path):
        path = tmp_path / "partial-anchor.alp.gz"
        write_alp(path, np.array([[0.0, 0.0, 0.0]]))
        _rewrite_metadata(
            path,
            lambda metadata: metadata.__setitem__("device_position", {"x": 1, "y": 2}),
        )

        with pytest.raises(AlfError, match=r"Geometry anchor.*malformed"):
            read_alf(path)

    def test_failed_write_leaves_no_partial_file(self, tmp_path, monkeypatch):
        """A truncated .gz at the final path reads as current output to every
        consumer that globs a sequence directory."""
        path = tmp_path / "a.alp.gz"
        write_alp(path, np.array([[1.0, 2.0, 3.0]]))
        good = path.read_bytes()

        import alf._io as mod

        real_replace = mod.os.replace

        def boom(src, dst):
            raise OSError("disk full")

        monkeypatch.setattr(mod.os, "replace", boom)
        with pytest.raises(OSError):
            write_alp(path, np.zeros((5000, 3)))
        monkeypatch.setattr(mod.os, "replace", real_replace)

        # the prior file is gone (unlink-before-write), but no partial replaced it
        assert not path.exists()
        assert list(tmp_path.glob("*.tmp*")) == []
        assert good  # sanity: the successful write really had produced bytes

    def test_alc_round_trips_through_the_shared_api(self, tmp_path):
        """The module has advertised .alc from its first line; without an
        accessor a consumer reaches into _payload or keeps its own parser."""
        path = tmp_path / "a.alc.gz"
        write_alc(path, np.array([0, 2, 1], dtype=np.uint8), ["ff0000", "00ff00", "0000ff"])
        alf = read_alf(path)
        assert alf.palette_indices.tolist() == [0, 2, 1]
        assert alf.palette == ["ff0000", "00ff00", "0000ff"]

    def test_stale_structural_keys_do_not_survive_pass_through(self, tmp_path):
        """Metadata is routinely copied forward. A stale _total_points or
        _format_version would describe the OLD payload while the arrays describe
        this one, and the reader's own cross-check would then reject a file this
        writer produced."""
        path = tmp_path / "a.all.gz"
        write_all(
            path,
            np.array([1, 2, 3], dtype=np.uint8),
            metadata={"_total_points": 999, "_format_version": 2, "custom": "kept"},
        )
        alf = read_alf(path)
        assert alf.total_points == 3
        assert alf.format_version == 1
        assert alf.metadata["custom"] == "kept"  # ordinary keys still pass through

    def test_alx_payload_is_length_checked(self, tmp_path):
        """.alx is 4 bytes/point; without a stride entry a truncated sidecar
        decodes into a valid-looking array of the wrong length, and the mapping
        it carries is index-correlated so a short read misattributes points."""
        path = tmp_path / "a.alx.gz"
        write_alx(path, np.array([1, 2, 3], dtype=np.uint32), alp_total_points=3, source_total_points=10)
        raw = gzip.decompress(path.read_bytes())
        path.write_bytes(gzip.compress(raw[:-4]))  # drop one whole index
        with pytest.raises(AlfError, match="payload holds"):
            read_alf(path)

    def test_rejected_alc_write_does_not_leave_the_previous_frame(self, tmp_path):
        path = tmp_path / "a.alc.gz"
        write_alc(path, np.array([0], dtype=np.uint8), ["ff0000"])
        with pytest.raises(ValueError):
            write_alc(path, np.array([0, 9], dtype=np.uint8), ["ff0000"])
        assert not path.exists()

    def test_instance_map_must_be_a_mapping(self, tmp_path):
        """A list serializes fine and then fails every str(id) lookup, so the
        coverage check would blame the ids for what is a type mistake."""
        with pytest.raises(ValueError, match="must be a mapping"):
            write_all(
                tmp_path / "a.all.gz",
                np.array([1], dtype=np.uint8),
                instance_ids=np.array([1], dtype=np.uint16),
                instance_map=["u1"],
            )

    def test_alc_no_colour_sentinel_is_legal(self, tmp_path):
        """The canonical colour writer writes len(palette) for points that matched no
        camera. A naive bounds check rejects every frame with an occluded point;
        a clamp recolours them with the last palette entry."""
        path = tmp_path / "a.alc.gz"
        write_alc(path, np.array([0, 1, 2], dtype=np.uint8), ["ff0000", "00ff00"])
        assert read_alf(path).palette_indices.tolist() == [0, 1, 2]

    def test_alc_palette_is_capped_at_255_to_leave_room_for_the_sentinel(self, tmp_path):
        """With 256 entries every uint8 value is a real colour, so occluded
        points have no representable index. That writer asserts the same
        bound."""
        with pytest.raises(ValueError, match="at most 255 palette entries"):
            write_alc(
                tmp_path / "a.alc.gz",
                np.array([0], dtype=np.uint8),
                [f"{i:02x}0000" for i in range(256)],
            )

    def test_alc_full_255_palette_still_has_a_usable_sentinel(self, tmp_path):
        path = tmp_path / "a.alc.gz"
        palette = [f"{i:02x}0000" for i in range(255)]
        write_alc(path, np.array([0, 254, 255], dtype=np.uint8), palette)
        assert read_alf(path).palette_indices.tolist() == [0, 254, 255]

    def test_alc_index_beyond_the_sentinel_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="out of range for a"):
            write_alc(tmp_path / "a.alc.gz", np.array([0, 5], dtype=np.uint8), ["ff0000"])

    def test_alx_round_trips_through_the_shared_api(self, tmp_path):
        path = tmp_path / "a.alx.gz"
        write_alx(
            path,
            np.array([0, 7, 4_000_000_000], dtype=np.uint64),
            alp_total_points=3,
            source_total_points=4_000_000_001,
        )
        assert read_alf(path).kept_indices.tolist() == [0, 7, 4_000_000_000]

    def test_inherited_deciles_are_validated_too(self, tmp_path):
        """Sequence-level values normally arrive by INHERITANCE, so validating
        only the explicit argument left the common path unguarded."""
        with pytest.raises(ValueError, match="non-decreasing"):
            write_ali(
                tmp_path / "a.ali.gz",
                np.arange(10, dtype=np.float64),
                lo=0.0,
                hi=9.0,
                metadata={"_intensities_every_10_percent": [200, 180, 160, 140, 120, 100, 80, 60, 40]},
            )

    def test_v2_impossible_by_stride_resolves_to_v1(self, tmp_path):
        """No _total_points, but a length not divisible by 3 rules v2 OUT, so
        exactly one reading survives and the answer is not a guess."""
        hb = json.dumps({"_format_version": 2}, separators=(",", ":")).encode()
        path = tmp_path / "000000.all"
        path.write_bytes(struct.pack("B", 1) + struct.pack("<I", len(hb)) + hb + bytes([1, 2, 3, 4]))
        alf = read_alf(path)
        assert alf.format_version == 1
        assert alf.format_version_verified is True

    def test_stale_int16_codec_also_refuses_to_report_a_bound(self, tmp_path):
        """A caller may gate on the bound BEFORE decoding, so reporting a
        millimetre bound for a float16 payload passes a gate the decode refuses."""
        path = tmp_path / "a.alp.gz"
        write_alp(path, np.array([[1.0, 2.0, 3.0]]), codec=GEOM_FP16)
        _rewrite_metadata(
            path,
            lambda m: (m.__setitem__("_geom_codec", "int16"), m.__setitem__("_geom_scale", 0.001)),
        )
        with pytest.raises(AlfError, match="does not use"):
            read_alf(path).max_geom_error_m

    def test_odd_byte_alp_payload_is_an_alf_error(self, tmp_path):
        """frombuffer raises a bare ValueError, escaping the contract callers
        use to quarantine a frame and keep going."""
        # No `_total_points`, so the declared-count cross-check (itself already
        # an AlfError) cannot fire and the raw frombuffer would be reached.
        hb = json.dumps({}, separators=(",", ":")).encode()
        path = tmp_path / "000000.alp"
        path.write_bytes(struct.pack("B", 1) + struct.pack("<I", len(hb)) + hb + bytes(5))
        with pytest.raises(AlfError, match="not a whole number of"):
            read_alf(path).xyz

    def test_non_numeric_anchor_component_is_an_alf_error(self, tmp_path):
        path = tmp_path / "a.alp.gz"
        write_alp(path, np.array([[1.0, 2.0, 3.0]]))
        with pytest.raises(AlfError, match="not JSON numbers"):
            read_alf(path).world_xyz(origin={"x": "bad", "y": 0.0, "z": 0.0})

    def test_descending_deciles_are_refused(self, tmp_path):
        """Percentiles are monotonic by definition; a descending sequence
        describes a distribution that cannot exist, and the viewer builds its
        contrast curve from it anyway."""
        with pytest.raises(ValueError, match="non-decreasing"):
            write_ali(
                tmp_path / "a.ali.gz",
                np.arange(10, dtype=np.float64),
                lo=0.0,
                hi=9.0,
                deciles=[200, 180, 160, 140, 120, 100, 80, 60, 40],
            )

    def test_ascending_deciles_still_write(self, tmp_path):
        path = tmp_path / "a.ali.gz"
        write_ali(
            path,
            np.arange(10, dtype=np.float64),
            lo=0.0,
            hi=9.0,
            deciles=[10, 20, 30, 40, 50, 60, 70, 80, 90],
        )
        assert read_alf(path).metadata["_intensities_every_10_percent"][0] == 10

    def test_corrupt_centroid_raises_rather_than_falling_back(self, tmp_path):
        """`{}` is falsey, so a truthiness check would skip validation and place
        an aerial frame by the sensor-pose convention its metadata contradicts."""
        path = tmp_path / "a.alp.gz"
        write_alp(
            path,
            np.array([[1.0, 2.0, 3.0]]),
            device_position=(100.0, 200.0, 5.0),
        )
        _rewrite_metadata(path, lambda m: m.__setitem__("world_centroid_epsg3857", {}))
        with pytest.raises(AlfError, match="missing required axis"):
            read_alf(path).world_xyz(sensor_pose=True)

    def test_length_matching_neither_stride_is_corruption(self, tmp_path):
        """With _total_points present, a length matching neither N nor 3N
        contradicts both layouts — that is truncation, not ambiguity."""
        hb = json.dumps({"_format_version": 2, "_total_points": 4}, separators=(",", ":")).encode()
        path = tmp_path / "000000.all"
        path.write_bytes(struct.pack("B", 1) + struct.pack("<I", len(hb)) + hb + bytes(7))
        with pytest.raises(AlfError, match="truncated or overlong"):
            read_alf(path).format_version

    def test_explicit_device_position_replaces_the_inherited_one(self, tmp_path):
        """A re-localised payload translated by the previous frame's origin puts
        the cloud wherever the last frame was."""
        first = tmp_path / "a.alp.gz"
        write_alp(first, np.array([[1.0, 2.0, 3.0]]), device_position=(100.0, 200.0, 5.0))
        inherited = dict(read_alf(first).metadata)
        second = tmp_path / "b.alp.gz"
        write_alp(
            second,
            np.array([[1.0, 2.0, 3.0]]),
            metadata=inherited,
            device_position=(0.0, 0.0, 0.0),
        )
        alf = read_alf(second)
        assert alf.device_position.tolist() == [0.0, 0.0, 0.0]
        assert alf.world_xyz(sensor_pose=True)[0] == pytest.approx([1.0, 2.0, 3.0])

    def test_mismatched_version_fields_resolve_by_payload_stride(self, tmp_path):
        """byte 1 + _format_version 2 is what the label-merge service writes.
        The payload is objective where the two headers are just assertions."""
        header = json.dumps({"_format_version": 2, "_total_points": 2}, separators=(",", ":"))
        hb = header.encode()
        # 3 bytes/point => genuinely v2 despite the v1 wire byte
        payload = bytes([7, 1, 0, 9, 2, 0])
        path = tmp_path / "000000.all"
        path.write_bytes(struct.pack("B", 1) + struct.pack("<I", len(hb)) + hb + payload)
        alf = read_alf(path)
        assert alf.format_version == 2
        assert alf.format_version_verified is True
        assert alf.categories.tolist() == [7, 9]

    def test_byte_only_v2_with_a_bad_stride_is_not_verified(self, tmp_path):
        """A version-byte-only v2 file has nothing corroborating it either: a
        payload that is not a whole number of 3-byte rows already contradicts
        the layout, so reporting it authoritative would mislead the preflight."""
        path = tmp_path / "000000.all"
        path.write_bytes(struct.pack("B", 2) + struct.pack("<I", 0) + bytes(4))
        assert read_alf(path).format_version_verified is False
        good = tmp_path / "000001.all"
        good.write_bytes(struct.pack("B", 2) + struct.pack("<I", 0) + bytes(6))
        assert read_alf(good).format_version_verified is True

    def test_self_consistent_file_is_verified(self, tmp_path):
        path = tmp_path / "a.all.gz"
        write_all(path, np.array([1, 2, 3], dtype=np.uint8))
        assert read_alf(path).format_version_verified is True

    def test_noncanonical_instance_map_keys_do_not_count_as_coverage(self, tmp_path):
        """The legacy batch exporter does an exact str(id) lookup, so "01" resolves to
        nothing there and emits object_uuid: null."""
        path = tmp_path / "a.all.gz"
        write_all(
            path,
            np.array([1], dtype=np.uint8),
            instance_ids=np.array([1], dtype=np.uint16),
            instance_map={"1": "u1"},
        )
        _rewrite_metadata(path, lambda m: m.__setitem__("instance_map", {"01": "u1"}))
        with pytest.raises(AlfError, match="missing entries for instance id"):
            read_alf(path).instance_ids

    def test_explicit_instance_map_replaces_the_inherited_one(self, tmp_path):
        """A rewrite passing read_alf(...).metadata back through must not publish
        the old identities alongside the new labels."""
        first = tmp_path / "a.all.gz"
        write_all(
            first,
            np.array([1], dtype=np.uint8),
            instance_ids=np.array([1], dtype=np.uint16),
            instance_map={"1": "OLD"},
        )
        inherited = dict(read_alf(first).metadata)
        second = tmp_path / "b.all.gz"
        write_all(
            second,
            np.array([1], dtype=np.uint8),
            instance_ids=np.array([1], dtype=np.uint16),
            instance_map={"1": "NEW"},
            metadata=inherited,
        )
        assert read_alf(second).instance_map == {"1": "NEW"}

    def test_partial_instance_map_is_refused_on_decode(self, tmp_path):
        """A map that CLAIMS to record identity and misses ids exports objects
        with object_uuid: null and reports success."""
        path = tmp_path / "a.all.gz"
        write_all(
            path,
            np.array([1, 2], dtype=np.uint8),
            instance_ids=np.array([1, 2], dtype=np.uint16),
            instance_map={"1": "u1", "2": "u2"},
        )
        _rewrite_metadata(path, lambda m: m.__setitem__("instance_map", {"1": "u1"}))
        alf = read_alf(path)
        assert alf.categories.tolist() == [1, 2]  # semantic labels stay readable
        with pytest.raises(AlfError, match="missing entries for instance id"):
            alf.instance_ids

    def test_categories_stay_readable_when_an_instance_id_is_reserved(self, tmp_path):
        """The panoptic half being unusable must not deny a semantic-only
        consumer category data that is perfectly fine — the accessor split this
        module documents, which the reserved-id check violated by living in the
        shared decode."""
        payload = bytes([7, 0xFF, 0xFF, 9, 0x01, 0x00])
        path = tmp_path / "000000.all"
        path.write_bytes(struct.pack("B", 2) + struct.pack("<I", 0) + payload)
        alf = read_alf(path)
        assert alf.categories.tolist() == [7, 9]
        with pytest.raises(AlfError, match="reserved instance id 65535"):
            alf.instance_ids

    def test_instance_map_property_checks_keys_and_coverage(self, tmp_path):
        """A caller resolving identities straight off this property would never
        reach `instance_ids`' checks, so it must enforce them itself."""
        path = tmp_path / "a.all.gz"
        write_all(
            path,
            np.array([1], dtype=np.uint8),
            instance_ids=np.array([1], dtype=np.uint16),
            instance_map={"1": "u1"},
        )
        _rewrite_metadata(path, lambda m: m.__setitem__("instance_map", {"01": "u1"}))
        with pytest.raises(AlfError, match="non-canonical key"):
            read_alf(path).instance_map

        _rewrite_metadata(path, lambda m: m.__setitem__("instance_map", {"2": "u2"}))
        with pytest.raises(AlfError, match="missing entries for emitted instance id"):
            read_alf(path).instance_map

    def test_absent_instance_map_still_reads(self, tmp_path):
        """Legacy v2 files identified by the wire byte never recorded identity;
        refusing them denies reads of stored data to enforce a later rule."""
        payload = bytes([7, 0x01, 0x00, 9, 0xFE, 0xFF])
        raw = struct.pack("B", 2) + struct.pack("<I", 0) + payload
        path = tmp_path / "000000.all"
        path.write_bytes(raw)
        assert read_alf(path).instance_ids.tolist() == [1, 65534]

    def test_reserved_instance_id_is_refused_on_decode_too(self, tmp_path):
        """Constraining only the writer lets the reader hand back labels whose
        identity the editor cannot resolve."""
        path = tmp_path / "a.all.gz"
        write_all(
            path,
            np.array([1, 2], dtype=np.uint8),
            instance_ids=np.array([1, 65534], dtype=np.uint16),
            instance_map={"1": "u1", "65534": "u2"},
        )
        raw = bytearray(gzip.decompress(path.read_bytes()))
        raw[-2:] = (65535).to_bytes(2, "little")
        path.write_bytes(gzip.compress(bytes(raw)))
        with pytest.raises(AlfError, match="reserved instance id 65535"):
            read_alf(path).instance_ids

    def test_geometry_decodes_once(self, tmp_path):
        """max_geom_error_m calls xyz for fp16, so check-then-use decoded the
        payload twice on frames that run to millions of points."""
        path = tmp_path / "a.alp.gz"
        write_alp(path, np.array([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]]))
        alf = read_alf(path)
        calls = []
        original = alf._decode_xyz
        alf._decode_xyz = lambda: (calls.append(1), original())[1]
        _ = alf.max_geom_error_m
        _ = alf.xyz
        _ = alf.xyz
        assert len(calls) == 1

    def test_v3_missing_offset_is_refused(self, tmp_path):
        """The grid is centred now, so a missing key decodes around the origin
        instead of the midpoint: 100..200 m comes back as -50..50 m."""
        path = tmp_path / "a.alp.gz"
        write_alp(
            path,
            np.array([[100.0, 0.0, 0.0], [200.0, 0.0, 0.0]]),
            codec=GEOM_INT16,
            frame_convention=FRAME_WORLD_ALIGNED,
        )
        _rewrite_metadata(path, lambda m: m.pop("_geom_offset"))
        with pytest.raises(AlfError, match="missing `_geom_offset`"):
            read_alf(path).xyz

    def test_non_json_literals_in_the_header_are_refused(self, tmp_path):
        """Python accepts NaN/Infinity; JSON has no such values, so every other
        implementation rejects these headers."""
        path = tmp_path / "a.alp.gz"
        write_alp(path, np.array([[1.0, 2.0, 3.0]]))
        _rewrite_metadata(path, lambda m: m.__setitem__("custom", float("nan")))
        with pytest.raises(AlfError, match="non-JSON literal"):
            read_alf(path)

    def test_palette_must_be_a_list_of_hex_strings(self, tmp_path):
        """A bare string passes len(), so 'ff0000' would read as six entries and
        make index 5 look in range."""
        path = tmp_path / "a.alc.gz"
        write_alc(path, np.array([0], dtype=np.uint8), ["ff0000"])
        _rewrite_metadata(path, lambda m: m.__setitem__("palette", "ff0000"))
        with pytest.raises(AlfError, match="must be a list"):
            read_alf(path).palette_indices

    def test_zero_width_bounds_over_varying_samples_are_refused(self):
        """lo=hi=0 over [0, 255] would zero real contrast and publish a header
        claiming a zero-width raw range."""
        with pytest.raises(ValueError, match="not constant"):
            map_intensity(np.array([0.0, 255.0]), lo=0.0, hi=0.0)

    def test_fp16_overflow_is_refused_even_with_the_budget_disabled(self, tmp_path):
        """check_budget=False permits a LARGE frame, not an unwritable
        coordinate. 70000 m overflows binary16 to Inf, and the file would carry
        inf in its own box — then be rejected by this module's own reader."""
        path = tmp_path / "a.alp.gz"
        with pytest.raises(ValueError, match="overflows float16"):
            write_alp(path, np.array([[70_000.0, 0.0, 0.0]]), check_budget=False)
        assert not path.exists()

    def test_int16_decode_overflow_is_refused(self, tmp_path):
        """A positive finite scale can still overflow: 1e308 times a stored 2 is
        Inf, and the fp16 path already refuses non-finite coordinates."""
        path = tmp_path / "a.alp.gz"
        # Two points, so centring leaves nonzero stored values to overflow — a
        # single point quantizes to (0,0,0) at its own centroid and cannot.
        write_alp(
            path, np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]), codec=GEOM_INT16, frame_convention=FRAME_WORLD_ALIGNED
        )
        _rewrite_metadata(path, lambda m: m.__setitem__("_geom_scale", 1e308))
        with pytest.raises(AlfError, match="non-finite"):
            read_alf(path).xyz

    def test_out_of_range_alc_index_is_refused_on_read(self, tmp_path):
        """The writer bounds these; external and legacy files do not."""
        path = tmp_path / "a.alc.gz"
        write_alc(path, np.array([0, 1], dtype=np.uint8), ["ff0000"])
        raw = bytearray(gzip.decompress(path.read_bytes()))
        raw[-1] = 5
        path.write_bytes(gzip.compress(bytes(raw)))
        with pytest.raises(AlfError, match="out of range for a"):
            read_alf(path).palette_indices

    def test_box_covers_the_decoded_points_not_the_input(self, tmp_path):
        """Both codecs round outward, and the server turns this header straight
        into item-metadata bounds — so a box from the pre-encoding extrema lets
        spatial filtering exclude real payload points."""
        xyz = np.array([[100.0, 0.005, 0.0], [-100.0, -0.005, 0.0]])
        for codec in (GEOM_INT16, GEOM_FP16):
            path = tmp_path / f"{codec}.alp.gz"
            write_alp(path, xyz, **_frame_kwargs(codec))
            alf = read_alf(path)
            lo = [float(v) for v in alf.metadata["box"].split(":")[:3]]
            hi = [float(v) for v in alf.metadata["box"].split(":")[3:]]
            decoded = alf.xyz
            assert (decoded >= np.array(lo) - 1e-12).all(), f"{codec}: point below box"
            assert (decoded <= np.array(hi) + 1e-12).all(), f"{codec}: point above box"

    def test_int16_codec_on_a_v1_envelope_is_refused(self, tmp_path):
        """Metadata is copied forward routinely, so a stale _geom_codec is a
        realistic shape — and fp16 [1,2,3] reads back as [15360, 16384, 16896]."""
        path = tmp_path / "a.alp.gz"
        write_alp(path, np.array([[1.0, 2.0, 3.0]]), codec=GEOM_FP16)
        _rewrite_metadata(
            path,
            lambda m: (m.__setitem__("_geom_codec", "int16"), m.__setitem__("_geom_scale", 1.0)),
        )
        with pytest.raises(AlfError, match="Fixed-point geometry is version 3"):
            read_alf(path).xyz

    def test_error_bound_validates_the_scale_it_reports(self, tmp_path):
        """A consumer may query the bound FIRST to gate on accuracy, so -0.5 for
        a corrupt negative scale would pass the gate that exists to catch it."""
        path = tmp_path / "a.alp.gz"
        write_alp(path, np.array([[1.0, 2.0, 3.0]]), codec=GEOM_INT16, frame_convention=FRAME_WORLD_ALIGNED)
        _rewrite_metadata(path, lambda m: m.__setitem__("_geom_scale", -1.0))
        with pytest.raises(AlfError, match="positive finite"):
            read_alf(path).max_geom_error_m

    def test_palette_accepts_rgb_tuples_and_rejects_junk(self, tmp_path):
        """generate_color_palette returns (r, g, b) tuples; str() would write the
        literal '(255, 0, 0)' — valid JSON no consumer can read as a colour."""
        path = tmp_path / "a.alc.gz"
        write_alc(path, np.array([0, 1], dtype=np.uint8), [(255, 0, 0), "#00FF00"])
        assert read_alf(path).palette == ["ff0000", "00ff00"]
        with pytest.raises(ValueError, match="six hex digits"):
            write_alc(tmp_path / "b.alc.gz", np.array([0], dtype=np.uint8), ["red"])

    def test_alx_rejects_a_count_that_disagrees_with_its_alp(self, tmp_path):
        """The sidecar is index-correlated with the geometry, so K points and
        K-1 indices is an independently valid file that misassociates every
        point after the gap."""
        with pytest.raises(ValueError, match=r"kept indices for an \.alp"):
            write_alx(
                tmp_path / "a.alx.gz",
                np.array([0, 1, 2]),
                alp_total_points=4,
                source_total_points=10,
            )

    def test_alx_accepts_a_matching_pair(self, tmp_path):
        path = tmp_path / "a.alx.gz"
        write_alx(path, np.array([0, 5, 9]), alp_total_points=3, source_total_points=10)
        assert read_alf(path).kept_indices.tolist() == [0, 5, 9]

    def test_boolean_mask_is_not_an_integer_stream(self, tmp_path):
        """np.mod(bool, 1) is 0 and the range is [0, 1], so a mask clears both
        checks and is written as valid-looking per-point data."""
        with pytest.raises(ValueError, match="not booleans"):
            write_all(
                tmp_path / "a.all.gz",
                np.array([1, 2], dtype=np.uint8),
                instance_ids=np.array([True, False]),
                instance_map={"1": "u1"},
            )
        with pytest.raises(ValueError, match="not booleans"):
            write_alx(
                tmp_path / "a.alx.gz",
                np.array([True, False]),
                alp_total_points=2,
                source_total_points=10,
            )

    def test_enormous_format_version_does_not_overflow(self, tmp_path):
        path = tmp_path / "a.all.gz"
        write_all(path, np.array([1, 2, 3], dtype=np.uint8))
        _rewrite_metadata(path, lambda m: m.__setitem__("_format_version", 10**400))
        with pytest.raises(AlfError, match="_format_version"):
            read_alf(path).categories

    def test_normalized_float_palette_is_refused_not_truncated(self, tmp_path):
        """(1.0, 0, 0) means red in a normalized palette; int() writes 010000,
        which the viewer then treats as an authoritative colour."""
        with pytest.raises(ValueError, match="must be an integer"):
            write_alc(tmp_path / "a.alc.gz", np.array([0], dtype=np.uint8), [(1.0, 0.0, 0.0)])
        # ints on the 0..255 scale are unambiguous and still accepted
        path = tmp_path / "b.alc.gz"
        write_alc(path, np.array([0], dtype=np.uint8), [(255, 0, 0)])
        assert read_alf(path).palette == ["ff0000"]

    def test_enormous_header_count_does_not_overflow(self, tmp_path):
        """float(10**400) raises OverflowError, escaping the quarantine contract
        before the AlfError path can run."""
        hb = json.dumps({"_total_points": 10**400}, separators=(",", ":")).encode()
        path = tmp_path / "000000.alp"
        path.write_bytes(struct.pack("B", 1) + struct.pack("<I", len(hb)) + hb + bytes(6))
        # AlfError, not OverflowError — the whole point is that a caller
        # quarantining bad frames still gets the contract's exception type.
        with pytest.raises(AlfError):
            read_alf(path)

    def test_oversized_integer_literal_is_an_alf_error(self, tmp_path):
        """CPython raises a plain ValueError past its int digit limit, not
        JSONDecodeError."""
        path = tmp_path / "000000.alp"
        body = b'{"n":' + b"9" * 5000 + b"}"
        path.write_bytes(struct.pack("B", 1) + struct.pack("<I", len(body)) + body)
        with pytest.raises(AlfError, match="not valid JSON"):
            read_alf(path)

    def test_non_numeric_geom_scale_is_an_alf_error(self, tmp_path):
        path = tmp_path / "a.alp.gz"
        write_alp(path, np.array([[1.0, 2.0, 3.0]]), codec=GEOM_INT16, frame_convention=FRAME_WORLD_ALIGNED)
        _rewrite_metadata(path, lambda m: m.__setitem__("_geom_scale", "bad"))
        with pytest.raises(AlfError, match="must be a JSON number"):
            read_alf(path).xyz

    def test_truncated_int16_payload_is_an_alf_error(self, tmp_path):
        hb = json.dumps(
            {
                "_frame_convention": "world-aligned",
                "_geom_codec": "int16",
                "_geom_scale": 0.01,
                "_geom_offset": [0, 0, 0],
            },
            separators=(",", ":"),
        ).encode()
        path = tmp_path / "000000.alp"
        path.write_bytes(struct.pack("B", 3) + struct.pack("<I", len(hb)) + hb + bytes(5))
        with pytest.raises(AlfError, match="not a whole number of"):
            read_alf(path).xyz

    def test_fractional_source_count_is_refused_not_truncated(self, tmp_path):
        """int() would write 1 for 1.9, normalizing away a value the reader's own
        domain check would then never see."""
        with pytest.raises(ValueError, match="must be whole"):
            write_alx(tmp_path / "a.alx.gz", np.array([0]), alp_total_points=1, source_total_points=1.9)

    def test_multi_column_per_point_arrays_are_refused(self, tmp_path):
        """ravel() would flatten an (N, 4) matrix into 4N values and record
        _total_points for the expanded size — a valid-looking file whose length
        no longer matches the cloud it is index-correlated with."""
        with pytest.raises(ValueError, match="must be 1-D"):
            write_ali(tmp_path / "a.ali.gz", np.zeros((4, 2), dtype=np.float64), lo=0.0, hi=1.0)
        with pytest.raises(ValueError, match="must be 1-D"):
            write_all(tmp_path / "a.all.gz", np.zeros((2, 2), dtype=np.uint8))

    def test_column_vector_is_still_accepted(self, tmp_path):
        path = tmp_path / "a.all.gz"
        write_all(path, np.array([[1], [2], [3]], dtype=np.uint8))
        assert read_alf(path).categories.tolist() == [1, 2, 3]

    def test_legacy_alp_named_label_file_gets_an_actionable_error(self, tmp_path):
        """Two legacy writers store segmentation LABELS as .alp.gz, so path
        inference calls them geometry. The reader cannot distinguish that from a
        truncated .alp -- a 2-point frame missing one point is 6 bytes, exactly
        the v2 label stride -- so it names both causes instead of guessing."""
        hb = json.dumps({"_total_points": 4}, separators=(",", ":")).encode()
        path = tmp_path / "000000.alp"
        path.write_bytes(struct.pack("B", 1) + struct.pack("<I", len(hb)) + hb + bytes(4))
        with pytest.raises(AlfError) as exc:
            read_alf(path)
        assert 'extension="all"' in str(exc.value)
        # and reading it as what it actually is works
        assert read_alf(path, extension="all").categories.tolist() == [0, 0, 0, 0]

    def test_instance_map_refuses_the_wrong_channel(self, tmp_path):
        """A .ali from a mixed glob returned None instead of surfacing the
        mix-up, masking it as a label file with no panoptic identities."""
        path = tmp_path / "a.ali.gz"
        write_ali(path, np.arange(4, dtype=np.float64), lo=0.0, hi=3.0)
        with pytest.raises(AlfError, match="belongs to"):
            read_alf(path).instance_map

    def test_unusable_instance_map_values_are_not_coverage(self, tmp_path):
        """The legacy batch exporter writes instance_lookup.get(key) straight into
        object_uuid, so a null or empty value exports garbage identity."""
        for bad in (None, "", "   ", 42):
            # The whole-map check now catches these first and names them more
            # precisely than the coverage check did.
            with pytest.raises(ValueError, match="unusable value"):
                write_all(
                    tmp_path / "a.all.gz",
                    np.array([1], dtype=np.uint8),
                    instance_ids=np.array([1], dtype=np.uint16),
                    instance_map={"1": bad},
                )

    def test_inherited_non_dict_instance_map_is_refused(self, tmp_path):
        """A string passes setdefault and then the coverage check by accident --
        `str(1) in "1"` is True -- so the writer emits a file its own reader
        rejects."""
        with pytest.raises(ValueError, match="inherited instance_map"):
            write_all(
                tmp_path / "a.all.gz",
                np.array([1], dtype=np.uint8),
                instance_ids=np.array([1], dtype=np.uint16),
                metadata={"instance_map": "1"},
            )

    def test_oversized_palette_is_refused_on_read(self, tmp_path):
        """At 256 entries the no-colour sentinel is unrepresentable, so index 255
        reads as a normal colour and occluded points have nowhere to go."""
        path = tmp_path / "a.alc.gz"
        write_alc(path, np.array([0], dtype=np.uint8), ["ff0000"])
        _rewrite_metadata(path, lambda m: m.__setitem__("palette", [f"{i:02x}0000" for i in range(256)]))
        with pytest.raises(AlfError, match="maximum is 255"):
            read_alf(path).palette_indices

    def test_star_import_exposes_the_alf_api(self):
        """AGENTS.md ("Module Export Hygiene") makes `__all__` the public API list
        and names `from package import *` as the check. Keeping the ALF names out
        of it hid read_alf/write_alp from the repo's own discovery mechanism to
        protect an import nobody performs by accident.

        Star-importing NumPy is the accepted cost, and it is paid only by a
        caller explicitly asking for every public API at once. The guarantee that
        matters — plain `import core` — is covered separately below."""
        import subprocess
        import sys

        probe = (
            "import sys\n"
            "exec('from alf import *')\n"
            "assert 'read_alf' in dir(), 'star import did not expose read_alf'\n"
            "assert 'write_alp' in dir(), 'star import did not expose write_alp'\n"
        )
        env = dict(os.environ, PYTHONPATH=str(_PACKAGE_ROOT))
        assert subprocess.run([sys.executable, "-c", probe], env=env, check=False).returncode == 0

    def test_duplicate_kept_indices_are_refused_on_read(self, tmp_path):
        """Size and range are both satisfied by [5, 5]."""
        path = tmp_path / "a.alx.gz"
        write_alx(path, np.array([5, 6]), alp_total_points=2, source_total_points=10)
        raw = bytearray(gzip.decompress(path.read_bytes()))
        raw[-4:] = (5).to_bytes(4, "little")
        path.write_bytes(gzip.compress(bytes(raw)))
        with pytest.raises(AlfError, match="duplicate kept indices"):
            read_alf(path).kept_indices

    def test_boolean_intensity_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="not booleans"):
            write_ali(tmp_path / "a.ali.gz", np.array([True, False]), lo=0.0, hi=1.0)

    def test_mixed_boolean_sequences_are_refused_everywhere(self, tmp_path):
        """np.asarray([True, 2]) promotes to int64, so a dtype check only ever
        caught UNIFORMLY boolean input — a mixed one becomes a real coordinate,
        category or intensity byte."""
        with pytest.raises(ValueError, match="not booleans"):
            write_alp(tmp_path / "a.alp.gz", [[True, 0, 0]])
        with pytest.raises(ValueError, match="not booleans"):
            write_all(tmp_path / "a.all.gz", [True, 2])
        with pytest.raises(ValueError, match="not booleans"):
            write_ali(tmp_path / "a.ali.gz", [True, 2], lo=0.0, hi=5.0)

    def test_origin_local_frame_places_with_sensor_pose(self, tmp_path):
        """write_alp without device_position produces an origin-local frame, and
        a caller correctly asserting the sensor-pose convention must be able to
        place it — routing {} into _as_anchor made "missing" look "corrupt"."""
        path = tmp_path / "a.alp.gz"
        write_alp(path, np.array([[1.0, 2.0, 3.0]]))
        assert read_alf(path).world_xyz(sensor_pose=True)[0] == pytest.approx([1.0, 2.0, 3.0])
        # a MALFORMED pose still fails
        bad = tmp_path / "b.alp.gz"
        write_alp(bad, np.array([[1.0, 2.0, 3.0]]))
        _rewrite_metadata(bad, lambda m: m.__setitem__("device_position", {"x": 1.0}))
        with pytest.raises(AlfError, match="missing required axis"):
            read_alf(bad).world_xyz(sensor_pose=True)

    def test_numpy_bool_in_a_dict_anchor_is_refused(self, tmp_path):
        """The dict path tested `bool` only, so a NumPy scalar coerced to 1.0."""
        path = tmp_path / "a.alp.gz"
        write_alp(path, np.array([[1.0, 2.0, 3.0]]))
        with pytest.raises(AlfError, match="not JSON numbers"):
            read_alf(path).world_xyz(origin={"x": np.bool_(True), "y": 0.0, "z": 0.0})

    def test_agreeing_v2_without_a_count_still_checks_the_stride(self, tmp_path):
        """Both fields saying v2 is not corroboration: 3-byte rows mean a 4-byte
        payload already contradicts the layout they assert."""
        hb = json.dumps({"_format_version": 2}, separators=(",", ":")).encode()
        bad = tmp_path / "000000.all"
        bad.write_bytes(struct.pack("B", 2) + struct.pack("<I", len(hb)) + hb + bytes(4))
        assert read_alf(bad).format_version_verified is False
        good = tmp_path / "000001.all"
        good.write_bytes(struct.pack("B", 2) + struct.pack("<I", len(hb)) + hb + bytes(6))
        assert read_alf(good).format_version_verified is True

    def test_numpy_booleans_are_refused(self, tmp_path):
        """isinstance(np.bool_(True), bool) is False, so a NumPy scalar — the
        shape a pandas/NumPy mask actually arrives in — slipped the check."""
        with pytest.raises(ValueError, match="not booleans"):
            write_all(tmp_path / "a.all.gz", [np.bool_(True), 2])
        with pytest.raises(ValueError, match="not booleans"):
            write_alp(tmp_path / "a.alp.gz", np.array([[True, 0, 0]], dtype=object))

    def test_ragged_anchor_is_an_alf_error(self, tmp_path):
        """np.asarray raises on a ragged sequence, and it sat outside the guard."""
        path = tmp_path / "a.alp.gz"
        write_alp(path, np.array([[1.0, 2.0, 3.0]]))
        with pytest.raises(AlfError, match="non-numeric component"):
            read_alf(path).world_xyz(origin=[[1], [2, 3], [4]])

    def test_boolean_intensity_bounds_are_refused(self, tmp_path):
        """math.isfinite(True) is True, and normalization then treats
        False/True as 0/1 — a valid-looking contrast channel from a caller bug."""
        import numpy as _np

        for lo, hi in ((False, True), (_np.bool_(False), 1.0), ("0", 1.0), (None, 1.0)):
            # np.bool_ is the real-world shape: a bound propagated from a
            # NumPy/pandas mask. isinstance(..., bool) is False for it while
            # math.isfinite accepts it, so the blacklist form let it through.
            with pytest.raises(ValueError, match="must be a real number"):
                write_ali(tmp_path / "a.ali.gz", np.array([1.0]), lo=lo, hi=hi)

    def test_empty_palette_is_refused(self, tmp_path):
        """`0 > len([])` is false, so payload byte 0 passes the range check."""
        path = tmp_path / "a.alc.gz"
        write_alc(path, np.array([0], dtype=np.uint8), ["ff0000"])
        _rewrite_metadata(path, lambda m: m.__setitem__("palette", []))
        with pytest.raises(AlfError, match="palette is empty"):
            read_alf(path).palette_indices

    def test_present_null_is_refused_on_every_field(self, tmp_path):
        """Absent means legacy and must keep working; present-and-null is a
        header that carries the field and says nothing, and collapsing the two
        lets malformed metadata take the permissive legacy path."""
        alp = tmp_path / "a.alp.gz"
        write_alp(alp, np.array([[1.0, 2.0, 3.0]]))
        _rewrite_metadata(alp, lambda m: m.__setitem__("device_position", None))
        with pytest.raises(AlfError, match=r"Geometry anchor.*malformed"):
            read_alf(alp)

        lbl = tmp_path / "a.all.gz"
        write_all(
            lbl,
            np.array([1], dtype=np.uint8),
            instance_ids=np.array([1], dtype=np.uint16),
            instance_map={"1": "u1"},
        )
        _rewrite_metadata(lbl, lambda m: m.__setitem__("instance_map", None))
        with pytest.raises(AlfError, match="present but null"):
            read_alf(lbl).instance_ids

        fv = tmp_path / "b.all.gz"
        write_all(fv, np.array([1, 2, 3], dtype=np.uint8))
        _rewrite_metadata(fv, lambda m: m.__setitem__("_format_version", None))
        with pytest.raises(AlfError, match="present but null"):
            read_alf(fv).categories

    def test_present_null_palette_and_codec_are_refused(self, tmp_path):
        """The last two fields still using a bare get(). A null palette read as
        "no palette" hides a corrupt colour file; a null _geom_codec defaulting
        to fp16 decodes a v3 int16 payload as float16 garbage."""
        alc = tmp_path / "a.alc.gz"
        write_alc(alc, np.array([0], dtype=np.uint8), ["ff0000"])
        _rewrite_metadata(alc, lambda m: m.__setitem__("palette", None))
        with pytest.raises(AlfError, match="present but null"):
            read_alf(alc).palette

        alp = tmp_path / "a.alp.gz"
        write_alp(alp, np.array([[1.0, 2.0, 3.0]]), codec=GEOM_INT16, frame_convention=FRAME_WORLD_ALIGNED)
        _rewrite_metadata(alp, lambda m: m.__setitem__("_geom_codec", None))
        with pytest.raises(AlfError, match="present but null"):
            read_alf(alp).xyz

    def test_absent_keys_still_read_as_legacy(self, tmp_path):
        """The other half of the rule: omitting them entirely must keep working,
        because that is what legacy files actually look like."""
        hb = json.dumps({"_total_points": 3}, separators=(",", ":")).encode()
        path = tmp_path / "000000.all"
        path.write_bytes(struct.pack("B", 1) + struct.pack("<I", len(hb)) + hb + bytes([1, 2, 3]))
        alf = read_alf(path)
        assert alf.categories.tolist() == [1, 2, 3]
        assert alf.instance_map is None

    def test_present_null_total_points_is_refused(self, tmp_path):
        """Treating it as absent disables the declared-count cross-check, so a
        payload of any whole-point length parses as valid."""
        path = tmp_path / "a.alp.gz"
        write_alp(path, np.array([[1.0, 2.0, 3.0]]))
        _rewrite_metadata(path, lambda m: m.__setitem__("_total_points", None))
        with pytest.raises(AlfError, match="present but null"):
            read_alf(path)

    def test_malformed_device_position_property_is_an_alf_error(self, tmp_path):
        """Reading the documented frame anchor directly must not abort a caller
        that quarantines bad frames by catching AlfError."""
        path = tmp_path / "a.alp.gz"
        write_alp(path, np.array([[1.0, 2.0, 3.0]]))
        _rewrite_metadata(path, lambda m: m.__setitem__("device_position", "bad"))
        with pytest.raises(AlfError):
            read_alf(path).device_position

    def test_string_geom_scale_is_refused(self, tmp_path):
        """float("0.1") coerces fine, letting a Python-only conversion define
        the file's semantics — the counts and format version reject strings."""
        path = tmp_path / "a.alp.gz"
        write_alp(path, np.array([[1.0, 2.0, 3.0]]), codec=GEOM_INT16, frame_convention=FRAME_WORLD_ALIGNED)
        _rewrite_metadata(path, lambda m: m.__setitem__("_geom_scale", "0.1"))
        with pytest.raises(AlfError, match="must be a JSON number"):
            read_alf(path).xyz

    def test_error_bound_validates_the_offset_not_just_its_presence(self, tmp_path):
        """A malformed offset returned a confident scale/2 while .xyz refused
        the same frame — the gate passing on geometry that cannot be produced."""
        # (a non-finite component never reaches here: `Infinity` is not JSON, so
        # the header parser rejects it first — see the non-JSON-literal test.)
        for bad in ("bad", [1, 2], [True, 0, 0]):
            path = tmp_path / "a.alp.gz"
            write_alp(path, np.array([[1.0, 2.0, 3.0]]), codec=GEOM_INT16, frame_convention=FRAME_WORLD_ALIGNED)
            _rewrite_metadata(path, lambda m, v=bad: m.__setitem__("_geom_offset", v))
            alf = read_alf(path)
            with pytest.raises(AlfError):
                alf.max_geom_error_m
            with pytest.raises(AlfError):
                alf.xyz

    def test_empty_palette_is_refused_on_the_property_too(self, tmp_path):
        path = tmp_path / "a.alc.gz"
        write_alc(path, np.array([0], dtype=np.uint8), ["ff0000"])
        _rewrite_metadata(path, lambda m: m.__setitem__("palette", []))
        with pytest.raises(AlfError, match="palette is empty"):
            read_alf(path).palette

    def test_error_bound_requires_the_offset_it_depends_on(self, tmp_path):
        """The preflight must not report a usable bound for a frame .xyz will
        refuse to reconstruct."""
        path = tmp_path / "a.alp.gz"
        write_alp(path, np.array([[1.0, 2.0, 3.0]]), codec=GEOM_INT16, frame_convention=FRAME_WORLD_ALIGNED)
        _rewrite_metadata(path, lambda m: m.pop("_geom_offset"))
        with pytest.raises(AlfError, match="missing `_geom_offset`"):
            read_alf(path).max_geom_error_m

    def test_unusable_entry_for_an_unemitted_id_is_refused(self, tmp_path):
        """The map is sequence-level, so entries for absent ids are legitimate —
        but the reader validates the WHOLE mapping, so a null value for an
        unemitted id writes a file the shared API cannot read back."""
        with pytest.raises(ValueError, match="unusable value"):
            write_all(
                tmp_path / "a.all.gz",
                np.array([1], dtype=np.uint8),
                instance_ids=np.array([1], dtype=np.uint16),
                instance_map={"1": "u1", "2": None},
            )

    def test_entries_for_unemitted_ids_are_kept(self, tmp_path):
        """Inheritance across a sequence depends on carrying them."""
        path = tmp_path / "a.all.gz"
        write_all(
            path,
            np.array([1], dtype=np.uint8),
            instance_ids=np.array([1], dtype=np.uint16),
            instance_map={"1": "u1", "2": "u2"},
        )
        assert read_alf(path).instance_map == {"1": "u1", "2": "u2"}

    def test_malformed_instance_map_property_is_validated(self, tmp_path):
        """The annotation promises Dict[str, str]; only instance_ids checked."""
        path = tmp_path / "a.all.gz"
        write_all(
            path,
            np.array([1], dtype=np.uint8),
            instance_ids=np.array([1], dtype=np.uint16),
            instance_map={"1": "u1"},
        )
        _rewrite_metadata(path, lambda m: m.__setitem__("instance_map", {"1": None}))
        with pytest.raises(AlfError, match="unusable value"):
            read_alf(path).instance_map

    def test_boolean_geom_offset_is_refused(self, tmp_path):
        """The offset is ADDED to every decoded point, so a boolean coerced to
        1.0 returns a plausible cloud shifted a metre off that axis."""
        path = tmp_path / "a.alp.gz"
        write_alp(path, np.array([[1.0, 2.0, 3.0]]), codec=GEOM_INT16, frame_convention=FRAME_WORLD_ALIGNED)
        _rewrite_metadata(path, lambda m: m.__setitem__("_geom_offset", [True, 0.0, 0.0]))
        with pytest.raises(AlfError, match="boolean component"):
            read_alf(path).xyz

    def test_oversized_geom_scale_is_an_alf_error(self, tmp_path):
        """float(10**400) raises OverflowError before the positive/finite check
        can turn it into AlfError.

        The 2**53 exactness check now fires first for a value this large, which
        is an equally correct refusal with a different message — so this asserts
        the CONTRACT (AlfError, never a bare OverflowError) rather than pinning
        whichever guard happens to run first."""
        path = tmp_path / "a.alp.gz"
        write_alp(path, np.array([[1.0, 2.0, 3.0]]), codec=GEOM_INT16, frame_convention=FRAME_WORLD_ALIGNED)
        _rewrite_metadata(path, lambda m: m.__setitem__("_geom_scale", 10**400))
        with pytest.raises(AlfError):
            read_alf(path).xyz
        with pytest.raises(AlfError):
            read_alf(path).max_geom_error_m

    def test_boolean_anchor_components_are_refused(self, tmp_path):
        """float(True) is 1.0, so the frame lands one metre off an axis."""
        path = tmp_path / "a.alp.gz"
        write_alp(path, np.array([[1.0, 2.0, 3.0]]))
        alf = read_alf(path)
        with pytest.raises(AlfError, match="not JSON numbers"):
            alf.world_xyz(origin={"x": True, "y": 0.0, "z": 0.0})
        # One message for every boolean shape now — the shared helper catches
        # bool-dtype arrays, mixed sequences and NumPy scalars alike.
        with pytest.raises(AlfError, match="boolean component"):
            alf.world_xyz(origin=np.array([True, False, True]))
        # MIXED sequences are the subtle case: np.asarray([True, 0, 0]) promotes
        # to int64, so a dtype check alone misses it and the frame shifts a
        # metre off that axis.
        with pytest.raises(AlfError, match="boolean component"):
            alf.world_xyz(origin=[True, 0, 0])

    def test_oversized_anchor_components_are_alf_errors(self, tmp_path):
        """float(10**400) and np.asarray both raise OverflowError, which is not
        a ValueError, so it escaped the quarantine contract."""
        path = tmp_path / "a.alp.gz"
        write_alp(path, np.array([[1.0, 2.0, 3.0]]))
        alf = read_alf(path)
        with pytest.raises(AlfError):
            alf.world_xyz(origin={"x": 10**400, "y": 0.0, "z": 0.0})
        with pytest.raises(AlfError):
            alf.world_xyz(origin=[10**400, 0, 0])

    def test_oversized_geom_offset_is_an_alf_error(self, tmp_path):
        path = tmp_path / "b.alp.gz"
        write_alp(path, np.array([[1.0, 2.0, 3.0]]), codec=GEOM_INT16, frame_convention=FRAME_WORLD_ALIGNED)
        _rewrite_metadata(path, lambda m: m.__setitem__("_geom_offset", [10**400, 0, 0]))
        with pytest.raises(AlfError):
            read_alf(path).xyz

    def test_unknown_path_suffix_requires_an_explicit_channel(self, tmp_path):
        """A cache file named by blob id would otherwise leave extension empty
        and disable every channel guard."""
        src = tmp_path / "a.all.gz"
        write_all(src, np.array([1, 2, 3, 4, 5, 6], dtype=np.uint8))
        blob = tmp_path / "9f86d081884c"
        blob.write_bytes(src.read_bytes())
        with pytest.raises(AlfError, match="Cannot infer the ALF channel"):
            read_alf(blob)
        assert read_alf(blob, extension="all").categories.tolist() == [1, 2, 3, 4, 5, 6]

    def test_malformed_device_position_is_an_alf_error(self, tmp_path):
        """A malformed placement is refused while the envelope is read."""
        path = tmp_path / "a.alp.gz"
        write_alp(path, np.array([[1.0, 2.0, 3.0]]))
        _rewrite_metadata(path, lambda m: m.__setitem__("device_position", "nonsense"))
        with pytest.raises(AlfError, match=r"Geometry anchor.*malformed"):
            read_alf(path)

    def test_writer_refuses_string_coordinates(self, tmp_path):
        """A CSV column that never got parsed writes a point at (1, 2, 3) and
        hides the parser bug — in the channel every other channel indexes."""
        with pytest.raises(ValueError, match="coordinates are strings"):
            write_alp(tmp_path / "a.alp.gz", np.array([["1", "2", "3"]]))
        with pytest.raises(ValueError, match="coordinates are strings"):
            write_alp(tmp_path / "b.alp.gz", [["1", "2", "3"]])
        # A numeric object array is still fine — that is how ragged input arrives.
        write_alp(tmp_path / "c.alp.gz", np.array([[1, 2.0, 3]], dtype=object))
        assert read_alf(tmp_path / "c.alp.gz").xyz.tolist() == [[1.0, 2.0, 3.0]]

    def test_writer_refuses_noncanonical_instance_map_keys(self, tmp_path):
        """`{"1": "u1", "01": "stale"}` passed the value and coverage checks, so
        the shared WRITER emitted a file the shared READER then rejected."""
        with pytest.raises(ValueError, match="non-canonical key"):
            write_all(
                tmp_path / "a.all.gz",
                np.array([1], dtype=np.uint8),
                instance_ids=np.array([1], dtype=np.uint16),
                instance_map={"1": "u1", "01": "stale"},
            )

    def test_writer_refuses_non_string_metadata_keys_recursively(self, tmp_path):
        """Python JSON-coerces mapping keys to strings, so ``1`` and ``"1"``
        become duplicate members that the paired reader must refuse."""
        collision = {1: "numeric", "1": "string"}
        for index, metadata in enumerate((collision, {"nested": collision})):
            path = tmp_path / f"bad-{index}.ali.gz"
            with pytest.raises(ValueError, match=r"metadata object key.*string"):
                write_ali(
                    path,
                    np.array([7.0]),
                    lo=0.0,
                    hi=10.0,
                    metadata=metadata,
                )
            assert not path.exists()

    def test_writer_validates_inherited_anchors(self, tmp_path):
        """A pass-through rewrite of an external header wrote clean and then
        raised on the first `.device_position` of the file it just produced."""
        for bad in ({"device_position": None}, {"device_position": {"x": 1}}):
            with pytest.raises(ValueError, match="inherited `device_position` is unusable"):
                write_alp(tmp_path / "a.alp.gz", np.array([[1.0, 2.0, 3.0]]), bad)
        with pytest.raises(ValueError, match="inherited `world_centroid_epsg3857` is unusable"):
            write_alp(
                tmp_path / "b.alp.gz",
                np.array([[1.0, 2.0, 3.0]]),
                {"world_centroid_epsg3857": {"x": 1}},
            )
        # An EXPLICIT device_position still overrides a bad inherited one.
        ok = tmp_path / "c.alp.gz"
        write_alp(
            ok,
            np.array([[1.0, 2.0, 3.0]]),
            {"device_position": None},
            device_position=(1.0, 2.0, 3.0),
        )
        assert read_alf(ok).device_position == pytest.approx([1.0, 2.0, 3.0])

    def test_writer_preserves_anchor_application_members_without_mutating_metadata(self, tmp_path):
        nested = {"source": "gps", "flags": ["rtk", {"fixed": True}]}
        metadata = {
            "device_position": {
                "z": 3,
                "note": nested,
                "x": 1,
                "quality": 0.875,
                "y": 2,
            }
        }
        before = json.dumps(metadata, sort_keys=True)

        inherited_path = tmp_path / "inherited.alp.gz"
        write_alp(inherited_path, np.array([[0.0, 0.0, 0.0]]), metadata)
        inherited = read_alf(inherited_path).metadata["device_position"]
        assert inherited == {
            "note": nested,
            "quality": 0.875,
            "x": 1.0,
            "y": 2.0,
            "z": 3.0,
        }

        explicit_path = tmp_path / "explicit.alp.gz"
        write_alp(
            explicit_path,
            np.array([[0.0, 0.0, 0.0]]),
            metadata,
            device_position=(4.0, 5.0, 6.0),
        )
        explicit = read_alf(explicit_path).metadata["device_position"]
        assert explicit == {
            "note": nested,
            "quality": 0.875,
            "x": 4.0,
            "y": 5.0,
            "z": 6.0,
        }
        assert json.dumps(metadata, sort_keys=True) == before

    def test_writer_preserves_application_members_from_explicit_anchor(self, tmp_path):
        """The public anchor argument owns more than its placement axes.

        A caller may attach provenance to the same x/y/z object without also
        duplicating that object under ``metadata``. Replacing the axes must not
        silently discard those application-owned members.
        """
        nested = {"source": "gps", "flags": ["rtk", {"fixed": True}]}
        anchor = {
            "z": 6.0,
            "note": nested,
            "x": 4.0,
            "quality": 0.875,
            "y": 5.0,
        }
        before = json.dumps(anchor, sort_keys=True)
        path = tmp_path / "explicit-only.alp.gz"

        write_alp(
            path,
            np.array([[0.0, 0.0, 0.0]]),
            device_position=anchor,
        )

        assert read_alf(path).metadata["device_position"] == {
            "note": nested,
            "quality": 0.875,
            "x": 4.0,
            "y": 5.0,
            "z": 6.0,
        }
        assert json.dumps(anchor, sort_keys=True) == before

    def test_writer_accepts_the_legacy_empty_anchor(self, tmp_path):
        """`{}` is what real producers emit and what the reader calls a zero
        origin. Refusing it on write meant `write_alp(metadata=alf.metadata)`
        could not rewrite a file `read_alf` accepts — the same confusion of
        "absent" with "corrupt" already fixed once on the read side."""
        path = tmp_path / "a.alp.gz"
        write_alp(path, np.array([[1.0, 2.0, 3.0]]), {"device_position": {}})
        assert read_alf(path).device_position == pytest.approx([0.0, 0.0, 0.0])
        # a nonempty MALFORMED anchor is still refused
        with pytest.raises(ValueError, match="inherited `device_position` is unusable"):
            write_alp(tmp_path / "b.alp.gz", np.array([[1.0, 2.0, 3.0]]), {"device_position": {"x": 1}})

    def test_zero_column_geometry_is_not_an_empty_frame(self, tmp_path):
        """A parser that selected no coordinate columns has size 0, so a bare
        size check wrote `_total_points: 0` and discarded every record."""
        with pytest.raises(ValueError, match="coordinate column"):
            write_alp(tmp_path / "a.alp.gz", np.empty((5, 0)))
        # a GENUINELY empty frame still writes
        empty = tmp_path / "b.alp.gz"
        write_alp(empty, np.empty((0, 3)))
        assert read_alf(empty).xyz.shape == (0, 3)

    def test_deeply_nested_header_is_an_alf_error(self, tmp_path):
        """RecursionError is not a ValueError, so it escaped the handler and
        aborted a pipeline whose quarantine is `except AlfError: skip`."""
        deep = "[" * 60000 + "]" * 60000
        hb = ('{"_total_points":0,"x":' + deep + "}").encode()
        path = tmp_path / "000000.alp"
        path.write_bytes(struct.pack("B", 1) + struct.pack("<I", len(hb)) + hb)
        with pytest.raises(AlfError, match="too deep"):
            read_alf(path)

    def test_reader_and_writer_share_the_256_collection_depth_limit(self, tmp_path):
        allowed = 0
        for _ in range(255):
            allowed = [allowed]
        output = tmp_path / "allowed.ali.gz"
        write_ali(
            output,
            np.array([], dtype=np.float64),
            lo=0.0,
            hi=1.0,
            metadata={"custom": allowed},
        )
        assert "custom" in read_alf(output).metadata

        too_deep = [allowed]
        with pytest.raises(ValueError, match="maximum JSON nesting depth"):
            write_ali(
                tmp_path / "refused.ali.gz",
                np.array([], dtype=np.float64),
                lo=0.0,
                hi=1.0,
                metadata={"custom": too_deep},
            )

        header = ('{"_total_points":0,"custom":' + "[" * 256 + "0" + "]" * 256 + "}").encode()
        raw = struct.pack("B", 1) + struct.pack("<I", len(header)) + header
        with pytest.raises(AlfError, match="header-nesting-too-deep"):
            read_alf(raw, extension="ali")

    def test_anchor_application_overflow_is_refused(self, tmp_path):
        """Finite locals plus a finite anchor can still exceed float64, and Inf
        in world coordinates poisons every bound and metric downstream."""
        hb = json.dumps(
            {
                "_total_points": 1,
                "_geom_codec": "int16",
                "_frame_convention": "world-aligned",
                "_geom_scale": 5e303,
                "_geom_offset": [0, 0, 0],
                "device_position": {"x": 1e308, "y": 0.0, "z": 0.0},
            },
            separators=(",", ":"),
        ).encode()
        path = tmp_path / "000000.alp"
        payload = np.array([32767, 0, 0], dtype="<i2").tobytes()
        path.write_bytes(struct.pack("B", 3) + struct.pack("<I", len(hb)) + hb + payload)
        alf = read_alf(path)
        assert np.isfinite(alf.xyz).all()  # locals are finite
        with pytest.raises(AlfError, match="overflowed to non-finite"):
            alf.world_xyz(sensor_pose=True)

    def test_empty_anchor_exemption_is_device_position_only(self, tmp_path):
        """`{}` means "zero origin" for a sensor pose and NOTHING for an aerial
        centroid — `world_xyz` requires all three axes of the latter. Exempting
        both keys just moved the writer/reader disagreement one key across."""
        with pytest.raises(ValueError, match="inherited `world_centroid_epsg3857` is unusable"):
            write_alp(
                tmp_path / "a.alp.gz",
                np.array([[1.0, 2.0, 3.0]]),
                {"world_centroid_epsg3857": {}},
            )

    def test_intensity_normalization_does_not_overflow(self, tmp_path):
        """`(values - lo) * 255` overflows to Inf on large-but-finite bounds
        even though the normalized result is representable, so the midpoint
        clipped to 255 and write_ali published a saturated channel."""
        path = tmp_path / "a.ali.gz"
        write_ali(path, np.array([5e307]), lo=0.0, hi=1e308)
        assert read_alf(path).intensity.tolist() == [127]

    def test_int16_writer_refuses_what_its_reader_rejects(self, tmp_path):
        """With check_budget=False and coordinates near float64 max, the derived
        scale reconstructs to Inf: the writer recorded an `-inf:...:inf` box and
        `read_alf(path).xyz` raised on the file it had just written."""
        m = np.finfo(np.float64).max
        with pytest.raises(ValueError, match="reconstruction overflows"):
            write_alp(
                tmp_path / "a.alp.gz",
                np.array([[-m, 0.0, 0.0], [m, 0.0, 0.0]]),
                codec="int16",
                frame_convention="world-aligned",
                check_budget=False,
            )

    def test_intensity_range_itself_can_overflow(self, tmp_path):
        """Dividing before scaling was not enough: `hi - lo` overflows too.
        Over [-1e308, 1e308] both the range and the endpoint subtraction went
        Inf, Inf/Inf gave NaN, and the cast published a channel with NO
        contrast — from bounds the writer had already accepted as finite."""
        path = tmp_path / "a.ali.gz"
        write_ali(path, np.array([-1e308, 0.0, 1e308]), lo=-1e308, hi=1e308)
        assert read_alf(path).intensity.tolist() == [0, 127, 255]
        # and the ordinary case is unchanged by the halving
        plain = tmp_path / "b.ali.gz"
        write_ali(plain, np.array([0.0, 5.0, 10.0]), lo=0.0, hi=10.0)
        assert read_alf(plain).intensity.tolist() == [0, 127, 255]

    def test_concurrent_writers_do_not_share_a_temp_file(self, tmp_path):
        """A PID-based temp name is shared by every THREAD in one process: the
        first os.replace unlinked it, so the loser either raised
        FileNotFoundError or kept writing into the already-published file."""
        import threading

        target = tmp_path / "race.alp.gz"
        errors: list = []

        def writer(i):
            try:
                write_alp(target, np.array([[float(i), 0.0, 0.0]]))
            except Exception as exc:
                errors.append(repr(exc))

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(24)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        # whichever writer won, the published file is complete and readable
        assert read_alf(target).xyz.shape == (1, 3)
        assert not list(tmp_path.glob(".*tmp*")), "temp files left behind"

    def test_error_bound_includes_offset_rounding(self, tmp_path):
        """`scale / 2` is the quantization error ALONE. Reconstruction is
        `quantized * scale + offset` in float64, so for a far off-origin frame
        the ULP at the offset dominates: a frame around -4.488e11 spanning 8 mm
        recorded 5e-5 while decoding with 6.1035e-5 of error. An accuracy-gated
        caller would have accepted geometry outside its threshold."""
        far = -4.488e11 + np.array([[0.0, 0, 0], [0.004, 0, 0], [0.008, 0, 0]])
        path = tmp_path / "a.alp.gz"
        write_alp(path, far, codec="int16", frame_convention="world-aligned")
        alf = read_alf(path)
        actual = float(np.max(np.abs(alf.xyz - far)))
        assert actual > 5e-05, "repro no longer exercises offset rounding"
        assert alf.max_geom_error_m >= actual, "stated bound understates the file"

        # An ego-centric frame is unaffected: no offset, no extra term.
        near = np.array([[0.0, 0, 0], [0.004, 0, 0], [0.008, 0, 0]])
        p2 = tmp_path / "b.alp.gz"
        write_alp(p2, near, codec="int16", frame_convention="world-aligned")
        near_alf = read_alf(p2)
        assert near_alf.max_geom_error_m == pytest.approx(5e-05)
        assert float(np.max(np.abs(near_alf.xyz - near))) <= near_alf.max_geom_error_m

    def test_stale_understated_bound_cannot_be_believed(self, tmp_path):
        """`_geom_max_err_m` is writer-owned, so a hand-rolled or stale header
        could otherwise report a bound the geometry does not meet."""
        far = -4.488e11 + np.array([[0.0, 0, 0], [0.008, 0, 0]])
        path = tmp_path / "a.alp.gz"
        write_alp(path, far, codec="int16", frame_convention="world-aligned")
        _rewrite_metadata(path, lambda m: m.__setitem__("_geom_max_err_m", 1e-12))
        alf = read_alf(path)
        assert alf.max_geom_error_m >= float(np.max(np.abs(alf.xyz - far)))

    def test_geom_report_matches_what_the_writer_achieves(self, tmp_path):
        """Measuring in CENTERED coordinates omits the float64 rounding that
        reconstruction adds at the frame's true magnitude. This was the third
        place to derive the int16 bound independently — writer, accessor,
        report — and the third to understate it."""
        far = -4.488e11 + np.array([[0.0, 0, 0], [0.004, 0, 0], [0.008, 0, 0]])
        path = tmp_path / "a.alp.gz"
        write_alp(path, far, codec="int16", frame_convention="world-aligned")
        achieved = float(np.max(np.abs(read_alf(path).xyz - far)))
        assert geom_error_report(far)[GEOM_INT16] >= achieved
        # ego-centric frames are unaffected by the offset term
        near = np.array([[0.0, 0, 0], [0.004, 0, 0]])
        assert geom_error_report(near)[GEOM_INT16] < 1e-04

    def test_oversized_recorded_bound_is_refused_during_parse(self, tmp_path):
        """A JSON integer has no width limit, so `10**400` is a valid number
        that float() refuses — and a raw OverflowError aborted callers whose
        quarantine is `except AlfError: skip_frame()`."""
        far = -4.488e11 + np.array([[0.0, 0, 0], [0.008, 0, 0]])
        path = tmp_path / "a.alp.gz"
        write_alp(path, far, codec="int16", frame_convention="world-aligned")
        _rewrite_metadata(path, lambda m: m.__setitem__("_geom_max_err_m", 10**400))
        with pytest.raises(AlfError, match="is not finite"):
            read_alf(path)

    def test_pandas_series_masks_are_refused(self, tmp_path):
        """`_contains_bool` only descended into ndarray/list/tuple, so a Series
        coerced to an object array and published the mask as category 1 and
        intensity 127 — past the invariant these writers advertise."""
        pd = pytest.importorskip("pandas")
        with pytest.raises(ValueError, match="not booleans"):
            write_all(tmp_path / "a.all.gz", pd.Series([True, 2]))
        with pytest.raises(ValueError, match="not booleans"):
            write_ali(tmp_path / "b.ali.gz", pd.Series([True, 2.0]), lo=0.0, hi=2.0)
        # an ordinary numeric Series is still a perfectly good input
        ok = tmp_path / "c.all.gz"
        write_all(ok, pd.Series([1, 2]))
        assert read_alf(ok).categories.tolist() == [1, 2]

    def test_reserved_id_is_refused_through_instance_map_too(self, tmp_path):
        """`instance_map` IS the panoptic half, so a consumer resolving through
        it never touches `instance_ids`. Checking only there let a MAPPED 65535
        in one public door while the other refused it."""
        hb = json.dumps({"_total_points": 1, "instance_map": {"65535": "uuid-x"}}, separators=(",", ":")).encode()
        path = tmp_path / "000000.all"
        path.write_bytes(struct.pack("B", 2) + struct.pack("<I", len(hb)) + hb + bytes([7, 0xFF, 0xFF]))
        alf = read_alf(path)
        # `instance_map` now refuses on the KEY domain (65535 is reserved), which
        # fires before the payload check; `instance_ids` refuses on the payload.
        # Different messages, same verdict — both doors are shut.
        with pytest.raises(AlfError, match="65535"):
            alf.instance_map
        with pytest.raises(AlfError, match="reserved instance id 65535"):
            alf.instance_ids
        assert alf.categories.tolist() == [7]  # the semantic half is still fine

    def test_explicit_instance_map_without_ids_is_refused(self, tmp_path):
        """Dropping it discarded the stable identities the caller asked for and
        reported success, hiding the omitted panoptic stream."""
        with pytest.raises(ValueError, match="instance_map was passed without instance_ids"):
            write_all(
                tmp_path / "a.all.gz",
                np.array([1, 2], dtype=np.uint8),
                instance_map={"1": "u1"},
            )
        # An INHERITED map is pass-through of someone else's header, and a v1
        # file has no ids for it to describe — dropping that one is correct.
        inherited = tmp_path / "b.all.gz"
        write_all(
            inherited,
            np.array([1, 2], dtype=np.uint8),
            metadata={"instance_map": {"1": "u1"}},
        )
        assert "instance_map" not in read_alf(inherited).metadata

    def test_intensity_normalization_spans_the_whole_float_range(self, tmp_path):
        """Halving fixed overflow and BROKE subnormals — `hi / 2` underflows to
        zero and the denominator collapses. Trading an overflow for an underflow
        is not fixing the class, so all three regimes are pinned here."""
        tiny = np.nextafter(0.0, 1.0)
        cases = (
            (np.array([0.0, tiny]), 0.0, tiny, [0, 255]),  # subnormal span
            (np.array([-1e308, 0.0, 1e308]), -1e308, 1e308, [0, 127, 255]),  # overflowing span
            (np.array([0.0, 5.0, 10.0]), 0.0, 10.0, [0, 127, 255]),  # ordinary
        )
        for i, (values, lo, hi, expect) in enumerate(cases):
            path = tmp_path / f"{i}.ali.gz"
            write_ali(path, values, lo=lo, hi=hi)
            assert read_alf(path).intensity.tolist() == expect, f"case {i}"

    def test_constant_frame_at_float64_max_is_encodable(self, tmp_path):
        """`(lo + hi) / 2` overflowed when both ends sat at the same extreme,
        giving an infinite offset that rejected a frame which is EXACTLY
        representable: that coordinate as the offset, an all-zero payload."""
        m = np.finfo(np.float64).max
        xyz = np.array([[m, m, m], [m, m, m]])
        path = tmp_path / "a.alp.gz"
        write_alp(path, xyz, codec="int16", frame_convention="world-aligned", check_budget=False)
        assert np.array_equal(read_alf(path).xyz, xyz)

    def test_format_version_refuses_non_label_channels(self, tmp_path):
        """A mixed-channel scan touching this on a v3 int16 `.alp` got a
        confident "label layout 1" while every other payload accessor already
        refuses a wrong-channel read."""
        path = tmp_path / "a.alp.gz"
        write_alp(path, np.array([[1.0, 2.0, 3.0]]), codec="int16", frame_convention="world-aligned")
        with pytest.raises(AlfError, match="that accessor belongs to"):
            read_alf(path).format_version
        # still answers on the channel it describes
        labels = tmp_path / "b.all.gz"
        write_all(labels, np.array([1, 2], dtype=np.uint8))
        assert read_alf(labels).format_version == 1

    def test_analytic_bound_never_collapses_to_zero(self, tmp_path):
        """The hypothetical `|offset| + 32767*scale` reach overflows for a large
        offset and scale even when the payload decodes finitely — an all-zero
        payload lands exactly ON the offset. `np.spacing(inf)` is NaN and
        `max(0.0, NaN)` returned 0.0, so this reported a ZERO error bound for a
        frame whose quantization step alone was 5e303. Worst possible direction
        for a number used as an accuracy preflight."""
        m = np.finfo(np.float64).max
        hb = json.dumps(
            {
                "_total_points": 1,
                "_geom_codec": "int16",
                "_frame_convention": "world-aligned",
                "_geom_scale": 1e304,
                "_geom_offset": [m, 0.0, 0.0],
                "_geom_max_err_m": 0.0,
            },
            separators=(",", ":"),
        ).encode()
        path = tmp_path / "000000.alp"
        payload = np.array([0, 0, 0], dtype="<i2").tobytes()
        path.write_bytes(struct.pack("B", 3) + struct.pack("<I", len(hb)) + hb + payload)
        alf = read_alf(path)
        assert np.isfinite(alf.xyz).all(), "repro must stay decodable"
        assert alf.max_geom_error_m >= 1e304 / 2.0

    def test_error_bound_is_measured_not_estimated(self, tmp_path):
        """This bound has been wrong in BOTH directions: modelled as scale/2 it
        understated far off-origin frames; charging one ULP at the decoded
        magnitude then overstated them, reporting ~2e292 for a constant frame
        at float64 max that round-trips with exactly zero error. Both break an
        accuracy gate. All four regressions are pinned together here so a fix
        for one cannot silently re-break another."""
        m = np.finfo(np.float64).max
        far = -4.488e11 + np.array([[0.0, 0, 0], [0.004, 0, 0], [0.008, 0, 0]])
        cases = (
            (np.array([[m, m, m], [m, m, m]]), False),
            (far, True),
            (np.array([[0.0, 0, 0], [0.004, 0, 0]]), True),
        )
        for i, (xyz, budget) in enumerate(cases):
            path = tmp_path / f"{i}.alp.gz"
            write_alp(path, xyz, codec="int16", frame_convention="world-aligned", check_budget=budget)
            alf = read_alf(path)
            achieved = float(np.max(np.abs(alf.xyz - xyz)))
            assert alf.max_geom_error_m >= achieved, f"case {i} understates"
            assert alf.max_geom_error_m < 1.0, f"case {i} overstates ({alf.max_geom_error_m})"

    def test_budget_gates_the_measured_error(self, tmp_path):
        """`scale / 2` can sit under the budget while float64 rounding at a
        large offset pushes the real error to it: a frame around 2**50 spanning
        ~30 km picks an ideal 0.228 m and reconstructs at 0.25 m, so the branch
        promising to reject the threshold wrote it anyway."""
        xyz = 2.0**50 + np.linspace(0, 30000, 64).reshape(-1, 1) * np.ones((1, 3))
        with pytest.raises(ValueError, match="at or past the"):
            write_alp(tmp_path / "a.alp.gz", xyz, codec="int16", frame_convention="world-aligned")
        # the same frame is writable when the caller opts out of the budget
        write_alp(tmp_path / "b.alp.gz", xyz, codec="int16", frame_convention="world-aligned", check_budget=False)

    def test_out_of_range_intensity_samples_are_refused(self, tmp_path):
        """`lo`/`hi` are documented as THE raw range, so a sample outside them
        is a range-calculation bug. Clipping turned that into a valid-looking
        channel with its contrast destroyed."""
        with pytest.raises(ValueError, match="outside the declared raw range"):
            write_ali(tmp_path / "a.ali.gz", np.array([0.0, 100.0]), lo=0.0, hi=1.0)
        ok = tmp_path / "b.ali.gz"
        write_ali(ok, np.array([0.0, 0.5, 1.0]), lo=0.0, hi=1.0)
        assert read_alf(ok).intensity.tolist() == [0, 127, 255]

    def test_instance_map_keys_must_be_in_the_registry_domain(self, tmp_path):
        """Spelling was checked; the DOMAIN was not. The payload validator
        reserves 65535, caps ids at 65534 and treats 0 as unset, so a map
        carrying those describes identities no frame can emit."""
        for bad in ({"1": "u1", "65535": "x"}, {"1": "u1", "0": "x"}, {"1": "u1", "99999": "x"}):
            with pytest.raises(ValueError, match="outside the valid instance id range"):
                write_all(
                    tmp_path / "a.all.gz",
                    np.array([1], dtype=np.uint8),
                    instance_ids=np.array([1], dtype=np.uint16),
                    instance_map=bad,
                )
        # An in-domain entry for an id THIS frame does not emit is the whole
        # point of a sequence-level map, so it must still write.
        ok = tmp_path / "b.all.gz"
        write_all(
            ok,
            np.array([1], dtype=np.uint8),
            instance_ids=np.array([1], dtype=np.uint16),
            instance_map={"1": "u1", "65534": "u2"},
        )
        assert read_alf(ok).metadata["instance_map"] == {"1": "u1", "65534": "u2"}

    def test_alx_counts_accept_numpy_integers(self, tmp_path):
        """These come from `np.sum(mask)` in every pipeline that would write a
        sidecar, and the int/float whitelist called `np.int64(2)` "not a
        number" — refusing a valid count for its container type."""
        path = tmp_path / "a.alx.gz"
        write_alx(
            path,
            np.array([0, 1], dtype=np.uint32),
            alp_total_points=np.int64(2),
            source_total_points=np.int64(5),
        )
        assert read_alf(path).kept_indices.tolist() == [0, 1]
        # booleans stay out: True IS an int, and a leaked mask would read as 1
        with pytest.raises(ValueError, match="must be a number"):
            write_alx(
                tmp_path / "b.alx.gz",
                np.array([0], dtype=np.uint32),
                alp_total_points=np.bool_(True),
                source_total_points=1,
            )

    def test_error_bound_memory_does_not_scale_with_frame_size(self, tmp_path):
        """The naive TwoSum allocated ~7 full-size float64 arrays, measured at
        ~195 B/point — multiple GB for a 10M-point frame, turning the
        documented accuracy preflight into an OOM on large geometry."""
        import tracemalloc

        from alf._io import _reconstruction_rounding

        peaks = []
        for n in (50_000, 200_000):
            xyz = -4.488e11 + np.random.default_rng(0).uniform(-50, 50, (n, 3))
            path = tmp_path / f"{n}.alp.gz"
            write_alp(path, xyz, codec="int16", frame_convention="world-aligned", check_budget=False)
            alf = read_alf(path)
            achieved = float(np.max(np.abs(alf.xyz - xyz)))
            tracemalloc.start()
            got = _reconstruction_rounding(alf._payload, alf.metadata["_geom_scale"], alf.metadata["_geom_offset"])
            peaks.append(tracemalloc.get_traced_memory()[1])
            tracemalloc.stop()
            # chunking must not change the answer: it is a max over blocks
            assert alf.max_geom_error_m >= achieved
            assert got > 0.0
        assert peaks[1] < peaks[0] * 2, f"memory scaled with frame size: {peaks}"

    def test_spacing_overflow_is_not_coordinate_overflow(self, tmp_path):
        """`np.spacing` steps toward +inf, so at float64 max it returns Inf for
        a perfectly ordinary finite number. Using it to BOUND a rounding error
        made this refuse geometry that decodes fine: scale = float64 max with a
        stored component of 1 multiplies exactly and `.xyz` succeeds.

        This trap was already fixed once for the reach term; missing the second
        use is why both now share `_ulp_below`."""
        m = np.finfo(np.float64).max
        hb = json.dumps(
            {
                "_total_points": 1,
                "_geom_codec": "int16",
                "_frame_convention": "world-aligned",
                "_geom_scale": m,
                "_geom_offset": [0.0, 0.0, 0.0],
            },
            separators=(",", ":"),
        ).encode()
        path = tmp_path / "000000.alp"
        head = struct.pack("B", 3) + struct.pack("<I", len(hb)) + hb
        path.write_bytes(head + np.array([1, 0, 0], dtype="<i2").tobytes())
        alf = read_alf(path)
        assert np.isfinite(alf.xyz).all(), "repro must stay decodable"
        assert math.isfinite(alf.max_geom_error_m)

        # A frame that genuinely cannot be represented must still be refused.
        beyond = tmp_path / "000001.alp"
        beyond.write_bytes(head + np.array([2, 0, 0], dtype="<i2").tobytes())
        with pytest.raises(AlfError):
            read_alf(beyond).xyz

    def test_reader_enforces_the_instance_map_key_domain(self, tmp_path):
        """The writer got this check first; an externally produced file whose
        payload happens to emit only valid ids still handed back sequence-level
        identities of "0", "-1", "65535" or "99999" — reserved or impossible,
        accepted purely because that frame did not emit them."""
        for key in ("0", "-1", "65535", "99999"):
            hb = json.dumps(
                {"_total_points": 1, "instance_map": {"1": "u1", key: "x"}},
                separators=(",", ":"),
            ).encode()
            path = tmp_path / f"{abs(int(key))}.all"
            path.write_bytes(struct.pack("B", 2) + struct.pack("<I", len(hb)) + hb + bytes([7, 0x01, 0x00]))
            with pytest.raises(AlfError):
                read_alf(path).instance_map
        # An in-domain entry for an id this frame does not emit still reads —
        # that is what a sequence-level map is for.
        hb = json.dumps(
            {"_total_points": 1, "instance_map": {"1": "u1", "65534": "u2"}},
            separators=(",", ":"),
        ).encode()
        ok = tmp_path / "ok.all"
        ok.write_bytes(struct.pack("B", 2) + struct.pack("<I", len(hb)) + hb + bytes([7, 0x01, 0x00]))
        assert read_alf(ok).instance_map == {"1": "u1", "65534": "u2"}

    def test_non_real_array_anchors_are_refused(self, tmp_path):
        """An ndarray never reached the list/tuple element check, so a COMPLEX
        anchor cast straight to float64: NumPy drops the imaginary part behind
        a ComplexWarning and `[1+9j, 2, 3]` becomes a confident (1, 2, 3). An
        anchor places the whole cloud, so that misframes silently.

        Every PAYLOAD path already refused complex via `_reject_non_real`;
        anchors were the one door left open."""
        base = np.array([[1.0, 2.0, 3.0]])
        for bad in (
            np.array([1 + 9j, 2, 3]),
            np.array(["1", "2", "3"]),
            np.array([1, 2, 3], dtype=object),
        ):
            with pytest.raises((AlfError, ValueError)):
                write_alp(tmp_path / "bad.alp.gz", base, device_position=bad)

        # `world_xyz(origin=...)` shares the validator and so shares the fix.
        path = tmp_path / "a.alp.gz"
        write_alp(path, base)
        with pytest.raises(AlfError):
            read_alf(path).world_xyz(origin=np.array([1 + 9j, 2, 3]))

        # Real numeric arrays and plain tuples are unaffected.
        for anchor_value in (np.array([1.0, 2.0, 3.0]), (1, 2, 3)):
            ok = tmp_path / "ok.alp.gz"
            write_alp(ok, base, device_position=anchor_value)
            assert read_alf(ok).device_position == pytest.approx([1.0, 2.0, 3.0])

    def test_ulp_below_is_sign_symmetric(self, tmp_path):
        """Stepping toward -inf is finite at +max and OVERFLOWS at -max, so the
        previous spacing fix covered half the number line: a frame decoding to
        exactly -float64_max was refused by its own accuracy preflight."""
        m = np.finfo(np.float64).max
        for stored, label in ((1, "positive"), (-1, "negative")):
            hb = json.dumps(
                {
                    "_total_points": 1,
                    "_geom_codec": "int16",
                    "_frame_convention": "world-aligned",
                    "_geom_scale": m,
                    "_geom_offset": [0.0, 0.0, 0.0],
                },
                separators=(",", ":"),
            ).encode()
            path = tmp_path / f"{label}.alp"
            payload = np.array([stored, 0, 0], dtype="<i2").tobytes()
            path.write_bytes(struct.pack("B", 3) + struct.pack("<I", len(hb)) + hb + payload)
            alf = read_alf(path)
            assert np.isfinite(alf.xyz).all(), f"{label} must stay decodable"
            assert math.isfinite(alf.max_geom_error_m), f"{label} bound not finite"

    def test_format_version_verified_refuses_non_label_channels(self, tmp_path):
        """It returned True — "layout verified" — for a payload with no label
        layout at all, bypassing the guard on `format_version` itself."""
        path = tmp_path / "a.alp.gz"
        write_alp(path, np.array([[1.0, 2.0, 3.0]]))
        with pytest.raises(AlfError, match="that accessor belongs to"):
            read_alf(path).format_version_verified
        labels = tmp_path / "b.all.gz"
        write_all(labels, np.array([1, 2], dtype=np.uint8))
        assert read_alf(labels).format_version_verified is True

    def test_integer_intensities_beyond_float64_exactness_are_refused(self, tmp_path):
        """Past 2**53 float64 cannot represent consecutive integers, so the
        cast collapsed distinct raw values: [2**53, 2**53+1] wrote [0, 0]."""
        with pytest.raises(ValueError, match=r"2\*\*53|consecutive integers"):
            write_ali(
                tmp_path / "a.ali.gz",
                np.array([2**53, 2**53 + 1], dtype=np.int64),
                lo=2**53,
                hi=2**53 + 1,
            )
        # A huge FLOAT range makes no consecutive-integer promise and stays legal.
        ok = tmp_path / "b.ali.gz"
        write_ali(ok, np.array([-1e308, 0.0, 1e308]), lo=-1e308, hi=1e308)
        assert read_alf(ok).intensity.tolist() == [0, 127, 255]

    def test_non_string_instance_map_keys_are_refused(self, tmp_path):
        """JSON members are strings, so `{1: "a", "1": "b"}` serializes both to
        "1" and json.loads keeps only the last — silently dropping an identity."""
        with pytest.raises(ValueError, match="non-string key"):
            write_all(
                tmp_path / "a.all.gz",
                np.array([1], dtype=np.uint8),
                instance_ids=np.array([1], dtype=np.uint16),
                instance_map={1: "uuid-a", "1": "uuid-b"},
            )

    def test_rejected_write_does_not_delete_a_concurrent_result(self, tmp_path):
        """Unlinking at entry satisfied "a rejected re-run leaves nothing stale"
        by also destroying whatever a parallel writer had just published, and
        emptying the path for the whole compression.

        SCOPE, stated precisely because this test was read as more than it is:
        the concurrent publish here happens BEFORE the fingerprint comparison,
        so what it proves is that the comparison works. It does NOT cover the
        window between `_clear_stale_output` and `unlink` — a writer publishing
        in that gap still loses its output.

        That window is real, acknowledged, and deliberately not closed: doing so
        needs same-path write serialization, which is a locking decision for the
        maintainers rather than a patch. Do not read a pass here as evidence the
        race is gone."""
        import alf._io as alf_io

        # Property 1: a rejected re-run still clears the caller's OWN stale file.
        path = tmp_path / "a.alp.gz"
        write_alp(path, np.array([[1.0, 2.0, 3.0]]))
        # An int16 write of an invalid return: still refused, now for the
        # specific reason that fixed-point cannot encode one. (Plain fp16 NaN is
        # valid content and no longer rejects anything, so it cannot drive this.)
        with pytest.raises(ValueError):
            write_alp(path, np.array([[np.nan, 0.0, 0.0]]), codec=GEOM_INT16, frame_convention=FRAME_WORLD_ALIGNED)
        assert not path.exists()

        # Property 2: a file published by someone ELSE mid-flight survives.
        other = tmp_path / "b.alp.gz"
        write_alp(other, np.array([[1.0, 2.0, 3.0]]))
        original = alf_io._as_xyz

        def publish_midflight(value, dest):
            result = original(value, dest)
            if np.isnan(result).any():
                write_alp(other, np.array([[9.0, 9.0, 9.0]]))
            return result

        alf_io._as_xyz = publish_midflight
        try:
            with pytest.raises(ValueError):
                write_alp(other, np.array([[np.nan, 0.0, 0.0]]), codec=GEOM_INT16, frame_convention=FRAME_WORLD_ALIGNED)
        finally:
            alf_io._as_xyz = original
        assert other.exists(), "a concurrent writer's valid output was deleted"
        assert read_alf(other).xyz.tolist() == [[9.0, 9.0, 9.0]]

    def test_destination_suffix_must_match_the_writer(self, tmp_path):
        """`read_alf` infers the channel from the filename, so intensity bytes
        written to a `.all.gz` path read back as CATEGORY IDS. Both are v1,
        one byte per point, same `_total_points` — every integrity check in
        this module passes and the mix-up surfaces as plausible labels."""
        with pytest.raises(ValueError, match="destination names a"):
            write_ali(tmp_path / "a.all.gz", np.array([1.0, 2.0]), lo=0.0, hi=2.0)
        with pytest.raises(ValueError, match="destination names a"):
            write_all(tmp_path / "b.ali.gz", np.array([1, 2], dtype=np.uint8))
        # Correct suffixes, and UNRECOGNISED ones (cache files named by blob id
        # are a real pattern here), are both left alone.
        write_ali(tmp_path / "c.ali.gz", np.array([1.0, 2.0]), lo=0.0, hi=2.0)
        write_ali(tmp_path / "blobid123", np.array([1.0, 2.0]), lo=0.0, hi=2.0)

    def test_any_array_like_anchor_is_validated(self, tmp_path):
        """Testing `isinstance(value, np.ndarray)` covered NumPy only, so a
        pandas Series walked into the float64 cast — dropping an imaginary
        part or parsing a numeric string. Coerce once, one dtype rule."""
        pd = pytest.importorskip("pandas")
        base = np.array([[1.0, 2.0, 3.0]])
        for bad in (pd.Series([1 + 9j, 2, 3]), pd.Series(["1", "2", "3"]), ["1", "2", "3"]):
            with pytest.raises((AlfError, ValueError)):
                write_alp(tmp_path / "bad.alp.gz", base, device_position=bad)
        ok = tmp_path / "ok.alp.gz"
        write_alp(ok, base, device_position=pd.Series([1.0, 2.0, 3.0]))
        assert read_alf(ok).device_position == pytest.approx([1.0, 2.0, 3.0])

    def test_masked_geometry_is_refused(self, tmp_path):
        """`np.asarray` DISCARDS the mask, so a row the caller marked invalid is
        written as an ordinary point — counted in `_total_points`, folded into
        the box, shifting every index-correlated stream after it."""
        masked = np.ma.array([[1.0, 2.0, 3.0], [100.0, 200.0, 300.0]], mask=[[0, 0, 0], [1, 1, 1]])
        with pytest.raises(ValueError, match="masked array"):
            write_alp(tmp_path / "a.alp.gz", masked)
        # A MaskedArray with nothing actually masked is ordinary data.
        ok = tmp_path / "b.alp.gz"
        write_alp(ok, np.ma.array([[1.0, 2.0, 3.0]]))
        assert read_alf(ok).xyz.tolist() == [[1.0, 2.0, 3.0]]

    def test_integer_geometry_beyond_float64_exactness_is_refused(self, tmp_path):
        """Past 2**53 the cast merges consecutive integers: int64 points at
        2**53 and 2**53+1 became identical, and the writer published a
        zero-span box and a 5e-05 bound having discarded a one-metre
        difference the caller supplied."""
        with pytest.raises(ValueError, match=r"2\*\*53|consecutive integers"):
            write_alp(
                tmp_path / "a.alp.gz",
                np.array([[2**53, 0, 0], [2**53 + 1, 0, 0]], dtype=np.int64),
                codec="int16",
                frame_convention="world-aligned",
                check_budget=False,
            )
        ok = tmp_path / "b.alp.gz"
        write_alp(ok, np.array([[1, 2, 3]], dtype=np.int64))
        assert read_alf(ok).xyz.tolist() == [[1.0, 2.0, 3.0]]

    def test_oversized_count_is_bounded_before_it_is_formatted(self, tmp_path):
        """A JSON integer has no width limit, so a 4299-digit `_total_points`
        must be refused before a payload calculation or error formatter turns
        it into a bare OverflowError/ValueError outside the ALF quarantine."""
        hb = ('{"_total_points":' + "1" + "0" * 4298 + "}").encode()
        path = tmp_path / "000000.alp"
        path.write_bytes(struct.pack("B", 1) + struct.pack("<I", len(hb)) + hb + bytes(6))
        with pytest.raises(AlfError, match="too large for binary64"):
            read_alf(path).xyz

    def test_masked_and_string_per_point_streams_are_refused(self, tmp_path):
        """These two rules went to geometry and to the anchors, and each time
        the non-geometry channels kept the hole. `np.asarray` drops a mask and
        publishes rows the caller marked invalid; an unparsed CSV column casts
        to real-looking measurements. Fixed once, in the shared coercion."""
        masked = np.ma.array([7, 8], mask=[False, True])
        with pytest.raises(ValueError, match="masked array"):
            write_all(tmp_path / "a.all.gz", masked)
        with pytest.raises(ValueError, match="masked array"):
            write_ali(tmp_path / "b.ali.gz", np.ma.array([1.0, 2.0], mask=[0, 1]), lo=0.0, hi=2.0)
        with pytest.raises(ValueError, match="masked array"):
            write_alx(
                tmp_path / "c.alx.gz",
                np.ma.array([0, 1], mask=[0, 1]),
                alp_total_points=2,
                source_total_points=5,
            )
        with pytest.raises(ValueError, match="holds strings"):
            write_ali(tmp_path / "d.ali.gz", np.array(["0", "10"]), lo=0, hi=10)
        with pytest.raises(ValueError, match="holds strings"):
            write_all(tmp_path / "e.all.gz", np.array(["1", "2"]))
        ok = tmp_path / "ok.all.gz"
        write_all(ok, np.array([1, 2], dtype=np.uint8))
        assert read_alf(ok).categories.tolist() == [1, 2]

    def test_declared_header_size_is_bounded(self, tmp_path):
        """The field is a uint32, so a corrupt file can declare 4 GB of header
        — which would be UTF-8-decoded and JSON-parsed on top of the already
        materialized file, OOMing the worker BEFORE the AlfError quarantine
        that exists to skip the frame could run."""
        path = tmp_path / "000000.alp"
        path.write_bytes(struct.pack("B", 1) + struct.pack("<I", 4_000_000_000) + b"{}")
        with pytest.raises(AlfError, match=r"past the .* limit"):
            read_alf(path)

    def test_int16_writer_chunks_without_changing_its_output(self, tmp_path):
        """Holding xyz, centered, the rint/clip temporaries, decoded and
        decoded-xyz simultaneously multiplies memory use. Quantization is
        elementwise and the error is a maximum, so
        chunking must not move a single byte."""
        n = 600_000
        xyz = np.random.default_rng(7).uniform(-30, 30, (n, 3))
        path = tmp_path / "a.alp.gz"
        write_alp(path, xyz, codec="int16", frame_convention="world-aligned")
        alf = read_alf(path)

        offset = np.array(alf.metadata["_geom_offset"])
        scale = alf.metadata["_geom_scale"]
        reference = np.clip(np.rint((xyz - offset) / scale), -32767, 32767).astype("<i2")
        raw = gzip.decompress(path.read_bytes())
        head = struct.unpack("<I", raw[1:5])[0]
        assert raw[5 + head :] == reference.tobytes(), "chunking changed the payload"

        decoded = reference.astype(np.float64) * scale + offset
        lo, hi = decoded.min(axis=0), decoded.max(axis=0)
        assert alf.metadata["box"] == f"{lo[0]}:{lo[1]}:{lo[2]}:{hi[0]}:{hi[1]}:{hi[2]}"
        assert alf.max_geom_error_m >= float(np.max(np.abs(alf.xyz - xyz)))

        # And the SCALE, which is derived before any of that: materializing
        # `centered` for `choose_geom_scale` was two more full-size float64
        # arrays, so the extent is accumulated blockwise. Same value either way.
        from alf._io import choose_geom_scale

        assert scale == choose_geom_scale(xyz - offset)

    def test_int16_write_memory_does_not_scale_with_frame_size(self, tmp_path):
        """`centered` plus `choose_geom_scale`'s own `np.abs` temporary were two
        full-size float64 arrays on top of the caller's input and the `_as_xyz`
        copy — ~1 GB at the documented 10M-point scale, before quantization
        started. A per-point RATIO, since the payload itself must grow."""
        import tracemalloc

        n = 600_000
        input_bytes = n * 3 * 8
        xyz = np.random.default_rng(0).uniform(-50, 50, (n, 3))
        tracemalloc.start()
        write_alp(tmp_path / "a.alp.gz", xyz, codec=GEOM_INT16, frame_convention=FRAME_WORLD_ALIGNED)
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        # Bounded against the INPUT, not an absolute figure. Measured: ~2.7x
        # with the extent accumulated blockwise, ~3.7x when `centered` and
        # `choose_geom_scale`'s `np.abs` temporary are both materialized. A
        # monotonic per-point ratio does NOT separate them — both fall as the
        # frame grows, so that form of the assertion passed either way.
        assert peak < input_bytes * 3.2, f"held {peak / input_bytes:.1f}x the input"

    def test_object_dtype_does_not_bypass_the_2_53_guards(self, tmp_path):
        """`dtype.kind` is `O` for a pandas nullable column, so a kind-based
        test declared these non-integral and waved them past both new guards —
        the same 2**53 collapse, one container across."""
        with pytest.raises(ValueError):
            write_alp(
                tmp_path / "a.alp.gz",
                np.array([[2**53, 0, 0], [2**53 + 1, 0, 0]], dtype=object),
                codec="int16",
                frame_convention="world-aligned",
                check_budget=False,
            )
        with pytest.raises(ValueError):
            write_ali(
                tmp_path / "b.ali.gz",
                np.array([2**53, 2**53 + 1], dtype=object),
                lo=2**53,
                hi=2**53 + 1,
            )
        # Small object arrays are how ragged input legitimately arrives.
        ok = tmp_path / "c.alp.gz"
        write_alp(ok, np.array([[1, 2, 3]], dtype=object))
        assert read_alf(ok).xyz.tolist() == [[1.0, 2.0, 3.0]]

    def test_geom_codec_refuses_non_geometry_channels(self, tmp_path):
        """The geometry-side counterpart of the guarded label properties: this
        returned a plausible "fp16" for an `.ali`/`.all`/`.alc`/`.alx`, letting
        a caller route a non-geometry file through codec-specific logic."""
        intensity = tmp_path / "a.ali.gz"
        write_ali(intensity, np.array([1.0, 2.0]), lo=0.0, hi=2.0)
        with pytest.raises(AlfError, match="that accessor belongs to"):
            read_alf(intensity).geom_codec
        geometry = tmp_path / "b.alp.gz"
        write_alp(geometry, np.array([[1.0, 2.0, 3.0]]))
        assert read_alf(geometry).geom_codec == GEOM_FP16

    def test_writer_refuses_a_header_its_reader_rejects(self, tmp_path):
        """I added the 10 MB reader limit last round and left the writer able
        to emit past it — the file gzipped down small and unremarkable, then
        failed to load."""
        with pytest.raises(ValueError, match=r"past the .* limit"):
            write_ali(
                tmp_path / "a.ali.gz",
                np.array([1.0, 2.0]),
                lo=0.0,
                hi=2.0,
                metadata={"blob": "x" * 10_000_000},
            )

    def test_object_arrays_do_not_bypass_the_string_guard(self, tmp_path):
        """`dtype.kind` is `O` for a pandas-derived or mixed array, so a numeric
        string reached the float64 cast. `_as_xyz` already combined the native
        and object tests; `_as_per_point` kept only the first half — the third
        time this pair diverged, so they share one predicate now."""
        with pytest.raises(ValueError, match="strings"):
            write_ali(tmp_path / "a.ali.gz", np.array(["0", 10], dtype=object), lo=0, hi=10)
        with pytest.raises(ValueError, match="strings"):
            write_all(tmp_path / "b.all.gz", np.array(["1", "2"], dtype=object))
        with pytest.raises(ValueError, match="strings"):
            write_alp(tmp_path / "c.alp.gz", np.array([["1", "2", "3"]], dtype=object))
        # A numeric object array is how ragged input legitimately arrives.
        ok = tmp_path / "d.ali.gz"
        write_ali(ok, np.array([0, 10.0], dtype=object), lo=0.0, hi=10.0)
        assert read_alf(ok).intensity.tolist() == [0, 255]

    def test_geom_report_is_chunked_without_changing_its_answer(self):
        """The writer and the reader's preflight already block this arithmetic;
        the report was the third full-frame pass over it. Both results are
        maxima, so blocking must move the peak and nothing else."""
        import tracemalloc

        rng = np.random.default_rng(5)
        xyz = rng.uniform(-50, 50, (700_000, 3))
        report = geom_error_report(xyz)

        # Recomputed the unchunked way, which is what the blocks must equal.
        from alf._io import choose_geom_offset

        offset = choose_geom_offset(xyz)
        scale = choose_geom_scale(xyz - offset)
        quantized = np.clip(np.rint((xyz - offset) / scale), -32767, 32767)
        assert report[GEOM_INT16] == pytest.approx(float(np.abs(quantized * scale + offset - xyz).max()))
        assert report[GEOM_FP16] == pytest.approx(float(np.abs(xyz.astype(np.float16).astype(np.float64) - xyz).max()))

        peaks = []
        for n in (200_000, 800_000):
            sample = rng.uniform(-50, 50, (n, 3))
            tracemalloc.start()
            geom_error_report(sample)
            peaks.append(tracemalloc.get_traced_memory()[1] / n)
            tracemalloc.stop()
        # Per-point cost must not grow with frame size.
        assert peaks[1] <= peaks[0], f"per-point memory grew: {peaks}"

    def test_anchors_reject_inexact_integers_in_every_shape(self, tmp_path):
        """An anchor PLACES the cloud, so an x of 2**53+1 silently becoming
        2**53 shifts every point in the frame by a metre. Geometry and
        intensity already had this rule; anchors are its third home.

        All four accepted shapes are pinned because the first version of this
        guard went in the sequence branch alone and left the `{x, y, z}` dict
        path open — the same one-of-two miss the rule exists to fix."""
        base = np.array([[1.0, 2.0, 3.0]])
        shapes = {
            "dict": lambda v: {"x": v, "y": 0, "z": 0},
            "tuple": lambda v: (v, 0, 0),
            "list": lambda v: [v, 0, 0],
            "array": lambda v: np.array([v, 0, 0]),
        }
        # Every SHAPE against every lossy KIND. The wide-float half of this rule
        # went into the sequence branch alone one round after the integer half
        # made the identical miss in the same function, so the matrix is the
        # assertion — not whichever cell a finding happened to name.
        kinds = [2**53 + 1]
        if np.finfo(np.longdouble).bits > 64:
            kinds.append(np.longdouble(2**63) + 1)
        frame = tmp_path / "frame.alp.gz"
        write_alp(frame, base)
        for kind in kinds:
            for _name, make in shapes.items():
                with pytest.raises((AlfError, ValueError)):
                    write_alp(tmp_path / "bad.alp.gz", base, device_position=make(kind))
                # `world_xyz(origin=...)` shares the validator, so it shares
                # every cell of the matrix too.
                with pytest.raises(AlfError):
                    read_alf(frame).world_xyz(origin=make(kind))

        # Ordinary anchors in every shape still work.
        for good in ((1, 2, 3), {"x": 1.0, "y": 2.0, "z": 3.0}, np.array([1.0, 2.0, 3.0])):
            ok = tmp_path / "ok.alp.gz"
            write_alp(ok, base, device_position=good)
            assert read_alf(ok).device_position == pytest.approx([1.0, 2.0, 3.0])

    def test_no_stale_byte_identical_claims_remain(self):
        """The promise was corrected in `README.md` and left standing in
        the module docstring and two inline comments. Three separate rounds
        found the same false claim in a different file, so this pins it."""
        from pathlib import Path as _Path

        module = _Path(alf_io_module.__file__).read_text()
        # The fingerprint comment legitimately says "byte-identically"; the
        # COMPATIBILITY claim is what must not come back.
        offenders = [
            line for line in module.splitlines() if "byte-identical" in line.lower() and "byte-identically" not in line
        ]
        assert offenders == [], f"stale byte-identical claim: {offenders}"

    def test_codec_parameters_reject_inexact_integers(self, tmp_path):
        """The codec parameters DEFINE the coordinate system, so an offset of
        2**53 + 1 converting to 2**53 decodes every point onto a different grid
        than the header states — and nothing downstream can tell, because the
        file is self-consistent about the wrong number.

        Fourth home for this rule after geometry, intensity and the anchors."""
        head = {"_total_points": 1, "_geom_codec": "int16", "_frame_convention": "world-aligned"}
        cases = (
            ("offset", {**head, "_geom_scale": 1, "_geom_offset": [2**53 + 1, 0, 0]}),
            ("scale", {**head, "_geom_scale": 2**53 + 1, "_geom_offset": [0, 0, 0]}),
        )
        for name, metadata in cases:
            hb = json.dumps(metadata, separators=(",", ":")).encode()
            path = tmp_path / f"{name}.alp"
            payload = np.array([1, 0, 0], dtype="<i2").tobytes()
            path.write_bytes(struct.pack("B", 3) + struct.pack("<I", len(hb)) + hb + payload)
            with pytest.raises(AlfError, match="cannot represent exactly"):
                read_alf(path).xyz

        # An ordinary v3 frame is untouched.
        ok = tmp_path / "ok.alp.gz"
        write_alp(ok, np.array([[1.0, 2.0, 3.0]]), codec=GEOM_INT16, frame_convention=FRAME_WORLD_ALIGNED)
        assert read_alf(ok).xyz[0].tolist() == pytest.approx([1.0, 2.0, 3.0], abs=1e-4)

    def test_codec_parameters_accept_exact_large_integers(self, tmp_path):
        """Magnitude is not loss: 2**53 and representable even integers above
        it survive the binary64 conversion and must not be rejected merely for
        crossing the consecutive-integer boundary."""
        header = {
            "_total_points": 1,
            "_geom_codec": "int16",
            "_frame_convention": "world-aligned",
            "_geom_scale": 2**53 + 2,
            "_geom_offset": [2**64, 0, 0],
        }
        encoded = json.dumps(header, separators=(",", ":")).encode()
        path = tmp_path / "exact.alp"
        path.write_bytes(
            struct.pack("B", 3) + struct.pack("<I", len(encoded)) + encoded + np.array([0, 0, 0], dtype="<i2").tobytes()
        )

        assert read_alf(path).xyz[0].tolist() == [float(2**64), 0.0, 0.0]

    def test_empty_payload_cannot_corroborate_a_version_mismatch(self, tmp_path):
        """An empty payload fits BOTH a 1-byte and a 3-byte stride, so it
        settles nothing. The reader must refuse the contradictory envelope
        before any accessor can treat either assertion as authoritative."""
        conflicting = json.dumps({"_total_points": 0, "_format_version": 2}, separators=(",", ":")).encode()
        path = tmp_path / "a.all"
        path.write_bytes(struct.pack("B", 1) + struct.pack("<I", len(conflicting)) + conflicting)
        with pytest.raises(AlfError, match="fits BOTH"):
            read_alf(path)

        # An empty file whose signals AGREE is still authoritative.
        agreeing = json.dumps({"_total_points": 0}, separators=(",", ":")).encode()
        ok = tmp_path / "b.all"
        ok.write_bytes(struct.pack("B", 1) + struct.pack("<I", len(agreeing)) + agreeing)
        assert read_alf(ok).format_version_verified is True

    def test_duplicate_header_keys_are_refused_at_any_depth(self, tmp_path):
        header = b'{"_total_points":1,"custom":{"sensor":"left","sensor":"right"}}'
        path = tmp_path / "duplicate.ali"
        path.write_bytes(struct.pack("B", 1) + struct.pack("<I", len(header)) + header + b"\x07")

        with pytest.raises(AlfError, match="duplicate JSON object key 'sensor'"):
            read_alf(path)

    def test_decompression_bomb_is_bounded(self, tmp_path, monkeypatch):
        """`gzip.decompress` materializes the WHOLE output before any header,
        count or payload check can look at it. These objects are
        untrusted and 199 KB of zeros expands to 200 MB at ~1000:1, so
        a hostile file OOMed the worker instead of reaching the AlfError
        quarantine this module promises."""
        monkeypatch.setattr(alf_io_module, "MAX_DECOMPRESSED_BYTES", 16 * 1024 * 1024)
        raw = struct.pack("B", 1) + struct.pack("<I", 2) + b"{}" + b"\0" * (64 * 1024 * 1024)
        bomb = tmp_path / "bomb.alp.gz"
        bomb.write_bytes(gzip.compress(raw))
        assert bomb.stat().st_size < 100_000, "repro must stay small on disk"
        with pytest.raises(AlfError, match="expands past"):
            read_alf(bomb)

        # An ordinary gzipped frame is unaffected.
        ok = tmp_path / "ok.alp.gz"
        write_alp(ok, np.random.default_rng(0).uniform(-10, 10, (5_000, 3)))
        assert read_alf(ok).xyz.shape == (5_000, 3)

    def test_rounding_bound_uses_the_larger_adjacent_spacing(self, tmp_path):
        """At a power of two the spacing ABOVE is twice the spacing below, so
        charging the lower one understates a value that rounded UP onto the
        boundary: 3 * 0.33333333333333337 is exactly 1.0 in float64 while the
        true product is 2**-53 above it. A bound that understates is the one
        failure this helper exists to prevent."""
        hb = json.dumps(
            {
                "_total_points": 1,
                "_geom_codec": "int16",
                "_frame_convention": "world-aligned",
                "_geom_scale": 0.33333333333333337,
                "_geom_offset": [0.0, 0.0, 0.0],
            },
            separators=(",", ":"),
        ).encode()
        path = tmp_path / "000000.alp"
        payload = np.array([3, 0, 0], dtype="<i2").tobytes()
        path.write_bytes(struct.pack("B", 3) + struct.pack("<I", len(hb)) + hb + payload)
        alf = read_alf(path)
        # The multiplication term must cover the real 2**-53 error, not half it.
        assert alf.max_geom_error_m >= 0.33333333333333337 / 2.0 + 2.0**-53

    def test_subnormal_scale_cannot_report_an_exact_bound(self, tmp_path):
        """The smallest positive float64 scale underflows both terms to zero,
        so a grid that demonstrably loses information reported an EXACT bound
        and an accuracy gate certified it."""
        hb = json.dumps(
            {
                "_total_points": 1,
                "_geom_codec": "int16",
                "_frame_convention": "world-aligned",
                "_geom_scale": 5e-324,
                "_geom_offset": [0.0, 0.0, 0.0],
            },
            separators=(",", ":"),
        ).encode()
        path = tmp_path / "000000.alp"
        payload = np.array([1, 0, 0], dtype="<i2").tobytes()
        path.write_bytes(struct.pack("B", 3) + struct.pack("<I", len(hb)) + hb + payload)
        assert read_alf(path).max_geom_error_m > 0.0

    def test_empty_int16_frame_records_no_error(self, tmp_path):
        """fp16 and `geom_error_report` both say 0.0 for an empty frame; int16
        recorded the scale floor, so a strict accuracy filter rejected empty
        frames for a loss that cannot have occurred."""
        path = tmp_path / "a.alp.gz"
        write_alp(path, np.zeros((0, 3)), codec=GEOM_INT16, frame_convention=FRAME_WORLD_ALIGNED)
        alf = read_alf(path)
        assert alf.metadata.get("_geom_max_err_m") == 0.0
        assert alf.max_geom_error_m == 0.0

    def test_fp16_bounds_pass_is_chunked_without_changing_the_box(self, tmp_path):
        """The DEFAULT codec path materialized a full (N, 3) float64 array
        solely to reduce its own extrema — ~240 MB at the documented
        10-million-point scale, on top of the input, the copy, the encoded
        array and the payload."""
        n = 400_000
        xyz = np.random.default_rng(11).uniform(-30, 30, (n, 3))
        path = tmp_path / "a.alp.gz"
        write_alp(path, xyz)

        reference = xyz.astype(np.float16).astype(np.float64)
        lo, hi = reference.min(axis=0), reference.max(axis=0)
        assert read_alf(path).metadata["box"] == f"{lo[0]}:{lo[1]}:{lo[2]}:{hi[0]}:{hi[1]}:{hi[2]}"

    def test_payload_is_a_view_not_a_copy(self, tmp_path):
        """`raw[end:]` duplicated almost the whole buffer while `raw` was still
        live, so a frame near the decompression cap transiently needed ~2x the
        payload — the bounded path OOMing on its way to succeeding."""
        n = 400_000
        path = tmp_path / "a.alp.gz"
        write_alp(path, np.random.default_rng(0).uniform(-10, 10, (n, 3)))
        alf = read_alf(path)
        assert isinstance(alf._payload, memoryview), "payload was copied, not viewed"
        # and it still behaves as bytes-like everywhere the reader needs it to
        assert len(alf._payload) == n * 6
        assert alf.xyz.shape == (n, 3)

    def test_wider_than_float64_geometry_is_refused(self, tmp_path):
        """`np.longdouble` is 80-bit on x86, so two coordinates a metre apart at
        2**63 are distinct on input and IDENTICAL after the cast — the writer
        then publishes coincident points and a 5e-05 bound having discarded the
        difference. Same failure as the integer guards, one dtype family across."""
        if np.finfo(np.longdouble).bits <= 64:
            pytest.skip("longdouble is float64 on this platform")
        with pytest.raises(ValueError, match="float64"):
            write_alp(
                tmp_path / "a.alp.gz",
                np.array([[2**63, 0, 0], [2**63 + 1, 0, 0]], dtype=np.longdouble),
                codec="int16",
                frame_convention="world-aligned",
                check_budget=False,
            )

        # But an extended-precision array whose values round-trip EXACTLY is
        # ordinary data. Rejecting on dtype alone blocked converters carrying
        # precision they never used — the missing half of this test, and the
        # reason the guard was too strict.
        ok = tmp_path / "b.alp.gz"
        write_alp(ok, np.array([[1.0, 2.0, 3.0]], dtype=np.longdouble))
        assert read_alf(ok).xyz[0].tolist() == pytest.approx([1.0, 2.0, 3.0], abs=1e-3)

    def test_empty_frame_bound_still_validates_the_codec(self, tmp_path):
        """The empty-frame shortcut skipped the codec checks, so a v3 file with
        a missing or stringly-typed `_geom_scale` reported a confident 0.0 while
        `.xyz` raised on the same file. This property is a PREFLIGHT, so
        certifying an unreadable frame is the failure it exists to prevent."""
        for name, metadata in (
            ("missing", {"_total_points": 0, "_geom_codec": "int16", "_geom_offset": [0, 0, 0]}),
            (
                "stringly",
                {
                    "_total_points": 0,
                    "_geom_codec": "int16",
                    "_frame_convention": "world-aligned",
                    "_geom_scale": "0.1",
                    "_geom_offset": [0, 0, 0],
                },
            ),
        ):
            hb = json.dumps(metadata, separators=(",", ":")).encode()
            path = tmp_path / f"{name}.alp"
            path.write_bytes(struct.pack("B", 3) + struct.pack("<I", len(hb)) + hb)
            with pytest.raises(AlfError):
                read_alf(path).max_geom_error_m

        # A VALID empty frame still reports no loss.
        ok = tmp_path / "ok.alp.gz"
        write_alp(ok, np.zeros((0, 3)), codec=GEOM_INT16, frame_convention=FRAME_WORLD_ALIGNED)
        assert read_alf(ok).max_geom_error_m == 0.0

    def test_lossy_narrowing_is_refused_on_every_coercion(self, tmp_path):
        """Two ways to lose distinctions in a float64 cast — wider floats and
        integers past 2**53 — and three coercions that all need both.

        Geometry grew each check first and the other paths inherited neither,
        TWICE: the integer rule took four rounds to reach anchors and codec
        params, and the wide-float rule was reported again for intensity and
        anchors the round after geometry got it. This asserts the whole matrix
        so the next dtype family is added in one place, not four."""
        wide = np.finfo(np.longdouble).bits > 64
        big_int = 2**53
        cases = [
            (
                "geometry",
                lambda v: write_alp(
                    tmp_path / "g.alp.gz", v, codec=GEOM_INT16, frame_convention=FRAME_WORLD_ALIGNED, check_budget=False
                ),
                np.array([[big_int, 0, 0], [big_int + 1, 0, 0]], dtype=np.int64),
                np.array([[2**63, 0, 0], [2**63 + 1, 0, 0]], dtype=np.longdouble),
            ),
            (
                "intensity",
                lambda v: write_ali(tmp_path / "i.ali.gz", v, lo=int(v[0]), hi=int(v[-1])),
                np.array([big_int, big_int + 1], dtype=np.int64),
                np.array([2**63, 2**63 + 1], dtype=np.longdouble),
            ),
            (
                "anchor",
                lambda v: write_alp(tmp_path / "a.alp.gz", np.array([[1.0, 2.0, 3.0]]), device_position=v),
                np.array([big_int + 1, 0, 0], dtype=np.int64),
                np.array([2**63 + 1, 0, 0], dtype=np.longdouble),
            ),
        ]
        for _name, write, integral, extended in cases:
            with pytest.raises((ValueError, AlfError)):
                write(integral)
            if wide:
                with pytest.raises((ValueError, AlfError)):
                    write(extended)

    def test_every_decoded_array_is_read_only(self, tmp_path):
        """The payload-backed accessors became read-only when `_payload` did;
        the CACHED DECODED copies did not. So `categories` raised on a v1
        `.all` and silently mutated the cache on a v2 one, and `xyz` was
        writable for both codecs — mutation semantics decided by payload
        layout, the same bug as "decided by compression" one layer down.

        Enumerated rather than sampled: this rule has now been applied in three
        separate rounds and missed a different subset each time."""
        v1 = tmp_path / "v1.all.gz"
        write_all(v1, np.array([1, 2], dtype=np.uint8))
        v2 = tmp_path / "v2.all.gz"
        write_all(
            v2,
            np.array([1, 2], dtype=np.uint8),
            instance_ids=np.array([1, 2], dtype=np.uint16),
            instance_map={"1": "u1", "2": "u2"},
        )
        fp16 = tmp_path / "a.alp.gz"
        write_alp(fp16, np.array([[1.0, 2.0, 3.0]]))
        int16 = tmp_path / "b.alp.gz"
        write_alp(int16, np.array([[1.0, 2.0, 3.0]]), codec=GEOM_INT16, frame_convention=FRAME_WORLD_ALIGNED)
        ali = tmp_path / "c.ali.gz"
        write_ali(ali, np.array([1.0, 2.0]), lo=0.0, hi=2.0)
        alc = tmp_path / "d.alc.gz"
        write_alc(alc, np.array([0], dtype=np.uint8), ["ff0000"])
        alx = tmp_path / "e.alx.gz"
        write_alx(alx, np.array([0, 1], dtype=np.uint32), alp_total_points=2, source_total_points=5)

        accessors = (
            ("v1 categories", lambda: read_alf(v1).categories),
            ("v2 categories", lambda: read_alf(v2).categories),
            ("v2 instance_ids", lambda: read_alf(v2).instance_ids),
            ("fp16 xyz", lambda: read_alf(fp16).xyz),
            ("int16 xyz", lambda: read_alf(int16).xyz),
            ("intensity", lambda: read_alf(ali).intensity),
            ("palette_indices", lambda: read_alf(alc).palette_indices),
            ("kept_indices", lambda: read_alf(alx).kept_indices),
        )
        for _name, read in accessors:
            array = read()
            with pytest.raises(ValueError):
                array[0] = [9.0, 9.0, 9.0] if array.ndim == 2 else 99

        # Freezing must not break anything that DERIVES from a cached array.
        alf = read_alf(fp16)
        assert alf.world_xyz(sensor_pose=True)[0].tolist() == pytest.approx([1.0, 2.0, 3.0])

    def test_decoded_payloads_are_read_only_for_both_storage_forms(self, tmp_path):
        """Switching the gzip branch to a bytearray made its memoryview
        WRITABLE, so every zero-copy accessor handed back mutable storage:
        `alf.intensity[0] = 123` silently rewrote the payload and changed later
        reads. The same statement RAISED on an uncompressed file, so whether a
        decoded array could be corrupted depended on how the object happened to
        be stored."""
        gzipped = tmp_path / "a.ali.gz"
        write_ali(gzipped, np.array([10.0, 200.0]), lo=0.0, hi=255.0)
        plain = tmp_path / "b.ali"
        plain.write_bytes(gzip.decompress(gzipped.read_bytes()))

        for path in (gzipped, plain):
            alf = read_alf(path)
            with pytest.raises(ValueError):
                alf.intensity[0] = 123
            assert alf.intensity.tolist() == [10, 200]

        # Every zero-copy accessor, not just the one reported.
        labels = tmp_path / "c.all.gz"
        write_all(labels, np.array([1, 2], dtype=np.uint8))
        palette = tmp_path / "d.alc.gz"
        write_alc(palette, np.array([0], dtype=np.uint8), ["ff0000"])
        sidecar = tmp_path / "e.alx.gz"
        write_alx(
            sidecar,
            np.array([0, 1], dtype=np.uint32),
            alp_total_points=2,
            source_total_points=5,
        )
        for accessor in (
            read_alf(labels).categories,
            read_alf(palette).palette_indices,
            read_alf(sidecar).kept_indices,
        ):
            with pytest.raises(ValueError):
                accessor[0] = 99

    def test_uncompressed_input_is_size_capped_too(self, tmp_path, monkeypatch):
        """The cap went to the gzip branch alone, so an uncompressed
        multi-gigabyte `.all` — a shape this reader explicitly supports, since
        live label diffs are uploaded uncompressed — sailed past it and OOMed
        the worker before any header check could quarantine the frame."""
        monkeypatch.setattr(alf_io_module, "MAX_DECOMPRESSED_BYTES", 1024)
        oversized = tmp_path / "big.alp"
        oversized.write_bytes(b"\0" * 4096)
        for source in (oversized, b"\0" * 4096, io.BytesIO(b"\0" * 4096)):
            with pytest.raises(AlfError):
                read_alf(source, extension="alp")

    def test_short_reads_do_not_truncate_a_stream(self, tmp_path):
        """A single `read(n)` may return FEWER bytes than asked for without
        being at EOF — normal for sockets and buffered network readers. Capping
        the read that way truncated the stream to its first chunk: a legacy
        file without `_total_points` parsed as a short prefix and quietly lost
        points, and one with a declared count was rejected as truncated.

        `source.read()` consumed the whole stream before the cap was added;
        this is the regression that introduced."""

        class Dribble(io.RawIOBase):
            """Returns three bytes at a time, never signalling EOF early."""

            def __init__(self, data):
                self.data = data
                self.pos = 0

            def readable(self):
                return True

            def read(self, n=-1):
                if self.pos >= len(self.data):
                    return b""
                chunk = self.data[self.pos : self.pos + 3]
                self.pos += len(chunk)
                return chunk

        path = tmp_path / "a.all.gz"
        write_all(path, np.arange(1, 9, dtype=np.uint8))
        raw = gzip.decompress(path.read_bytes())
        header_len = struct.unpack("<I", raw[1:5])[0]
        # No `_total_points`, so the length comes from the payload alone — the
        # shape that loses points silently rather than raising.
        bare = raw[:1] + struct.pack("<I", 0) + raw[5 + header_len :]

        whole = read_alf(io.BytesIO(bare), extension="all").categories.tolist()
        assert read_alf(Dribble(bare), extension="all").categories.tolist() == whole
        # and through the gzip branch, which reads from the same stream
        assert read_alf(Dribble(path.read_bytes()), extension="all").categories.tolist() == whole

    def test_no_input_path_doubles_the_payload_in_memory(self, tmp_path):
        """`bytes(buffer)` allocates a second contiguous copy while the first is
        still live. That doubling was removed from the gzip branch, then
        reintroduced in the stream loop written to fix short-read truncation —
        so this asserts every input path at once rather than the one reported.

        A per-byte ratio, not a fixed threshold: the failure is memory that
        scales at 2x the payload, and only a comparison across sizes catches
        that independently of interpreter overhead."""
        import tracemalloc

        payloads = (8 * 1024 * 1024, 32 * 1024 * 1024)
        ratios = {"path": [], "bytes": [], "stream": []}
        for size in payloads:
            blob = struct.pack("B", 1) + struct.pack("<I", 0) + b"\x01" * size
            uncompressed = tmp_path / f"{size}.all"
            uncompressed.write_bytes(blob)
            sources = {
                "path": lambda u=uncompressed: read_alf(u, extension="all"),
                "bytes": lambda b=blob: read_alf(b, extension="all"),
                "stream": lambda b=blob: read_alf(io.BytesIO(b), extension="all"),
            }
            for label, read in sources.items():
                tracemalloc.start()
                assert read().categories.shape == (size,)
                ratios[label].append(tracemalloc.get_traced_memory()[1] / size)
                tracemalloc.stop()

        for name, seen in ratios.items():
            assert seen[1] < 2.0, f"{name} holds {seen[1]:.1f}x the payload"

    def test_all_false_masks_are_ordinary_data(self, tmp_path):
        """`mask is not nomask` was too strict. An EXPLICIT all-False mask
        exposes nothing through `np.asarray`, so refusing it rejected valid
        data across all five channels — a false positive I introduced while
        fixing the real one."""
        for _name, write in (
            ("categories", lambda v: write_all(tmp_path / "a.all.gz", v)),
            ("intensity", lambda v: write_ali(tmp_path / "b.ali.gz", v, lo=0.0, hi=8.0)),
            (
                "indices",
                lambda v: write_alx(tmp_path / "c.alx.gz", v, alp_total_points=2, source_total_points=9),
            ),
        ):
            write(np.ma.array([7, 8], mask=[False, False]))
        write_alp(tmp_path / "d.alp.gz", np.ma.array([[1.0, 2.0, 3.0]], mask=[[0, 0, 0]]))

        # Anything ACTUALLY masked is still refused.
        with pytest.raises(ValueError, match="masked array"):
            write_all(tmp_path / "e.all.gz", np.ma.array([7, 8], mask=[False, True]))
        with pytest.raises(ValueError, match="masked array"):
            write_alp(
                tmp_path / "f.alp.gz",
                np.ma.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], mask=[[0, 0, 0], [1, 1, 1]]),
            )

    def test_object_arrays_holding_wide_floats_are_refused(self, tmp_path):
        """The CONTAINER is object, so the dtype test sees nothing — but a
        `np.longdouble` scalar inside narrows exactly as lossily. Object arrays
        already needed elementwise inspection for integers; this is the same
        inspection one scalar type across."""
        if np.finfo(np.longdouble).bits <= 64:
            pytest.skip("longdouble is float64 on this platform")
        big = np.longdouble(2**63)
        with pytest.raises(ValueError):
            write_alp(
                tmp_path / "a.alp.gz",
                np.array([[big, 0, 0], [big + 1, 0, 0]], dtype=object),
                codec=GEOM_INT16,
                frame_convention=FRAME_WORLD_ALIGNED,
                check_budget=False,
            )
        with pytest.raises(ValueError):
            write_ali(
                tmp_path / "b.ali.gz",
                np.array([big, big + 1], dtype=object),
                lo=2**63,
                hi=2**63 + 1,
            )

    def test_buffer_input_is_capped_before_it_is_copied(self, tmp_path):
        """`bytes(source)` duplicated the buffer before the cap could look at
        it, so an oversized input OOMed on the way to being rejected. Checking
        the length first removes the copy entirely — the reader wants a
        read-only view anyway."""
        import tracemalloc

        # A BYTEARRAY, not bytes. `bytes(b)` on an immutable bytes object
        # returns that same object, so the `bytes` input is the one buffer kind
        # where this copy cannot happen — testing it measured 0.0x for both the
        # fixed and broken versions and proved nothing.
        size = 8 * 1024 * 1024
        blob = bytearray(struct.pack("B", 1) + struct.pack("<I", 0) + b"\x01" * size)
        tracemalloc.start()
        assert read_alf(blob, extension="all").categories.shape == (size,)
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        # Half the payload, not the whole of it: the blob is allocated OUTSIDE
        # the traced window, so a full `bytes(source)` copy lands at ~1.0x and
        # a `peak < size` bound cannot tell it from the zero-copy view. Verified
        # by reverting the fix and watching this fail.
        assert peak < size / 2, f"buffer input copied: {peak / size:.2f}x the payload"

    def test_integral_looking_literals_are_judged_on_the_token(self, tmp_path):
        """`float("1.0000000000000001")` is exactly 1.0 and `float("1e-400")` is
        0.0, so a fractional count validated AFTER conversion looks like a clean
        integer — the evidence is gone before the check runs.

        Both directions, because judging integrality by whether `int(token)`
        parses rejects `1.0`, which is what my first attempt at this did."""
        reject = ("1.0000000000000001", "1e-400", "1.5")
        accept = {"1.0": 1, "1e0": 1, "1": 1, "2e1": 20}
        for token in reject:
            header = ('{"_total_points":' + token + "}").encode()
            path = tmp_path / "bad.all"
            path.write_bytes(struct.pack("B", 1) + struct.pack("<I", len(header)) + header + b"\x05")
            with pytest.raises(AlfError, match="integer"):
                read_alf(path)
        for token, expected in accept.items():
            header = ('{"_total_points":' + token + "}").encode()
            path = tmp_path / "ok.all"
            path.write_bytes(struct.pack("B", 1) + struct.pack("<I", len(header)) + header + b"\x05" * expected)
            assert read_alf(path).total_points == expected

    def test_inherited_array_anchor_does_not_crash(self, tmp_path):
        """`header[key] == {}` on a NumPy (3,) array returns a boolean ARRAY,
        so the `if` raised "truth value is ambiguous" before the shared anchor
        validator ran. The same shape works through the explicit argument, so
        converters assembling metadata from NumPy could not use the documented
        metadata-preserving write path at all."""
        path = tmp_path / "a.alp.gz"
        write_alp(
            path,
            np.array([[1.0, 2.0, 3.0]]),
            {"device_position": np.array([1.0, 2.0, 3.0])},
        )
        assert read_alf(path).device_position == pytest.approx([1.0, 2.0, 3.0])

        # The `{}` legacy exemption still applies to an actual empty mapping.
        legacy = tmp_path / "b.alp.gz"
        write_alp(legacy, np.array([[1.0, 2.0, 3.0]]), {"device_position": {}})
        assert read_alf(legacy).device_position == pytest.approx([0.0, 0.0, 0.0])

    def test_masked_anchor_components_are_refused(self, tmp_path):
        """`np.asarray` drops the mask and hands back the BACKING value, so a
        masked x of 100 placed the whole frame 100 m away — a component the
        caller explicitly marked invalid deciding where the cloud goes.
        Geometry and the per-point channels already ask this."""
        base = np.array([[1.0, 2.0, 3.0]])
        masked = np.ma.array([100.0, 0.0, 0.0], mask=[True, False, False])
        with pytest.raises(AlfError, match="masked"):
            write_alp(tmp_path / "a.alp.gz", base, device_position=masked)

        frame = tmp_path / "b.alp.gz"
        write_alp(frame, base)
        with pytest.raises(AlfError, match="masked"):
            read_alf(frame).world_xyz(origin=masked)

        # An all-False mask is ordinary data here too.
        ok = tmp_path / "c.alp.gz"
        write_alp(ok, base, device_position=np.ma.array([1.0, 2.0, 3.0], mask=[0, 0, 0]))
        assert read_alf(ok).device_position == pytest.approx([1.0, 2.0, 3.0])

    def test_concatenated_gzip_members_are_refused(self, tmp_path, monkeypatch):
        """`GzipFile` decodes every concatenated member into ONE byte run, so a
        second ALF's envelope becomes the first one's payload. For a legacy file
        with no `_total_points` any length is valid, so `.ali` silently returned
        the next file's JSON header as intensity samples."""
        single = tmp_path / "one.ali.gz"
        write_ali(single, np.array([1.0, 2.0]), lo=0.0, hi=2.0)
        raw = gzip.decompress(single.read_bytes())
        header_len = struct.unpack("<I", raw[1:5])[0]
        metadata = json.loads(raw[5 : 5 + header_len])
        metadata.pop("_total_points", None)
        rebuilt = json.dumps(metadata, separators=(",", ":")).encode()
        member = gzip.compress(raw[:1] + struct.pack("<I", len(rebuilt)) + rebuilt + raw[5 + header_len :])

        legacy = tmp_path / "legacy.ali.gz"
        legacy.write_bytes(member)
        assert read_alf(legacy).intensity.tolist() == [127, 255]

        joined = tmp_path / "cat.ali.gz"
        joined.write_bytes(member + member)
        with pytest.raises(AlfError, match="more than one gzip member"):
            read_alf(joined)

        # And when the first member ends EXACTLY on an input-block boundary.
        # `unused_data` is empty there and the rest of the input was never fed
        # to the decompressor at all, so the second member went unseen and the
        # concatenation was accepted after all. Driven by shrinking the block
        # rather than building a 16 MB fixture.
        for block in (len(member), len(member) + 7):
            monkeypatch.setattr(alf_io_module, "_GZIP_CHUNK_BYTES", block)
            with pytest.raises(AlfError, match="more than one gzip member"):
                read_alf(joined)
            assert read_alf(legacy).intensity.tolist() == [127, 255]

        # A truncated member must still be AlfError, not a bare zlib error.
        chopped = tmp_path / "chopped.ali.gz"
        chopped.write_bytes(member[: len(member) // 2])
        with pytest.raises(AlfError):
            read_alf(chopped)

    def test_subnormal_half_scale_rounds_upward(self, tmp_path):
        """`2.5e-323 / 2` is 1e-323 where the exact half is 2.5 minimum
        subnormals, so the reported bound sat BELOW the real quantization
        error. Flooring only the zero case fixed one end of that."""
        hb = json.dumps(
            {
                "_total_points": 1,
                "_geom_codec": "int16",
                "_frame_convention": "world-aligned",
                "_geom_scale": 2.5e-323,
                "_geom_offset": [0.0, 0.0, 0.0],
            },
            separators=(",", ":"),
        ).encode()
        path = tmp_path / "000000.alp"
        payload = np.array([0, 0, 0], dtype="<i2").tobytes()
        path.write_bytes(struct.pack("B", 3) + struct.pack("<I", len(hb)) + hb + payload)
        assert read_alf(path).max_geom_error_m >= 1.5e-323

    def test_object_arrays_of_exact_wide_floats_are_accepted(self, tmp_path):
        """The native branch gained a value comparison last round and this one
        did not, so an object array of exactly-representable longdoubles was
        refused while the equivalent NATIVE array was accepted — one rule
        disagreeing with itself depending on the container."""
        if np.finfo(np.longdouble).bits <= 64:
            pytest.skip("longdouble is float64 on this platform")
        exact = np.array([[np.longdouble(1), np.longdouble(2), np.longdouble(3)]], dtype=object)
        ok = tmp_path / "a.alp.gz"
        write_alp(ok, exact)
        assert read_alf(ok).xyz[0].tolist() == pytest.approx([1.0, 2.0, 3.0], abs=1e-3)

        # Lossy ones are still refused.
        lossy = np.array([[np.longdouble(2**63), 0, 0], [np.longdouble(2**63) + 1, 0, 0]], dtype=object)
        with pytest.raises(ValueError):
            write_alp(
                tmp_path / "b.alp.gz", lossy, codec=GEOM_INT16, frame_convention=FRAME_WORLD_ALIGNED, check_budget=False
            )

    def test_public_geometry_helpers_are_exported(self):
        """`choose_geom_offset`, `choose_geom_scale` and `map_intensity` are
        non-underscored public APIs that callers had to reach into the module
        for. AGENTS.md makes `__all__` the export list."""
        import alf as core_pkg

        for name in ("choose_geom_offset", "choose_geom_scale", "map_intensity"):
            assert name in core_pkg.__all__, f"{name} missing from __all__"
            assert getattr(core_pkg, name) is not None

    def test_both_public_geometry_helpers_validate_their_input(self):
        """Exporting these made them entry points, so they answer for their own
        input. Both reduced over axis 0 without checking: an (N, 4)
        XYZ-plus-intensity matrix let a large intensity choose `_geom_scale`,
        a masked array silently omitted its largest component, and NaN/Inf
        produced a `nan` offset and an `inf` scale that poison every coordinate
        derived from them.

        Non-finite is NOT in this list any more. Invalid returns are valid
        content that v3 stores as `INT16_NO_RETURN`, so these helpers skip them
        when sizing the grid rather than refusing the frame — sized from a slot
        that holds no position, the grid would be wrong. The shapes below are
        still refused because each is a genuine mistake about the input."""
        bad = (
            np.array([[0.0, 0, 0, 1e6]]),
            np.ma.array([[0.0, 0, 0], [100.0, 0, 0]], mask=[[0, 0, 0], [1, 1, 1]]),
            np.array([["1", "2", "3"]]),
        )
        for helper in (choose_geom_offset, choose_geom_scale):
            for value in bad:
                with pytest.raises(ValueError):
                    helper(value)

        assert choose_geom_offset(np.array([[0.0, 0, 0], [10.0, 10, 10]])).tolist() == [
            5.0,
            5.0,
            5.0,
        ]
        assert choose_geom_scale(np.array([[0.0, 0, 0], [100.0, 0, 0]])) > 0.0
        assert choose_geom_offset(np.zeros((0, 3))).tolist() == [0.0, 0.0, 0.0]

        # The writer derives its extent blockwise and must still land on the
        # same step the public helper gives for the same centred coordinates.
        rng = np.random.default_rng(3)
        xyz = rng.uniform(-30, 30, (20_000, 3))
        offset = choose_geom_offset(xyz)
        assert choose_geom_scale(xyz - offset) == alf_io_module._scale_from_extent(float(np.max(np.abs(xyz - offset))))

    def test_public_offset_helper_validates_its_shape(self):
        """Exporting this made it a public entry point, and it reduced over
        axis 0 without checking what it was given: the flat `[1, 2, 3]` single
        point `write_alp` accepts became the SCALAR 2, and an (N, 4)
        XYZ-plus-intensity matrix became four components. Both then broadcast
        through centring and produce wrong geometry without raising."""
        assert choose_geom_offset(np.array([1.0, 2.0, 3.0])).tolist() == [1.0, 2.0, 3.0]
        with pytest.raises(ValueError):
            choose_geom_offset(np.array([[1.0, 2.0, 3.0, 9.0]]))
        assert choose_geom_offset(np.array([[0.0, 0, 0], [10.0, 10, 10]])).tolist() == [
            5.0,
            5.0,
            5.0,
        ]
        assert choose_geom_offset(np.zeros((0, 3))).tolist() == [0.0, 0.0, 0.0]

    def test_writer_refuses_an_envelope_its_reader_rejects(self, tmp_path, monkeypatch):
        """`read_alf` refuses anything past the decompression cap, so writing
        past it produces a file the paired reader cannot consume. The writer
        already honours the HEADER cap for exactly this reason."""
        monkeypatch.setattr(alf_io_module, "MAX_DECOMPRESSED_BYTES", 1024)
        with pytest.raises(ValueError, match="past the"):
            write_ali(tmp_path / "a.ali.gz", np.arange(2000, dtype=np.float64), lo=0.0, hi=1999.0)
        with pytest.raises(ValueError, match="past the"):
            write_alp(tmp_path / "b.alp.gz", np.random.default_rng(0).uniform(-5, 5, (500, 3)))

    def test_public_map_intensity_validates_its_input(self, tmp_path):
        """A package-level public API now, and a direct caller got NONE of the
        mask, complex, string or lossy-narrowing checks `write_ali` performs
        before calling it. Second helper this happened to after
        `choose_geom_offset` — exporting a function is an API decision and the
        validation has to come with it."""
        with pytest.raises(ValueError, match="masked"):
            map_intensity(np.ma.array([0.0, 1.0], mask=[False, True]), 0.0, 1.0)
        with pytest.raises(ValueError, match="strings"):
            map_intensity(np.array(["0", "10"]), 0, 10)
        with pytest.raises(ValueError):
            map_intensity(np.array([0 + 0j, 10 + 9j]), 0, 10)
        if np.finfo(np.longdouble).bits > 64:
            with pytest.raises(ValueError):
                map_intensity(
                    np.array([np.longdouble(2**63), np.longdouble(2**63) + 1]),
                    2**63,
                    2**63 + 1,
                )
        assert map_intensity(np.array([0.0, 5.0, 10.0]), 0.0, 10.0).tolist() == [0, 127, 255]

    def test_header_tokens_that_lose_precision_are_refused(self, tmp_path):
        """`9007199254740993.0` is an ordinary JSON token that reads back as
        `...992.0` — one metre out for an anchor, a different grid for a codec
        parameter. `_reject_inexact_int` catches the integer spelling; this is
        the decimal one, which reaches `float()` through a different door.

        Anchors AND codec params, since the same defect was in both and only
        the anchors were reported."""
        lossy = "9007199254740993.0"
        cases = (
            ("device_position", '{"_total_points":1,"device_position":{"x":%s,"y":0.0,"z":0.0}}'),
            (
                "world_centroid_epsg3857",
                '{"_total_points":1,"world_centroid_epsg3857":{"x":%s,"y":0.0,"z":0.0}}',
            ),
        )
        for name, template in cases:
            hb = (template % lossy).encode()
            path = tmp_path / f"{name}.alp"
            payload = np.array([1.0, 2.0, 3.0], dtype=np.float16).tobytes()
            path.write_bytes(struct.pack("B", 1) + struct.pack("<I", len(hb)) + hb + payload)
            with pytest.raises(AlfError, match="cannot represent"):
                if name == "device_position":
                    read_alf(path).world_xyz(sensor_pose=True)
                else:
                    read_alf(path).world_xyz()

        hb = (
            '{"_total_points":1,"_geom_codec":"int16","_frame_convention":"world-aligned",'
            '"_geom_scale":1,'
            f'"_geom_offset":[{lossy},0,0]}}'
        ).encode()
        codec = tmp_path / "codec.alp"
        codec.write_bytes(
            struct.pack("B", 3) + struct.pack("<I", len(hb)) + hb + np.array([0, 0, 0], dtype="<i2").tobytes()
        )
        with pytest.raises(AlfError, match="cannot represent"):
            read_alf(codec).xyz

        # Tokens float64 DOES preserve are untouched — a decimal fraction is
        # inherently approximate in binary and `1e+308` round-trips exactly.
        # Two earlier versions of this check rejected both.
        ok = tmp_path / "ok.alp.gz"
        write_alp(ok, np.array([[0.1, 0.2, 0.3]]), device_position=(-448799999999.996, 0, 0))
        assert read_alf(ok).device_position[0] == pytest.approx(-448799999999.996)

    def test_compressed_size_is_not_the_decompressed_cap(self, tmp_path, monkeypatch):
        """The cap is about DECOMPRESSED bytes. Applying it to a gzip file's
        size on disk rejects a legitimate one whose payload is incompressible —
        deflate framing makes the archive LARGER than its contents, so a
        1024-byte envelope written at a 1024-byte cap became a 1047-byte `.gz`
        that the reader refused. The writer's own output, unreadable."""
        monkeypatch.setattr(alf_io_module, "MAX_DECOMPRESSED_BYTES", 1024)
        envelope = struct.pack("B", 1) + struct.pack("<I", 0) + os.urandom(1019)
        archive = tmp_path / "a.all.gz"
        archive.write_bytes(gzip.compress(envelope))
        assert archive.stat().st_size > 1024, "repro needs an incompressible payload"
        assert read_alf(archive, extension="all").categories.shape == (1019,)

        # An oversized RAW file is still refused before it is read...
        raw_file = tmp_path / "big.all"
        raw_file.write_bytes(b"\0" * 4096)
        with pytest.raises(AlfError):
            read_alf(raw_file, extension="all")

        # ...and a small archive that EXPANDS past the cap still is too.
        bomb = tmp_path / "bomb.all.gz"
        bomb.write_bytes(gzip.compress(struct.pack("B", 1) + struct.pack("<I", 0) + b"\0" * 100_000))
        with pytest.raises(AlfError, match="expands past"):
            read_alf(bomb, extension="all")

    def test_float_spelled_scale_token_is_checked(self, tmp_path):
        """The int branch and `_geom_offset` both consult the retained token;
        the FLOAT branch of the scale did not, so `9007199254740993.0` was
        accepted as `...992.0` and every stored component decoded against a
        scale the file does not declare."""
        hb = (
            b'{"_total_points":1,"_geom_codec":"int16","_frame_convention":"world-aligned",'
            b'"_geom_scale":9007199254740993.0,"_geom_offset":[0,0,0]}'
        )
        path = tmp_path / "000000.alp"
        payload = np.array([1, 0, 0], dtype="<i2").tobytes()
        path.write_bytes(struct.pack("B", 3) + struct.pack("<I", len(hb)) + hb + payload)
        with pytest.raises(AlfError, match="cannot represent"):
            read_alf(path).xyz

    def test_expanded_integral_geometry_tokens_are_exact_claims(self, tmp_path):
        """`.0` and `e0` cannot inherit a shortest-scientific escape.

        These tokens spell an exact mathematical integer, but binary64 reads
        both as a nearby value. A normalized writer token such as
        `1.0000000000000001e+18` remains a binary64 round-trip spelling.
        """
        headers = (
            (
                b'{"_total_points":1,"_geom_codec":"int16","_frame_convention":"world-aligned","_geom_scale":1,'
                b'"_geom_offset":[1000000000000000100.0,0,0]}'
            ),
            (
                b'{"_total_points":1,"_geom_codec":"int16","_frame_convention":"world-aligned",'
                b'"_geom_scale":1000000000000000100e0,"_geom_offset":[0,0,0]}'
            ),
        )
        payload = np.array([1, 0, 0], dtype="<i2").tobytes()
        for index, header in enumerate(headers):
            path = tmp_path / f"lossy-{index}.alp"
            path.write_bytes(struct.pack("B", 3) + struct.pack("<I", len(header)) + header + payload)
            with pytest.raises(AlfError, match="cannot represent"):
                _ = read_alf(path).xyz

        writer = tmp_path / "writer.alp.gz"
        exact = float(1000000000000000128)
        write_alp(
            writer,
            np.array([[exact, 0, 0]], dtype=np.float64),
            codec=GEOM_INT16,
            frame_convention=FRAME_WORLD_ALIGNED,
            check_budget=False,
        )
        assert b"1.0000000000000001e+18" in gzip.decompress(writer.read_bytes())
        assert read_alf(writer).xyz[0, 0] == exact

    def test_nonfinite_anchor_token_precedes_decoded_shape_validation(self, tmp_path):
        header = b'{"_total_points":1,"device_position":{"x":1e309,"y":0,"z":0}}'
        path = tmp_path / "nonfinite-anchor.alp"
        path.write_bytes(
            struct.pack("B", 1) + struct.pack("<I", len(header)) + header + np.zeros(3, dtype=np.float16).tobytes()
        )

        with pytest.raises(AlfError, match="too large to represent") as caught:
            read_alf(path)
        assert "Geometry anchor" not in str(caught.value)

    def test_int16_decode_does_not_hold_two_full_arrays(self, tmp_path):
        """`astype(...) * scale + offset` holds the multiplication temporary and
        the addition result at once — ~240 MB each at the documented 10M-point
        scale, on top of the still-live payload."""
        import tracemalloc

        n = 400_000
        xyz = np.random.default_rng(0).uniform(-50, 50, (n, 3))
        path = tmp_path / "a.alp.gz"
        write_alp(path, xyz, codec=GEOM_INT16, frame_convention=FRAME_WORLD_ALIGNED)

        alf = read_alf(path)
        tracemalloc.start()
        decoded = alf.xyz
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()

        result_bytes = n * 3 * 8
        assert peak < result_bytes * 1.6, f"held {peak / result_bytes:.1f}x the result"
        # In-place must not change the answer.
        metadata = alf.metadata
        expected = np.frombuffer(alf._payload, dtype="<i2").reshape(-1, 3).astype(np.float64)
        expected = expected * metadata["_geom_scale"] + np.array(metadata["_geom_offset"])
        assert np.array_equal(decoded, expected)

    def test_every_input_kind_applies_the_same_size_rule(self, tmp_path, monkeypatch):
        """The cap is about DECOMPRESSED bytes, and deflate expands
        incompressible data slightly — so measuring an ARCHIVE against it
        rejects the writer's own output. Fixed for paths, then found again for
        buffers; asserting all five kinds together rather than a third round."""
        monkeypatch.setattr(alf_io_module, "MAX_DECOMPRESSED_BYTES", 1024)
        envelope = struct.pack("B", 1) + struct.pack("<I", 0) + os.urandom(1006)
        archive = gzip.compress(envelope)
        assert len(archive) > 1024, "repro needs the archive to exceed the cap"
        on_disk = tmp_path / "a.ali.gz"
        on_disk.write_bytes(archive)

        for source in (
            on_disk,
            archive,
            bytearray(archive),
            memoryview(archive),
            io.BytesIO(archive),
        ):
            assert read_alf(source, extension="ali").intensity.shape == (1006,)

        # An oversized RAW input is still refused on every branch...
        raw = struct.pack("B", 1) + struct.pack("<I", 0) + b"\0" * 4096
        (tmp_path / "big.ali").write_bytes(raw)
        for source in (tmp_path / "big.ali", raw, io.BytesIO(raw)):
            with pytest.raises(AlfError):
                read_alf(source, extension="ali")

        # ...and so is a small archive that EXPANDS past the cap.
        bomb = gzip.compress(struct.pack("B", 1) + struct.pack("<I", 0) + b"\0" * 100_000)
        for source in (bomb, io.BytesIO(bomb)):
            with pytest.raises(AlfError, match="expands past"):
                read_alf(source, extension="ali")

    def test_mixed_type_anchor_components_keep_their_precision(self, tmp_path):
        """`np.ravel` on a mixed sequence promotes to float64 FIRST, so
        `(2**53 + 1, 0.5, 0)` reached the checks with x already rounded and the
        frame shipped a metre out. The dict path inspected its components and
        the sequence path did not."""
        base = np.array([[1.0, 2.0, 3.0]])
        lossy = 2**53 + 1
        for value in (
            {"x": lossy, "y": 0.5, "z": 0},
            (lossy, 0.5, 0),
            [lossy, 0.5, 0],
        ):
            with pytest.raises((AlfError, ValueError)):
                write_alp(tmp_path / "bad.alp.gz", base, device_position=value)

        # A caller-built float64 ARRAY is not detectable and should not be: it
        # rounds at construction, before this library is called at all.
        assert np.array([lossy, 0.5, 0])[0] == float(2**53)

        for value in ({"x": 1, "y": 0.5, "z": 0}, (1, 0.5, 0), [1, 0.5, 0]):
            ok = tmp_path / "ok.alp.gz"
            write_alp(ok, base, device_position=value)
            assert read_alf(ok).device_position.tolist() == [1.0, 0.5, 0.0]

    def test_fractional_tokens_that_shift_are_refused(self, tmp_path):
        """Restricting the token check to INTEGER-valued literals was a
        narrowing I added defensively and did not need: `9007199254740993.5`
        reads back as `...994.0` and shifts a frame half a metre, while the
        repr comparison accepts `0.1` and `0.30000000000000004` on merit."""
        hb = b'{"_total_points":1,"device_position":{"x":9007199254740993.5,"y":0.0,"z":0.0}}'
        path = tmp_path / "000000.alp"
        payload = np.array([1.0, 2.0, 3.0], dtype=np.float16).tobytes()
        path.write_bytes(struct.pack("B", 1) + struct.pack("<I", len(hb)) + hb + payload)
        with pytest.raises(AlfError, match="cannot represent"):
            read_alf(path).world_xyz(sensor_pose=True)

        ok = tmp_path / "ok.alp.gz"
        write_alp(ok, np.array([[0.1, 0.2, 0.3]]), device_position=(-448799999999.996, 0, 0))
        assert read_alf(ok).device_position[0] == pytest.approx(-448799999999.996)

    def test_datetime_payloads_are_refused(self, tmp_path):
        """Refusing only COMPLEX let every other non-numeric dtype through: a
        `datetime64[D]` column casts to its epoch day count, so three dates
        published as coordinates 19723, 19724, 19725 — entirely plausible,
        entirely wrong. Stated as an allow-list now."""
        dates = np.array([["2024-01-01", "2024-01-02", "2024-01-03"]], dtype="datetime64[D]")
        with pytest.raises(ValueError, match="not real numeric data"):
            write_alp(tmp_path / "a.alp.gz", dates, codec=GEOM_INT16, frame_convention=FRAME_WORLD_ALIGNED)
        with pytest.raises(ValueError, match="not real numeric data"):
            write_alp(tmp_path / "b.alp.gz", np.array([[1, 2, 3]], dtype="timedelta64[s]"))
        flat = np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[D]")
        with pytest.raises(ValueError, match="not real numeric data"):
            write_ali(tmp_path / "c.ali.gz", flat, lo=0, hi=1)
        with pytest.raises(ValueError, match="not real numeric data"):
            write_all(tmp_path / "d.all.gz", flat)

    def test_only_lossy_integers_are_refused(self, tmp_path):
        """`2**53` itself IS exactly representable, and so is every even
        integer above it — a magnitude test refused an int64 frame at exactly
        2**53 that round-trips perfectly, while the identical float64 input was
        accepted. The wide-float branch was given a value comparison two rounds
        ago and this one kept the cruder test.

        Fifth false positive in this loop, all the same shape: a guard that
        checks a proxy for the property instead of the property."""
        write_alp(
            tmp_path / "exact.alp.gz",
            np.array([[2**53, 0, 0]], dtype=np.int64),
            codec=GEOM_INT16,
            frame_convention=FRAME_WORLD_ALIGNED,
            check_budget=False,
        )
        write_alp(tmp_path / "small.alp.gz", np.array([[1, 2, 3]], dtype=np.int64))
        with pytest.raises(ValueError):
            write_alp(
                tmp_path / "lossy.alp.gz",
                np.array([[2**53, 0, 0], [2**53 + 1, 0, 0]], dtype=np.int64),
                codec=GEOM_INT16,
                frame_convention=FRAME_WORLD_ALIGNED,
                check_budget=False,
            )

    def test_memoryview_offsets_are_byte_offsets(self, tmp_path):
        """`len()` and slicing on a `memoryview` count ELEMENTS, so a view cast
        to `I` made every byte offset advance four times too far and the size
        check undercount by the item size — `raw[end:]` returned the tail of
        the payload rather than the payload."""
        source = tmp_path / "a.ali.gz"
        write_ali(source, np.arange(20, dtype=np.float64), lo=0.0, hi=19.0)
        raw = gzip.decompress(source.read_bytes())
        expected = read_alf(raw, extension="ali").intensity.tolist()

        for view in (memoryview(raw), memoryview(bytearray(raw))):
            assert read_alf(view, extension="ali").intensity.tolist() == expected

        padded = raw + b"\0" * ((4 - len(raw) % 4) % 4)
        wide = memoryview(padded).cast("I")
        assert len(wide) * wide.itemsize == len(padded), "repro needs a wide view"
        decoded = read_alf(wide, extension="ali").intensity.tolist()
        assert decoded[: len(expected)] == expected

    def test_path_is_read_through_one_descriptor(self, tmp_path):
        """`stat()`, a separate open for the magic bytes, then `read_bytes()`
        can describe three different files if the path is replaced in between —
        and the final read was unbounded, so a file that was small at check
        time could grow past the cap and OOM the worker the cap protects."""
        import tracemalloc

        path = tmp_path / "a.alp.gz"
        write_alp(path, np.random.default_rng(0).uniform(-10, 10, (200_000, 3)))
        tracemalloc.start()
        alf = read_alf(path)
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        assert alf.xyz.shape == (200_000, 3)
        # `read(limit + 1)` PRE-ALLOCATES the limit, so bounding by the cap
        # rather than the file's own size reserved 512 MB to read this frame.
        assert peak < path.stat().st_size * 8, f"read allocated {peak} bytes"

    def test_growth_past_the_stat_snapshot_is_refused(self, tmp_path, monkeypatch):
        """The bounded read asked whether the file had passed 512 MB, not
        whether it had grown since the `fstat` — so a legacy channel appended
        to in that window was parsed as its old prefix plus one byte and the
        rest vanished. `.ali` and `.all` without `_total_points` accept ANY
        payload length, so nothing downstream would have noticed.

        The comment above that check already claimed it detected growth. The
        code compared against the wrong bound."""
        header = b"{}"
        path = tmp_path / "a.ali"
        path.write_bytes(struct.pack("B", 1) + struct.pack("<I", len(header)) + header + b"\x01" * 10)
        assert len(read_alf(path, extension="ali").intensity) == 10

        # A stat that under-reports stands in for the append-after-stat race.
        real_fstat = alf_io_module.os.fstat

        class Stale:
            def __init__(self, stat_result):
                self.st_size = stat_result.st_size - 4
                # The REAL mode. Only `st_size` is being falsified; the reader
                # keys the strict length rule off `S_ISREG`, so a fake without a
                # mode would route this regular file down the advisory-size path
                # and test nothing.
                self.st_mode = stat_result.st_mode

        monkeypatch.setattr(alf_io_module.os, "fstat", lambda fd: Stale(real_fstat(fd)))
        with pytest.raises(AlfError, match="grew while being read"):
            read_alf(path, extension="ali")

    def test_writer_output_always_reads_back(self, tmp_path):
        """The module's whole premise: one writer, one reader, and the writer
        never emits what the reader refuses. All three of this round's findings
        were instances of that invariant breaking."""
        path = tmp_path / "a.all.gz"
        write_all(
            path,
            np.array([1, 2], dtype=np.uint8),
            instance_ids=np.array([1, 2], dtype=np.uint16),
            instance_map={"1": "u1", "2": "u2"},
        )
        alf = read_alf(path)
        assert alf.instance_map == {"1": "u1", "2": "u2"}
        assert alf.instance_ids.tolist() == [1, 2]

    def test_complex_arrays_are_refused(self, tmp_path):
        """astype(float64) drops every imaginary component with only a
        ComplexWarning, which callers need not treat as fatal — so
        [[1+9j, 2, 3]] publishes a plausible point at (1, 2, 3)."""
        with pytest.raises(ValueError, match="not complex"):
            write_alp(tmp_path / "a.alp.gz", np.array([[1 + 9j, 2, 3]]))
        with pytest.raises(ValueError, match="not complex"):
            write_all(tmp_path / "a.all.gz", np.array([1 + 9j, 2]))
        with pytest.raises(ValueError, match="not complex"):
            write_ali(tmp_path / "a.ali.gz", np.array([1 + 9j, 2]), lo=0.0, hi=5.0)

    def test_error_bound_checks_the_payload_stride(self, tmp_path):
        """Valid codec keys but a truncated payload: the preflight must not
        report a usable bound for geometry .xyz will refuse."""
        hb = json.dumps(
            {
                "_frame_convention": "world-aligned",
                "_geom_codec": "int16",
                "_geom_scale": 0.01,
                "_geom_offset": [0, 0, 0],
            },
            separators=(",", ":"),
        ).encode()
        path = tmp_path / "000000.alp"
        path.write_bytes(struct.pack("B", 3) + struct.pack("<I", len(hb)) + hb + bytes(8))
        alf = read_alf(path)
        with pytest.raises(AlfError, match="6-byte XYZ triples"):
            alf.max_geom_error_m
        with pytest.raises(AlfError):
            alf.xyz

    def test_error_bound_defers_to_the_decode_for_overflow(self, tmp_path):
        """A positive finite scale still yields Inf at a stored 32767, so
        enumerating the decode's rules here missed a case for the third time.
        The preflight now runs the decode instead of re-deriving it."""
        hb = json.dumps(
            {
                "_frame_convention": "world-aligned",
                "_geom_codec": "int16",
                "_geom_scale": 1e308,
                "_geom_offset": [0, 0, 0],
            },
            separators=(",", ":"),
        ).encode()
        path = tmp_path / "000000.alp"
        payload = np.array([32767, 0, 0], dtype="<i2").tobytes()
        path.write_bytes(struct.pack("B", 3) + struct.pack("<I", len(hb)) + hb + payload)
        alf = read_alf(path)
        with pytest.raises(AlfError):
            alf.xyz
        with pytest.raises(AlfError):
            alf.max_geom_error_m

    def test_boolean_geometry_is_refused(self, tmp_path):
        """astype(float64) turns a mask into 0.0/1.0 COORDINATES, and geometry
        is the channel every other stream is indexed against."""
        with pytest.raises(ValueError, match="not booleans"):
            write_alp(tmp_path / "a.alp.gz", np.zeros((3, 3), dtype=bool))

    def test_duplicate_kept_indices_are_refused(self, tmp_path):
        """Two labelled points would address the same raw point while another
        kept point has no mapping at all."""
        with pytest.raises(ValueError, match="duplicates"):
            write_alx(
                tmp_path / "a.alx.gz",
                np.array([5, 5]),
                alp_total_points=2,
                source_total_points=10,
            )
        # reordered subsets remain legal
        path = tmp_path / "b.alx.gz"
        write_alx(path, np.array([9, 0, 4]), alp_total_points=3, source_total_points=10)
        assert read_alf(path).kept_indices.tolist() == [9, 0, 4]

    def test_verified_flag_validates_the_format_version(self, tmp_path):
        """It is the documented preflight for whether the layout is
        authoritative, so it must not be the one path trusting a corrupt
        header."""
        path = tmp_path / "a.all.gz"
        write_all(path, np.array([1, 2, 3], dtype=np.uint8))
        for bad in (True, 1.9, "1"):
            _rewrite_metadata(path, lambda m, v=bad: m.__setitem__("_format_version", v))
            with pytest.raises(AlfError):
                read_alf(path).format_version_verified

    def test_xyz_refuses_the_wrong_channel(self, tmp_path):
        """Six intensity bytes would otherwise decode as one plausible fp16
        point, and world_xyz() inherits the same wrong-channel result."""
        path = tmp_path / "a.ali.gz"
        write_ali(path, np.arange(6, dtype=np.float64), lo=0.0, hi=5.0)
        alf = read_alf(path)
        assert alf.intensity.size == 6
        with pytest.raises(AlfError, match="belongs to"):
            alf.xyz
        with pytest.raises(AlfError, match="belongs to"):
            alf.world_xyz(sensor_pose=True)

    def test_bytes_input_requires_an_explicit_channel(self, tmp_path):
        """An empty extension disables EVERY channel guard, so a raw .all blob
        could be read through .xyz."""
        path = tmp_path / "a.all.gz"
        write_all(path, np.array([1, 2, 3, 4, 5, 6], dtype=np.uint8))
        blob = path.read_bytes()
        with pytest.raises(AlfError, match="needs an explicit `extension=`"):
            read_alf(blob)
        # with the channel named, it reads correctly and the guards work
        alf = read_alf(blob, extension="all")
        assert alf.categories.tolist() == [1, 2, 3, 4, 5, 6]
        with pytest.raises(AlfError, match="belongs to"):
            alf.xyz

    def test_alc_without_a_palette_is_refused(self, tmp_path):
        """.alc bytes are only offsets into the header palette, so without it
        the indices cannot be resolved or even range-checked."""
        path = tmp_path / "a.alc.gz"
        write_alc(path, np.array([0], dtype=np.uint8), ["ff0000"])
        _rewrite_metadata(path, lambda m: m.pop("palette"))
        with pytest.raises(AlfError, match="no `palette`"):
            read_alf(path).palette_indices

    def test_boolean_geom_scale_is_refused(self, tmp_path):
        """float(True) is 1.0, so a boolean scale decodes every stored int16 as
        one metre and reports a confident 0.5 m bound."""
        path = tmp_path / "a.alp.gz"
        write_alp(path, np.array([[1.0, 2.0, 3.0]]), codec=GEOM_INT16, frame_convention=FRAME_WORLD_ALIGNED)
        _rewrite_metadata(path, lambda m: m.__setitem__("_geom_scale", True))
        with pytest.raises(AlfError, match="must be a JSON number"):
            read_alf(path).xyz

    def test_boolean_alp_count_is_refused(self, tmp_path):
        """True == 1, so a one-entry sidecar would pair with any cloud."""
        with pytest.raises(ValueError, match="alp_total_points must be a number"):
            write_alx(
                tmp_path / "a.alx.gz",
                np.array([0]),
                alp_total_points=True,
                source_total_points=10,
            )

    def test_non_hex_palette_entry_is_refused_on_read(self, tmp_path):
        """ "zzzzzz" passes a length check and then range-checks indices against
        a palette the viewer cannot parse."""
        path = tmp_path / "a.alc.gz"
        write_alc(path, np.array([0], dtype=np.uint8), ["ff0000"])
        _rewrite_metadata(path, lambda m: m.__setitem__("palette", ["zzzzzz"]))
        with pytest.raises(AlfError, match="hexadecimal"):
            read_alf(path).palette_indices

    def test_over_prefixed_palette_entry_is_refused_by_the_writer_too(self, tmp_path):
        """Same `lstrip` bug on the write side — it would emit the unparseable
        colour rather than merely pass one through."""
        with pytest.raises(ValueError, match="six hex digits"):
            write_alc(tmp_path / "a.alc.gz", np.array([0], dtype=np.uint8), ["##ff0000"])

    def test_palette_is_returned_normalized(self, tmp_path):
        """The accessor documents `rrggbb`, so it must return `rrggbb`. Handing
        back "#00FF00" verbatim breaks a consumer that prepends '#' or calls
        bytes.fromhex on it."""
        path = tmp_path / "a.alc.gz"
        write_alc(path, np.array([0, 1], dtype=np.uint8), ["ff0000", "00ff00"])
        _rewrite_metadata(path, lambda m: m.__setitem__("palette", ["#00FF00", "AABBCC"]))
        assert read_alf(path).palette == ["00ff00", "aabbcc"]

    def test_over_prefixed_palette_entry_is_refused(self, tmp_path):
        """`lstrip("#")` strips ANY number of leading hashes, so "##ff0000"
        validated clean and reached the viewer as an unparseable colour."""
        path = tmp_path / "a.alc.gz"
        write_alc(path, np.array([0], dtype=np.uint8), ["ff0000"])
        _rewrite_metadata(path, lambda m: m.__setitem__("palette", ["##ff0000"]))
        with pytest.raises(AlfError, match="hexadecimal"):
            read_alf(path).palette

    def test_byte_source_is_not_embedded_in_error_messages(self):
        """These frames run to hundreds of MB; interpolating one allocates and
        logs the whole payload while the caller quarantines a single frame."""
        hb = json.dumps({"_total_points": 99}, separators=(",", ":")).encode()
        blob = struct.pack("B", 1) + struct.pack("<I", len(hb)) + hb + bytes(60_000)
        with pytest.raises(AlfError) as exc:
            read_alf(blob, extension="alp")
        message = str(exc.value)
        assert "<bytes of" in message
        assert len(message) < 500, f"error message embedded the payload: {len(message)} chars"

    def test_alx_without_a_source_count_is_refused(self, tmp_path):
        """Without it the range check does not run at all, so a same-length
        sidecar sails through and an export indexes the raw cloud unvalidated."""
        path = tmp_path / "a.alx.gz"
        write_alx(path, np.array([0, 1]), alp_total_points=2, source_total_points=10)
        _rewrite_metadata(path, lambda m: m.pop("_source_total_points"))
        with pytest.raises(AlfError, match="missing `_source_total_points`"):
            read_alf(path).kept_indices

    def test_truncated_alx_payload_is_an_alf_error(self, tmp_path):
        """A 6-byte payload is neither a whole number of 4-byte indices nor the
        declared count's length. Both guards raise AlfError; the declared-count
        one runs first and names the mismatch more precisely."""
        hb = json.dumps({"_total_points": 2, "_source_total_points": 10}, separators=(",", ":")).encode()
        path = tmp_path / "000000.alx"
        path.write_bytes(struct.pack("B", 1) + struct.pack("<I", len(hb)) + hb + bytes(6))
        with pytest.raises(AlfError, match="payload holds"):
            read_alf(path)

    def test_alx_without_a_declared_count_is_refused(self, tmp_path):
        """Without it the one-mapping-per-point check silently does not run,
        leaving a one-entry sidecar pairable with a full cloud."""
        hb = json.dumps({"_source_total_points": 10}, separators=(",", ":")).encode()
        path = tmp_path / "000000.alx"
        path.write_bytes(struct.pack("B", 1) + struct.pack("<I", len(hb)) + hb + bytes(8))
        with pytest.raises(AlfError, match="missing `_total_points`"):
            read_alf(path)

    def test_non_numeric_geom_offset_is_an_alf_error(self, tmp_path):
        path = tmp_path / "a.alp.gz"
        write_alp(path, np.array([[1.0, 2.0, 3.0]]), codec=GEOM_INT16, frame_convention=FRAME_WORLD_ALIGNED)
        _rewrite_metadata(path, lambda m: m.__setitem__("_geom_offset", ["bad", 0, 0]))
        with pytest.raises(AlfError, match="non-numeric component"):
            read_alf(path).xyz

    def test_replacing_the_anchor_drops_a_stale_centroid(self, tmp_path):
        """world_centroid_epsg3857 OUTRANKS device_position, so carrying it
        through would leave the frame on the old aerial origin despite the
        explicit replacement."""
        first = tmp_path / "a.alp.gz"
        write_alp(
            first,
            np.array([[1.0, 2.0, 3.0]]),
            {"world_centroid_epsg3857": {"x": -13_600_000.0, "y": 4_500_000.0, "z": 12.0}},
        )
        inherited = dict(read_alf(first).metadata)
        second = tmp_path / "b.alp.gz"
        write_alp(
            second,
            np.array([[1.0, 2.0, 3.0]]),
            metadata=inherited,
            device_position=(100.0, 200.0, 5.0),
        )
        alf = read_alf(second)
        assert "world_centroid_epsg3857" not in alf.metadata
        assert alf.world_xyz(sensor_pose=True)[0] == pytest.approx([101.0, 202.0, 8.0])

    def test_source_count_domain_is_validated_like_total_points(self, tmp_path):
        """It is used as a count before the range check, so it needs the same
        domain — the three languages coerce, throw and truncate differently."""
        for bad in (-1, 1.9, True, "12"):
            path = tmp_path / "a.alx.gz"
            write_alx(path, np.array([0, 1]), alp_total_points=2, source_total_points=10)
            _rewrite_metadata(path, lambda m, v=bad: m.__setitem__("_source_total_points", v))
            with pytest.raises(AlfError, match="_source_total_points"):
                read_alf(path).kept_indices

    def test_alx_out_of_range_index_is_refused_on_read(self, tmp_path):
        """The writer records _source_total_points so a reader can re-check it.
        Downstream nothing catches this: the export addresses the raw cloud with
        the index and reads the wrong point."""
        path = tmp_path / "a.alx.gz"
        write_alx(path, np.array([0, 1, 2]), alp_total_points=3, source_total_points=10)
        raw = bytearray(gzip.decompress(path.read_bytes()))
        raw[-4:] = (999).to_bytes(4, "little")
        path.write_bytes(gzip.compress(bytes(raw)))
        with pytest.raises(AlfError, match="out of range for a source cloud"):
            read_alf(path).kept_indices

    def test_alx_index_beyond_the_source_cloud_is_refused(self, tmp_path):
        """Right length, wrong range: an export would read the
        authoritative raw data out of range or land on the wrong point."""
        with pytest.raises(ValueError, match="out of range for a source cloud"):
            write_alx(
                tmp_path / "a.alx.gz",
                np.array([0, 1, 999]),
                alp_total_points=3,
                source_total_points=100,
            )

    def test_alx_refuses_negative_indices(self, tmp_path):
        """A negative index wraps through uint32 into a real, wrong point."""
        with pytest.raises(ValueError, match="uint32"):
            write_alx(tmp_path / "a.alx.gz", np.array([0, -1]), alp_total_points=2, source_total_points=10)

    def test_corrupt_gzip_arrives_as_alf_error(self, tmp_path):
        """Truncation mid-transfer is how these objects actually go bad; a raw
        EOFError aborts a run that would otherwise quarantine the frame."""
        path = tmp_path / "a.alp.gz"
        write_alp(path, np.zeros((500, 3)))
        blob = path.read_bytes()
        path.write_bytes(blob[: len(blob) // 2])
        with pytest.raises(AlfError, match="Corrupt gzip"):
            read_alf(path)

    def test_fractional_total_points_is_rejected(self, tmp_path):
        """int() truncates 1.9 to 1, and a one-point payload would then satisfy
        the very cross-check that guards against length mismatch."""
        path = tmp_path / "a.alp.gz"
        write_alp(path, np.array([[1.0, 2.0, 3.0]]))
        _rewrite_metadata(path, lambda m: m.__setitem__("_total_points", 1.9))
        # Matches on "integer", which both the token check and the value check
        # share — the token check now fires first, and pinning either one's
        # exact wording makes this fail on a correct refusal.
        with pytest.raises(AlfError, match="integer"):
            read_alf(path)

    def test_incomplete_device_position_is_refused_for_sensor_pose(self, tmp_path):
        """Asserting device_position IS the world origin, then defaulting a
        missing axis to zero, is the same silent shift by a different door."""
        path = tmp_path / "a.alp.gz"
        write_alp(path, np.array([[1.0, 2.0, 3.0]]))
        _rewrite_metadata(path, lambda m: m.__setitem__("device_position", {"x": 100.0}))
        with pytest.raises(AlfError, match="missing required axis"):
            read_alf(path).world_xyz(sensor_pose=True)

    def test_out_of_range_deciles_are_refused(self, tmp_path):
        for bad in ([12.9] * 9, [300] * 9):
            with pytest.raises(ValueError, match=r"whole numbers|0\.\.255"):
                write_ali(
                    tmp_path / "a.ali.gz",
                    np.arange(10, dtype=np.float64),
                    lo=0.0,
                    hi=9.0,
                    deciles=bad,
                )

    def test_reserved_instance_id_65535_is_refused(self, tmp_path):
        """persistent_instance_registry.dart:25 caps valid local ids at 65534, so
        65535 clears every uint16 check and then loses its instance in the
        editor."""
        with pytest.raises(ValueError, match="uint16"):
            write_all(
                tmp_path / "a.all.gz",
                np.array([1, 2], dtype=np.uint8),
                instance_ids=np.array([1, 65535]),
                instance_map={"1": "u1", "65535": "u2"},
            )

    def test_top_of_the_valid_instance_range_still_writes(self, tmp_path):
        path = tmp_path / "a.all.gz"
        write_all(
            path,
            np.array([1, 2], dtype=np.uint8),
            instance_ids=np.array([1, 65534]),
            instance_map={"1": "u1", "65534": "u2"},
        )
        assert read_alf(path).instance_ids.tolist() == [1, 65534]

    def test_every_public_alf_entry_point_is_exported(self, tmp_path):
        """The package-level import convention only helps if it is complete —
        one omitted name and callers go back to reaching into the module."""
        import alf as core_pkg

        for name in (
            "read_alf",
            "iter_frames",
            "write_alp",
            "write_ali",
            "write_all",
            "write_alc",
            "write_alx",
            "geom_error_report",
            "AlfFile",
            "AlfError",
        ):
            assert hasattr(core_pkg, name), f"alf.{name} is not exported"
            assert name in dir(core_pkg), f"alf.{name} missing from dir()"
            # And in `__all__`. Building the list imports nothing — it is
            # strings — so the lazy boundary survives; only a star import
            # resolves the names.
            assert name in core_pkg.__all__, f"alf.{name} missing from __all__"

    def test_core_import_does_not_pull_numpy(self):
        """AGENTS.md pins this package to requests + PyYAML so the
        CSV/config/API helpers stay usable where NumPy is absent; alf_guard uses
        `math` alone for the same reason. An eager alf_io import would break
        `import core` outright for scripts that never touch lidar."""
        import subprocess
        import sys

        probe = (
            "import sys; import alf; "
            "assert 'numpy' not in sys.modules, 'alf eagerly imported numpy'; "
            "alf.read_alf; "
            "assert 'numpy' in sys.modules, 'lazy export did not resolve'"
        )
        env = dict(os.environ, PYTHONPATH=str(_PACKAGE_ROOT))
        assert subprocess.run([sys.executable, "-c", probe], env=env, check=False).returncode == 0

    def test_degenerate_fixed_point_scale_is_rejected(self, tmp_path):
        """Zero collapses the cloud onto the offset; negative mirrors it."""
        for bad in (0.0, -0.01):
            path = tmp_path / "a.alp.gz"
            write_alp(path, np.array([[1.0, 2.0, 3.0]]), codec=GEOM_INT16, frame_convention=FRAME_WORLD_ALIGNED)
            _rewrite_metadata(path, lambda m, v=bad: m.__setitem__("_geom_scale", v))
            with pytest.raises(AlfError, match="positive finite"):
                read_alf(path).xyz

    def test_fractional_format_version_is_not_truncated_to_v1(self, tmp_path):
        """`int(1.5)` is 1, which would sail through the known-layouts guard and
        decode an unknown payload as v1 categories."""
        path = tmp_path / "a.all.gz"
        write_all(path, np.array([1, 2, 3], dtype=np.uint8))
        _rewrite_metadata(path, lambda m: m.__setitem__("_format_version", 1.5))
        with pytest.raises(AlfError, match="must be an integer"):
            read_alf(path).categories

    def test_non_finite_fp16_coordinates_are_returned_on_read(self, tmp_path):
        """A NaN in float16 geometry is a valid invalid-return marker.

        The previous assertion incorrectly required a refusal for this content.
        """
        path = tmp_path / "a.alp.gz"
        write_alp(path, np.array([[1.0, 2.0, 3.0]]))
        raw = bytearray(gzip.decompress(path.read_bytes()))
        raw[-2:] = np.array([np.float16("nan")], dtype="<f2").tobytes()
        path.write_bytes(gzip.compress(bytes(raw)))
        points = read_alf(path).xyz
        assert points.shape == (1, 3)
        assert np.isnan(points[0, 2])
        assert points[0, :2].tolist() == [1.0, 2.0]

    def test_malformed_header_arrives_as_alf_error(self, tmp_path):
        """Callers quarantine bad frames by catching AlfError; a leaked
        JSONDecodeError aborts the run instead of skipping one file."""
        path = tmp_path / "a.alp.gz"
        write_alp(path, np.array([[1.0, 2.0, 3.0]]))
        raw = gzip.decompress(path.read_bytes())
        (mlen,) = struct.unpack("<I", raw[1:5])
        broken = raw[:5] + b"{not json" + b" " * (mlen - 9) + raw[5 + mlen :]
        path.write_bytes(gzip.compress(broken))
        with pytest.raises(AlfError, match="not valid JSON"):
            read_alf(path)

    def test_non_object_header_root_is_rejected(self, tmp_path):
        path = tmp_path / "a.alp.gz"
        write_alp(path, np.array([[1.0, 2.0, 3.0]]))
        raw = gzip.decompress(path.read_bytes())
        (mlen,) = struct.unpack("<I", raw[1:5])
        body = b"[1,2,3]"
        rebuilt = raw[:1] + struct.pack("<I", len(body)) + body + raw[5 + mlen :]
        path.write_bytes(gzip.compress(rebuilt))
        with pytest.raises(AlfError, match="must be a JSON object"):
            read_alf(path)

    def test_column_vector_intensity_is_accepted(self, tmp_path):
        """(N, 1) is what a caller has after slicing one field out of a
        structured array; np.sort would stay 2-D and int(ordered[i]) would raise."""
        path = tmp_path / "a.ali.gz"
        write_ali(path, np.arange(10, dtype=np.float64).reshape(-1, 1), lo=0.0, hi=9.0)
        assert read_alf(path).intensity.size == 10

    def test_accessors_refuse_the_wrong_channel(self, tmp_path):
        """`.categories` off an `.ali` returns intensity readings as class ids —
        same bytes, plausible values, no error."""
        ali = tmp_path / "a.ali.gz"
        write_ali(ali, np.arange(6, dtype=np.float64), lo=0.0, hi=5.0)
        alf = read_alf(ali)
        assert alf.intensity.size == 6  # the right accessor still works
        with pytest.raises(AlfError, match="belongs to"):
            alf.categories
        with pytest.raises(AlfError, match="belongs to"):
            alf.instance_ids

    def test_reversed_intensity_bounds_raise(self):
        with pytest.raises(ValueError, match="reversed"):
            map_intensity(np.array([10.0, 20.0, 30.0]), lo=100.0, hi=0.0)

    def test_constant_intensity_still_zero_fills(self):
        out = map_intensity(np.array([7.0, 7.0]), lo=7.0, hi=7.0)
        assert out.tolist() == [0, 0]

    def test_panoptic_streams_decode_once(self, tmp_path):
        """Reading both v2 streams must not decode the payload twice."""
        path = tmp_path / "a.all.gz"
        cats = np.array([1, 2, 3], dtype=np.uint8)
        ids = np.array([0, 255, 65534], dtype=np.uint16)
        write_all(path, cats, instance_ids=ids, instance_map={"255": "u1", "65534": "u2"})
        alf = read_alf(path)
        calls = []
        original = alf._decode_labels_v2
        alf._decode_labels_v2 = lambda: (calls.append(1), original())[1]
        assert alf.categories.tolist() == [1, 2, 3]
        assert alf.instance_ids.tolist() == [0, 255, 65534]
        assert alf.categories.tolist() == [1, 2, 3]
        assert len(calls) == 1

    def test_v3_offset_must_have_three_components(self, tmp_path):
        """A scalar `_geom_offset` broadcasts across all axes and returns a
        uniformly shifted cloud rather than an error."""
        path = tmp_path / "a.alp.gz"
        write_alp(path, np.array([[1.0, 2.0, 3.0]]), codec=GEOM_INT16, frame_convention=FRAME_WORLD_ALIGNED)
        _rewrite_metadata(path, lambda m: m.__setitem__("_geom_offset", 5.0))
        with pytest.raises(AlfError, match="3 components"):
            read_alf(path).xyz

    def test_v3_offset_must_be_finite(self, tmp_path):
        path = tmp_path / "a.alp.gz"
        write_alp(path, np.array([[1.0, 2.0, 3.0]]), codec=GEOM_INT16, frame_convention=FRAME_WORLD_ALIGNED)
        _rewrite_metadata(path, lambda m: m.__setitem__("_geom_offset", [float("inf"), 0.0, 0.0]))
        # Caught at the PARSE now, which is strictly stronger: `Infinity` is not
        # JSON at all, so no header can carry it regardless of which key it lands
        # in. The per-key finiteness check remains as defence in depth for an
        # AlfFile constructed programmatically rather than read from bytes.
        with pytest.raises(AlfError, match="non-JSON literal"):
            read_alf(path).xyz


class TestBoundedReadAndPromotionHoles:
    """Three ways a guard that already existed failed to cover its own case.

    Each is the same shape: a rule written for one direction, one container or
    one branch, and the mirror case walking straight past it — silently, which
    is the only failure mode this module treats as unacceptable.
    """

    def _legacy_channel(self, path, payload):
        """An uncompressed `.ali` with NO `_total_points` — the live label-diff
        shape. Any payload length is structurally valid, so a dropped byte has
        nothing downstream to catch it."""
        header = b"{}"
        path.write_bytes(struct.pack("B", 1) + struct.pack("<I", len(header)) + header + payload)
        return path

    def test_shrinkage_during_the_bounded_read_is_refused(self, tmp_path, monkeypatch):
        """The growth check was fixed and the mirror case was not.

        `read(size + 1)` catches a file that GREW past the `fstat` snapshot, but
        `len(raw) < size` — the file truncated mid-transfer, which is exactly how
        these objects go bad — passed unnoticed. A legacy channel without
        `_total_points` accepts any payload length, so 100 intensity samples
        parsed as 60 and every index-correlated channel behind them shifted,
        with nothing raised."""
        path = self._legacy_channel(tmp_path / "a.ali", bytes(range(100)))
        assert len(read_alf(path, extension="ali").intensity) == 100

        full = path.stat().st_size
        real_fstat = alf_io_module.os.fstat

        class Snapshot:
            """The size from BEFORE the truncation, with the real mode."""

            def __init__(self, stat_result):
                self.st_size = stat_result.st_size
                self.st_mode = stat_result.st_mode

        def truncating_fstat(fd):
            stat_result = real_fstat(fd)
            # The file loses 40 bytes in the window between the size being
            # observed and the payload being read.
            with open(path, "r+b") as handle:
                handle.truncate(full - 40)
            return Snapshot(stat_result)

        monkeypatch.setattr(alf_io_module.os, "fstat", truncating_fstat)
        with pytest.raises(AlfError, match="shrank while being read"):
            read_alf(path, extension="ali")

    def test_advisory_size_sources_are_not_held_to_the_strict_length(self, tmp_path):
        """`st_size` is only a byte count for a REGULAR file.

        A FIFO reports 0 no matter how much it delivers, so the strict rule
        cannot apply — and it never reached that rule anyway: the `seek(0)` after
        the magic-byte peek raised `io.UnsupportedOperation`, which is not the
        `AlfError` a pipeline quarantines on. Same contract leak as the
        memoryview cast, one branch over."""
        import threading

        payload = bytes(range(100))
        header = b"{}"
        body = struct.pack("B", 1) + struct.pack("<I", len(header)) + header + payload
        fifo = tmp_path / "channel.ali"
        os.mkfifo(fifo)
        threading.Thread(target=lambda: fifo.write_bytes(body), daemon=True).start()

        assert len(read_alf(fifo, extension="ali").intensity) == 100

    def test_non_contiguous_memoryviews_raise_alf_error_not_type_error(self):
        """`cast()` is restricted to C-contiguous views and raises a bare
        `TypeError` otherwise.

        The magic-byte probe called `probe.cast("B")` BEFORE the contiguity
        guard, and the guard itself sat inside an `itemsize != 1 or ndim != 1`
        branch that a strided byte view does not enter — so the rejection was
        unreachable for exactly the views that needed it. A pipeline whose
        quarantine is `except AlfError: skip_frame()` tore down the whole run."""
        buffer = bytearray(b"\x1f\x8b" + b"x" * 200)
        with pytest.raises(AlfError, match="contiguous byte view"):
            read_alf(memoryview(buffer)[::2], extension="ali")

        # FORTRAN-contiguous, not merely "non-contiguous". `memoryview.contiguous`
        # is True for these while `cast()` still refuses them, so a guard written
        # with the looser predicate leaked the identical TypeError.
        transposed = np.zeros((4, 4), dtype=np.uint8).T
        with pytest.raises(AlfError, match="contiguous byte view"):
            read_alf(memoryview(transposed), extension="ali")

    def test_contiguous_views_still_read(self, tmp_path):
        """The guard must not have been bought by refusing valid input: a plain
        buffer, a widened cast and a C-contiguous 2-D view all still parse."""
        path = tmp_path / "a.ali.gz"
        write_ali(path, np.array([10.0, 200.0]), lo=0.0, hi=255.0)
        raw = path.read_bytes()
        padded = raw + b"\x00" * ((4 - len(raw) % 4) % 4)

        assert read_alf(memoryview(raw), extension="ali").intensity.tolist() == [10, 200]
        assert read_alf(memoryview(bytearray(padded)).cast("I"), extension="ali").intensity.tolist() == [10, 200]
        rows = np.frombuffer(padded, dtype=np.uint8).reshape(1, -1)
        assert read_alf(memoryview(rows), extension="ali").intensity.tolist() == [10, 200]

    def test_mixed_python_sequences_are_checked_before_numpy_promotes_them(self, tmp_path):
        """`np.asarray` promotes a MIXED sequence before the precision guard can
        look at it.

        `[[2**53 + 1, 0.5, 0.0]]` is float64 with x already rounded to `2**53`
        by the time `_narrows_lossily_to_float64` runs, so the point published a
        metre out — while the equivalent `np.array(..., dtype=np.int64)` was
        correctly refused. The guard disagreed with itself depending on whether
        the caller had already built an array."""
        with pytest.raises(ValueError, match="cannot narrow to float64"):
            write_alp(
                tmp_path / "a.alp.gz", [[2**53 + 1, 0.5, 0.0]], codec=GEOM_INT16, frame_convention=FRAME_WORLD_ALIGNED
            )
        with pytest.raises(ValueError, match="cannot narrow to float64"):
            write_alp(tmp_path / "b.alp.gz", ((2**53 + 1, 0.5, 0.0),), codec=GEOM_FP16)

        # Every per-point channel has the identical hole, and each is correlated
        # with the cloud BY INDEX — so a value that shifted during coercion is as
        # wrong there as a coordinate is.
        with pytest.raises(ValueError, match="cannot narrow to float64"):
            write_ali(tmp_path / "c.ali.gz", [2**53 + 1, 0.5], lo=0.0, hi=2.0**54)

        # An integer too large to become a float at all overflows rather than
        # comparing, which is the most lossy case there is.
        with pytest.raises(ValueError, match="cannot narrow to float64"):
            write_alp(
                tmp_path / "d.alp.gz", [[10**400, 0.5, 0.0]], codec=GEOM_INT16, frame_convention=FRAME_WORLD_ALIGNED
            )

    def test_the_promotion_guard_agrees_across_containers(self, tmp_path):
        """The VALUE decides, not the magnitude, and the container must not.

        `2**53` itself is exactly representable and so is every multiple up to
        the next binade — the native-dtype branch was corrected for that once
        already. Re-deriving a magnitude test for the raw-sequence case would
        have refused frames that round-trip perfectly, and left a list
        disagreeing with the object array holding the same numbers."""
        exact = [[float(2**53), 0.5, 0.0]]
        containers = {
            "list": [[2**53, 0.5, 0.0]],
            "int64": np.array([[2**53, 0, 0]], dtype=np.int64),
            "object": np.array([[2**53, 0, 0]], dtype=object),
            "float64": np.array(exact),
        }
        for name, value in containers.items():
            write_alp(tmp_path / f"ok-{name}.alp.gz", value, codec=GEOM_INT16, frame_convention=FRAME_WORLD_ALIGNED)

        lossy = {
            "list": [[2**53 + 1, 0.0, 0.0]],
            "int64": np.array([[2**53 + 1, 0, 0]], dtype=np.int64),
            "object": np.array([[2**53 + 1, 0, 0]], dtype=object),
        }
        for name, value in lossy.items():
            with pytest.raises(ValueError, match="cannot narrow to float64"):
                write_alp(
                    tmp_path / f"bad-{name}.alp.gz", value, codec=GEOM_INT16, frame_convention=FRAME_WORLD_ALIGNED
                )


class _CountingStream:
    """A forward-only, NON-SEEKABLE reader that records how much was consumed.

    No `seek`/`tell` at all, deliberately: if the header path ever reaches for
    one it fails here rather than silently working on files and breaking on the
    sockets and `botocore` bodies this exists for.
    """

    def __init__(self, payload: bytes):
        self._buffer = io.BytesIO(payload)
        self.bytes_read = 0

    def read(self, size: int = -1) -> bytes:
        chunk = self._buffer.read(size)
        self.bytes_read += len(chunk)
        return chunk


def _gzip_with_oversized_payload(path, header_metadata, payload_bytes):
    """A real gzip ALF whose payload expands past `MAX_DECOMPRESSED_BYTES`.

    Written INCREMENTALLY, so the test does not need 600 MB of its own to prove
    that the reader does not need it either.
    """
    header = json.dumps(header_metadata, separators=(",", ":")).encode("utf-8")
    envelope = struct.pack("B", 1) + struct.pack("<I", len(header)) + header
    engine = zlib.compressobj(9, zlib.DEFLATED, 31)
    block = bytes(8 * 1024 * 1024)
    with open(path, "wb") as handle:
        handle.write(engine.compress(envelope))
        for _ in range(payload_bytes // len(block)):
            handle.write(engine.compress(block))
        handle.write(engine.flush())
    return path


class TestHeaderOnlyRead:
    """`read_alf_header` must be cheap, and cheap in the way it claims.

    The point of this entry point is a RESOURCE property, so these assert the
    resource property directly. A test that only checked the returned metadata
    would pass identically on the full-read path and prove nothing at all.
    """

    def _frame(self, tmp_path, name="a.alp.gz", **kwargs):
        path = tmp_path / name
        write_alp(path, np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), {"seq": "s1"}, **kwargs)
        return path

    def test_a_malformed_count_is_refused_here_and_not_at_an_accessor(self):
        """`_total_points` is decidable from the header alone, so refusing it is
        a header-scoped rule and this function claims to enforce every one.

        It did not. The check lived in the `total_points` property, so a
        malformed count sailed through a scan and raised later, from an
        accessor, on whichever frame someone happened to look at closely — and
        the TypeScript reader refused the same bytes at the same call, which is
        a conformance split between two implementations the support matrix
        calls equal. Found while building the Dart entry point against both.
        """
        for bad in (None, -1, 1.9, "4", True):
            header = json.dumps({"_total_points": bad}, separators=(",", ":")).encode()
            raw = struct.pack("B", 1) + struct.pack("<I", len(header)) + header
            with pytest.raises(AlfError):
                read_alf_header(raw, extension="alp")
            # The full reader refuses it too — a header-only read that accepted
            # what `read_alf` rejects is the drift this pins.
            with pytest.raises(AlfError):
                read_alf(raw, extension="alp")

    def test_channel_scoped_header_rules_are_enforced_here_too(self):
        """The envelope rules apply to every channel; these depend on which one
        you opened, and they lived only inside the channel decoders — which a
        header-only read never reaches. So a scan accepted `.alx` files with no
        declared count and `.all` files naming a label layout that does not
        exist, both of which `read_alf` refuses and both of which are decidable
        from the header alone. Every implementation had this gap."""
        cases = [
            ({"_source_total_points": 10}, b"\x00\x00\x00\x00", "alx"),
            ({"_total_points": 1, "_format_version": 9}, b"\x01", "all"),
        ]
        for header, payload, extension in cases:
            encoded = json.dumps(header, separators=(",", ":")).encode()
            raw = struct.pack("B", 1) + struct.pack("<I", len(encoded)) + encoded + payload
            with pytest.raises(AlfError):
                read_alf(raw, extension=extension)
            with pytest.raises(AlfError):
                read_alf_header(raw, extension=extension)

    def test_a_lazily_refused_rule_is_not_promoted_into_the_header_read(self):
        """`.alx` without `_source_total_points` is refused when its indices are
        DECODED, not when the file is read. A header-only read reaches neither,
        and adding it here would make a scan stricter than the read it stands in
        for — the opposite error, and just as wrong."""
        header = json.dumps({"_total_points": 1}, separators=(",", ":")).encode()
        raw = struct.pack("B", 1) + struct.pack("<I", len(header)) + header + b"\x00\x00\x00\x00"
        read_alf(raw, extension="alx")  # accepted at the read
        read_alf_header(raw, extension="alx")  # so accepted here as well

    def test_an_absent_count_is_still_absent_rather_than_an_error(self):
        """Only .alx requires the field. Eager validation must not turn "the
        header omits it" into a refusal for the four channels that may."""
        header = json.dumps({"seq": "s1"}, separators=(",", ":")).encode()
        raw = struct.pack("B", 1) + struct.pack("<I", len(header)) + header
        assert read_alf_header(raw, extension="alp").total_points is None

    def test_payload_is_never_decompressed(self, tmp_path):
        """The strongest available proof: a file the FULL reader cannot read.

        A payload past `MAX_DECOMPRESSED_BYTES` is one `read_alf` refuses by
        construction, so a header read that succeeds on the same bytes cannot
        have inflated it — there is no passing this test by decompressing and
        being quick about it.

        Sized relative to the cap so a future budget increase cannot make the
        fixture small enough to pass full decompression. The failure must come
        from the resource limit, not an unrelated payload-length check."""
        oversized = alf_io_module.MAX_DECOMPRESSED_BYTES + 64 * 1024 * 1024
        path = _gzip_with_oversized_payload(tmp_path / "huge.ali.gz", {"_total_points": oversized}, oversized)
        assert path.stat().st_size < 4 * 1024 * 1024  # a compressed run of zeros

        with pytest.raises(AlfError, match="expands past the"):
            read_alf(path)

        header = read_alf_header(path)
        assert header.total_points == oversized
        assert header.header_only is True

        # And the COMPRESSED side is bounded too: one block, not the whole object.
        stream = _CountingStream(path.read_bytes())
        assert read_alf_header(stream, extension="ali").total_points == oversized
        assert stream.bytes_read <= alf_io_module._HEADER_PREFIX_CHUNK_BYTES

    def test_the_decompressor_is_never_asked_for_more_than_the_envelope(self, monkeypatch, tmp_path):
        """The early stop bounds how much INPUT is fed; `max_length` bounds how
        much OUTPUT a single call may produce. Without it one 64 KB block of
        compressed zeros inflates to ~64 MB before the loop can notice it has
        enough — bounded, but not the "few KB per frame" this promises. White-box
        on purpose: the bound is the feature, so it is asserted, not inferred."""
        requested = []
        real = zlib.decompressobj

        class Recording:
            def __init__(self, engine):
                self._engine = engine

            def decompress(self, data, max_length=0):
                requested.append(max_length)
                return self._engine.decompress(data, max_length)

            def __getattr__(self, name):
                return getattr(self._engine, name)

        monkeypatch.setattr(alf_io_module.zlib, "decompressobj", lambda **kw: Recording(real(**kw)))
        path = _gzip_with_oversized_payload(
            tmp_path / "bounded.ali.gz", {"_total_points": 600_000_000}, 64 * 1024 * 1024
        )
        assert read_alf_header(path).total_points == 600_000_000
        assert requested, "the gzip branch was never exercised"
        # POSITIVE on every call. zlib spells "unbounded" as max_length=0, so an
        # assertion on the maximum alone passes when the bound is removed — the
        # recorded value goes DOWN to 0, not up. Every call must carry a real
        # cap, and that cap must be the envelope's own size.
        assert all(0 < asked <= 5 + alf_io_module._MAX_HEADER_BYTES for asked in requested), requested
        assert max(requested) < 1024

    def test_reads_only_a_few_kilobytes_of_a_large_stream(self, tmp_path):
        """The byte-level proof, on the non-seekable source this exists for."""
        path = tmp_path / "big.ali"
        payload = bytes(4 * 1024 * 1024)
        header = json.dumps({"_total_points": len(payload)}, separators=(",", ":")).encode()
        path.write_bytes(struct.pack("B", 1) + struct.pack("<I", len(header)) + header + payload)

        stream = _CountingStream(path.read_bytes())
        assert read_alf_header(stream, extension="ali").total_points == len(payload)
        # One 64 KB block covers any real envelope; the file is 4 MB.
        assert stream.bytes_read <= alf_io_module._HEADER_PREFIX_CHUNK_BYTES

        whole = _CountingStream(path.read_bytes())
        assert read_alf(whole, extension="ali").total_points == len(payload)
        assert whole.bytes_read == path.stat().st_size

    def test_every_source_kind_agrees_with_the_full_read(self, tmp_path):
        """Path, bytes, memoryview and stream — compressed and not."""
        gz = self._frame(tmp_path, device_position=[10.0, 20.0, 30.0])
        plain = tmp_path / "b.ali"
        write_ali(plain, np.array([1.0, 2.0, 3.0]), lo=0.0, hi=3.0)

        for path, extension in ((gz, "alp"), (plain, "ali")):
            raw = path.read_bytes()
            expected = read_alf(path)
            for source in (path, raw, bytearray(raw), memoryview(raw), _CountingStream(raw)):
                got = read_alf_header(source, extension=extension)
                assert got.version == expected.version
                assert got.metadata == expected.metadata
                assert got.total_points == expected.total_points

        assert read_alf_header(gz).device_position.tolist() == [10.0, 20.0, 30.0]

    def test_payload_accessors_refuse_instead_of_answering(self, tmp_path):
        """Empty-payload substitution would make every one of these LIE."""
        gz = self._frame(tmp_path)
        header = read_alf_header(gz)
        for name in ("xyz", "max_geom_error_m"):
            with pytest.raises(AlfError, match="never decompresses the payload"):
                getattr(header, name)
        with pytest.raises(AlfError, match="never decompresses the payload"):
            header.world_xyz(sensor_pose=True)

        labels = tmp_path / "c.all.gz"
        write_all(labels, np.array([1, 2, 3], dtype=np.uint8))
        for name in ("categories", "instance_ids"):
            with pytest.raises(AlfError, match="never decompresses the payload"):
                getattr(read_alf_header(labels), name)

        colours = tmp_path / "d.alc.gz"
        write_alc(colours, np.array([0, 0], dtype=np.uint8), ["ff0000"])
        with pytest.raises(AlfError, match="never decompresses the payload"):
            read_alf_header(colours).palette_indices

        indexes = tmp_path / "e.alx.gz"
        write_alx(indexes, np.array([0, 2], dtype=np.uint32), alp_total_points=2, source_total_points=3)
        with pytest.raises(AlfError, match="never decompresses the payload"):
            read_alf_header(indexes).kept_indices

        intensities = tmp_path / "f.ali.gz"
        write_ali(intensities, np.array([1.0, 2.0]), lo=0.0, hi=2.0)
        with pytest.raises(AlfError, match="never decompresses the payload"):
            read_alf_header(intensities).intensity

    def test_header_derived_accessors_still_work(self, tmp_path):
        """The guard must not have been bought by refusing the whole API."""
        path = tmp_path / "g.alp.gz"
        write_alp(path, np.array([[1.0, 2.0, 3.0]]), codec=GEOM_INT16, frame_convention=FRAME_WORLD_ALIGNED)
        header = read_alf_header(path)
        assert header.geom_codec == GEOM_INT16
        assert header.total_points == 1
        assert header.version == 3
        assert isinstance(header.metadata["_geom_scale"], float)

        labels = tmp_path / "h.all.gz"
        write_all(labels, np.array([1, 2], dtype=np.uint8))
        # The header answers on its own here, so the sentinel is never consulted.
        assert read_alf_header(labels).format_version == 1

    def test_a_payload_dependent_header_field_refuses_rather_than_guessing(self, tmp_path):
        """`format_version` falls back to the payload STRIDE when the wire byte
        and `_format_version` disagree. With no payload read there is no stride,
        and an empty stand-in would "measure" zero bytes and pick a layout —
        so the sentinel has to make it raise."""
        header = json.dumps({"_format_version": 2, "_total_points": 2}, separators=(",", ":")).encode()
        body = struct.pack("B", 1) + struct.pack("<I", len(header)) + header + bytes(6)
        path = tmp_path / "i.all"
        path.write_bytes(body)

        assert read_alf(path).format_version == 2  # resolved by the 3-byte stride
        with pytest.raises(AlfError, match="never decompresses the payload"):
            read_alf_header(path).format_version

    def test_header_scoped_refusals_are_not_bypassed(self, tmp_path):
        """A header-only read must not become a way around header validation."""

        def envelope(header_bytes, version=1, payload=b""):
            return struct.pack("B", version) + struct.pack("<I", len(header_bytes)) + header_bytes + payload

        cases = {
            "bad-version": (envelope(b"{}", version=9), "ali", "Unsupported ALF version byte"),
            "v3-on-ali": (envelope(b"{}", version=3), "ali", "fixed-point"),
            "v2-on-alp": (envelope(b"{}", version=2), "alp", "interleaved label layout"),
            "not-json": (envelope(b"{oops"), "ali", "not valid JSON"),
            "not-utf8": (envelope(b'{"k":"\xff\xfe"}'), "ali", "not valid UTF-8"),
            "nan-literal": (envelope(b'{"k":NaN}'), "ali", "non-JSON literal"),
            "duplicate-key": (envelope(b'{"k":1,"k":2}'), "ali", "duplicate JSON object key"),
            "root-not-object": (envelope(b"[1,2]"), "ali", "must be a JSON object"),
            "too-deep": (envelope(b"[" * 300 + b"]" * 300), "ali", "header-nesting-too-deep"),
            "huge-int": (envelope(b'{"k":' + b"1" * 400 + b"}"), "ali", "is not finite"),
            "short": (b"\x01\x02", "ali", "Too short to be ALF"),
            "truncated-header": (
                struct.pack("B", 1) + struct.pack("<I", 50) + b"{}",
                "ali",
                "Truncated header",
            ),
        }
        for name, (body, extension, message) in cases.items():
            path = tmp_path / f"{name}.{extension}"
            path.write_bytes(body)
            with pytest.raises(AlfError, match=message):
                read_alf_header(path)
            with pytest.raises(AlfError, match=message):
                read_alf_header(bytes(body), extension=extension)

        # `.alp` anchor validation is header-scoped too, and channel-specific.
        anchor = json.dumps({"device_position": [1, "x", 3]}, separators=(",", ":")).encode()
        bad_anchor = tmp_path / "anchor.alp"
        bad_anchor.write_bytes(envelope(anchor))
        with pytest.raises(AlfError, match="device_position"):
            read_alf_header(bad_anchor)

    def test_an_oversized_declared_header_is_refused_before_it_is_accumulated(self, tmp_path):
        """The uint32 length field can declare 4 GB. The full reader catches
        that after materializing the file; this path would otherwise sit in the
        read loop inflating toward it."""
        # With MEGABYTES behind the bad length. An envelope followed by nothing
        # hits EOF on the first block and would be refused by the shared parser
        # no matter what this loop did — so the guard has to be tested against a
        # source it could actually keep reading.
        path = tmp_path / "j.ali.gz"
        declared = struct.pack("B", 1) + struct.pack("<I", 3_000_000_000)
        # INCOMPRESSIBLE filler. 64 MB of zeros gzips to under one 64 KB block,
        # so the reader would consume the whole file in a single read and the
        # assertion below could not tell a bounded reader from an unbounded one.
        path.write_bytes(gzip.compress(declared + b"{}" + os.urandom(4 * 1024 * 1024)))
        assert path.stat().st_size > 8 * alf_io_module._HEADER_PREFIX_CHUNK_BYTES

        stream = _CountingStream(path.read_bytes())
        with pytest.raises(AlfError, match="Header is too large"):
            read_alf_header(stream, extension="ali")
        assert stream.bytes_read <= alf_io_module._HEADER_PREFIX_CHUNK_BYTES

    def test_a_stream_that_never_delivers_the_header_is_bounded(self, monkeypatch):
        """gzip's FNAME field is a NUL-terminated string with NO length limit, so
        a member can consume unbounded input while emitting zero output — the
        decompressor is still happily waiting for the filename to end. Without a
        cap on COMPRESSED bytes consumed, a header that never arrives reads the
        whole object, which is the exact cost this entry point exists to avoid.
        The output-side bounds cannot see this at all: nothing is ever output."""
        monkeypatch.setattr(alf_io_module, "MAX_DECOMPRESSED_BYTES", 1024 * 1024)
        # FLG=0x08 sets FNAME; the name then runs on without a terminating NUL.
        never_ends = b"\x1f\x8b\x08\x08\x00\x00\x00\x00\x00\x03" + b"a" * (3 * 1024 * 1024)
        assert len(never_ends) > alf_io_module._limit_for(True)

        stream = _CountingStream(never_ends)
        with pytest.raises(AlfError, match="without reaching the end of the ALF header"):
            read_alf_header(stream, extension="ali")
        # Stopped at the cap rather than draining the source.
        assert stream.bytes_read <= alf_io_module._limit_for(True) + alf_io_module._HEADER_PREFIX_CHUNK_BYTES

    def test_a_truncated_gzip_envelope_arrives_as_alf_error(self, tmp_path):
        """Mid-transfer truncation is how these objects go bad, and a caller's
        quarantine is `except AlfError`. A raw zlib.error would abort the run."""
        path = tmp_path / "k.alp.gz"
        write_alp(path, np.array([[1.0, 2.0, 3.0]]))
        whole = path.read_bytes()

        with pytest.raises(AlfError):
            read_alf_header(whole[: len(whole) // 2], extension="alp")
        # A stream that is CORRUPT rather than short: zlib raises from inside the
        # read, and that has to be translated. Byte 10 is the first deflate byte,
        # so the failure lands while the envelope is still being produced.
        corrupt = bytearray(whole)
        corrupt[10] ^= 0xFF
        with pytest.raises(AlfError, match="Corrupt gzip stream"):
            read_alf_header(bytes(corrupt), extension="alp")
        # A member that decodes cleanly but stops inside the declared header.
        head = json.dumps({"seq": "x" * 5000}, separators=(",", ":")).encode()
        body = struct.pack("B", 1) + struct.pack("<I", len(head)) + head[:100]
        with pytest.raises(AlfError, match="Truncated header"):
            read_alf_header(gzip.compress(body), extension="ali")

    def test_a_missing_path_is_not_diagnosed_as_a_corrupt_stream(self, tmp_path):
        """The gzip handler must scope to the INFLATE. Wrapped around the read
        loop it caught the open() as well, and reported a file that does not
        exist as "Corrupt gzip stream: No such file or directory" — a misleading
        diagnosis, and a divergence from `read_alf`, which lets it through."""
        missing = tmp_path / "gone.ali"
        with pytest.raises(FileNotFoundError):
            read_alf(missing)
        with pytest.raises(FileNotFoundError):
            read_alf_header(missing)

    def test_non_contiguous_buffers_are_refused_here_too(self):
        """The shared normalizer, so the rule cannot hold in one reader only."""
        buffer = bytearray(b"\x1f\x8b" + b"x" * 200)
        with pytest.raises(AlfError, match="contiguous byte view"):
            read_alf_header(memoryview(buffer)[::2], extension="ali")
        with pytest.raises(AlfError, match="contiguous byte view"):
            read_alf_header(memoryview(np.zeros((4, 4), dtype=np.uint8).T), extension="ali")

    def test_the_channel_rules_are_the_same_as_the_full_read(self, tmp_path):
        raw = gzip.compress(struct.pack("B", 1) + struct.pack("<I", 2) + b"{}")
        with pytest.raises(AlfError, match="needs an explicit `extension=`"):
            read_alf_header(raw)
        with pytest.raises(AlfError, match="Unknown ALF channel"):
            read_alf_header(raw, extension="nope")
        blob = tmp_path / "cache-9f3c"
        blob.write_bytes(raw)
        with pytest.raises(AlfError, match="Cannot infer the ALF channel"):
            read_alf_header(blob)

    def test_the_payload_scoped_cross_check_is_documented_as_absent(self, tmp_path):
        """A header-only read CANNOT run `_check_payload_length` — its evidence
        is the bytes it declined to fetch. Pinned so the weaker contract is a
        stated property rather than something a caller discovers."""
        header = json.dumps({"_total_points": 999}, separators=(",", ":")).encode()
        path = tmp_path / "l.ali"
        path.write_bytes(struct.pack("B", 1) + struct.pack("<I", len(header)) + header + b"\x00\x00")

        with pytest.raises(AlfError, match="declares more points than"):
            read_alf(path)
        # Accepted here, and honestly labelled.
        lightweight = read_alf_header(path)
        assert lightweight.total_points == 999
        assert lightweight.header_only is True


def test_non_numeric_objects_are_not_diagnosed_as_strings(tmp_path):
    """`Decimal`, `Fraction` and generators are refused — correctly — but they
    were refused with "holds strings, not numbers", which sends the caller
    hunting for an unparsed CSV column they do not have. The strings case keeps
    its own wording, including the object-array-of-strings shape that has no
    string DTYPE to give it away."""
    import decimal
    import fractions

    for value in (
        [[decimal.Decimal("9007199254740993"), 0.5, 0.0]],
        [[fractions.Fraction(3), 0.5, 0.0]],
        ([1.0, 2.0, 3.0] for _ in range(1)),
    ):
        with pytest.raises(ValueError, match="coordinates are non-numeric objects"):
            write_alp(tmp_path / "a.alp.gz", value, codec=GEOM_INT16, frame_convention=FRAME_WORLD_ALIGNED)
    with pytest.raises(ValueError, match="raw_intensity holds non-numeric objects"):
        write_ali(tmp_path / "b.ali.gz", [decimal.Decimal(1), 0.5], lo=0.0, hi=2.0)

    for value in ([["1", "2", "3"]], np.array([["1", "2", "3"]], dtype=object)):
        with pytest.raises(ValueError, match="coordinates are strings, not numbers"):
            write_alp(tmp_path / "c.alp.gz", value, codec=GEOM_INT16, frame_convention=FRAME_WORLD_ALIGNED)
    with pytest.raises(ValueError, match="raw_intensity holds strings, not numbers"):
        write_ali(tmp_path / "d.ali.gz", ["1", "2"], lo=0.0, hi=2.0)


class TestInvalidReturnsAreValidContent:
    """A NaN component in float16 `.alp` geometry is a lidar's INVALID RETURN.

    Synthetic frames cover finite, mixed, and entirely invalid returns.
    Accepting them must preserve every point index and its companion channels.
    """

    def _alp(self, points, header=None, version=1, dtype="<f2"):
        blob = json.dumps(
            header if header is not None else {"_total_points": len(points)},
            separators=(",", ":"),
        ).encode("utf-8")
        return (
            struct.pack("B", version) + struct.pack("<I", len(blob)) + blob + np.asarray(points, dtype=dtype).tobytes()
        )

    def test_reading_keeps_invalid_returns_in_place(self):
        """IN PLACE, not dropped. Every other channel is correlated with geometry
        BY INDEX, so compacting the invalid returns away here would misalign
        `.ali`, `.all`, `.alc` and `.alx` for the whole frame."""
        blob = self._alp([[1.5, -2.5, 3.75], [float("nan")] * 3, [4.0, 5.0, 6.0], [float("nan"), 0.5, -0.25]])
        points = read_alf(blob, extension="alp").xyz
        assert points.shape == (4, 3)
        assert points[0].tolist() == [1.5, -2.5, 3.75]
        assert np.isnan(points[1]).all()
        assert points[2].tolist() == [4.0, 5.0, 6.0]
        assert np.isnan(points[3, 0]) and points[3, 1:].tolist() == [0.5, -0.25]
        kept = points[~np.isnan(points).any(axis=1)]
        assert kept.tolist() == [[1.5, -2.5, 3.75], [4.0, 5.0, 6.0]]

    def test_both_infinities_survive_distinctly(self):
        """A reader that collapses +Inf, -Inf and NaN to one marker has changed
        what the file says."""
        points = read_alf(
            self._alp([[float("inf"), float("-inf"), 0.0], [1.0, float("nan"), -1.0]]),
            extension="alp",
        ).xyz
        assert points[0, 0] == np.inf and points[0, 1] == -np.inf
        assert np.isnan(points[1, 1])

    def test_the_error_bound_is_finite_and_ignores_invalid_returns(self):
        """`max_geom_error_m` is the documented accuracy preflight. A plain max
        over a frame with one invalid return is `nan`, and `nan <= threshold` is
        False for every threshold — so the gate silently stopped answering."""
        clean = read_alf(self._alp([[100.0, 0.0, 0.0]]), extension="alp").max_geom_error_m
        withnan = read_alf(self._alp([[100.0, 0.0, 0.0], [float("nan")] * 3]), extension="alp").max_geom_error_m
        assert withnan == clean
        assert math.isfinite(withnan) and withnan > 0

        # An Inf component is excluded from the magnitude too, or the bound is Inf.
        withinf = read_alf(self._alp([[100.0, 0.0, 0.0], [float("inf"), 0.0, 0.0]]), extension="alp").max_geom_error_m
        assert withinf == clean

    def test_an_all_invalid_frame_reports_the_subnormal_floor(self):
        """No finite component means no magnitude to describe. Stating the floor
        is honest; `nan` and `inf` are not, and neither is inventing one."""
        bound = read_alf(self._alp([[float("nan")] * 3]), extension="alp").max_geom_error_m
        assert bound == alf_io_module.FP16_SUBNORMAL_HALF_ULP

    def test_world_placement_carries_the_marker_through(self):
        """`world_xyz`'s overflow check exists to catch a FINITE local plus a
        finite anchor summing past float64. An invalid return is non-finite
        before the anchor and after it — refusing that would make every frame
        unreadable again through a second door."""
        blob = self._alp(
            [[1.0, 2.0, 3.0], [float("nan")] * 3],
            {"_total_points": 2, "device_position": [10.0, 20.0, 30.0]},
        )
        world = read_alf(blob, extension="alp").world_xyz(sensor_pose=True)
        assert world[0].tolist() == [11.0, 22.0, 33.0]
        assert np.isnan(world[1]).all()

    def test_a_genuine_overflow_is_still_refused(self):
        """The other half of that check must survive: a FINITE local coordinate
        that becomes non-finite once the anchor is applied is a real failure.

        Built on int16, because fp16 cannot reach it — binary16 caps at 65504,
        which is negligible beside float64's maximum, so no finite fp16 local can
        overflow any finite anchor. Fixed-point can: `_geom_scale` is an
        arbitrary finite double."""
        blob = self._alp(
            [[1, 0, 0]],
            {
                "_geom_codec": "int16",
                "_frame_convention": "world-aligned",
                "_geom_scale": 1.7e308,
                "_geom_offset": [0.0, 0.0, 0.0],
                "_total_points": 1,
                "device_position": [1.7e308, 0.0, 0.0],
            },
            version=3,
            dtype="<i2",
        )
        parsed = read_alf(blob, extension="alp")
        assert math.isfinite(parsed.xyz[0, 0])  # the LOCAL frame is fine
        with pytest.raises(AlfError, match="overflowed to non-finite"):
            parsed.world_xyz(sensor_pose=True)

    def test_the_writer_round_trips_invalid_returns(self, tmp_path):
        """A format whose reader accepts what its writer cannot produce cannot
        round-trip valid input data."""
        original = np.array([[1.5, -2.5, 3.75], [np.nan, np.nan, np.nan], [4.0, 5.0, 6.0], [np.inf, 0.5, -0.25]])
        path = tmp_path / "a.alp.gz"
        write_alp(path, original)
        back = read_alf(path).xyz
        assert np.array_equal(original, back, equal_nan=True)

    def test_the_box_describes_the_finite_extent(self, tmp_path):
        """The server turns `box` straight into item-metadata spatial bounds, so
        a `nan` or `inf` in it takes the frame out of every spatial query.

        BOTH are excluded, and they need separate coverage: `np.nanmin` ignores
        NaN on its own, so a test using only NaN passes whether or not anything
        masks the infinities — and an Inf reaching the box is the case that
        makes the frame publish no box at all instead of the finite one it has.
        """
        path = tmp_path / "b.alp.gz"
        header = write_alp(path, np.array([[1.0, 2.0, 3.0], [np.nan, np.nan, np.nan], [-1.0, 0.0, 5.0]]))
        assert header["box"] == "-1.0:0.0:3.0:1.0:2.0:5.0"
        assert header["_box_frame"] == "local"

        infinite = tmp_path / "b-inf.alp.gz"
        header = write_alp(
            infinite,
            np.array([[1.0, 2.0, 3.0], [np.inf, -np.inf, np.nan], [-1.0, 0.0, 5.0]]),
        )
        assert header["box"] == "-1.0:0.0:3.0:1.0:2.0:5.0"
        assert header["_box_frame"] == "local"

    def test_a_frame_with_no_finite_extent_publishes_no_box(self, tmp_path):
        """`nan:nan:nan:...` is unparseable to every consumer, and inventing a
        zero box would claim an extent the frame does not have."""
        path = tmp_path / "c.alp.gz"
        header = write_alp(path, np.array([[np.nan, np.nan, np.nan]]))
        assert "box" not in header and "_box_frame" not in header
        assert np.isnan(read_alf(path).xyz).all()

    def test_the_fp16_budget_guard_still_guards(self, tmp_path):
        """`np.max` over a frame with one invalid return is `nan`, and every
        comparison inside the guard is False against `nan` — so one NaN
        disabled the localization check entirely, on exactly the frames that
        carry them."""
        with pytest.raises(ValueError, match="localized coordinate component"):
            write_alp(tmp_path / "d.alp.gz", np.array([[600.0, 0.0, 0.0], [np.nan] * 3]))

    def test_a_finite_coordinate_overflowing_fp16_is_still_refused(self, tmp_path):
        """Representability, not the marker: 70000 m is a real coordinate that
        binary16 cannot hold, and writing it as Inf would invent an invalid
        return that the source data never had."""
        with pytest.raises(ValueError, match="overflows float16"):
            write_alp(tmp_path / "e.alp.gz", np.array([[70000.0, 0.0, 0.0]]), check_budget=False)

    def test_int16_encodes_the_marker_with_a_reserved_word(self, tmp_path):
        """It could not, until the corpus was measured; now it can.

        `rint(nan/scale).astype(int16)` is still platform-defined — the fix is
        not to cast the NaN but to never let the cast decide. Invalid rows are
        masked and `INT16_NO_RETURN` written explicitly, so no undefined value
        reaches the payload.
        """
        path = tmp_path / "f.alp.gz"
        xyz = np.array([[1.0, 2.0, 3.0], [np.nan] * 3])
        write_alp(path, xyz, codec=GEOM_INT16, frame_convention=FRAME_WORLD_ALIGNED)
        decoded = read_alf(path).xyz
        assert np.isfinite(decoded[0]).all()
        assert np.isnan(decoded[1]).all()

    def test_the_sizing_helpers_skip_invalid_returns_rather_than_refuse(self):
        """Sizing the grid from a slot that holds no position gives a wrong grid.

        These used to refuse any non-finite input, on the reasoning that v3 had
        no encoding for it. Now they size from the real coordinates and ignore
        the rest — and crucially they must ignore them WITHOUT filtering into a
        copy, which is what the writer's memory bound measures.
        """
        xyz = np.array([[10.0, 20.0, 30.0], [np.nan] * 3, [-10.0, -20.0, -30.0]])
        clean = np.array([[10.0, 20.0, 30.0], [-10.0, -20.0, -30.0]])
        np.testing.assert_allclose(choose_geom_offset(xyz), choose_geom_offset(clean))
        assert choose_geom_scale(xyz) == choose_geom_scale(clean)

    def test_a_frame_of_nothing_but_invalid_returns_is_writable(self):
        """Every beam missed. A real, if uninteresting, frame."""
        assert np.array_equal(choose_geom_offset(np.full((4, 3), np.nan)), np.zeros(3))
        assert choose_geom_scale(np.full((4, 3), np.nan)) == MIN_GEOM_SCALE_M

    def test_an_int16_decode_that_overflows_is_still_refused(self):
        """A different rule from §5.1, and no permission there relaxes it: int16
        cannot ENCODE a non-finite value, so one out of the decode is always the
        codec parameters overflowing."""
        header = {
            "_geom_codec": "int16",
            "_frame_convention": "world-aligned",
            "_geom_scale": 1e308,
            "_geom_offset": [0.0, 0.0, 0.0],
            "_total_points": 1,
        }
        blob = self._alp([[2, 0, 0]], header, version=3, dtype="<i2")
        with pytest.raises(AlfError, match=r"int16 \.alp decodes to non-finite"):
            read_alf(blob, extension="alp").xyz


def test_alx_writer_refuses_binary64_inexact_source_count(tmp_path):
    with pytest.raises(ValueError, match="binary64"):
        write_alx(
            tmp_path / "bad.alx.gz",
            np.array([0], dtype=np.uint32),
            alp_total_points=1,
            source_total_points=9007199254740993,
        )


def test_reserved_exponent_guard_bounds_before_bigint_expansion(monkeypatch):
    hostile = alf_io_module._HeaderFloat("1e100000000")
    zero = alf_io_module._HeaderFloat("0e10000000000")

    def forbid_bigint_expansion(_value):
        raise AssertionError("hostile exponent reached int(Decimal)")

    monkeypatch.setattr(alf_io_module, "int", forbid_bigint_expansion, raising=False)

    with pytest.raises(AlfError, match="too large for binary64"):
        alf_io_module._reject_inexact_count_token(hostile, "_total_points")
    alf_io_module._reject_inexact_count_token(zero, "_total_points")


@pytest.mark.parametrize("channel", ["alp", "ali", "all", "alc", "alx"])
def test_writers_refuse_metadata_outside_finite_binary64(tmp_path, channel):
    path = tmp_path / f"bad.{channel}.gz"
    metadata = {"custom": {"outside_binary64": 10**400}}
    writers = {
        "alp": lambda: write_alp(path, np.zeros((0, 3)), metadata),
        "ali": lambda: write_ali(path, np.zeros(0), lo=0.0, hi=1.0, metadata=metadata),
        "all": lambda: write_all(path, np.zeros(0), metadata=metadata),
        "alc": lambda: write_alc(path, np.zeros(0), ["ff0000"], metadata=metadata),
        "alx": lambda: write_alx(
            path,
            np.zeros(0),
            alp_total_points=0,
            source_total_points=0,
            metadata=metadata,
        ),
    }

    with pytest.raises(ValueError, match="finite binary64"):
        writers[channel]()
    assert not path.exists()


class TestMeasuredMemoryLimits:
    """The two bounds, and the fact that they are bounds rather than claims.

    Synthetic payloads above the old 512 MB cap must read within the configured
    budget. A size limit is a resource policy, not evidence of file corruption.
    These checks measure allocation overhead as well as the accepted sizes.
    """

    def _frame_of_payload(self, tmp_path, payload_bytes, name="big.ali.gz"):
        return _gzip_with_oversized_payload(tmp_path / name, {"_total_points": payload_bytes}, payload_bytes)

    def test_a_hundred_million_point_payload_is_under_the_cap(self):
        """The default budget accommodates 100 million six-byte points."""
        assert alf_io_module.MAX_DECOMPRESSED_BYTES > 600_000_000

    def test_a_large_synthetic_frame_reads(self, tmp_path):
        path = self._frame_of_payload(tmp_path, 600 * 1024 * 1024)
        assert read_alf(path).total_points == 600 * 1024 * 1024

    def test_the_refusal_no_longer_claims_frames_cannot_be_this_large(self, tmp_path):
        """The refusal describes a configurable budget, not an impossible size."""
        alf_io_module.MAX_DECOMPRESSED_BYTES = 1024
        try:
            path = self._frame_of_payload(tmp_path, 8 * 1024 * 1024, name="over.ali.gz")
            with pytest.raises(AlfError) as caught:
                read_alf(path)
        finally:
            alf_io_module.MAX_DECOMPRESSED_BYTES = 1024 * 1024 * 1024
        message = str(caught.value)
        assert "do not reach this size" not in message
        assert "set_memory_limits" in message

    def test_decoding_is_bounded_separately_from_reading(self, tmp_path):
        """Reading a frame and DECODING one are different allocations.

        6 wire bytes per point become 24 in the float64 array, so a frame that
        fits the read cap can still be 4x it once decoded. Only the read was
        bounded before; the Dart reader has bounded both since it learned v3.
        Raising the read cap without this would have traded a clean refusal for
        an OOM kill on a 3 GB worker — no diagnostic, mid-request.
        """
        xyz = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
        path = tmp_path / "small.alp.gz"
        write_alp(path, xyz, {"seq": "s"})
        alf.set_memory_limits(decoded_geometry_bytes=8)
        try:
            parsed = read_alf(path)
            with pytest.raises(AlfError, match="decoded geometry too large"):
                parsed.xyz
        finally:
            alf.set_memory_limits(decoded_geometry_bytes=3 * 1024 * 1024 * 1024)

    def test_the_decode_bound_refuses_before_allocating(self, tmp_path):
        """Derived from the payload length, so nothing large is built first."""
        xyz = np.zeros((1000, 3), dtype=np.float64)
        path = tmp_path / "k.alp.gz"
        write_alp(path, xyz, {"seq": "s"})
        parsed = read_alf(path)
        alf.set_memory_limits(decoded_geometry_bytes=1000 * 3 * 8 - 1)
        try:
            with pytest.raises(AlfError, match="decoded geometry too large"):
                parsed.xyz
        finally:
            alf.set_memory_limits(decoded_geometry_bytes=3 * 1024 * 1024 * 1024)
        alf.set_memory_limits(decoded_geometry_bytes=1000 * 3 * 8)
        assert read_alf(path).xyz.shape == (1000, 3)
        alf.set_memory_limits(decoded_geometry_bytes=3 * 1024 * 1024 * 1024)

    def test_the_setter_is_live_not_a_snapshot(self, tmp_path):
        """A knob that reads as working and does nothing is the failure mode
        this package keeps finding in other people's code."""
        before = alf.memory_limits()["decompressed_bytes"]
        alf.set_memory_limits(decompressed_bytes=4096)
        try:
            assert alf_io_module.MAX_DECOMPRESSED_BYTES == 4096
            path = self._frame_of_payload(tmp_path, 8 * 1024 * 1024, name="live.ali.gz")
            with pytest.raises(AlfError):
                read_alf(path)
        finally:
            alf.set_memory_limits(decompressed_bytes=before)
        assert alf.memory_limits()["decompressed_bytes"] == before

    def test_the_setter_refuses_nonsense(self):
        for bad in (0, -1, 1.5, True, "512MB"):
            with pytest.raises(ValueError):
                alf.set_memory_limits(decompressed_bytes=bad)
            with pytest.raises(ValueError):
                alf.set_memory_limits(decoded_geometry_bytes=bad)

    def test_header_only_read_is_unaffected_by_the_decode_bound(self, tmp_path):
        """The escape hatch the decode refusal points at must actually work."""
        xyz = np.zeros((1000, 3), dtype=np.float64)
        path = tmp_path / "h.alp.gz"
        write_alp(path, xyz, {"seq": "s"})
        alf.set_memory_limits(decoded_geometry_bytes=8)
        try:
            assert read_alf_header(path).total_points == 1000
        finally:
            alf.set_memory_limits(decoded_geometry_bytes=3 * 1024 * 1024 * 1024)


def test_read_alf_header_rejects_stream_whose_read_returns_non_bytes() -> None:
    """A reader whose `.read()` hands back something other than bytes must fail
    fast, not spin. Before the guard, `out += chunk` in the envelope loop fell
    through to the chunk type's `__radd__` (a `MagicMock` here), `len(out)`
    stuck at 0, and `read_alf_header` looped forever allocating.
    """
    from unittest.mock import MagicMock

    bogus = MagicMock()  # .read() returns a MagicMock, never b""
    with pytest.raises(alf.AlfError, match="not bytes"):
        alf.read_alf_header(bogus, extension="alp")
    # The guard fires on the FIRST read; anything more means we started looping.
    assert bogus.read.call_count == 1


class _StrReader:
    """A text-mode style reader: `.read()` returns str, which is not bytes."""

    def read(self, n: int = -1) -> str:  # signature mirrors BinaryIO.read
        return "\x01\x00\x00\x00\x02{}"


def test_read_alf_header_rejects_text_mode_reader() -> None:
    with pytest.raises(alf.AlfError, match="not bytes"):
        alf.read_alf_header(_StrReader(), extension="alp")  # type: ignore[arg-type]
