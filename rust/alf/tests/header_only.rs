//! `read_alf_header_bytes` against the shared conformance corpus.
//!
//! The corpus is this repository's bytes-on-disk evidence, and it is the right
//! thing to hold a header-only reader to — not because header-only reading is a
//! wire feature (it is not; `spec/support.md` says so) but because the corpus
//! is the only place that already knows, for hundreds of files, what the right
//! answer is. The property asserted here is the one that matters:
//!
//!   A header-only read must never DISAGREE with a full read. It may decline to
//!   have an opinion — that is the payload-scoped half of the contract — but
//!   where it has one, it must be the same opinion with the same refusal id.
//!
//! The dangerous drift is a header-only reader that validates less: it hands
//! back a header the full reader refuses, and a caller scanning a sequence
//! never learns the file was bad.

use std::collections::BTreeSet;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use alf::{read_alf_bytes, read_alf_header, read_alf_header_bytes, Channel};

fn corpus() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/conformance/data")
        .canonicalize()
        .expect("conformance corpus")
}

/// Every corpus file in a directory, with the channel its name declares.
fn cases(dir: &Path) -> Vec<(String, PathBuf, Channel)> {
    let mut found = Vec::new();
    for entry in fs::read_dir(dir).expect("corpus directory") {
        let path = entry.expect("corpus entry").path();
        let name = path.file_name().unwrap().to_str().unwrap().to_owned();
        if !(name.ends_with(".gz") || name.ends_with(".bin")) {
            continue;
        }
        let stem = name.trim_end_matches(".gz").trim_end_matches(".bin");
        let channel = match Channel::from_path(stem) {
            Some(channel) => channel,
            None => continue,
        };
        let case = stem.rsplit_once('.').unwrap().0.to_owned();
        found.push((case, path, channel));
    }
    found.sort_by(|left, right| left.1.cmp(&right.1));
    found
}

fn expected_refusal(dir: &Path, case: &str) -> Option<String> {
    let text = fs::read_to_string(dir.join(format!("{case}.json"))).ok()?;
    let value: serde_json::Value = serde_json::from_str(&text).ok()?;
    Some(value.get("refused")?.as_str()?.to_owned())
}

#[test]
fn a_header_read_agrees_with_a_full_read_on_every_valid_corpus_file() {
    let dir = corpus();
    let mut checked = 0;
    for (_case, path, channel) in cases(&dir) {
        let bytes = fs::read(&path).expect("corpus bytes");
        let whole = read_alf_bytes(&bytes, channel).expect("valid corpus file");
        let head = read_alf_header_bytes(&bytes, channel).expect("header-only read");
        assert_eq!(head.version, whole.version, "{path:?} version");
        assert_eq!(head.channel, whole.channel, "{path:?} channel");
        assert_eq!(head.metadata, whole.metadata, "{path:?} metadata");
        assert_eq!(
            head.frame_convention, whole.frame_convention,
            "{path:?} frame convention"
        );
        assert_eq!(
            head.frame_rotation, whole.frame_rotation,
            "{path:?} frame rotation"
        );
        checked += 1;
    }
    assert!(checked >= 37, "expected the valid corpus, got {checked}");
}

#[test]
fn a_header_read_never_disagrees_with_a_full_read_on_an_invalid_file() {
    let dir = corpus().join("invalid");
    let mut header_scoped = BTreeSet::new();
    let mut checked = 0;

    for (case, path, channel) in cases(&dir) {
        let want = match expected_refusal(&dir, &case) {
            Some(want) => want,
            None => continue,
        };
        let bytes = fs::read(&path).expect("corpus bytes");
        let full = read_alf_bytes(&bytes, channel)
            .err()
            .unwrap_or_else(|| panic!("{case}: the full reader accepted an invalid file"));
        assert_eq!(full.id, want, "{case}: full reader");

        match read_alf_header_bytes(&bytes, channel) {
            // Declining to have an opinion is allowed: the rule this file
            // breaks may be payload-scoped, and the payload was never read.
            Ok(_) => {}
            // Having a DIFFERENT opinion is not. That is the drift.
            Err(error) => {
                assert_eq!(
                    error.id, want,
                    "{case}: the header read refused a different rule than the full read"
                );
                header_scoped.insert(error.id.to_owned());
            }
        }
        checked += 1;
    }

    assert!(checked >= 60, "expected the invalid corpus, got {checked}");
    // Naming a handful keeps the assertion above from passing on a reader that
    // has quietly stopped refusing anything at all.
    for id in [
        "frame-convention-missing",
        "frame-rotation-not-normalized",
        "header-key-duplicate",
        "header-not-an-object",
        "geom-number-inexact",
        "version-unsupported",
        // Channel-scoped, and skipped by every implementation's header reader
        // until now: they lived inside the channel decoders, which a
        // header-only read never reaches.
        "total-points-missing",
        "label-layout-unsupported",
    ] {
        assert!(
            header_scoped.contains(id),
            "{id} is header-scoped but the header reader did not refuse it; \
             it refused: {header_scoped:?}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// The resource property, on files whose PAYLOAD is missing or wrong. A reader
// that inflated to the end could not answer any of these.
// ─────────────────────────────────────────────────────────────────────────────

fn envelope(version: u8, header: &str, payload: &[u8]) -> Vec<u8> {
    let mut out = vec![version];
    out.extend_from_slice(&(header.len() as u32).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(payload);
    out
}

fn gzip(bytes: &[u8]) -> Vec<u8> {
    let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    encoder.write_all(bytes).expect("gzip");
    encoder.finish().expect("gzip")
}

/// A frame whose payload is large and highly compressible, so that a bounded
/// read is visibly bounded.
fn big_frame() -> Vec<u8> {
    let payload: Vec<u8> = (0..3 * 1024 * 1024).map(|i| (i % 251) as u8).collect();
    gzip(&envelope(
        1,
        r#"{"_total_points":524288,"seq":"s1"}"#,
        &payload,
    ))
}

#[test]
fn a_member_truncated_after_its_header_still_yields_the_header() {
    let whole = big_frame();
    let cut = &whole[..whole.len() / 2];
    assert!(
        read_alf_bytes(cut, Channel::Alp).is_err(),
        "the full reader must not accept a truncated member"
    );
    let head = read_alf_header_bytes(cut, Channel::Alp).expect("header survives truncation");
    assert_eq!(head.metadata.get("seq").unwrap().as_str(), Some("s1"));
}

#[test]
fn a_corrupt_payload_crc32_is_invisible_here_and_fatal_to_a_full_read() {
    let mut bytes = big_frame();
    let end = bytes.len();
    bytes[end - 8] ^= 0xff;
    let full = read_alf_bytes(&bytes, Channel::Alp).expect_err("CRC32 must be checked");
    assert_eq!(full.id, "gzip-malformed");
    assert_eq!(
        read_alf_header_bytes(&bytes, Channel::Alp)
            .expect("a header read never reaches the trailer")
            .version,
        1
    );
}

#[test]
fn a_small_members_trailer_is_reached_and_therefore_checked() {
    // The CRC test above uses a multi-megabyte payload, so the probe stops long
    // before the trailer and the trailer is never seen. A SMALL member ends
    // inside the probe, `flate2` reads its trailer on the way past, and the
    // corruption is caught — for free, and only because the file was small.
    //
    // Pinned so the docs cannot claim more than the code does: "CRC32 and ISIZE
    // go unverified" is true of a member longer than the request and false of a
    // shorter one, and nothing may depend on it either way.
    let mut small = gzip(&envelope(1, r#"{"_total_points":0}"#, &[]));
    let end = small.len();
    small[end - 8] ^= 0xff;
    let refused = read_alf_header_bytes(&small, Channel::Alp)
        .expect_err("a member that ends inside the probe has its trailer read");
    assert_eq!(refused.id, "gzip-malformed");
    // And the full reader agrees, so the subset property is intact: this is a
    // refusal `read_alf_bytes` also makes, not one invented here.
    assert_eq!(
        read_alf_bytes(&small, Channel::Alp)
            .expect_err("full read")
            .id,
        "gzip-malformed"
    );
}

#[test]
fn a_second_gzip_member_is_undetected_here_and_refused_by_a_full_read() {
    let one = gzip(&envelope(1, r#"{"_total_points":0}"#, &[]));
    let two: Vec<u8> = one.iter().chain(one.iter()).copied().collect();
    let full = read_alf_bytes(&two, Channel::Alp).expect_err("two headers, one file");
    assert_eq!(full.id, "multiple-gzip-members");
    assert!(read_alf_header_bytes(&two, Channel::Alp).is_ok());
}

#[test]
fn the_declared_count_is_not_cross_checked_against_the_payload() {
    let bytes = envelope(1, r#"{"_total_points":4}"#, &[0u8; 6]);
    assert!(read_alf_bytes(&bytes, Channel::Alp).is_err());
    assert_eq!(
        read_alf_header_bytes(&bytes, Channel::Alp)
            .expect("the count is header-scoped, the cross-check is not")
            .total_points,
        Some(4.0)
    );
}

#[test]
fn a_malformed_count_is_refused_here_rather_than_by_a_later_accessor() {
    // `check_count_keys` deliberately passes over a non-NUMBER, leaving it to
    // whichever decoder knows whether the key was required at all. A
    // header-only read reaches no decoder, so it has to refuse directly —
    // Python's reader had exactly this gap and TypeScript's did not.
    for bad in ["null", "-1", "1.9", "\"4\"", "true"] {
        let bytes = envelope(1, &format!(r#"{{"_total_points":{bad}}}"#), &[]);
        assert!(
            read_alf_bytes(&bytes, Channel::Alp).is_err(),
            "the full read accepted _total_points: {bad}"
        );
        assert!(
            read_alf_header_bytes(&bytes, Channel::Alp).is_err(),
            "the header read accepted _total_points: {bad}"
        );
    }
    // An ABSENT count is not a malformed one — only `.alx` requires the field.
    assert_eq!(
        read_alf_header_bytes(&envelope(1, r#"{"seq":"s1"}"#, &[]), Channel::Alp)
            .expect("a terse header is a valid header")
            .total_points,
        None
    );
}

#[test]
fn a_header_past_the_probe_still_reads_in_a_second_inflate() {
    // Over 64 KiB: the first pass returns a prefix the envelope parser would
    // call truncated, and the retry is what makes it whole. A reader that
    // trusted one pass would refuse a legal file.
    let padding = "x".repeat(80 * 1024);
    let header = format!(r#"{{"_total_points":0,"pad":"{padding}"}}"#);
    let bytes = gzip(&envelope(1, &header, &[]));
    let head = read_alf_header_bytes(&bytes, Channel::Alp).expect("a long header");
    assert_eq!(
        head.metadata.get("pad").unwrap().as_str(),
        Some(&padding[..])
    );
    assert_eq!(
        head.metadata,
        read_alf_bytes(&bytes, Channel::Alp).unwrap().metadata
    );
}

#[test]
fn the_path_entry_point_reads_a_prefix_rather_than_the_file() {
    let dir = std::env::temp_dir().join("alf-header-only-test");
    fs::create_dir_all(&dir).expect("scratch dir");
    let path = dir.join("frame_000000.alp.gz");
    fs::write(&path, big_frame()).expect("write frame");

    // The channel comes from the suffix, ignoring `.gz`, exactly as `read_alf`.
    let head = read_alf_header(path.to_str().unwrap(), "").expect("path read");
    assert_eq!(head.channel, Channel::Alp);
    assert_eq!(head.metadata.get("seq").unwrap().as_str(), Some("s1"));
    assert_eq!(head.total_points, Some(524288.0));
    assert_eq!(
        head.metadata,
        read_alf_header_bytes(&big_frame(), Channel::Alp)
            .unwrap()
            .metadata
    );

    // The same answer for an UNCOMPRESSED file, where the prefix read takes a
    // different branch.
    let plain = dir.join("frame_000001.alp");
    fs::write(
        &plain,
        envelope(1, r#"{"_total_points":0,"seq":"s1"}"#, &[]),
    )
    .expect("write");
    assert_eq!(
        read_alf_header(plain.to_str().unwrap(), "")
            .expect("uncompressed path read")
            .total_points,
        Some(0.0)
    );

    fs::remove_dir_all(&dir).ok();
}

/// A gzip member that spends `empty_blocks * 5` compressed bytes producing
/// NOTHING before its real content. Empty non-final stored blocks are legal
/// DEFLATE; no encoder emits them, and an author may emit as many as they like.
fn low_yield_gzip(inner: &[u8], empty_blocks: usize) -> Vec<u8> {
    let mut out = vec![0x1f, 0x8b, 0x08, 0, 0, 0, 0, 0, 0, 0xff];
    for _ in 0..empty_blocks {
        out.extend_from_slice(&[0x00, 0x00, 0x00, 0xff, 0xff]);
    }
    out.push(0x01);
    out.extend_from_slice(&(inner.len() as u16).to_le_bytes());
    out.extend_from_slice(&(!(inner.len() as u16)).to_le_bytes());
    out.extend_from_slice(inner);
    out.extend_from_slice(&crc32(inner).to_le_bytes());
    out.extend_from_slice(&(inner.len() as u32).to_le_bytes());
    out
}

fn crc32(bytes: &[u8]) -> u32 {
    let mut table = [0u32; 256];
    for (i, slot) in table.iter_mut().enumerate() {
        let mut c = i as u32;
        for _ in 0..8 {
            c = if c & 1 != 0 {
                0xEDB8_8320 ^ (c >> 1)
            } else {
                c >> 1
            };
        }
        *slot = c;
    }
    let mut crc = 0xFFFF_FFFFu32;
    for byte in bytes {
        crc = table[((crc ^ u32::from(*byte)) & 0xff) as usize] ^ (crc >> 8);
    }
    crc ^ 0xFFFF_FFFF
}

#[test]
fn a_low_yield_member_the_full_reader_accepts_is_not_refused_here() {
    // THE PROPERTY THIS NEARLY BROKE. A first attempt at bounding the work used
    // a tight per-request input budget, which made the header read refuse a
    // member `read_alf_bytes` reads without complaint — inverting the one
    // guarantee this entry point makes. The bound has to be the SAME one the
    // full reader uses, or the subset claim is false.
    let inner = envelope(1, r#"{"_total_points":0,"seq":"s1"}"#, &[]);
    let bytes = low_yield_gzip(&inner, 40_000); // 200 KB producing nothing
    assert_eq!(
        read_alf_bytes(&bytes, Channel::Alp)
            .expect("the full reader accepts it")
            .metadata
            .get("seq")
            .and_then(|v| v.as_str()),
        Some("s1")
    );
    assert_eq!(
        read_alf_header_bytes(&bytes, Channel::Alp)
            .expect("so the header reader must too")
            .metadata
            .get("seq")
            .and_then(|v| v.as_str()),
        Some("s1")
    );
}

#[test]
fn a_huge_uncompressed_frame_still_yields_its_header() {
    // The whole point of the entry point: a frame's SIZE is a fact about a
    // payload this never reads. An earlier revision of the compressed-input
    // ceiling was applied unconditionally, so a 2 GiB uncompressed frame with a
    // perfectly good header came back `gzip-malformed` — reporting "Compressed
    // ALF input" about a file that is neither compressed nor malformed.
    //
    // `set_len` only sets the size; nothing is written and nothing is
    // allocated, so this costs a few kilobytes on disk and reads 64 KiB of it.
    let dir = std::env::temp_dir().join("alf-huge-header-test");
    fs::create_dir_all(&dir).expect("scratch dir");
    let path = dir.join("frame_000000.alp");
    let envelope = envelope(1, r#"{"_total_points":0,"seq":"big"}"#, &[]);
    {
        let file = fs::File::create(&path).expect("create");
        use std::io::Write as _;
        (&file).write_all(&envelope).expect("write envelope");
        file.set_len(2 * 1024 * 1024 * 1024).expect("grow");
    }

    let head = read_alf_header(path.to_str().unwrap(), "").expect("header of a huge frame");
    assert_eq!(
        head.metadata.get("seq").and_then(|v| v.as_str()),
        Some("big")
    );
    assert_eq!(head.total_points, Some(0.0));

    fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_missing_or_unnamed_path_is_an_error_not_a_panic() {
    assert!(read_alf_header("/nonexistent/frame_000000.alp.gz", "").is_err());
    assert!(read_alf_header("frame_000000.unknown", "").is_err());
}
