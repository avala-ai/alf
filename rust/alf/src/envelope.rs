//! `[version:1B][header_len:4B LE][UTF-8 JSON header][payload]`, optionally gzipped.

use std::fs::{self, File};
use std::io::{Read, Seek};

use crate::error::{refuse, AlfError, Result};
use crate::header::MAX_HEADER_BYTES;

/// Decompressed-size ceiling, mirroring the Python reference's measured cap.
pub const MAX_DECOMPRESSED_BYTES: usize = 1024 * 1024 * 1024;

/// Compressed-input ceiling, matching Python's `_limit_for(True)` and
/// TypeScript's `MAX_COMPRESSED_INPUT_BYTES`.
///
/// The decompressed cap alone does not bound the WORK a member can demand:
/// DEFLATE may emit nothing at all for arbitrarily much input — a member of
/// empty non-final stored blocks costs five bytes each and produces zero — so a
/// reader driven by how much output it has walks the whole file regardless.
/// Measured here: a 100 MiB member of such blocks took 132 ms to yield the
/// 35 bytes of envelope at its end, and it scales linearly.
///
/// The megabyte of slack over the decompressed cap is there because deflate
/// expands incompressible data slightly, so measuring the archive against the
/// decompressed limit would reject a writer's own legitimate output.
pub const MAX_COMPRESSED_INPUT_BYTES: usize = MAX_DECOMPRESSED_BYTES + 1024 * 1024;

// Compile-time, because it is a fact about the constants rather than about any
// input: measuring an archive against the DECOMPRESSED cap would reject a
// writer's own output, since deflate grows data it cannot compress.
const _: () = assert!(MAX_COMPRESSED_INPUT_BYTES > MAX_DECOMPRESSED_BYTES);

/// Refuse an input that is absurd before any of it is inflated.
fn check_input_size(len: usize) -> Result<()> {
    if len > MAX_COMPRESSED_INPUT_BYTES {
        refuse!(
            "gzip-malformed",
            "Compressed ALF input is {len} bytes, past the conservative \
             {MAX_COMPRESSED_INPUT_BYTES}-byte input limit"
        );
    }
    Ok(())
}

pub struct Envelope {
    pub version: u8,
    pub header_bytes: Vec<u8>,
    pub payload: Vec<u8>,
}

/// gzip magic. Checked rather than trusting the filename: the invalid corpus
/// deliberately names its files `.bin`.
fn is_gzip(bytes: &[u8]) -> bool {
    bytes.len() >= 2 && bytes[0] == 0x1f && bytes[1] == 0x8b
}

/// Inflate exactly one member, refusing a second.
///
/// A concatenation of two gzip members is well-formed gzip that most readers
/// silently treat as one stream. For this format it means two different headers
/// claiming to describe the same file, so it is refused rather than resolved.
fn gunzip_single_member(bytes: &[u8]) -> Result<Vec<u8>> {
    check_input_size(bytes.len())?;
    fn inflate<R: Read>(mut reader: R) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        // Bounded: a decompression bomb must not be able to allocate freely.
        let mut limited = (&mut reader).take(MAX_DECOMPRESSED_BYTES as u64 + 1);
        limited.read_to_end(&mut out).map_err(|err| {
            AlfError::new("gzip-malformed", format!("Corrupt gzip stream: {err}"))
        })?;
        if out.len() > MAX_DECOMPRESSED_BYTES {
            return Err(AlfError::new(
                "gzip-malformed",
                format!(
                    "Corrupt gzip stream: decompressed output exceeds the \
                     {MAX_DECOMPRESSED_BYTES}-byte limit"
                ),
            ));
        }
        Ok(out)
    }

    let first = inflate(flate2::read::GzDecoder::new(bytes))?;
    // Two members are well-formed gzip that most readers silently join into one
    // stream. Here it means two headers claiming to describe one file, so it is
    // refused rather than resolved. Detected by DECODING BOTH WAYS: a
    // single-member read stops at the first trailer, a multi-member read does
    // not, so a longer multi-member result is proof a second member exists —
    // and it needs no assumptions about how much the decoder buffered ahead.
    // A FAILED multi-member read is not a second member. `ali-gzip-zero-padded`
    // is a valid single member followed by zero bytes, which `MultiGzDecoder`
    // rejects as a malformed second header — so its error means "there was only
    // one", and only a SUCCESSFUL longer read proves otherwise.
    let all = match inflate(flate2::read::MultiGzDecoder::new(bytes)) {
        Ok(all) => all,
        Err(_) => return Ok(first),
    };
    if all.len() != first.len() {
        refuse!(
            "multiple-gzip-members",
            "Stream holds more than one gzip member; two members mean two headers \
             describing one file"
        );
    }
    Ok(first)
}

/// Bytes inflated on the first probe of a gzipped file.
///
/// Large enough that essentially every real header lands in one pass, small
/// enough that a 1 GiB frame costs kilobytes. A longer header costs one more,
/// exactly-sized inflate — not a full decompression.
const HEADER_PROBE_BYTES: usize = 64 * 1024;

/// Inflate at most `at_least` bytes of one member and stop.
///
/// `flate2`'s decoder is a `Read`, so `take` is the whole mechanism: the
/// inflater does the work the reader asks for and no more. Bounded by
/// construction — there is no separate ceiling to enforce, because the limit
/// IS the request.
///
/// WHAT IT GIVES UP, and it is not small: the member's CRC32 and ISIZE go
/// unverified, because checking either means inflating to the end — the exact
/// cost this exists to avoid. Second-member detection goes with them. So this
/// establishes that the bytes it returned decompressed correctly and NOTHING
/// about the bytes it did not read.
fn gunzip_prefix(bytes: &[u8], at_least: usize) -> Result<Vec<u8>> {
    // The SAME ceiling the full decoder applies, and needed here for a reason
    // specific to a prefix read: `take` bounds the OUTPUT, and output can be
    // held at zero for as much input as an author likes. See
    // `MAX_COMPRESSED_INPUT_BYTES`.
    check_input_size(bytes.len())?;
    let mut out = Vec::new();
    flate2::read::GzDecoder::new(bytes)
        .take(at_least as u64)
        .read_to_end(&mut out)
        .map_err(|err| AlfError::new("gzip-malformed", format!("Corrupt gzip stream: {err}")))?;
    Ok(out)
}

/// Parse the envelope prefix WITHOUT its payload.
///
/// The payload of the returned [`Envelope`] is empty and means "not read",
/// never "empty file" — only [`read`] can tell those apart, because telling
/// them apart requires the bytes this declined to fetch.
pub fn read_header(bytes: &[u8]) -> Result<Envelope> {
    let inflated;
    let raw: &[u8] = if is_gzip(bytes) {
        let probe = gunzip_prefix(bytes, HEADER_PROBE_BYTES)?;
        // Re-inflate only when the DECLARED header genuinely outran the probe.
        // The length is read here rather than by catching the truncation
        // refusal below, because a genuinely truncated file would otherwise be
        // retried and then reported as though the probe size were at fault.
        inflated = match declared_header_len(&probe) {
            Some(declared) if 5 + declared > probe.len() => gunzip_prefix(bytes, 5 + declared)?,
            _ => probe,
        };
        &inflated
    } else {
        // NOT `to_vec`. A header-only read of an uncompressed frame must not
        // copy the frame; only the header is taken below.
        bytes
    };
    split(raw, false)
}

/// [`read_header`] against a FILE, reading only as much of it as the header
/// needs.
///
/// The point of a header-only read is not paying for the payload, and
/// `read_alf_header_bytes` on `std::fs::read(path)` pays for it in full before
/// the parser is even reached — a 1 GiB frame to answer a question the first
/// few hundred bytes settle. This reads a bounded prefix of the file instead,
/// inflating a bounded prefix of that.
pub fn read_header_from_path(path: &str) -> Result<Envelope> {
    let probe = read_prefix(path, HEADER_PROBE_BYTES)?;
    // Re-read only when the DECLARED header genuinely outran the probe. The
    // file is reopened rather than the stream rewound because the gzip decoder
    // has consumed an unknown amount of it, and "how much compressed input did
    // that take" is not a question the decoder answers.
    match declared_header_len(&probe) {
        Some(declared) if 5 + declared > probe.len() => {
            split(&read_prefix(path, 5 + declared)?, false)
        }
        _ => split(&probe, false),
    }
}

/// At most `at_least` DECOMPRESSED bytes from the front of `path`.
fn read_prefix(path: &str, at_least: usize) -> Result<Vec<u8>> {
    let mut file = File::open(path)
        .map_err(|err| AlfError::new("", format!("Cannot read {path:?}: {err}")))?;
    let mut magic = [0u8; 2];
    // A short read is not an error here: a file too small to hold the magic is
    // also too small to be gzip, and `split` gives it the right refusal.
    let seen = read_up_to(&mut file, &mut magic)
        .map_err(|err| AlfError::new("", format!("Cannot read {path:?}: {err}")))?;
    let compressed = is_gzip(&magic[..seen]);
    file.rewind()
        .map_err(|err| AlfError::new("", format!("Cannot read {path:?}: {err}")))?;

    let mut out = Vec::new();
    if !compressed {
        // NO SIZE CHECK on this branch, and that is the point of the whole
        // entry point: nothing here reads past the envelope, so a frame's
        // total size is a fact about a payload this never touches. An earlier
        // revision checked it unconditionally and refused a 2 GiB uncompressed
        // frame with a perfectly good header — reporting `gzip-malformed`, and
        // "Compressed ALF input", about a file that is neither.
        file.take(at_least as u64)
            .read_to_end(&mut out)
            .map_err(|err| AlfError::new("", format!("Cannot read {path:?}: {err}")))?;
        return Ok(out);
    }
    // Bounded before a byte is inflated, like the in-memory gzip path. Read from
    // `metadata` rather than by reading the file: the point is not to read it.
    if let Ok(meta) = fs::metadata(path) {
        check_input_size(usize::try_from(meta.len()).unwrap_or(usize::MAX))?;
    }
    flate2::read::GzDecoder::new(file)
        .take(at_least as u64)
        .read_to_end(&mut out)
        .map_err(|err| AlfError::new("gzip-malformed", format!("Corrupt gzip stream: {err}")))?;
    Ok(out)
}

/// `Read::read` may return fewer bytes than asked for without being at EOF.
fn read_up_to<R: Read>(source: &mut R, into: &mut [u8]) -> std::io::Result<usize> {
    let mut filled = 0;
    while filled < into.len() {
        match source.read(&mut into[filled..])? {
            0 => break,
            n => filled += n,
        }
    }
    Ok(filled)
}

/// The declared header length, when the prefix is long enough to state one and
/// the claim is inside the limit. `None` means "do not act on this" — the
/// refusals belong to [`split`], which reports them once and precisely.
fn declared_header_len(raw: &[u8]) -> Option<usize> {
    if raw.len() < 5 {
        return None;
    }
    let declared = u32::from_le_bytes([raw[1], raw[2], raw[3], raw[4]]) as usize;
    (declared <= MAX_HEADER_BYTES).then_some(declared)
}

pub fn read(bytes: &[u8]) -> Result<Envelope> {
    let raw = if is_gzip(bytes) {
        gunzip_single_member(bytes)?
    } else {
        bytes.to_vec()
    };
    split(&raw, true)
}

/// The envelope rules that apply once the bytes are in hand.
///
/// Shared by both entry points so a header-only read cannot accept a prefix the
/// full read would refuse. `keep_payload` is the ONLY difference between them.
fn split(raw: &[u8], keep_payload: bool) -> Result<Envelope> {
    if raw.len() < 5 {
        refuse!(
            "header-truncated",
            "Truncated header: {} bytes is too short for the 5-byte envelope prefix",
            raw.len()
        );
    }
    let version = raw[0];
    let declared = u32::from_le_bytes([raw[1], raw[2], raw[3], raw[4]]) as usize;

    // Declared length is checked BEFORE slicing, so an absurd claim costs a
    // comparison rather than an allocation.
    if declared > MAX_HEADER_BYTES {
        refuse!(
            "header-too-large",
            "Header is too large: declares {declared} bytes, past the \
             {MAX_HEADER_BYTES}-byte limit"
        );
    }
    if raw.len() < 5 + declared {
        refuse!(
            "header-truncated",
            "Truncated header: declares {declared} bytes but only {} remain",
            raw.len() - 5
        );
    }

    Ok(Envelope {
        version,
        header_bytes: raw[5..5 + declared].to_vec(),
        payload: if keep_payload {
            raw[5 + declared..].to_vec()
        } else {
            Vec::new()
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_input_ceiling_is_the_same_for_a_prefix_read_and_a_full_one() {
        // Tested on the rule rather than on a 1 GiB buffer. What matters is
        // that ONE bound governs both entry points: a prefix read with a
        // tighter one would refuse members the full reader accepts, which
        // inverts the guarantee the header-only entry point makes.
        assert!(check_input_size(MAX_COMPRESSED_INPUT_BYTES).is_ok());
        let refused =
            check_input_size(MAX_COMPRESSED_INPUT_BYTES + 1).expect_err("past the ceiling");
        assert_eq!(refused.id, "gzip-malformed");
    }
}
