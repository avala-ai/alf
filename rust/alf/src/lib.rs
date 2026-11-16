//! Reference Rust reader for ALF (Arrays in Local Frames).
//!
//! The fourth language adapter, peer to `python/`, `dart/` and `typescript/`,
//! held to the same conformance corpus. It exists because an application
//! decodes `.alp` geometry inside a WASM module, and a decoder living in the
//! consuming repository is a second implementation of the format that drifts
//! from this one — which is exactly what happened: that copy still refused the
//! NaN binary16 carries as content, and had no handling for the v3 invalid-return
//! sentinel at all.
//!
//! Channel is inferred from the filename suffix, or stated explicitly. The
//! invalid corpus names its files `.bin` precisely so nothing can infer a
//! channel from them, which is why `extension` is a parameter rather than a
//! convenience.

pub mod decimal;
pub mod envelope;
pub mod error;
pub mod geometry;
pub mod header;

use std::collections::BTreeMap;

use error::refuse;
pub use error::{AlfError, Result};
use header::Value;

pub use geometry::{GeomCodec, GeometryDecoder, INT16_NO_RETURN};

const INT16_MAX: i16 = 32767;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Channel {
    Alp,
    Ali,
    Alc,
    All,
    Alx,
}

impl Channel {
    pub fn as_str(self) -> &'static str {
        match self {
            Channel::Alp => "alp",
            Channel::Ali => "ali",
            Channel::Alc => "alc",
            Channel::All => "all",
            Channel::Alx => "alx",
        }
    }

    fn from_str(text: &str) -> Option<Self> {
        match text {
            "alp" => Some(Channel::Alp),
            "ali" => Some(Channel::Ali),
            "alc" => Some(Channel::Alc),
            "all" => Some(Channel::All),
            "alx" => Some(Channel::Alx),
            _ => None,
        }
    }

    /// The load-bearing suffix of a filename, ignoring any `.gz`.
    pub fn from_path(path: &str) -> Option<Self> {
        let trimmed = path.strip_suffix(".gz").unwrap_or(path);
        let suffix = trimmed.rsplit('.').next()?;
        Channel::from_str(suffix)
    }
}

/// A unit quaternion, with its components NAMED.
///
/// Deliberately not `[f64; 4]`. Every array spelling of a quaternion has to
/// choose between `[x, y, z, w]` and `[w, x, y, z]`, both are in wide use, and
/// the two are indistinguishable in the values — so a caller that guesses wrong
/// misorients the entire cloud with nothing to catch it. That is the ambiguity
/// `_frame_rotation` is an object on the wire to remove, and handing it back as
/// a bare array would put it straight back at the API boundary.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrameRotation {
    pub w: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

/// A decoded ALF file. Channel-specific fields are populated for their channel.
#[derive(Clone, Debug)]
pub struct AlfFile {
    pub version: u8,
    pub channel: Channel,
    pub metadata: BTreeMap<String, Value>,

    // .alp
    pub geom_codec: Option<&'static str>,
    pub xyz: Vec<[f64; 3]>,
    pub max_geom_error_m: Option<f64>,
    /// How the local points reach the world. `None` for v1/v2, which do not say.
    pub frame_convention: Option<String>,
    /// The device rotation, present only on a `device-aligned` frame.
    pub frame_rotation: Option<FrameRotation>,

    // .ali
    pub intensity: Vec<u8>,

    // .alc
    pub palette: Vec<String>,
    pub palette_indices: Vec<u8>,

    // .all
    pub format_version: Option<u8>,
    pub categories: Vec<u8>,
    pub instance_ids: Option<Vec<u16>>,
    pub instance_map: Option<BTreeMap<String, Value>>,

    // .alx
    pub kept_indices: Vec<u32>,
}

impl AlfFile {
    fn empty(version: u8, channel: Channel, metadata: BTreeMap<String, Value>) -> Self {
        AlfFile {
            version,
            channel,
            metadata,
            geom_codec: None,
            xyz: Vec::new(),
            max_geom_error_m: None,
            frame_convention: None,
            frame_rotation: None,
            intensity: Vec::new(),
            palette: Vec::new(),
            palette_indices: Vec::new(),
            format_version: None,
            categories: Vec::new(),
            instance_ids: None,
            instance_map: None,
            kept_indices: Vec::new(),
        }
    }
}

/// An ALF envelope read WITHOUT its payload.
///
/// A distinct type from [`AlfFile`], deliberately, and this is where the Rust
/// and TypeScript readers differ from the Python reference: Python returns an
/// `AlfFile` whose payload accessors raise, because it cannot say "the same
/// thing, minus those fields" in its types. Rust can. `head.xyz` does not
/// compile rather than handing back an empty `Vec` that reads as "no points".
#[derive(Clone, Debug)]
pub struct AlfHeader {
    pub version: u8,
    pub channel: Channel,
    pub metadata: BTreeMap<String, Value>,
    /// Exact integral binary64 count, or `None` when omitted. It is not a
    /// machine-sized allocation length: header scans can observe larger counts.
    pub total_points: Option<f64>,
    /// How the local points reach the world. `None` for v1/v2, which do not say.
    pub frame_convention: Option<String>,
    /// The device rotation, present only on a `device-aligned` frame.
    ///
    /// Surfaced here and not left in `metadata` for the same reason
    /// [`FrameRotation`] exists at all: a caller pulling `w`/`x`/`y`/`z` out of
    /// a raw JSON object is doing by hand exactly the thing the named type
    /// prevents, and a header-only scan is where a frame convention is most
    /// likely to be read across a whole sequence.
    pub frame_rotation: Option<FrameRotation>,
}

/// The declared `.all` layout, or `None` when the header omits it.
///
/// The RANGE check only, and deliberately not the wire-byte-versus-field
/// disagreement `decode_all` also makes. Both are decidable from the header
/// here, but the other three implementations resolve that disagreement with the
/// payload, so enforcing it in a header-only read would make Rust's scans
/// refuse files the other scans accept — and header-only behaviour is not
/// corpus-backed, so nothing would catch the split. A subset that is not
/// maximal is still a subset; a divergence is not.
fn checked_label_layout(metadata: &BTreeMap<String, Value>) -> Result<Option<i64>> {
    let declared = match metadata.get("_format_version") {
        Some(value) => header_integer(value, "_format_version", "label-layout-unsupported")?,
        None => return Ok(None),
    };
    if declared != 1 && declared != 2 {
        refuse!(
            "label-layout-unsupported",
            "Unsupported `.all` _format_version {declared}"
        );
    }
    Ok(Some(declared))
}

/// Channel rules a header-only read must ALSO enforce.
///
/// The envelope rules apply to every channel; these depend on which one you
/// opened, and they lived only inside the channel decoders — so
/// [`read_alf_header_bytes`], which reaches no decoder, accepted files
/// [`read_alf_bytes`] refuses. That is the "validates less" drift the
/// header-only contract exists to rule out, and it was present in every
/// implementation.
///
/// Only rules the full reader enforces EAGERLY belong here. A `.alx` missing
/// `_source_total_points` is refused when its indices are decoded rather than
/// when the file is read, so adding it would make a header scan stricter than
/// the read it stands in for.
fn check_header_scoped_channel_rules(
    metadata: &BTreeMap<String, Value>,
    channel: Channel,
) -> Result<()> {
    if channel == Channel::Alx && !metadata.contains_key("_total_points") {
        refuse!(
            "total-points-missing",
            "Header is missing `_total_points`; a decimation sidecar indexes a \
             different cloud, so its own payload cannot state the count"
        );
    }
    if channel == Channel::All {
        checked_label_layout(metadata)?;
    }
    Ok(())
}

/// Everything about an envelope that is decidable WITHOUT its payload.
///
/// Shared by [`read_alf_bytes`] and [`read_alf_header_bytes`] so the two cannot
/// drift, and the direction they would drift in is the dangerous one: a
/// header-only reader that validated less would hand back a header the full
/// reader refuses, and a caller scanning a sequence would never learn the file
/// was bad. Payload-scoped rules live in the channel decoders, which only
/// [`read_alf_bytes`] reaches, because their evidence is bytes a header-only
/// read never fetched.
fn parse_header(
    version: u8,
    channel: Channel,
    metadata: BTreeMap<String, Value>,
) -> Result<AlfHeader> {
    // Version/channel cross-check. Versions 1 and 2 are `.all` LABEL layouts and
    // 3 is the `.alp`-only GEOMETRY codec, so the same byte means different
    // things per channel and a mismatch is a file whose bytes cannot be trusted.
    match version {
        1 => {}
        2 => {
            if channel != Channel::All {
                refuse!(
                    "label-version-on-non-label-channel",
                    "Version byte 2 is the `.all` interleaved label layout, but this file is \
                     .{}. Refusing to decode it as geometry",
                    channel.as_str()
                );
            }
        }
        3 => {
            if channel != Channel::Alp {
                refuse!(
                    "geometry-version-on-non-geometry-channel",
                    "Version byte 3 is fixed-point `.alp` geometry, but this file is .{}",
                    channel.as_str()
                );
            }
        }
        other => refuse!(
            "version-unsupported",
            "Unsupported ALF version byte {other}"
        ),
    }

    // ORDER MATTERS. A non-finite number breaks a different rule depending on
    // where it sits: `1e309` under `device_position` is `geom-number-inexact`,
    // the same token elsewhere is `header-not-json`. The specific check runs
    // first so the general one cannot claim a case that belongs to it.
    // .alp ONLY. These keys define a coordinate grid on a geometry file; the
    // same key on a `.ali` is metadata nobody decodes against, and refusing it
    // there would reject a conforming file over a number it never uses.
    if channel == Channel::Alp {
        check_geometry_anchor_shapes(&metadata)?;
        // Before the grid parameters: a file that does not say where its points
        // are is refused for that, which is decidable first and from less.
        check_frame_convention(&metadata, version)?;
        // `_geom_scale`/`_geom_offset` are the fixed-point grid. On a v1 fp16
        // file they are stale metadata nothing decodes against, so a bad number
        // there is the generic rule's business, not this one
        // (`v1-stale-geometry-number-outside-binary64`).
        check_geometry_numbers(&metadata, version == 3)?;
    }
    // Counts before the generic sweep, for the same reason: `1e400` under
    // `_total_points` is `total-points-not-an-integer`, not `header-not-json`.
    check_count_keys(&metadata)?;
    reject_nonfinite_tree(&metadata)?;

    // EAGERLY, rather than when a caller asks. The count is decidable from the
    // header alone, which makes refusing a malformed one a header-scoped rule —
    // and `check_count_keys` deliberately passes over a non-NUMBER, leaving that
    // to whichever decoder knows whether the key was required. A header-only
    // read reaches no decoder, so without this a `null` count would survive a
    // scan and surface later from whichever frame a caller looked at closely.
    // Python's reader had exactly that gap; TypeScript's did not.
    let total_points = declared_points(&metadata)?;
    check_header_scoped_channel_rules(&metadata, channel)?;

    let mut frame_convention = None;
    let mut frame_rotation = None;
    if channel == Channel::Alp {
        // Validated above; surfaced here because a reader that only VALIDATES
        // the declaration still cannot answer with it, and the bug this field
        // exists for was a field that was present and ignored.
        frame_convention = metadata
            .get("_frame_convention")
            .and_then(|value| value.as_str())
            .map(str::to_owned);
        frame_rotation = metadata.get("_frame_rotation").and_then(|value| {
            let map = value.as_object()?;
            Some(FrameRotation {
                w: map.get("w")?.as_f64()?,
                x: map.get("x")?.as_f64()?,
                y: map.get("y")?.as_f64()?,
                z: map.get("z")?.as_f64()?,
            })
        });
    }

    Ok(AlfHeader {
        version,
        channel,
        metadata,
        total_points,
        frame_convention,
        frame_rotation,
    })
}

/// Read one ALF file from bytes. `extension` states the channel when the
/// filename cannot ("alp", "ali", ...); empty means infer from `path`.
pub fn read_alf_bytes(bytes: &[u8], channel: Channel) -> Result<AlfFile> {
    let env = envelope::read(bytes)?;
    let head = parse_header(env.version, channel, header::parse(&env.header_bytes)?)?;

    let mut file = AlfFile::empty(head.version, channel, head.metadata);
    file.frame_convention = head.frame_convention;
    file.frame_rotation = head.frame_rotation;
    match channel {
        Channel::Alp => decode_alp(&mut file, &env.payload)?,
        Channel::Ali => decode_ali(&mut file, &env.payload)?,
        Channel::Alc => decode_alc(&mut file, &env.payload)?,
        Channel::All => decode_all(&mut file, &env.payload)?,
        Channel::Alx => decode_alx(&mut file, &env.payload)?,
    }
    Ok(file)
}

/// Parse ONLY an ALF file's envelope, without inflating or reading its payload.
///
/// For the scan-many-frames case: a prefetch that needs `_total_points`,
/// `device_position`, `_geom_scale` or a `palette` across a thousand frames
/// should not pay a full decompression for each. [`read_alf_bytes`] inflates
/// everything before it returns anything; this stops after the header — about
/// 64 KiB of a gzipped frame however large the frame is.
///
/// ITS REFUSALS ARE A STRICT SUBSET OF [`read_alf_bytes`]'s, AND THAT SPLIT IS
/// THE CONTRACT. Every header-scoped rule still applies — both share
/// `parse_header`, so they agree by construction rather than by review. The
/// payload-scoped rules cannot apply, because their evidence is the bytes this
/// declined to read:
///
/// * the declared count is not cross-checked against the payload length;
/// * a second gzip member goes undetected;
/// * the member's CRC32 and ISIZE go unverified.
///
/// So a header this accepts is a header [`read_alf_bytes`] would also accept. A
/// FILE it accepts is not necessarily a file [`read_alf_bytes`] would: a frame
/// truncated mid-download still yields its header here.
///
/// A separate function rather than a flag on the existing one, on purpose: a
/// boolean lets the weaker contract be selected by a variable at a call site
/// that reads exactly like the strong one.
pub fn read_alf_header_bytes(bytes: &[u8], channel: Channel) -> Result<AlfHeader> {
    let env = envelope::read_header(bytes)?;
    parse_header(env.version, channel, header::parse(&env.header_bytes)?)
}

/// The channel a path read should use: the caller's, or the filename's.
fn resolve_channel(path: &str, extension: &str) -> Result<Channel> {
    if extension.is_empty() {
        Channel::from_path(path).ok_or_else(|| {
            AlfError::new(
                "",
                format!("Cannot infer an ALF channel from {path:?}; pass one explicitly"),
            )
        })
    } else {
        Channel::from_str(extension)
            .ok_or_else(|| AlfError::new("", format!("Unknown ALF channel {extension:?}")))
    }
}

/// Read from a path, inferring the channel from the suffix when not stated.
pub fn read_alf(path: &str, extension: &str) -> Result<AlfFile> {
    let channel = resolve_channel(path, extension)?;
    let bytes = std::fs::read(path)
        .map_err(|err| AlfError::new("", format!("Cannot read {path:?}: {err}")))?;
    read_alf_bytes(&bytes, channel)
}

/// [`read_alf_header_bytes`] from a path, reading only as much of the file as
/// the header needs.
///
/// Not `read_alf_header_bytes(&std::fs::read(path)?, ..)`: that pays for the
/// whole payload before the parser is reached, which is the cost this entry
/// point exists to avoid. This streams a bounded prefix instead.
pub fn read_alf_header(path: &str, extension: &str) -> Result<AlfHeader> {
    let channel = resolve_channel(path, extension)?;
    let env = envelope::read_header_from_path(path)?;
    parse_header(env.version, channel, header::parse(&env.header_bytes)?)
}

// ─────────────────────────────────────────────────────────────────────────────
// Header helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Header numbers that DEFINE a coordinate grid, and must therefore state
/// exactly what float64 will read back.
///
/// `9007199254740993.0` is an ordinary JSON token that becomes `...992.0` once
/// read — one metre out for an anchor, a different grid for a codec parameter.
/// Magnitude alone is fine: `1e+308` round-trips unchanged and is nobody's
/// precision claim.
const GEOMETRY_NUMBER_KEYS: [&str; 4] = [
    "device_position",
    "world_centroid_epsg3857",
    "_geom_offset",
    "_geom_scale",
];

fn check_geometry_numbers(metadata: &BTreeMap<String, Value>, fixed_point: bool) -> Result<()> {
    for key in GEOMETRY_NUMBER_KEYS {
        if !fixed_point && (key == "_geom_scale" || key == "_geom_offset") {
            continue;
        }
        let value = match metadata.get(key) {
            Some(value) => value,
            None => continue,
        };
        match value {
            // An anchor's grid is its x/y/z. Any OTHER key it carries is
            // ordinary metadata that nothing decodes against, so a bad number
            // there belongs to the generic non-finite rule, not this one
            // (`anchor-extra-number-outside-binary64`).
            Value::Object(map) => {
                for axis in ["x", "y", "z"] {
                    if let Some(component) = map.get(axis) {
                        check_exact(key, component)?;
                    }
                }
            }
            Value::Array(items) => {
                for item in items {
                    check_exact(key, item)?;
                }
            }
            _ => check_exact(key, value)?,
        }
    }
    Ok(())
}

/// An anchor states a position, so it needs all three axes.
///
/// `{"x": 1, "y": 2}` is not a partial position, it is an unusable one: the
/// reader would have to invent a `z` and would place the cloud somewhere the
/// file never claimed.
/// The v3 frame convention: how a `.alp`'s local points reach the world.
///
/// A `.alp`'s points are the world points with an origin subtracted, which is
/// HALF of a rigid transform. Producers disagreed about the other half — some
/// translated and stopped, leaving points on world axes, some also rotated into
/// the device frame — and nothing in the bytes told them apart. The two agree
/// exactly whenever the device heading is identity, so a consumer that omitted
/// the rotation matched a producer that omitted it and the pair looked right
/// until the vehicle turned. See `spec/format.md` §5.3.
fn check_frame_convention(metadata: &BTreeMap<String, Value>, version: u8) -> Result<()> {
    let declared = metadata.get("_frame_convention");
    let rotation = metadata.get("_frame_rotation");

    // Versions 1 and 2 are grandfathered: they predate the field and a deployed
    // file cannot be told retroactively what it meant. What they may not do is
    // CARRY it — this reader applies no rotation to v1/v2 geometry, so the
    // declaration would be a promise the format does not keep.
    if version != 3 {
        if declared.is_some() || rotation.is_some() {
            let named = if declared.is_some() {
                "_frame_convention"
            } else {
                "_frame_rotation"
            };
            refuse!(
                "frame-convention-on-legacy-version",
                "`{named}` is a version-3 field, but this .alp is version {version}. Readers \
                 apply no rotation to v1/v2 geometry, so honouring it is exactly what they \
                 will not do"
            );
        }
        return Ok(());
    }

    let declared = match declared {
        Some(value) => value,
        None => refuse!(
            "frame-convention-missing",
            "version-3 .alp is missing `_frame_convention`; its points are the world points \
             with an origin subtracted, and the file must say whether the device rotation was \
             applied as well"
        ),
    };
    let declared = match declared.as_str() {
        Some(text) if text == "world-aligned" || text == "device-aligned" => text,
        _ => refuse!(
            "frame-convention-unsupported",
            "Unsupported `_frame_convention`; expected \"world-aligned\" or \"device-aligned\""
        ),
    };

    if declared == "world-aligned" {
        if rotation.is_some() {
            refuse!(
                "frame-rotation-unexpected",
                "`_frame_convention: world-aligned` declares that no rotation was applied, but \
                 `_frame_rotation` is present"
            );
        }
        return Ok(());
    }

    let rotation = match rotation {
        Some(value) => value,
        None => refuse!(
            "frame-rotation-missing",
            "`_frame_convention: device-aligned` requires `_frame_rotation`; without it the \
             points cannot be returned to the world at all"
        ),
    };
    let map = match rotation.as_object() {
        Some(map) => map,
        None => refuse!(
            "frame-rotation-malformed",
            "`_frame_rotation` must be an object with members w, x, y and z"
        ),
    };
    // EXACTLY the four members: a stray `qw` beside `w` is a producer mixing two
    // spellings, not application data, and only one of them would be read.
    if map.len() != 4 {
        refuse!(
            "frame-rotation-malformed",
            "`_frame_rotation` must carry exactly w, x, y and z"
        );
    }
    let mut norm_squared = 0.0_f64;
    for axis in ["w", "x", "y", "z"] {
        let component = match map.get(axis) {
            Some(value) => value,
            None => refuse!(
                "frame-rotation-malformed",
                "`_frame_rotation` is missing `{axis}`"
            ),
        };
        // The same exact-token rule the grid parameters are under: a component
        // whose stated decimal changes when decoded turns the whole frame.
        check_exact("_frame_rotation", component)?;
        let number = match component.as_f64() {
            Some(number) if number.is_finite() => number,
            _ => refuse!(
                "frame-rotation-malformed",
                "`_frame_rotation.{axis}` must be a finite JSON number"
            ),
        };
        norm_squared += number * number;
    }
    // Conjugation by a non-unit quaternion scales by |q|^2 as well as rotating,
    // uniformly across the frame and nowhere else in the file. Refused rather
    // than normalized: normalizing would hide the producer's fault, not fix it.
    if (norm_squared - 1.0).abs() > 1e-9 {
        refuse!(
            "frame-rotation-not-normalized",
            "`_frame_rotation` is not a unit quaternion (|q|^2 = {norm_squared})"
        );
    }
    Ok(())
}

fn check_geometry_anchor_shapes(metadata: &BTreeMap<String, Value>) -> Result<()> {
    for key in ["device_position", "world_centroid_epsg3857"] {
        let value = match metadata.get(key) {
            Some(value) => value,
            None => continue,
        };
        let map = match value {
            Value::Object(map) => map,
            _ => refuse!(
                "geometry-anchor-malformed",
                "Geometry anchor `{key}` must be an object with x, y and z"
            ),
        };
        // An EMPTY anchor is legacy-valid: it states no position, which older
        // writers used to mean "local frame, no offset". A PARTIAL one is the
        // unusable case — `{"x":1,"y":2}` would make the reader invent a `z`.
        if map.is_empty() {
            continue;
        }
        for axis in ["x", "y", "z"] {
            match map.get(axis) {
                Some(Value::Number(..)) => {}
                _ => refuse!(
                    "geometry-anchor-malformed",
                    "Geometry anchor `{key}` is missing a numeric `{axis}`"
                ),
            }
        }
    }
    Ok(())
}

/// `_total_points` / `_source_total_points`: integer-valued, and exactly what
/// float64 reads back.
fn check_count_keys(metadata: &BTreeMap<String, Value>) -> Result<()> {
    for key in ["_total_points", "_source_total_points"] {
        let value = match metadata.get(key) {
            Some(value) => value,
            None => continue,
        };
        let (decoded, token) = match value {
            Value::Number(decoded, token) => (*decoded, token.as_str()),
            // A non-number is caught with the right identifier by the channel
            // decoder, which knows whether the key was required at all.
            _ => continue,
        };
        let exact = match decimal::parse(token) {
            Some(exact) => exact,
            None => continue,
        };
        // A hostile exponent is refused on the LITERAL, before any float. Every
        // finite binary64 integer has at most 309 decimal digits, so a token
        // claiming more is out of range whichever way it is spelled — and
        // `1e-400`, which silently decodes to 0.0, is not a count either.
        let digit_span = exact.digits.len() as i64 + exact.exponent;
        if !exact.is_zero() && (digit_span > 309 || exact.exponent < 0) {
            refuse!(
                "total-points-not-an-integer",
                "`{key}` is {token:?}, which is not a non-negative integer binary64 can hold"
            );
        }
        if !decoded.is_finite() {
            refuse!(
                "total-points-not-an-integer",
                "`{key}` is {token:?}, an integer too large for binary64"
            );
        }
        if decoded.fract() != 0.0 || decoded < 0.0 {
            refuse!(
                "total-points-not-an-integer",
                "`{key}` must be a non-negative integer, got {token:?}"
            );
        }
        if decimal::from_f64_integer(decoded).as_ref() != Some(&exact) {
            refuse!(
                "total-points-not-an-integer",
                "`{key}` is {token:?}, which binary64 cannot represent exactly"
            );
        }
    }
    Ok(())
}

/// True for a writer's own normalised scientific spelling, e.g. `1.5e+20`.
///
/// Matches `-?[0-9](\.[0-9]*[1-9])?[eE][+-]?[1-9][0-9]*` without a regex crate.
fn is_canonical_scientific(token: &str) -> bool {
    let bytes = token.as_bytes();
    let mut i = 0usize;
    if i < bytes.len() && bytes[i] == b'-' {
        i += 1;
    }
    if i >= bytes.len() || !bytes[i].is_ascii_digit() {
        return false;
    }
    i += 1;
    if i < bytes.len() && bytes[i] == b'.' {
        i += 1;
        let start = i;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
        }
        // The fraction must end in a non-zero digit, or it is not shortest.
        if i == start || bytes[i - 1] == b'0' {
            return false;
        }
    }
    if i >= bytes.len() || (bytes[i] != b'e' && bytes[i] != b'E') {
        return false;
    }
    i += 1;
    if i < bytes.len() && (bytes[i] == b'+' || bytes[i] == b'-') {
        i += 1;
    }
    if i >= bytes.len() || !(b'1'..=b'9').contains(&bytes[i]) {
        return false;
    }
    i += 1;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        i += 1;
    }
    i == bytes.len()
}

fn check_exact(key: &str, value: &Value) -> Result<()> {
    let (decoded, token) = match value {
        Value::Number(decoded, token) => (*decoded, token.as_str()),
        _ => return Ok(()),
    };
    let exact = match decimal::parse(token) {
        Some(exact) => exact,
        None => return Ok(()),
    };
    if !decoded.is_finite() {
        refuse!(
            "geom-number-inexact",
            "`{key}` is {token:?}, too large to represent as finite float64"
        );
    }

    // A raw integer token states an exact integer, even when the decoded float's
    // SHORTEST spelling happens to equal the original text — `18446744073709552000`
    // reads back as the different integer `18446744073709551616` while printing
    // as `1.8446744073709552e19`, so the shortest-repr comparison below cannot
    // see the shifted grid on its own.
    let plain_integer = !token.contains('.') && !token.contains('e') && !token.contains('E');
    if plain_integer {
        if decimal::from_f64_integer(decoded).as_ref() == Some(&exact) {
            return Ok(());
        }
        refuse!(
            "geom-number-inexact",
            "`{key}` is {token:?} in the header but float64 cannot represent exactly what \
             that integer token states"
        );
    }

    if exact.is_integral() {
        if decimal::from_f64_integer(decoded).as_ref() == Some(&exact) {
            return Ok(());
        }
        // Expanded `.0` and exponent-zero spellings are still exact integer
        // claims; they do not inherit the escape below merely because both forms
        // canonicalise to the same text.
        if is_canonical_scientific(token)
            && decimal::from_f64_shortest(decoded).as_ref() == Some(&exact)
        {
            return Ok(());
        }
        refuse!(
            "geom-number-inexact",
            "`{key}` is {token:?} in the header but float64 cannot represent exactly what \
             that integral token states"
        );
    }

    if decimal::from_f64_shortest(decoded).as_ref() != Some(&exact) {
        refuse!(
            "geom-number-inexact",
            "`{key}` is {token:?} in the header but {decoded} once read — float64 cannot \
             represent exactly what the token states"
        );
    }
    Ok(())
}

/// Refuse any header number outside finite binary64, walking iteratively so a
/// deep header cannot recurse the stack.
fn reject_nonfinite_tree(metadata: &BTreeMap<String, Value>) -> Result<()> {
    let mut pending: Vec<&Value> = metadata.values().collect();
    while let Some(value) = pending.pop() {
        match value {
            Value::Number(parsed, token) => {
                if !parsed.is_finite() {
                    refuse!(
                        "header-not-json",
                        "Header number {token:?} is not finite; JSON has no NaN or Infinity"
                    );
                }
            }
            Value::Array(items) => pending.extend(items.iter()),
            Value::Object(map) => pending.extend(map.values()),
            _ => {}
        }
    }
    Ok(())
}

/// A header integer, refused when float64 cannot hold it exactly.
///
/// JSON numbers arrive as float64. `9007199254740993` parses to
/// `9007199254740992` — a DIFFERENT integer — so the token is compared against
/// the round-trip rather than trusting the parsed value.
fn header_integer(value: &Value, key: &str, id: &'static str) -> Result<i64> {
    let (parsed, token) = match value {
        Value::Number(parsed, token) => (*parsed, token.as_str()),
        Value::Null => refuse!(id, "`{key}` is present but null; it must be an integer"),
        _ => refuse!(id, "`{key}` must be an integer"),
    };
    // VALUE-based, not notation-based. `7e0` and `700e-2` both denote the
    // integer 7, and a writer is free to emit either — refusing them because the
    // token contains an `e` would reject a conforming file over its spelling.
    if parsed.fract() != 0.0 {
        refuse!(id, "`{key}` must be an integer, got {token:?}");
    }
    // NOT a magnitude cap. `18014398509481984` is 2**54 — past 2**53, and
    // exactly representable, because it is a power of two. Magnitude is not
    // precision loss; the round-trip below is what actually decides.
    // The one thing notation DOES decide: a plain integer token names an exact
    // integer, so if re-rendering the parsed value differs, float64 silently
    // changed which integer this is (`9007199254740993` -> `...992`). Exponent
    // and fractional spellings are already covered by the checks above.
    let plain_integer = !token.contains('.') && !token.contains('e') && !token.contains('E');
    if plain_integer {
        let round_trip = format!("{}", parsed as i64);
        if round_trip != token.trim_start_matches('+') {
            refuse!(
                id,
                "`{key}` is {token:?}, which binary64 cannot represent exactly"
            );
        }
    }
    Ok(parsed as i64)
}

/// The declared count, or `None` when the header does not state one.
///
/// Only `.alx` requires it: a decimation sidecar's payload cannot be
/// self-describing because its whole purpose is to index a DIFFERENT cloud. For
/// every other channel the payload length states the count, so a header that
/// omits it is merely terse — `ali-zero-length-header` is a valid file with no
/// header at all.
fn declared_points(metadata: &BTreeMap<String, Value>) -> Result<Option<f64>> {
    let value = match metadata.get("_total_points") {
        Some(value) => value,
        None => return Ok(None),
    };
    header_count(value, "_total_points").map(Some)
}

fn total_points(file: &AlfFile) -> Result<f64> {
    let value = file.metadata.get("_total_points").ok_or_else(|| {
        AlfError::new(
            "total-points-missing",
            "Header is missing `_total_points`; the payload length cannot be checked without it",
        )
    })?;
    header_count(value, "_total_points")
}

fn header_count(value: &Value, key: &str) -> Result<f64> {
    // Raw-token exactness was checked by check_count_keys at the envelope.
    // Preserve that value: narrowing to i64/usize clamps or truncates counts
    // which the format permits, and can turn a huge payload claim into zero.
    match value
        .as_f64()
        .filter(|n| n.is_finite() && *n >= 0.0 && n.fract() == 0.0)
    {
        Some(count) => Ok(count),
        None => refuse!(
            "total-points-not-an-integer",
            "`{key}` must be a non-negative integer"
        ),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Channels
// ─────────────────────────────────────────────────────────────────────────────

fn decode_alp(file: &mut AlfFile, payload: &[u8]) -> Result<()> {
    if !payload.len().is_multiple_of(2) {
        refuse!(
            "payload-not-whole-points",
            "Payload is not a whole number of 16-bit words"
        );
    }
    let words = payload.len() / 2;
    if !words.is_multiple_of(3) {
        refuse!(
            "payload-not-whole-points",
            "Payload holds {words} values, not a multiple of 3 values per point"
        );
    }
    let stored = words / 3;
    if let Some(declared) = declared_points(&file.metadata)? {
        if stored as f64 != declared {
            refuse!(
                "payload-length-mismatch",
                "Header declares {declared} points but the payload holds {stored}"
            );
        }
    }

    // Codec selection. Version 3 REQUIRES int16 and int16 requires version 3, so
    // a file cannot claim one and store the other.
    let declared_codec = file
        .metadata
        .get("_geom_codec")
        .and_then(|v| v.as_str())
        .unwrap_or("fp16");
    let codec = match declared_codec {
        "fp16" => "fp16",
        "int16" => "int16",
        other => refuse!("codec-unsupported", "Unknown .alp geometry codec {other:?}"),
    };
    if file.version == 3 && codec != "int16" {
        refuse!(
            "codec-version-disagreement",
            "Version-3 .alp declares codec {codec:?}; v3 is fixed-point geometry and \
             requires `_geom_codec: 'int16'`"
        );
    }
    if codec == "int16" && file.version != 3 {
        refuse!(
            "codec-version-disagreement",
            "Fixed-point geometry is version 3, but this file declares version {}",
            file.version
        );
    }
    file.geom_codec = Some(codec);

    let mut xyz = Vec::with_capacity(stored);
    if codec == "fp16" {
        let decoder = GeometryDecoder::fp16();
        for point in 0..stored {
            let mut coords = [0f64; 3];
            // `axis` is load-bearing, not an index accident: it addresses the
            // payload word AND the component it lands in, so zipping would hide
            // the correspondence this decoder is entirely about.
            #[allow(clippy::needless_range_loop)]
            for axis in 0..3 {
                let at = (point * 3 + axis) * 2;
                let bits = u16::from_le_bytes([payload[at], payload[at + 1]]);
                // Through the SHARED primitive, so the file reader and the
                // word-level boundary cannot disagree about what a word means.
                // Non-finite is content: a NaN here is the invalid-return marker
                // (spec/format.md §5.1), kept IN PLACE so entry i still describes
                // the same point as entry i of every companion channel.
                coords[axis] = decoder.component(bits, axis);
            }
            xyz.push(coords);
        }
        file.max_geom_error_m = fp16_error_bound(&xyz);
    } else {
        let scale = geom_scale(file)?;
        let offset = geom_offset(file)?;
        // Built here so the parameter validation lives in ONE place; the reader
        // has already refused a non-positive scale above, and this would refuse
        // it again rather than decode against it.
        let decoder = GeometryDecoder::int16(scale, offset)
            .map_err(|message| AlfError::new("geom-scale-not-positive", message))?;
        let mut partial_marker = false;
        let mut reconstruction = 0f64;
        for point in 0..stored {
            let mut raw = [0i16; 3];
            #[allow(clippy::needless_range_loop)]
            for axis in 0..3 {
                let at = (point * 3 + axis) * 2;
                raw[axis] = i16::from_le_bytes([payload[at], payload[at + 1]]);
            }
            let words = [raw[0] as u16, raw[1] as u16, raw[2] as u16];
            if decoder.is_no_return(&words, 0) {
                xyz.push([f64::NAN; 3]);
                continue;
            }
            // A PARTIAL marker is not an absent measurement — no conforming
            // writer can emit the reserved word as a coordinate, so the file is
            // corrupt. The primitive deliberately does not decide this: a word
            // cannot be corrupt, only a file can.
            if raw.contains(&INT16_NO_RETURN) {
                partial_marker = true;
                break;
            }
            let mut coords = [0f64; 3];
            #[allow(clippy::needless_range_loop)]
            for axis in 0..3 {
                let quantized = raw[axis] as f64;
                let product = quantized * scale;
                let total = product + offset[axis];
                // TwoSum recovers the addition's EXACT rounding error; half the
                // larger adjacent ULP bounds the multiplication's. Measured on
                // the words actually stored, never on the extremes of the int16
                // range — the extremes overstate the bound by orders of
                // magnitude for a frame that never reaches them.
                let near = total - product;
                let add_error = (product - (total - near)) + (offset[axis] - near);
                let rounding = add_error.abs() + adjacent_ulp(product) / 2.0;
                if rounding.is_finite() {
                    reconstruction = reconstruction.max(rounding);
                }
                coords[axis] = total;
            }
            if !coords.iter().all(|c| c.is_finite()) {
                refuse!(
                    "coordinate-non-finite",
                    "int16 .alp decodes to non-finite coordinates; `_geom_scale`/`_geom_offset` \
                     overflow the representable range"
                );
            }
            xyz.push(coords);
        }
        if partial_marker {
            refuse!(
                "geom-partial-no-return",
                "int16 .alp contains a point with SOME but not all components equal to the \
                 reserved invalid-return marker ({INT16_NO_RETURN}). No conforming writer can \
                 emit that word as a coordinate, so this payload is corrupt"
            );
        }
        file.max_geom_error_m = Some(int16_error_bound(file, scale, reconstruction, &xyz));
    }
    file.xyz = xyz;
    Ok(())
}

fn geom_scale(file: &AlfFile) -> Result<f64> {
    let value = file.metadata.get("_geom_scale").ok_or_else(|| {
        AlfError::new(
            "geom-scale-missing",
            "Fixed-point geometry is missing `_geom_scale`",
        )
    })?;
    let scale = value.as_f64().ok_or_else(|| {
        AlfError::new(
            "geom-scale-not-positive",
            "`_geom_scale` must be a positive finite number",
        )
    })?;
    if !scale.is_finite() || scale <= 0.0 {
        refuse!(
            "geom-scale-not-positive",
            "`_geom_scale` must be a positive finite number, got {scale}"
        );
    }
    Ok(scale)
}

fn geom_offset(file: &AlfFile) -> Result<[f64; 3]> {
    let value = file.metadata.get("_geom_offset").ok_or_else(|| {
        AlfError::new(
            "geom-offset-malformed",
            "Fixed-point geometry is missing `_geom_offset`",
        )
    })?;
    let items = value.as_array().ok_or_else(|| {
        AlfError::new(
            "geom-offset-malformed",
            "`_geom_offset` must have exactly 3 components",
        )
    })?;
    if items.len() != 3 {
        refuse!(
            "geom-offset-malformed",
            "`_geom_offset` must have exactly 3 components, got {}",
            items.len()
        );
    }
    let mut out = [0f64; 3];
    for (index, item) in items.iter().enumerate() {
        let value = item.as_f64().ok_or_else(|| {
            AlfError::new(
                "geom-offset-malformed",
                "`_geom_offset` must have exactly 3 finite numeric components",
            )
        })?;
        if !value.is_finite() {
            refuse!(
                "geom-offset-malformed",
                "`_geom_offset` must have exactly 3 finite numeric components"
            );
        }
        out[index] = value;
    }
    Ok(out)
}

/// Half the largest adjacent binary16 ULP over the finite components.
fn fp16_error_bound(xyz: &[[f64; 3]]) -> Option<f64> {
    let mut largest = 0f64;
    for point in xyz {
        for &component in point {
            if component.is_finite() {
                largest = largest.max(component.abs());
            }
        }
    }
    if xyz.is_empty() {
        return Some(0.0);
    }
    // With no finite components, use zero's subnormal spacing. Invalid
    // returns have no magnitude, but a nonempty fp16 frame is not lossless.
    let half = half::f16::from_f64(largest);
    let next = half::f16::from_bits(half.to_bits() + 1);
    let spacing = if next.is_finite() {
        next.to_f64() - largest
    } else {
        // At 65504 the upward neighbor is infinity; the finite gap below is
        // still 32, so the frame's half-ULP remains 16.
        largest - half::f16::from_bits(half.to_bits() - 1).to_f64()
    };
    Some(spacing / 2.0)
}

/// The larger finite adjacent binary64 gap, using the lower gap at MAX.
fn adjacent_ulp(value: f64) -> f64 {
    if !value.is_finite() {
        return 0.0;
    }
    let bits = value.abs().to_bits();
    let next = f64::from_bits(bits + 1);
    if next.is_finite() {
        next - value.abs()
    } else {
        value.abs() - f64::from_bits(bits - 1)
    }
}

/// `scale/2` PLUS the measured reconstruction rounding, floored against the
/// header's own stated bound.
///
/// Half the step is only the quantisation error. Recovering `stored * scale +
/// offset` in float64 rounds twice more, and at a scale near 1e19 that second
/// rounding is a whole ULP — enough to make a bound that omits it understate the
/// error it claims to bound.
fn int16_error_bound(file: &AlfFile, scale: f64, reconstruction: f64, xyz: &[[f64; 3]]) -> f64 {
    if xyz.is_empty() {
        return 0.0;
    }
    let half_step = scale / 2.0;
    let mut analytic = half_step + reconstruction;
    if scale < 1e-300 {
        // Halving can underflow to zero or round a subnormal downward.
        // Match the other readers' upward floor before honoring a recorded
        // value, which may be absent or may understate this grid's error.
        analytic = analytic.max(f64::from_bits(half_step.to_bits() + 1));
    }
    let recorded = file
        .metadata
        .get("_geom_max_err_m")
        .and_then(|v| v.as_f64())
        .filter(|v| v.is_finite())
        .unwrap_or(0.0);
    analytic.max(recorded)
}

fn decode_ali(file: &mut AlfFile, payload: &[u8]) -> Result<()> {
    let declared = declared_points(&file.metadata)?.unwrap_or(payload.len() as f64);
    if payload.len() as f64 != declared {
        refuse!(
            "payload-length-mismatch",
            "Header declares {declared} points but the payload holds {}",
            payload.len()
        );
    }
    file.intensity = payload.to_vec();
    Ok(())
}

fn decode_alc(file: &mut AlfFile, payload: &[u8]) -> Result<()> {
    let declared = declared_points(&file.metadata)?.unwrap_or(payload.len() as f64);
    if payload.len() as f64 != declared {
        refuse!(
            "payload-length-mismatch",
            "Header declares {declared} points but the payload holds {}",
            payload.len()
        );
    }
    let entries = file
        .metadata
        .get("palette")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    let mut palette = Vec::with_capacity(entries.len());
    for entry in &entries {
        let text = entry.as_str().ok_or_else(|| {
            AlfError::new(
                "palette-entry-malformed",
                "Palette entries are six hexadecimal digits",
            )
        })?;
        let body = text.strip_prefix('#').unwrap_or(text);
        if body.len() != 6 || !body.bytes().all(|b| b.is_ascii_hexdigit()) {
            refuse!(
                "palette-entry-malformed",
                "Palette entry {text:?} is not six hexadecimal digits"
            );
        }
        // Normalised, not echoed. `#FF0000` and `ff0000` name one colour, and a
        // reader that returned them differently would make two identical
        // palettes compare unequal.
        palette.push(body.to_ascii_lowercase());
    }
    for &index in payload {
        if palette.is_empty() || index as usize >= palette.len() {
            refuse!(
                "palette-index-out-of-range",
                "Palette index {index} is out of range for a {}-entry palette",
                palette.len()
            );
        }
    }
    file.palette = palette;
    file.palette_indices = payload.to_vec();
    Ok(())
}

fn decode_all(file: &mut AlfFile, payload: &[u8]) -> Result<()> {
    // The layout disagreement is decided BEFORE the count: a file claiming both
    // layouts is ambiguous whether or not it also states `_total_points`.
    let declared_layout = file
        .metadata
        .get("_format_version")
        .map(|v| header_integer(v, "_format_version", "label-layout-unsupported"))
        .transpose()?;

    // The wire byte SELECTS the layout: 1 is one byte per point, 2 is the
    // interleaved three. `_format_version` may restate it but may not contradict
    // it — a file claiming both decodes two different ways and the payload
    // length cannot arbitrate, least of all when it is empty and fits either.
    let wire_layout = if file.version == 2 { 2u8 } else { 1u8 };
    if let Some(declared) = declared_layout {
        if declared != 1 && declared != 2 {
            refuse!(
                "label-layout-unsupported",
                "Unsupported `.all` _format_version {declared}"
            );
        }
        if declared as u8 != wire_layout {
            refuse!(
                "label-layout-ambiguous",
                "Wire version {} selects label layout {wire_layout} but `_format_version`                  says {declared}; the payload fits BOTH layouts and cannot arbitrate",
                file.version
            );
        }
    }
    let layout = wire_layout;
    let stride = if layout == 1 { 1 } else { 3 };
    let stored = payload.len() / stride;
    let declared = declared_points(&file.metadata)?.unwrap_or(stored as f64);
    if payload.len() as f64 != declared * stride as f64 {
        refuse!(
            "payload-length-mismatch",
            "Header declares {declared} points but the payload holds {} bytes",
            payload.len()
        );
    }

    file.format_version = Some(layout);
    if layout == 1 {
        file.categories = payload.to_vec();
        file.instance_ids = None;
    } else {
        let mut categories = Vec::with_capacity(stored);
        let mut ids = Vec::with_capacity(stored);
        for point in 0..stored {
            categories.push(payload[point * 3]);
            let id = u16::from_le_bytes([payload[point * 3 + 1], payload[point * 3 + 2]]);
            if id == u16::MAX {
                refuse!(
                    "instance-id-reserved",
                    "Point {point} uses the reserved instance id 65535"
                );
            }
            ids.push(id);
        }
        file.categories = categories;
        file.instance_ids = Some(ids);
    }

    if let Some(map) = file
        .metadata
        .get("instance_map")
        .and_then(|v| v.as_object())
    {
        for key in map.keys() {
            if key.parse::<u32>().is_err() || (key.len() > 1 && key.starts_with('0')) {
                refuse!(
                    "instance-map-malformed",
                    "`instance_map` has non-canonical key(s): {key:?}"
                );
            }
        }
        if let Some(ids) = &file.instance_ids {
            for id in ids {
                if *id != 0 && !map.contains_key(&id.to_string()) {
                    refuse!(
                        "instance-map-incomplete",
                        "`instance_map` is missing entries for instance id {id}"
                    );
                }
            }
        }
        file.instance_map = Some(map.clone());
    }
    Ok(())
}

fn decode_alx(file: &mut AlfFile, payload: &[u8]) -> Result<()> {
    let source = file.metadata.get("_source_total_points").ok_or_else(|| {
        AlfError::new(
            "source-total-points-missing",
            "Decimation sidecar is missing `_source_total_points`",
        )
    })?;
    let source_total = header_count(source, "_source_total_points")?;
    let declared = total_points(file)?;

    if !payload.len().is_multiple_of(4) {
        refuse!(
            "payload-not-whole-points",
            "Payload is not a whole number of 32-bit indices"
        );
    }
    let count = payload.len() / 4;
    if count as f64 != declared {
        refuse!(
            "payload-length-mismatch",
            "Header declares {declared} kept indices but the payload holds {count}"
        );
    }

    let mut kept = Vec::with_capacity(count);
    let mut seen = std::collections::BTreeSet::new();
    for index in 0..count {
        let at = index * 4;
        let value = u32::from_le_bytes([
            payload[at],
            payload[at + 1],
            payload[at + 2],
            payload[at + 3],
        ]);
        if value as f64 >= source_total {
            refuse!(
                "kept-index-out-of-range",
                "Kept index {value} is out of range for a source cloud of {source_total} points"
            );
        }
        if !seen.insert(value) {
            refuse!(
                "kept-index-duplicate",
                "Sidecar contains duplicate kept indices ({value})"
            );
        }
        kept.push(value);
    }
    file.kept_indices = kept;
    Ok(())
}

/// Exposed so the writer-side clamp and the reader agree on the range.
pub const fn int16_max() -> i16 {
    INT16_MAX
}
