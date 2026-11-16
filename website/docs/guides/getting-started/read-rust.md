# Read in Rust

`rust/alf` is a **reader**. Its public API is `read_alf` / `read_alf_bytes` and nothing else, so it
earns the reader conformance gate and is `n/a` for the writer one rather than ❌ on a gate it never
entered.

## Why it exists

The crate provides a conformance-verified decoder for native Rust and WebAssembly applications,
including binary16 invalid returns and the v3 fixed-point sentinel. Applications can use it
without maintaining their own envelope parser.

`wasm32-unknown-unknown` is a supported target. Its only inflate dependency is `flate2`
with the `rust_backend` feature, because a C zlib
would not build for wasm32.

## Add it

The crate is ready for its first crates.io release, but version 0.1.0 has not been bootstrapped there
yet. Until it is available, depend on it from Git:

```toml
[dependencies]
alf = { git = "https://github.com/avala-ai/alf.git", package = "avala-alf" }
```

## Read a frame

```rust
use alf::{read_alf_bytes, Channel};

let file = read_alf_bytes(&bytes, Channel::Alp)?;

file.version;             // wire version byte
file.geom_codec;          // Option<&'static str> — "fp16" or "int16"
file.xyz;                 // Vec<[f64; 3]>, LOCAL to the frame
file.max_geom_error_m;    // Option<f64>
file.metadata;            // BTreeMap<String, Value>
```

`AlfFile` is one struct with per-channel fields, populated for the channel you asked for:

```rust
file.intensity;        // Vec<u8>            .ali
file.palette;          // Vec<String>        .alc
file.palette_indices;  // Vec<u8>            .alc
file.format_version;   // Option<u8>         .all — 1 or 2
file.categories;       // Vec<u8>            .all
file.instance_ids;     // Option<Vec<u16>>   .all v2
file.instance_map;     // Option<BTreeMap<String, Value>>
file.kept_indices;     // Vec<u32>           .alx
```

## From a path

```rust
use alf::read_alf;

let file = read_alf("frame_000000.alp.gz", "")?;   // empty: infer from the suffix
let file = read_alf("payload.bin", "alp")?;        // or state it
```

`extension` is a parameter rather than a convenience because it has to be. The invalid corpus names
its files `.bin` precisely so nothing can infer a channel from them — a reader that guessed would
pass cases it should refuse. `Channel::from_path` ignores a trailing `.gz` when it does infer.

## Errors carry the identifier

```rust
match read_alf_bytes(&bytes, Channel::Alp) {
    Ok(file) => ...,
    Err(err) => {
        err.id;        // &'static str — "header-not-json", "geom-scale-not-positive", ...
        err.message;   // String, for humans
    }
}
```

Rust carries the corpus identifier **inside** the error, which is a deliberate difference from the
other SDKs. Every other implementation maps its human-readable message onto an identifier in its
conformance runner, with a table of substring needles — and that table is a second source of truth
that drifts, because a reworded message silently stops matching and the runner reports an unnamed
refusal. Here the message and the identifier are constructed together at the single place the rule
is enforced, so neither can drift from the other.

An empty `id` means an error no invalid corpus case covers. The runner surfaces that as a gap rather
than a pass.

## Geometry decoding, directly

The geometry primitives are public, because the WASM boundary needs them without going through a
whole-file read:

```rust
use alf::{GeomCodec, GeometryDecoder, INT16_NO_RETURN};

let decoder = GeometryDecoder::int16(scale, offset)?;   // or ::fp16()
decoder.codec();
decoder.is_fixed_point();
decoder.is_no_return(&words, index);
decoder.point(&words, index);      // Option<[f64; 3]> — None for an invalid return
```

`INT16_NO_RETURN` is `-32768`, the reserved word marking a point with no laser return. See
[Invalid returns](../invalid-returns.md) for the rules that go with it — in particular that a point
with _some but not all_ words equal to the marker must be refused, not reconstructed.

## Malformed input refuses; it does not panic

The crate has a panic-safety suite, and it is not decoration. A panic across the WASM boundary is
not a refusal a caller can handle, so `tests/test_rust_panic_safety.py` requires every malformed
corpus file to produce a named refusal instead.

## Header parsing

The header is parsed with a raw event reader rather than `serde_json`'s map, because the corpus
needs **duplicate keys to be detectable** and a map cannot express that. `serde_json` is kept only
for emitting the canonical conformance answer. `preserve_order` would not be enough.

The nesting depth bound is enforced before a recursive parser is invoked, and the declared header
size bound before any allocation — see [Why it fails closed](../failing-closed.md).

## Reading only the header

Scanning a directory for `_total_points`, `device_position` or a `palette` should not cost a full
decompression per frame — or, for the path entry point, a full **read** per frame.

```rust
use alf::{read_alf_header, read_alf_header_bytes, Channel};

let head = read_alf_header(path, "")?;              // streams a prefix of the file
let head = read_alf_header_bytes(&bytes, Channel::Alp)?;

head.version;
head.metadata;
head.total_points;      // Option<usize>
head.frame_convention;  // Option<String>
head.frame_rotation;    // Option<FrameRotation>, with its components NAMED
head.xyz;               // does not COMPILE — a header-only read has no payload
```

It returns an `AlfHeader`, not an `AlfFile`. Python returns an `AlfFile` whose payload accessors
raise, because it cannot say "the same thing, minus those fields" in its types. Rust can, so
`head.xyz` fails to compile rather than handing back an empty `Vec` that reads as "no points".

`read_alf_header` reads a **bounded prefix of the file**, not the file. `read_alf` loads the whole
frame before parsing anything, which is the cost this exists to avoid — so the path form is not
`read_alf_header_bytes(&std::fs::read(path)?, ..)` and deliberately so.

`frame_rotation` is surfaced here rather than left in `metadata` for the same reason `FrameRotation`
exists at all: pulling `w`/`x`/`y`/`z` out of a raw JSON object by hand is the thing the named type
prevents, and a header-only scan is exactly where a frame convention gets read across a whole
sequence.

**Its refusals are a strict subset of `read_alf_bytes`'s, and that split is the contract.** Both
share one `parse_header`, so every header-scoped rule applies to both by construction. The
payload-scoped rules cannot apply — the declared count is not cross-checked against the payload
length, a second gzip member goes undetected, and the member's CRC32 and ISIZE go unverified.

That is asserted against the corpus rather than asserted in prose: for every valid file the header
read must agree with the full read, and for every invalid one it must refuse with the **same**
identifier or decline to have an opinion — never a different one.

## One gap worth knowing

There is **no writer in Rust**. The [support matrix](/spec/support) records it: the reader row is ✅
across every channel and the writer column is ❌ for `.alp` and `read` for the sidecars. The crate
documents itself as a reader, which is why it earns the reader conformance gate and is `n/a` for the
writer one rather than ❌ on a gate it never entered.

## Verify it

```bash
(cd rust/alf && cargo build --release)
python3 tests/conformance/run.py --runner "rust/alf/target/release/alf-conformance"
```

The subshell matters. A bare `cd rust/alf && …` leaves the second command looking for
`rust/alf/tests/conformance/run.py`, and its runner path resolving under `rust/alf/rust/alf`.
