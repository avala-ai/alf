# ALF — Arrays in Local Frames

[![CI][ci-badge]][ci] [![Conformance][conformance-badge]][conformance]
[![License][license-badge]][license]

A container format for point-cloud sequences in a **local** coordinate frame, streamed one frame at
a time. Each file is one typed array plus a JSON header, and that is the whole of the name: an
`.alp` is an _array of local points_.

**Documentation lives at [alf.dev](https://alf.dev)** — the specification, the per-language guides,
the refusal catalogue and the conformance story, rendered from this repository.

## Why ALF

Annotation-grade point clouds delivered to a browser, one frame at a time: small files, a
GPU-friendly layout, and a frame-local origin. It is **not** a geospatial archive — LAS/LAZ/COPC own
that and do it better. ALF gives up global coordinates on purpose, and gets its precision back from
the trade. The format grew up on ego-centric automotive lidar sweeps, but nothing in it is
lidar-specific: photogrammetry, RGBD and aerial tiles use the same channels.

- **One typed array per file**, keyed by extension — five channels, below.
- **Fails closed.** Where guessing would return plausible-but-wrong data, every reader raises.
- **Two geometry codecs, identical 6 bytes per point** — float16 (v1, the default) and int16 fixed
  point (v3, opt-in), with up to 82× less quantization error for the same bytes.
- **Header-only reads.** Scanning a directory of gigabyte frames costs kilobytes per frame.
- **One specification, four implementations, one shared conformance corpus** — Python, Dart,
  TypeScript and Rust are graded against the same 117 cases.

## Channels

| Channel | Holds                                                                        | Payload            |
| ------- | ---------------------------------------------------------------------------- | ------------------ |
| `.alp`  | Array of Local **P**oints — XYZ geometry                                     | 6 bytes/point      |
| `.ali`  | Array of Local **I**ntensity                                                 | 1 byte/point       |
| `.alc`  | Array of Local **C**olor — palette indexes                                   | 1 byte/point       |
| `.all`  | Array of Local **L**abels — semantic segmentation, optionally + instance ids | 1 or 3 bytes/point |
| `.alx`  | Array of Local inde**X**es — decimation sidecar                              | 4 bytes/kept point |

Channels are **index-correlated**: entry _i_ of every channel describes point _i_. That single
invariant is what most of the validation here exists to defend — a length mismatch does not corrupt
one file, it silently misattributes every label and intensity reading after the gap.

## Quick start

```python
from alf import read_alf, write_alp

alf = read_alf("frame_000000.alp.gz")
alf.xyz                            # (N, 3) float64, LOCAL to device_position
alf.max_geom_error_m               # the frame's measured error bound
alf.world_xyz(sensor_pose=True)    # or origin=(x, y, z) — never guesses the frame

write_alp(path, local_xyz, metadata, device_position=(x, y, z))   # fp16/v1
write_alp(path, local_xyz, codec="int16",                         # v3, opt-in
          frame_convention="world-aligned")   # ...and v3 must say which frame
```

`import alf` does not load NumPy; touching a codec does.

TypeScript, Dart and Rust expose the same API — see their package READMEs
([`typescript/alf/`](typescript/alf/), [`dart/alf/`](dart/alf/), [`rust/alf/`](rust/alf/)) and the
[getting-started guides](https://alf.dev/guides/) for each.

### Reading only the header

Scanning a sequence for `_total_points`, `device_position` or a `palette` should not cost a full
download and a full decompression per frame:

```python
from alf import read_alf_header

head = read_alf_header("frame_000000.alp.gz")   # or bytes, or any .read(n) stream
head.total_points, head.device_position, head.metadata
head.xyz                                        # AlfError — the payload was never read
```

It reads the envelope and stops, never inflating the payload, and works on forward-only sources —
sockets, pipes, object-store streaming bodies — because it never seeks. Every header-scoped refusal
is identical to `read_alf`'s (the two share one envelope parser), but the payload-scoped rules
cannot run: the count is not cross-checked against the payload, concatenated gzip members are not
detected, and a gzip member's CRC32/ISIZE are never verified, because checking them means inflating
to the end — the cost the function exists to avoid. Use `read_alf` when the points matter.

The same entry point exists in all four languages, with one deliberate difference: TypeScript's
`readAlfHeader`, Dart's `readAlfHeader` and Rust's `read_alf_header` return an `AlfHeader` rather
than an `AlfFile`, so a payload accessor on a header-only read does not compile. Python returns an
`AlfFile` whose payload accessors raise, because a dataclass cannot express "these fields are not
present here". Rust's path entry point additionally streams a bounded prefix of the file rather than
the file. `head.header_only` reports which kind of file you are holding.

## Installation

The packages are not on PyPI, npm, pub.dev or crates.io yet — code existing in a repository is not
evidence that a package is installable, which is why this README carries no registry badges. The
reserved registry coordinates are:

| Registry  | Distribution name | Consumer API                                          |
| --------- | ----------------- | ----------------------------------------------------- |
| PyPI      | `avala-alf`       | `pip install avala-alf`, then `import alf`            |
| npm       | `@avala-ai/alf`   | `import … from "@avala-ai/alf"`                       |
| pub.dev   | `alf`             | `import 'package:alf/alf.dart'`                       |
| crates.io | `avala-alf`       | `cargo add avala-alf --rename alf`, then `use alf::…` |

Until they are live, install from source — see [CONTRIBUTING.md](CONTRIBUTING.md) for the per-language
setup, and [RELEASING.md](RELEASING.md) for what each registry needs before the first publish.

## Wire format

```
byte 0      version
bytes 1-4   header length, uint32 little-endian
bytes 5..N  UTF-8 JSON header
rest        payload, keyed by extension AND (for .all) `_format_version`
```

| Version | Meaning                                         | Valid on    |
| ------- | ----------------------------------------------- | ----------- |
| 1       | `.all` categories only; `.alp` float16 geometry | any channel |
| 2       | `.all` interleaved category + instance id       | `.all`      |
| 3       | `.alp` int16 fixed-point geometry               | `.alp` only |

Versions 1 and 2 are **label layouts** and share the wire byte across channels. Version 3 describes
the **geometry codec**, so a 3 on `.ali`/`.all`/`.alc` is a corrupt file, not a newer label format,
and is refused. The JSON header root is an object with a maximum collection nesting depth of 256,
enforced by readers before invoking a recursive parser and by writers so they cannot publish an
envelope their paired reader rejects.

Files are usually gzipped (`.alp.gz`). The reader detects gzip by magic number, not by filename,
because live label diffs are uploaded uncompressed while the merged object is compressed and both
arrive as `.all`.

The normative text is [`spec/format.md`](spec/format.md), served as-is at
[alf.dev/spec](https://alf.dev/spec/format).

## Precision

Two geometry codecs, identical 6 bytes per point:

- **v1 — float16.** Error scales with each component's magnitude (half-ULP ≈ `|c| × 2⁻¹¹`): ~4 mm at
  8 m, ~3 cm at 100 m, ~25 cm at 512 m. A typical automotive lidar is ±1–2 cm, so past ~30 m the
  _format_ dominates the sensor, while near the origin it spends bits on precision no sensor
  delivers. Both halves of that trade are wrong.
- **v3 — int16 fixed point.** `decoded = stored * _geom_scale + _geom_offset`, where `_geom_offset`
  is the frame's per-axis bounding-box midpoint — the grid is built around the data rather than
  around zero, and the full int16 range is spent on the frame's own span. Error becomes uniform
  across the frame:

| Frame                | float16 worst | int16 worst |         |
| -------------------- | ------------- | ----------- | ------- |
| ±50 m                | 15.6 mm       | 0.76 mm     | **20×** |
| ±100 m               | 31.3 mm       | 1.53 mm     | **20×** |
| off-centre 100–200 m | 62.5 mm       | 0.76 mm     | **82×** |

The off-centre row is what the midpoint offset buys. Measured, not asserted — `geom_error_report()`
computes this and the suite runs it.

One reading rule: `_geom_scale / 2` is the **quantization** term only; floating-point reconstruction
rounding lands on top of it, and for a far off-origin frame that term dominates. Compare an accuracy
threshold against `max_geom_error_m` — the bound the writer _measured_ — never against the scale.

## Invalid returns (NaN geometry)

A NaN component in `.alp` geometry means **invalid return** — a ray that did not come back. It is
valid content with a meaning, not damage, and it is the normal shape of production data. `.xyz`
returns invalid returns as-is, at their original index, so consumers filter what they want to drop:

```python
points = alf.xyz
points = points[~np.isnan(points).any(axis=1)]
```

They keep their index on purpose: every other channel is correlated with geometry **by index**, so
compacting them away in the reader would misalign the frame. `max_geom_error_m` and the header `box`
are computed over the finite components only, and a frame with no finite extent publishes no `box`
at all.

Version 3 expresses an invalid return with a reserved word: `_geom_scale` is derived as
`max|component| / 32767`, so a real coordinate quantises into `[-32767, +32767]` and `-32768`
(`INT16_NO_RETURN`) is unreachable as data. A point whose three words are all the marker decodes to
NaN in all three components, so the same frame written in either codec decodes to the same array. A
point with _some_ but not all components non-finite is refused in both directions
(`geom-partial-no-return`): half a coordinate is not a position. See
[`spec/format.md`](spec/format.md) §5.1–5.2.

## It fails closed

Where guessing would return plausible-but-wrong data, ALF raises. That is the whole design stance,
and it is why this repository exists as one implementation rather than the twenty-four hand-rolled
parsers it replaced — three languages disagreeing about which malformed files to accept is worse
than any one of their answers.

The conformance suite's [invalid corpus](tests/conformance/) is the enforcement: each entry breaks
**exactly one** rule and names the refusal a conforming reader must produce, because "both decoders
raised an error" is not agreement.

## Status

**v1 is the default. v3 is opt-in** —
`write_alp(path, xyz, codec="int16", frame_convention="world-aligned")` — because not every consumer
decodes it yet. Writing v3 for a reader that cannot decode it publishes unreadable geometry, so the
default stays where every existing consumer can follow.

An **implementation** parses the format itself, so it owes the shared conformance suite a passing
run. Application adoption is a separate check: every consumer of a written file must use a
compatible implementation and pass its integration tests.

| Implementation                 | v1  | v2 labels | v3 geometry | Verification                                                                             |
| ------------------------------ | --- | --------- | ----------- | ---------------------------------------------------------------------------------------- |
| Python (`python/`)             | ✅  | ✅        | ✅          | 117/117 conformance cases + unit/regression tests                                        |
| Dart (`dart/alf/`)             | ✅  | ✅        | ✅          | 117/117 conformance cases + package unit/regression tests; pure Dart VM + web/Wasm codec |
| TypeScript (`typescript/alf/`) | ✅  | ✅        | ✅          | 117/117 conformance cases + package unit/regression tests; browser/Node-compatible ESM   |
| Rust (`rust/alf/`)             | ✅  | ✅        | ✅          | 117/117 conformance cases + panic-safety suite; executed Wasm count and geometry checks  |

These checks establish SDK support, not deployment readiness. Each adopter maintains its own
consumer and producer inventory, including any raw-buffer or precompiled WebAssembly path.
See [`spec/adoption.md`](spec/adoption.md) for the migration evidence required before enabling v3
writes. Publishing the SDKs does not change the default codec.

## Repository layout

```
python/alf/        reference implementation
dart/alf/          pure Dart implementation (VM + web/Wasm)
dart/conformance/  private Dart conformance and encode-roundtrip adapters
typescript/alf/    browser/Node-compatible ESM implementation
rust/alf/          Rust reader, builds for wasm32-unknown-unknown
tests/             the suite that measures the claims above
tests/conformance/ cross-language contract: valid corpus + invalid corpus
spec/              normative format specification — served AS-IS at alf.dev/spec
website/           the alf.dev site: guides, reference, and the spec above
```

## Contributing

Pull requests are welcome. [CONTRIBUTING.md](CONTRIBUTING.md) has the development setup, the exact
gates CI runs, and the process for adding a validation rule or a language SDK.
[GOVERNANCE.md](GOVERNANCE.md) describes how specification changes are decided, and
[SECURITY.md](SECURITY.md) how to report a vulnerability — in a format whose decoders parse
untrusted bytes, a reader that accepts what the specification refuses is a security issue.

## License

Apache-2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE). The specification, the conformance suite
and all four SDKs are under the same licence.

[ci-badge]: https://github.com/avala-ai/alf/actions/workflows/ci.yml/badge.svg
[ci]: https://github.com/avala-ai/alf/actions/workflows/ci.yml
[conformance-badge]: https://img.shields.io/badge/conformance-117%2F117%20%C3%97%204%20languages-brightgreen
[conformance]: https://alf.dev/reference/conformance
[license-badge]: https://img.shields.io/badge/license-Apache--2.0-blue
[license]: LICENSE
