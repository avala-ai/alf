# ALF wire-format specification

This document is normative for ALF 0.1. The Python, Dart, and TypeScript
packages are implementations of this contract; none of them defines a private
variant of it. A rule marked as a refusal is part of the format, not merely an
implementation preference.

## 1. Scope

ALF stores one frame-local, index-correlated array per file. It defines five
channels:

| Suffix | Channel                            | Payload element               |
| ------ | ---------------------------------- | ----------------------------- |
| `.alp` | point geometry                     | XYZ triple, 6 bytes per point |
| `.ali` | normalized intensity               | uint8                         |
| `.all` | semantic or panoptic labels        | uint8, or uint8 + uint16 LE   |
| `.alc` | palette colour index               | uint8                         |
| `.alx` | source index for a decimated point | uint32 LE                     |

Entry `i` in every sidecar describes point `i` in the corresponding `.alp`.
A reader must therefore reject any declared point count that disagrees with a
payload's channel stride.

ALF is a transport-neutral container. File access, HTTP, caching, rendering,
and annotation merge policy are outside this specification.

## 2. Envelope

Every uncompressed ALF file is:

```text
offset  size  meaning
0       1     version byte
1       4     UTF-8 JSON header length, uint32 little-endian
5       N     JSON header; the root value must be an object
5 + N   ...   channel payload
```

For compatibility with deployed v1 files, a zero header length is interpreted
as an empty object (`{}`). A non-empty header must be valid JSON whose root is
an object. Malformed JSON syntax is refused as `header-not-json`.

The serialized JSON header may be at most 10,000,000 bytes. A larger declared
header is refused as `header-too-large`, even when all declared bytes are
present and form valid JSON; readers apply this bound before UTF-8 decoding or
JSON parsing allocates another copy.

The maximum collection nesting depth is 256 JSON objects/arrays, counting the
header's root object as depth 1. Readers enforce this before invoking a
recursive parser and refuse deeper input as `header-nesting-too-deep`; writers
enforce the same bound so they cannot publish a header their paired reader
rejects.

The channel is selected by the filename suffix or an equivalent explicit
caller argument; it is not duplicated in the envelope. Readers must not infer
a channel from payload shape.

Duplicate member names are forbidden in the header root and in every nested
JSON object. JSON parsers commonly keep only the last occurrence, which makes
the interpreted header depend on parser behavior and silently loses metadata.
Readers refuse such a header as `header-key-duplicate`.
Writer APIs require every application-metadata object key to be a string at
every depth before serialization. They do not coerce keys such as integer `1`
to `"1"`, because that can collide with an existing string key and emit the
duplicate-member shape readers must refuse.

An ALF envelope may be wrapped in one gzip member. Readers detect gzip from
the `1f 8b` magic bytes rather than a filename or content-encoding header.
Concatenated members are refused: otherwise the second envelope becomes bytes
in the first channel's payload. Gzip CRC32 and uncompressed-size checks apply;
a corrupt single member is refused as `gzip-malformed`, while a second member is
refused as `multiple-gzip-members`.

## 3. Version registry

| Byte | Meaning                                                         | Valid channel |
| ---- | --------------------------------------------------------------- | ------------- |
| `1`  | baseline channel layout; `.alp` float16; `.all` semantic labels | any           |
| `2`  | interleaved panoptic label layout                               | `.all` only   |
| `3`  | int16 fixed-point geometry                                      | `.alp` only   |

Version 2 on a non-`.all` channel and version 3 on a non-`.alp` channel are
refusals. Unknown version bytes are refusals. These checks happen before a
payload is exposed.

## 4. Common header fields

`_total_points`, when present, is a non-negative integral JSON number. Integral
spellings such as `1`, `1.0`, and `1e0` are equivalent. The exact integer must
survive conversion to binary64; this permits representable values above the
consecutive-integer boundary (for example `2^54`) but refuses
`9007199254740993`. Its value times the channel stride must equal the payload
length. `.alx` requires the field. The same numeric rule applies to
`_source_total_points`. `_format_version` shares only the exact-integral
binary64 spelling rule; it is a label-layout selector, not a non-negative
count. Its supported values are 1 and 2. Other exact integral values are
refused as `label-layout-unsupported`; in particular, negative values such as
`-1.0` are `label-layout-unsupported`, not `total-points-not-an-integer`.
A zero significand remains exact zero for any syntactically valid decimal
exponent, including a hostile-length exponent such as `0e10000000000`;
implementations detect that zero before applying exponent work limits because
no expansion or large integer is needed to establish its value.
Non-zero exponent spellings outside finite binary64, such as `1e309`, are
refused as `total-points-not-an-integer`, not as generic malformed metadata.
Readers bound the exponent before expanding an integral token into an integer;
a compact token such as `1e100000000` must not allocate a hundred-million-digit
intermediate. For a negative exponent, readers first compare the shift with the
significand's available trailing zeros. Thus `1` followed by 10,000 zeros and
`e-10000` is exactly `1` and remains valid, while `1e-10000000` is known to be
fractional without constructing a ten-million-digit divisor.

Metadata may contain additional JSON fields. Reserved fields beginning with
`_` have the meanings defined here and must not be repurposed by applications.
All metadata numbers must decode to finite binary64 values in the shared
cross-language metadata domain. In particular, an arbitrary-precision integer
outside finite binary64 (for example `10^400`) is refused as `header-not-json`;
reserved count fields instead retain their more specific
`total-points-not-an-integer` refusal. Writers enforce the same exact,
finite-binary64 rule for count arguments before serializing them, while still
permitting exactly representable wide integers such as `2^64`. Writers also
validate application metadata numbers against the finite-binary64 domain before
serialization, so a successful writer never emits a header its paired reader
must refuse.

Geometry headers may place local points with `device_position` or
`world_centroid_epsg3857`, represented as an `x`/`y`/`z` object or a
three-component array. For compatibility with deployed files, the legacy
`device_position: {}` form is also valid and means the zero origin `[0, 0, 0]`;
writers preserve it when it is inherited from metadata. This is the only empty
or partial-object exception. Readers otherwise validate the complete shape
while reading the envelope and refuse a partial object or wrong-length array
as `geometry-anchor-malformed`; a malformed placement cannot remain latent
until an application later calls a placement accessor. Their numeric tokens
follow the same binary64 precision rule as v3 grid parameters: a token whose
stated decimal value changes during decoding is refused as
`geom-number-inexact`, because it would shift the whole frame rather than one
sample. This anchor rule applies only to `.alp`; the same member names on
another channel remain ordinary application metadata. For an x/y/z anchor
object, exact geometry-token validation applies only to those three axes.
Additional members remain application metadata and follow the generic
finite-binary64 rule. Writers normalize or replace the three axes but preserve
every additional member and must not mutate caller-owned nested metadata. Each
written axis token must state the exact binary64 value; an
unsafe integral axis therefore uses a preserving spelling such as scientific
notation rather than a nearby decimal integer that merely parses back to the
same binary64 value. As for every JSON object, the x/y/z members may appear in
any order without changing the anchor.

## 5. Geometry (`.alp`)

The payload is three little-endian 16-bit values per point in XYZ order.
Payload length must be divisible by 6.

### 5.1 Version 1: binary16

Each component is IEEE 754 binary16, little-endian, INCLUDING its non-finite
encodings. `_geom_codec` is absent or the string `fp16`; fixed-point metadata
must not override a version-1 payload.

**A NaN component means "invalid return". It is valid content, and readers MUST
NOT refuse it.** A reader that rejects a `.alp` because its geometry contains
NaN is non-conforming. This is not a tolerance for damaged files: it is what a
lidar produces for a ray that did not come back. Frames may contain invalid
returns at any fraction of their point indices, including every index.

Infinity is likewise permitted and MUST NOT be refused. It carries no defined
meaning, and readers
should treat it as they treat NaN: a component with no position.

Three rules follow, and each of them is a rule because getting it wrong is
silent:

1. **Invalid returns keep their INDEX.** Channels are correlated by index —
   entry _i_ of `.ali`, `.all`, `.alc` describes point _i_ of `.alp` — so a
   reader or writer that drops or compacts invalid returns misaligns every other
   channel of the frame. The point count includes them.
2. **Magnitude-derived values are computed over the FINITE components only.**
   The reported worst-case component error (below) and the header `box` both
   describe real coordinates; a component with no position contributes to
   neither. An implementation that takes a plain maximum gets NaN, and NaN
   compares false against every threshold a consumer checks it with, so the
   error bound stops meaning anything precisely when it is needed.
3. **A frame with no finite component on some axis publishes NO `box`.** The
   header `box` is a string of six numbers; `nan` or `inf` inside it is
   unparseable, and consumers turn it directly into spatial bounds.

Writers MUST be able to emit non-finite components in this codec. A format whose
reader accepts a shape its writer cannot produce cannot round-trip its own
data, and the writer interoperability gate depends on being able to hand a
candidate writer exactly the frames the corpus contains.

The reported worst-case component error is the binary16 half-ULP at the
largest FINITE decoded component magnitude, with the binary16 subnormal floor.
A frame whose components are all non-finite has no magnitude to describe and
reports the subnormal floor, as an all-zero frame does.

### 5.2 Version 3: fixed point

Each component is a signed int16, little-endian, reconstructed as:

```text
decoded[axis] = stored[axis] * _geom_scale + _geom_offset[axis]
```

Required header fields:

- `_geom_codec`: exactly `int16`
- `_geom_scale`: positive finite JSON number
- `_geom_offset`: exactly three finite JSON numbers

**Version 3 represents an invalid return with a reserved word.** The encoding is
symmetric — `_geom_scale` is derived as `max|component| / 32767`, so a real
coordinate quantises into `[-32767, +32767]` and `-32768` is unreachable. Two's
complement leaves it without a positive counterpart, which is what makes it
usable as a marker rather than a value.

```
INT16_NO_RETURN = -32768
```

The rules are normative:

- **A point whose three words are all `INT16_NO_RETURN` is an invalid return.**
  Readers MUST decode it to non-finite (NaN in all three components) and MUST
  NOT apply the reconstruction to it. It is the same content §5.1 defines, in a
  different codec, so the same frame written in either codec decodes to the same
  array.
- **Writers MUST encode a wholly non-finite point as the sentinel**, and MUST
  NOT drop it, compact the frame, or substitute a finite value. `round((NaN -
offset) / scale)` narrowed to int16 is platform-defined, so an implementation
  that lets the cast decide will emit an arbitrary coordinate that looks exactly
  like data: the invalid rows must be masked and the sentinel written
  explicitly.
- **A point with SOME but not all components non-finite MUST be refused.** Half a
  coordinate is not a position. The sentinel represents a whole missing point,
  so a partly non-finite coordinate has no encoding.
- **Readers MUST refuse a point with SOME but not all words equal to the
  marker.** It is the mirror of the writer rule above, and the one corruption
  nothing downstream can catch: `-32768` is reserved, so a conforming writer
  never emits it as a coordinate, yet reconstructing it produces an entirely
  ordinary position — `[-32768, 0, 0]` at `_geom_scale: 0.01` decodes to
  `-327.68`. No later check can distinguish that from measured data, so a
  reader that stays silent here converts corruption into a plausible point.
- **`_geom_scale` and `_geom_offset` are derived from the real coordinates only.**
  Sizing the grid from a slot that holds no position gives a wrong grid; a frame
  whose returns are all invalid has no magnitude and takes the scale floor. This
  covers BOTH spellings of an invalid return: a reduction that skips NaN but not
  infinities silently takes the all-invalid path on a frame that has real
  positions, and centres the grid on zero.
- **A non-finite v3 decode OUTSIDE a sentinel row is still corruption**,
  specifically `_geom_scale`/`_geom_offset` overflowing binary64, and readers
  refuse it as `coordinate-non-finite`.

Why this is specified rather than left to version 1: invalid returns are
structural, not incidental. A scanning lidar can emit a dense range image with
`A` azimuth samples per beam: point `i` is beam `i // A`, azimuth `i % A`, and
every companion channel is indexed the same way. Requiring producers to
filter before encoding would destroy the beam/azimuth mapping AND desynchronise
`.ali` and `.all` against the geometry, which is the one invariant §2 states.
Without the sentinel, version 3 could not preserve this representation when
any ray has no return.

This changed the meaning of a v3 wire pattern that no conforming writer had ever
emitted, so no file's interpretation changed.

Geometry codec parameters must not claim numeric precision that disappears
when decoded as binary64. Readers preserve enough of the raw JSON number token
to compare its decimal claim with the decoded binary64 value. They refuse
tokens such as `9007199254740993`, `9007199254740993.0`, and
`9007199254740993.5`; accepting one would silently move the coordinate grid.
Ordinary binary-inexact decimals such as `0.1`, and large values whose shortest
representation remains `1e+308`, stay valid.

Plain integers are exact integer claims. Expanded integral decimal and
exponent-zero tokens are exact integer claims too, so
`1000000000000000100.0` and `1000000000000000100e0` are invalid: their exact
mathematical integer differs from the integer represented by the decoded
binary64 value. An integral claim is valid when those values are equal,
regardless of digit count or expanded spelling. For example,
`18446744073709551616`, `18446744073709551616.0`, and
`18446744073709551616e0` all state the same exactly representable `2^64` value.

Normalized scientific notation is instead a binary64-preserving writer
spelling when it is the decoded value's shortest round-trip decimal: one digit
precedes the optional fraction and the exponent is normalized and non-zero.
For example, a writer may serialize the binary64 integer
`1000000000000000128` as `1.0000000000000001e+18`; readers must accept that
output even though an expanded exact-integer claim of `1000000000000000100`
would be refused. Normalized shape alone is not an exemption: the spelling
must equal the decoded binary64 value's shortest scientific representation,
so the lossy token `9.007199254740993e15` is refused.
Readers cancel fractional decimal placement and insignificant trailing zeroes
before bounding exponent work, so `0.` followed by 9,980 zeroes and
`18446744073709551616e10000` is the same valid value without requiring a large
power-of-ten allocation.
Conversely, the raw integer token `18446744073709552000` is invalid even though
some binary64 runtimes use it as a shortest decimal spelling: its expanded
integer claim differs from the exact decoded value `18446744073709551616`.
A zero geometry significand likewise remains exact zero regardless of exponent
length; `0e10000000000` is therefore a valid offset component and is
short-circuited before hostile-exponent work limits.

`_geom_max_err_m` is optional for readers so legacy and hand-written v3 files
remain decodable. When present, it records the writer-measured maximum
component error. Current writers must emit it; readers must still derive a
conservative analytic bound when it is absent or understates reconstruction
rounding.

Unknown `_geom_codec` strings are refused as `codec-unsupported`. A known
codec that disagrees with the version byte is refused as
`codec-version-disagreement`. By default a writer derives one
shared scale from the largest absolute component after subtracting the per-axis
bounding-box midpoint. A writer API may also accept a caller-selected scale and
offset; this is conforming, not a private codec variant, provided the scale is
positive and finite, the offset has three finite components, every quantized
sample fits signed int16, reconstruction stays finite, and the emitted
`_geom_max_err_m` records the measured error for that selected grid. Current
writers record at least the independently measured maximum component
reconstruction error and, for a non-empty fixed-point payload, `_geom_scale/2`.
The half-scale term is rounded upward when its exact value falls between two
binary64 values. In particular, a positive `_geom_scale` whose half underflows
must still produce a positive `_geom_max_err_m`; the minimum positive binary64
scale therefore has the minimum positive binary64 error floor, not zero.
This comparison uses the raw emitted field before any reader derives or
normalizes a conservative bound. An interoperability gate may allow only an
ULP-scaled arithmetic allowance for this comparison; a fixed absolute tolerance
would erase the complete error bound for sufficiently small valid grids.
Current writers retain v1 as the default. Before opting in to v3, verify every
consumer of the written files using the requirements in
[adoption.md](adoption.md), and keep that deployment inventory with the application.

### 5.3 Version 3: the frame convention

A `.alp`'s points are the world points with an origin subtracted. That is what
makes them small enough to be worth 6 bytes, and it is **half** of a rigid
transform. Producers disagreed about the other half: some subtracted the origin
and stopped, leaving points on world axes, while others also rotated into the
device frame. Both called the result "the frame", and nothing in the bytes told
them apart.

The two conventions **agree exactly whenever the device heading is identity**,
so a sequence with identity headings cannot distinguish them. A
consumer that omitted the rotation therefore matched a producer that omitted it,
and the pair looked correct until the vehicle turned — at which point geometry
and annotations drifted apart by the accumulated heading, with nothing raising.

Version 3 refuses to be ambiguous.

**A version-3 `.alp` MUST carry `_frame_convention`**, a string with exactly one
of two values:

| Value            | Meaning                                                  | World placement                               |
| ---------------- | -------------------------------------------------------- | --------------------------------------------- |
| `world-aligned`  | only the origin was subtracted; points are on world axes | `world = local + anchor`                      |
| `device-aligned` | the points were also rotated into the device frame       | `world = R(_frame_rotation) · local + anchor` |

Its absence is refused as `frame-convention-missing`, and an unrecognised value
as `frame-convention-unsupported`. The field is REQUIRED rather than defaulted
because a default is the thing that failed: absence is exactly what both
producers had, and reading it as either convention re-creates the bug for the
other one.

**`device-aligned` requires `_frame_rotation`** (`frame-rotation-missing`), an
object carrying exactly the four members `w`, `x`, `y` and `z`, each a finite
JSON number. Any other shape, a missing member, or an additional member is
refused as `frame-rotation-malformed`.

An OBJECT rather than a four-element array, deliberately. Every array spelling of
a quaternion must choose between `[x, y, z, w]` and `[w, x, y, z]`, both are in
wide use, and the two are indistinguishable in the bytes — so a reader and a
writer that disagree produce a rotation that is wrong in a way no length check
catches. Naming the members removes the choice.

Naming the members settles which number is which. It does **not** settle what the
four of them mean, and that is a second ambiguity with the same shape: two
readers can agree on every component and still rotate in opposite directions.
So `R(_frame_rotation)` is defined here rather than left to the reader.

**The convention is Hamilton, and the rotation is active.** Hamilton means
`ij = k`; the JPL convention has `ij = -k` and produces the transpose, which is
the rotation the other way. Active means `R(q)` is applied to the coordinates
rather than to the axes: `_frame_rotation` is the device's orientation in the
world, and `R(q)` carries a point's device-frame components onto world axes.
Writing `local` as the pure quaternion `(0, x, y, z)`:

```text
(0, world - anchor) = q ⊗ (0, local) ⊗ q*        q* = (w, -x, -y, -z)
```

That conjugation is the matrix below, which a reader should build once per frame
and apply with a single multiply rather than forming two quaternion products per
point:

```text
        ⎡ 1 - 2(y² + z²)    2(xy - zw)      2(xz + yw)  ⎤
R(q) =  ⎢   2(xy + zw)    1 - 2(x² + z²)    2(yz - xw)  ⎥      world = R(q)·local + anchor
        ⎣   2(xz - yw)      2(yz + xw)    1 - 2(x² + y²)⎦
```

`q` and `-q` are the same rotation — the components appear in even-degree terms
only — so a producer may write either, and a reader must not read the sign of
`w` as information.

**The pinned vector.** A conforming implementation MUST reproduce this exactly.
It is chosen so that every wrong reading is visibly wrong rather than close:

| `_frame_rotation`            | `local`     | `world - anchor` |
| ---------------------------- | ----------- | ---------------- |
| `w=0.5, x=0.5, y=0.5, z=0.5` | `(1, 2, 3)` | `(3, 1, 2)`      |

Those components are exact in binary64 and the norm is exactly 1, so the check
is not a tolerance question. The rotation is 120° about `(1, 1, 1)/√3`, which
cycles the axes — `x → y → z → x` — and the transposed reading (passive, or
Hamilton components fed to a JPL formula) yields `(2, 3, 1)` instead. An
axis-aligned test point or a 180° rotation would have agreed under both and
pinned nothing, which is the same trap the convention itself set: the readings
coincide on the easy cases and separate on the ones that reach production.

Its components follow the same exact-binary64 token rule as the grid parameters
(`geom-number-inexact`): a component whose stated decimal changes when decoded
turns the whole frame, not one sample.

**`_frame_rotation` MUST be a unit quaternion.** Conjugation by a non-unit
quaternion maps a vector to |q|² times its rotation, so it scales as well as
rotates — at |q| = 1.05 that is a 10% stretch, 10 m of error at 100 m, applied
uniformly to the frame and to nothing else in the file. Readers refuse
`frame-rotation-not-normalized` rather than normalizing: normalizing would
repair a producer that has a bug, and the bug would stay. The bound is on the
squared norm, which is the factor actually applied, and is `1e-9` — far looser
than the rounding of a genuinely normalized quaternion and far tighter than any
error worth a metre.

Normalization belongs in the **writer**, which knows what the caller meant. A
pose quaternion that has been through several compositions is a unit quaternion
to about `1e-7`, so a writer that passed one through untouched would emit files
its own reader refuses. A writer normalizes such a value and records the
normalized form; it refuses one that is not nearly unit already, because drift
is a small number and |q| = 2 is a units or ordering mistake that normalizing
would turn into a confident wrong rotation.

**`world-aligned` MUST NOT carry `_frame_rotation`** (`frame-rotation-unexpected`).
The two statements cannot both be what the producer meant, and the reading that
drops the rotation returns plausible, wrongly-oriented geometry.

Identity is a legal `device-aligned` rotation and is not the same as
`world-aligned`. A device pointing along +x has that heading, and it is exactly
the case in which the two conventions agree — so a reader that treats identity as
"no declaration" passes every straight-line frame and fails at the corner.

**Versions 1 and 2 are grandfathered.** They predate the field, both conventions
were written, and a deployed file cannot be told retroactively what it meant.
Readers apply no rotation to v1/v2 geometry. What such a file may NOT do is carry
`_frame_convention` or `_frame_rotation`: honouring the declaration is precisely
what a conforming reader will not do, so the declaration would be a promise the
format does not keep. Either field on a v1 or v2 `.alp` is refused as
`frame-convention-on-legacy-version`.

These checks are header-scoped and are applied while the envelope is read, so a
header-only reader enforces them too and a frame that cannot say where its points
are is refused before any payload is decoded. They precede the payload-scoped
geometry checks: a file that neither states its convention nor carries a valid
`_geom_scale` is refused for the convention, because that is decidable first and
from less.

## 6. Intensity (`.ali`)

The payload is one uint8 per point. Writers normalize a declared finite raw
range to `[0, 255]` and may record nine normalized deciles in
`_intensities_every_10_percent`. Bounds may not be reversed; values may not be
non-finite or outside the supplied bounds.

## 7. Labels (`.all`)

Layout 1 stores one uint8 category per point.

Layout 2 stores one three-byte row per point:

```text
category:uint8, instance_id:uint16 little-endian
```

Layout 2 uses wire byte 2 and `_format_version: 2`. Readers may encounter
legacy production files in which the two signals differ; they may resolve the
layout only when objective payload evidence leaves one possible
interpretation. A declared `_total_points` can establish the unique stride. If
the count is absent, a payload length not divisible by three rules out layout
2 and resolves to layout 1; a length divisible by three still fits both
layouts and is refused as ambiguous.

Instance id 0 means no instance. Id 65535 is reserved and refused. When an
`instance_map` object is present, every key must be the canonical decimal
spelling of an id in 1..65534 (no sign or leading zero), every value must be a
non-empty string, and every emitted non-zero instance id must have an entry.

## 8. Colour (`.alc`)

The payload is one uint8 palette index per point. The canonical `palette`
header field is a non-empty list of at most 255 six-digit RGB hex strings
without `#`. For compatibility, readers accept exactly one optional leading
`#` and normalize it away; two or more leading hashes are malformed. Current
writers emit the canonical form.
Index `palette.length` is the no-colour sentinel; larger indexes are refused.

## 9. Decimation index (`.alx`)

The payload is one uint32 little-endian source index per kept point. The header
requires `_total_points` and `_source_total_points`. Kept indexes must be
unique and strictly less than `_source_total_points`; order may differ from
source order.

## 10. Conformance and refusals

The shared corpus in `tests/conformance/` is executable evidence for this
specification. A language runner prints one canonical JSON document on stdout.
A required rejection is an answer, not a crash:

```json
{ "refused": "stable-refusal-id" }
```

The runner exits zero for a named refusal. A crash, unnamed error, extra stdout,
or different refusal id fails the case. The support matrix may say a feature is
verified only when this corpus exercises it and the language runner passes.
Canonical geometry output rounds the exact decoded binary64 rational to six
decimal places with ties-to-even semantics. It must not first multiply in
binary64, because an intermediate rounding can move a value onto the wrong side
of a decimal midpoint.

Writer verification is separate: another implementation must decode the
candidate writer's output and compare canonical meaning. Canonical writer-owned
header fields must also be inspected before a compatibility reader normalizes
them. Application metadata is compared after portable binary64 normalization,
excluding only fields the writer explicitly regenerates from its typed payload
or API arguments; `.alp` anchor arrays and x/y/z objects are the same metadata
after their documented shape normalization. The candidate receives decoded
typed values and metadata, not an already-valid ALF input or sibling fixture it
could copy as its output. A self-round-trip does not prove interoperability.
