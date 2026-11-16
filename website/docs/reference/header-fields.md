# Header fields

The header is a JSON object. Fields beginning with `_` are **reserved** — they have the meanings
defined here and must not be repurposed by applications — and a handful of unprefixed names
(`box`, `palette`, `instance_map`, `device_position`, `world_centroid_epsg3857`) are reserved too,
for historical reasons.

Everything else in the header is application metadata and is preserved by writers.

The normative text is [`spec/format.md` §4–§9](/spec/format). This page is an index.

## The numeric domains

Three different rules apply to numbers in an ALF header, and knowing which one a field is under
explains most of its refusals.

| Domain             | Applies to                                                                                             | Rule                                                                                                                                                                                                                                                        |
| ------------------ | ------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Portable**       | every number in the header                                                                             | Must decode to a finite binary64 value. An arbitrary-precision integer outside that range (`10^400`) is refused as `header-not-json`.                                                                                                                       |
| **Exact integral** | `_total_points`, `_source_total_points`, `_format_version`                                             | Must be an exact integer whose value survives conversion to binary64. Spellings `1`, `1.0` and `1e0` are equivalent. Values above the consecutive-integer boundary are allowed when exactly representable (2⁵⁴), and refused when not (`9007199254740993`). |
| **Exact geometry** | `_geom_scale`, `_geom_offset`, the `_frame_rotation` components, and the x/y/z axes of a `.alp` anchor | The token's stated decimal value must equal the decoded binary64 value. A mismatch is `geom-number-inexact` — because it moves the whole coordinate grid, not one sample.                                                                                   |

The exact-geometry rule applies **only on `.alp`**. The same member names on another channel are
ordinary application metadata under the portable rule.

## Common to every channel

### `_total_points`

Non-negative exact integer. Its value times the channel stride must equal the payload length; a
disagreement is `payload-length-mismatch`.

Optional on `.alp`, `.ali`, `.alc` and `.all` — when absent, the length cross-check simply cannot
run. **Required on `.alx`** (`total-points-missing`).

Refusals: `total-points-not-an-integer` for a fractional value, a present `null`, a value outside
finite binary64, or a hostile exponent spelling.

## Geometry — `.alp`

### `_geom_codec`

The string `fp16` or `int16`. Absent means `fp16`.

Must agree with the version byte: `fp16` requires version 1, `int16` requires version 3. A
disagreement is `codec-version-disagreement`, and an unknown string is `codec-unsupported`. On a
version-1 payload, fixed-point metadata must not override the codec.

### `_geom_scale`

**v3 only, required.** A positive finite number: the step, in metres, between adjacent int16 values.

By default a writer derives it as `max|component| / 32767` after subtracting the per-axis
bounding-box midpoint. Missing is `geom-scale-missing`; zero or negative is `geom-scale-not-positive`
— zero collapses the frame onto its offset and negative mirrors it.

### `_geom_offset`

**v3 only, required.** Exactly three finite numbers: the origin the int16 grid is built around,
per axis. Default is the frame's per-axis bounding-box midpoint.

A scalar or 2-vector is `geom-offset-malformed`, because it would broadcast across axes and yield a
uniformly shifted cloud rather than an error.

Reconstruction is `decoded[axis] = stored[axis] * _geom_scale + _geom_offset[axis]`.

### `_geom_max_err_m`

The **writer-measured** maximum component error, in metres.

Optional for readers, so legacy and hand-written v3 files stay decodable — but current writers must
emit it, and a reader that does not find one must derive a conservative analytic bound. It records
at least the independently measured maximum component reconstruction error and, for a non-empty
fixed-point payload, `_geom_scale / 2` rounded upward.

Under the **portable** rule, not the exact-geometry one: it describes the grid rather than defining
it, so a token that rounds moves nobody's coordinates. Readers accept any finite decoded value and
combine it with their own analytic bound; a value outside finite binary64 is refused as ordinary
malformed metadata (`header-not-json`), not as `geom-number-inexact`.

> Compare an accuracy threshold against this field, **never** against `_geom_scale`. The scale is
> the quantization term only; reconstruction rounding lands on top of it and dominates for a far
> off-origin frame.

A positive scale whose half underflows must still produce a positive error — the minimum positive
binary64 scale has the minimum positive binary64 error floor, not zero.

### `_frame_convention`

**v3 only, required.** The string `world-aligned` or `device-aligned`. It says whether the points
were rotated into the device frame as well as translated — the half of the rigid transform the
coordinates cannot reveal.

| Value            | World placement                               |
| ---------------- | --------------------------------------------- |
| `world-aligned`  | `world = local + anchor`                      |
| `device-aligned` | `world = R(_frame_rotation) · local + anchor` |

Absent on v3 is `frame-convention-missing`; an unrecognised value is
`frame-convention-unsupported`. **Required rather than defaulted**, because absence is exactly what
both historical producers had, so a default re-creates the bug for one of them.

Present on a v1 or v2 `.alp` is `frame-convention-on-legacy-version` — readers apply no rotation to
those versions, so the declaration would be a promise the format does not keep.

### `_frame_rotation`

**Required when `_frame_convention` is `device-aligned`**, and forbidden otherwise
(`frame-rotation-unexpected`). An object carrying exactly `w`, `x`, `y` and `z`, each a finite JSON
number under the exact-geometry token rule.

An object rather than an array because every array spelling of a quaternion must choose between
`[x, y, z, w]` and `[w, x, y, z]`, and the two are indistinguishable in the bytes. Exactly four
members, because a stray `qw` beside `w` is a producer mixing spellings and only one would be read.

Naming the members settles which number is which; it does not settle what they mean. The convention
is **Hamilton** (`ij = k`) and the rotation is **active** — `world = R(q) · local + anchor`, where
`R(q)` is the matrix form of `q ⊗ (0, local) ⊗ q*`:

```text
        ⎡ 1 - 2(y² + z²)    2(xy - zw)      2(xz + yw)  ⎤
R(q) =  ⎢   2(xy + zw)    1 - 2(x² + z²)    2(yz - xw)  ⎥
        ⎣   2(xz - yw)      2(yz + xw)    1 - 2(x² + y²)⎦
```

The JPL convention has `ij = -k` and yields the transpose, which is a rotation the other way — a
plausible cloud, wrong by the device heading, from a reader that agrees on every component. The
pinned vector, exact in binary64: `w=x=y=z=0.5` takes `local = (1, 2, 3)` to `(3, 1, 2)`, not
`(2, 3, 1)`. `q` and `-q` are the same rotation, so the sign of `w` carries no information.

Must be a **unit quaternion**: conjugation by a non-unit one scales by |q|² as well as rotating, so
it moves every coordinate in the frame by a factor nothing else in the file disagrees with. Readers
refuse `frame-rotation-not-normalized` beyond `1e-9` on the squared norm. **Writers normalize** —
they know what the caller meant, and a pose quaternion that has been through several compositions is
unit only to about `1e-7`. A writer refuses one that is not nearly unit already, because drift is a
small number and |q| = 2 is a units or ordering mistake.

Identity is a legal `device-aligned` rotation and is not the same as `world-aligned`.

### `device_position` and `world_centroid_epsg3857`

Where the frame's local points sit. Either an `x`/`y`/`z` object or a three-component array; members
may appear in any order.

The complete shape is validated **while reading the envelope**, not when an accessor is called
(`geometry-anchor-malformed`) — a malformed placement must not stay latent until an application
tries to place the cloud.

One exception, for deployed files: **`device_position: {}` is valid** and means the zero origin
`[0, 0, 0]`. It is the only empty-or-partial anchor object the format accepts, and writers preserve
it when it is inherited from metadata.

For an x/y/z object, the exact-geometry rule covers those three axes only. Additional members stay
application metadata under the portable rule, and writers preserve every one of them while
normalizing or replacing the axes.

### `box` and `_box_frame`

`box` is six numbers joined by `:` — `xlo:ylo:zlo:xhi:yhi:zhi` — and `_box_frame` states the frame
they are in (`"local"`).

Two properties matter more than the format:

- **It describes the decoded coordinates, never the writer's input.** Both codecs are lossy and
  round in either direction, so a pre-encoding extremum can sit _inside_ the decoded extent. A frame
  spanning ±0.005 m decodes at ±0.006104 m under int16, and consumers turn this header straight into
  spatial bounds — so filtering against a box computed pre-encoding would exclude real payload
  points.
- **A frame with no finite component on some axis publishes no `box` at all**, and no `_box_frame`
  either. `nan` or `inf` inside the string is unparseable. An inherited box from a previous frame is
  actively removed rather than left, because metadata is often copied forward and would otherwise
  publish the _old_ spatial extent for a frame that now has none.

## Intensity — `.ali`

### `_min_intensity` / `_max_intensity`

Always `0` and `255` — the normalized range, by mapping definition.

These are **not** the raw bounds the normalization used. The raw range is a `write_ali` argument and
is deliberately not recorded: no legacy writer emits raw bounds, so adding them would change every
ported `.ali` envelope.

### `_intensities_every_10_percent`

Nine normalized deciles. In every existing writer this is a **sequence-level** distribution
accumulated across all frames and stamped onto each one, not a per-frame statistic.

## Labels — `.all`

### `_format_version`

The label layout: `1` for one uint8 category per point, `2` for a three-byte
`category:uint8, instance_id:uint16 LE` row.

Under the exact-integral rule for spelling, but it is a **layout selector, not a count** — so a
negative value such as `-1.0` is `label-layout-unsupported`, not `total-points-not-an-integer`.
Any other exact integral value is likewise `label-layout-unsupported`.

Falls back to the envelope version byte when absent. Both signals exist in the wild and neither is
universal — the Dart writer sets the byte _and_ the key, while some legacy parsers identify v2 from
the byte alone. Keying only on metadata would read those files as v1, returning the interleaved
stream as categories.

When the two signals **disagree**, a reader may resolve the layout only when objective payload
evidence leaves one interpretation: a declared `_total_points` can establish the stride, and a
payload length not divisible by three rules out layout 2. A length divisible by three fits both and
is `label-layout-ambiguous`.

### `instance_map`

An object mapping instance ids to names. Every key must be the **canonical decimal spelling** of an
id in 1..65534 — no sign, no leading zero (`instance-map-malformed`) — every value must be a
non-empty string, and **every emitted non-zero instance id must have an entry**
(`instance-map-incomplete`).

Instance id `0` means _no instance_. Id `65535` is reserved and refused (`instance-id-reserved`).

## Colour — `.alc`

### `palette`

A non-empty list of at most 255 six-digit RGB hex strings **without** `#`.

Readers accept exactly one optional leading `#` for compatibility and normalize it away; two or more
are malformed. Current writers emit the canonical hash-free form, and the conformance writer gate
inspects it in the raw envelope before a compatibility reader can normalize it.

Index `palette.length` is the **no-colour sentinel**, written for points that matched no camera or
were out of view. Larger indexes are `palette-index-out-of-range`.

## Decimation — `.alx`

### `_source_total_points`

Required. The size of the cloud the kept indexes address (`source-total-points-missing`). Under the
exact-integral rule.

Kept indexes must be unique (`kept-index-duplicate`) and strictly less than this value
(`kept-index-out-of-range`). Order may differ from source order.

## Application metadata

Anything not listed above. It is preserved across a write, compared after portable binary64
normalization by the conformance writer gate, and subject to two rules:

- **Every number must decode to a finite binary64 value.** Writers validate this before serializing,
  so a successful writer never emits a header its paired reader must refuse.
- **Every key must be a string, at every depth.** Writers do not coerce an integer key `1` to `"1"`,
  because that can collide with an existing string key and produce the duplicate-member shape
  readers must refuse (`header-key-duplicate`).

Nesting is bounded at 256 levels counting the root object as depth 1, enforced before a recursive
parser is invoked (`header-nesting-too-deep`), and the serialized header at 10,000,000 bytes,
enforced before decoding allocates another copy (`header-too-large`).
