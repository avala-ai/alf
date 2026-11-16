# Invalid returns

A NaN component in `.alp` geometry means **invalid return** — a ray that did not come back. It is
valid content with a defined meaning, not damage, and a reader that refuses a file because its
geometry contains NaN is non-conforming.

Frames may contain invalid returns at any fraction of their point indices, including every index.
The format preserves those slots so companion channels keep describing the same points.

Infinity is likewise permitted and must not be refused. It carries no defined meaning,
and readers should treat it as they treat NaN:
a component with no position.

## Why it is structural

A scanning lidar can emit a dense range image. With `A` azimuth samples per beam, point `i` is
beam `i // A`, azimuth `i % A`, and every companion channel is indexed the same way. A missing
return still occupies its original slot.

Requiring producers to filter before encoding would destroy the beam/azimuth mapping **and**
desynchronise `.ali` and `.all` against the geometry — which is the one invariant the whole format
is built on. So the empty slots stay, and they need an encoding.

## Three rules, and each is a rule because getting it wrong is silent

### 1. Invalid returns keep their index

Channels are correlated by index. A reader or writer that drops or compacts invalid returns
misaligns every other channel of the frame. The point count includes them.

```python
points = alf.xyz                                  # (N, 3), invalid returns in place
points = points[~np.isnan(points).any(axis=1)]    # filter in YOUR code, if you want to
```

Filtering is the consumer's decision, taken with the sidecars in hand. It is not the reader's.

### 2. Magnitude-derived values are computed over the finite components only

The reported worst-case component error and the header `box` both describe real coordinates; a
component with no position contributes to neither.

An implementation that takes a plain maximum gets NaN — and NaN compares false against every
threshold a consumer checks it with, so the error bound stops meaning anything precisely when it is
needed.

The reported bound is the binary16 half-ULP at the largest **finite** decoded component magnitude,
with the binary16 subnormal floor. A frame whose components are all non-finite has no magnitude to
describe and reports that floor, exactly as an all-zero frame does.

### 3. A frame with no finite component on some axis publishes no `box`

The header `box` is a string of six numbers, and consumers turn it directly into spatial bounds.
`nan` or `inf` inside it is unparseable, so the field is omitted rather than poisoned.

## Writers must be able to emit them

A format whose reader accepts a shape its writer cannot produce cannot round-trip its own data. The
writer interoperability gate depends on being able to hand a candidate writer exactly the frames the
corpus contains, so emitting non-finite components in float16 is a requirement, not an allowance.

## In version 3: a reserved word

Fixed point has no natural NaN, so version 3 spends a codepoint on one.

```text
INT16_NO_RETURN = -32768
```

The encoding is symmetric on purpose — `_geom_scale` is derived as `max|component| / 32767`, so a
real coordinate quantises into `[-32767, +32767]` and `-32768` is unreachable. Two's complement
leaves it without a positive counterpart, which is what makes it usable as a marker rather than a
value.

The rules are normative:

- **A point whose three words are all `INT16_NO_RETURN` is an invalid return.** Readers decode it to
  NaN in all three components and must not apply the reconstruction to it. The same frame written in
  either codec decodes to the same array.
- **Writers must encode a wholly non-finite point as the sentinel** — never drop it, compact the
  frame, or substitute a finite value. `round((NaN - offset) / scale)` narrowed to int16 is
  platform-defined, so an implementation that lets the cast decide emits an arbitrary coordinate
  that looks exactly like data. The invalid rows must be masked and the sentinel written explicitly.
- **A point with some but not all components non-finite is refused** (`geom-partial-no-return`).
  Half a coordinate is not a position. The sentinel describes a whole missing point, so a partly
  non-finite coordinate has no encoding.
- **Readers must refuse a point with some but not all words equal to the marker.** This is the
  corruption nothing downstream can catch: `-32768` is reserved, so a conforming writer never emits
  it as a coordinate, yet reconstructing it produces an entirely ordinary position —
  `[-32768, 0, 0]` at `_geom_scale: 0.01` decodes to `-327.68`. No later check distinguishes that
  from measured data, so a reader that stays silent converts corruption into a plausible point.
- **`_geom_scale` and `_geom_offset` are derived from the real coordinates only.** Sizing the grid
  from a slot that holds no position gives a wrong grid. This covers _both_ spellings: a reduction
  that skips NaN but not infinities silently takes the all-invalid path on a frame that has real
  positions, and centres the grid on zero.
- **A non-finite v3 decode outside a sentinel row is still corruption** — specifically
  `_geom_scale`/`_geom_offset` overflowing binary64 — and is refused as `coordinate-non-finite`.

Because no conforming writer had ever emitted this wire pattern, defining it changed no existing
file's interpretation.

## What this means at a call site

Both codecs carry an invalid return, so you do not have to choose a codec based on whether your
frames are filtered:

```python
from alf import write_alp

write_alp(path, xyz_with_nans)                     # v1: NaN components, natively
write_alp(path, xyz_with_nans, codec="int16",      # v3: sentinel rows, same meaning
          frame_convention="world-aligned")
```

What v3 refuses is a **partly** non-finite point, in every implementation. If you are holding one,
you are holding data no sensor produced — decide what it means before encoding it, because the
format deliberately declines to guess.

The public grid helpers need no masking. `choose_geom_offset` drops rows that are not wholly
finite, and `choose_geom_scale` takes its extent from the finite values, falling back to the scale
floor when a frame has none — because a grid sized from a slot that holds no position is a wrong
grid, and the fix for that is to exclude those slots rather than to refuse the frame:

```python
choose_geom_offset(np.array([[1.0, 2.0, 3.0], [np.nan] * 3, [5.0, 6.0, 7.0]]))
# array([3., 4., 5.])  — the midpoint of the two real rows
```

The normative text is [`spec/format.md` §5.1–5.2](/spec/format).
