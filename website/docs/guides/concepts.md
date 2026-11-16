# Concepts

Everything ALF does follows from one sentence, so it is worth stating first.

## Channels are index-correlated

A frame is a set of sibling files that share a point ordering. Entry _i_ of every channel describes
point _i_ of the `.alp`.

| Suffix | Channel                                         | Payload element             | Bytes            |
| ------ | ----------------------------------------------- | --------------------------- | ---------------- |
| `.alp` | Array of Local **P**oints — XYZ geometry        | XYZ triple                  | 6 per point      |
| `.ali` | Array of Local **I**ntensity                    | uint8                       | 1 per point      |
| `.alc` | Array of Local **C**olour — palette indexes     | uint8                       | 1 per point      |
| `.all` | Array of Local **L**abels                       | uint8, or uint8 + uint16 LE | 1 or 3 per point |
| `.alx` | Array of Local inde**X**es — decimation sidecar | uint32 LE                   | 4 per kept point |

That single invariant is what most of the validation in this format exists to defend. A length
mismatch does not corrupt one file — it silently misattributes every label and intensity reading
after the gap, and the file stays perfectly readable while it does. So a reader must reject any
declared point count that disagrees with a payload's channel stride, and a reader must never
drop, compact or reorder points on the way out.

The extensions are load-bearing in stored data: readers infer the channel from the suffix and
stored objects are named by it. They do not change.

## The envelope

Every uncompressed ALF file is the same five-byte preamble followed by a header and a payload.

```text
offset  size  meaning
0       1     version byte
1       4     UTF-8 JSON header length, uint32 little-endian
5       N     JSON header; the root value must be an object
5 + N   ...   channel payload
```

Three consequences worth knowing before you write a reader:

**The channel is not in the envelope.** It comes from the filename suffix or an explicit caller
argument, and a reader must not infer it from the payload's shape. `readAlf` therefore takes the
channel as a required parameter in every SDK.

**Compression is detected by magic, not by name.** A gzip-wrapped envelope starts `1f 8b`, and that
is what readers look at — not the filename, not a `Content-Encoding` header. Live label diffs are
uploaded uncompressed while the merged object is compressed, and both arrive as `.all`. Exactly one
gzip member is allowed; a second one is refused as `multiple-gzip-members`, because otherwise the
second envelope silently becomes bytes in the first channel's payload.

**The header has bounds, and they are checked before allocation.** At most 10,000,000 serialized
bytes, at most 256 levels of collection nesting counting the root object as depth 1. Both are
applied before UTF-8 decoding or a recursive parser gets a chance to allocate.

Duplicate member names are forbidden — in the root and at every nesting depth. Most JSON parsers
keep the last occurrence, which makes the interpreted header depend on which parser you used and
silently drops metadata; ALF refuses it as `header-key-duplicate` instead.

## The version byte

| Byte | Meaning                                                 | Valid channel |
| ---- | ------------------------------------------------------- | ------------- |
| `1`  | baseline layout; `.alp` float16, `.all` semantic labels | any           |
| `2`  | interleaved panoptic label layout                       | `.all` only   |
| `3`  | int16 fixed-point geometry                              | `.alp` only   |

A version byte on the wrong channel is a refusal, and so is an unrecognised one. There is **no
skip-unknown mechanism**, by design: a reader that skipped what it did not understand would return
a partial frame, and a partial point cloud is indistinguishable from a small one.

That design choice is the whole reason [adoption](/spec/adoption) is a document. Adding a version
means every reader on a file's path must be updated _before_ anything writes it, which is why v3
geometry is opt-in and v1 remains the default.

## Local, not global

Coordinates in a `.alp` are local to the frame. The header carries the placement — `device_position`
or `world_centroid_epsg3857` — as an `x`/`y`/`z` object or a three-component array, and readers
validate the complete shape while reading the envelope rather than leaving a malformed placement
latent until an accessor is called.

Giving up global coordinates is not a limitation that was worked around; it is the trade that buys
the precision. A frame-local origin lets 16 bits per axis describe a 100 m span to under two
millimetres. See [Precision and the two codecs](./precision.md).

One consequence: no SDK will guess a world frame for you. Python's `world_xyz()` requires either an
explicit `origin=` or `sensor_pose=True`, because silently picking one of the two anchor fields is
how a sequence ends up half in one frame and half in another.

## Metadata

Fields beginning with `_` are reserved and have the meanings the specification gives them.
Applications may add their own fields alongside, with one rule that catches people out: **every
metadata number must decode to a finite binary64 value**, and geometry-bearing numbers must
additionally _state_ their exact binary64 value.

The second rule exists because a geometry parameter that loses precision on decode does not move
one sample — it moves the whole frame. See the [header field index](../reference/header-fields.md)
for what each reserved field means and which rule applies to it.

## What a frame looks like on disk

```text
frame_000000.alp.gz     6 bytes/point   geometry, local to device_position
frame_000000.ali.gz     1 byte/point    intensity
frame_000000.all.gz     1 or 3 bytes    labels, and optionally instance ids
frame_000000.alc.gz     1 byte/point    palette index
frame_000000.alx.gz     4 bytes/kept    which source points survived decimation
```

Every sidecar is optional. What is not optional is that any sidecar present has the same point
count as the geometry it accompanies — which is the invariant this page opened with.
