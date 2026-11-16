# Write in Python

Every writer takes a path, validates before it writes, and returns **the header it actually wrote**
— so a caller can log or assert on the recorded error bound rather than recomputing it.

All five writers gzip their output. Files are conventionally named `frame_000000.alp.gz`.

## Geometry — `.alp`

```python
from alf import write_alp

header = write_alp(
    "frame_000000.alp.gz",
    local_xyz,                       # (N, 3) float, ALREADY local to device_position
    metadata,                        # optional application metadata
    device_position=(x, y, z),       # the frame's placement
)

header["_geom_max_err_m"]            # what the writer measured
```

`local_xyz` must already be local. The writer will not localize for you, and the `box` header is
computed here from the same array that gets encoded — which makes the bug class where bounds
describe world coordinates while the payload holds local ones unrepresentable rather than fixed once
per writer.

Pass only coordinates. An `(N, 4)` array — XYZ plus intensity or ring — is refused rather than
silently reinterpreted as `4N/3` points.

### Choosing the codec

```python
write_alp(path, local_xyz)                      # v1 float16 — the default

write_alp(                                      # v3 fixed point — opt-in
    path,
    local_xyz,
    codec="int16",
    frame_convention="world-aligned",           # required: which frame these points are in
)
```

**Selecting v3 means declaring the frame convention.** `local_xyz` is the world points with an
origin subtracted, and that is half of a rigid transform — the file must say whether you also
rotated them into the device frame. Pass `"world-aligned"` if you only subtracted the origin, or
`"device-aligned"` with a `frame_rotation` unit quaternion if you rotated as well. The writer
normalizes the quaternion; the reader refuses a non-unit one. See
[the frame convention](../precision.md#the-frame-convention).

**v1 is the default and v3 is opt-in**, because the format has no skip-unknown mechanism: writing v3
for a reader that cannot decode it publishes either unreadable geometry or, worse, coordinates that
look plausible and are wrong by metres. See [Precision and the two codecs](../precision.md) for the
error figures and [adoption](/spec/adoption) for the checks required on each consumer's read path.

Before switching a dataset, measure what it would cost:

```python
from alf import geom_error_report

geom_error_report(local_xyz)     # {'fp16': ..., 'int16': ...}, worst-case metres per component
```

### The error budget

Geometry writers enforce the shared annotation-grade error budget by default. Conversion tooling
that genuinely needs to exceed it can opt out explicitly:

```python
write_alp(path, local_xyz, check_budget=False)
```

That relaxes the budget check only. Finite/range validation, the anchor rules and the exact-binary64
number rules all still apply.

### Anchors

`device_position` replaces the anchor **axes** while preserving non-axis application metadata inside
that object, and removes a competing inherited `world_centroid_epsg3857`. Both behaviours matter
when rewriting a frame: carrying the old anchor through would translate new coordinates by the old
origin, and `world_xyz()` would then place the cloud wherever the previous frame was.

Each written axis token states its exact binary64 value, so an unsafe integral axis is written in a
preserving spelling — scientific notation — rather than a nearby decimal integer that merely parses
back to the same value.

## Intensity — `.ali`

```python
from alf import write_ali

write_ali(path, raw_intensity, lo=0.0, hi=255.0, deciles=None, metadata=None)
```

`lo`/`hi` are the **raw** range the normalization used. They are deliberately not recorded in the
header: no legacy writer emits raw bounds, so adding them would change every ported `.ali` envelope.
Bounds may not be reversed, and values may not be non-finite or outside them.

`deciles` records nine normalized deciles in `_intensities_every_10_percent`.

To normalize separately — the same float division the canonical generator uses, which two earlier
writers got a consistent one low by using integer division:

```python
from alf import map_intensity

normalized = map_intensity(raw, lo, hi)     # (N,) uint8
```

## Labels — `.all`

```python
from alf import write_all

write_all(path, categories)                                  # v1: semantic only
write_all(path, categories, instance_ids=ids,                # v2: panoptic
          instance_map={"1": "car", "2": "pedestrian"})
```

Instance id `0` means _no instance_. Id `65535` is reserved and refused. When `instance_map` is
present, every key must be the canonical decimal spelling of an id in 1..65534 — no sign, no leading
zero — every value must be a non-empty string, and **every emitted non-zero id must have an entry**.

Categories are validated before casting. `asarray(300, dtype=uint8)` wraps to 44, so an out-of-range
category is a refusal rather than a quietly renumbered label.

## Colour — `.alc`

```python
from alf import write_alc

write_alc(path, palette_indices, ["ff0000", "00ff00", "0000ff"])
```

The palette is a non-empty list of at most 255 six-digit RGB hex strings **without** `#`. Readers
accept exactly one optional leading `#` for compatibility and normalize it away; writers emit the
canonical form.

`len(palette)` is the **no-colour sentinel** — written for points that matched no camera or were out
of view — and is accepted. Anything beyond it is refused, because it renders as whatever the
viewer's bounds handling does with it, which is a colour bug nobody traces back to the file.

## Decimation index — `.alx`

```python
from alf import write_alx

write_alx(
    path,
    kept_indices,
    alp_total_points=len(kept_indices),   # points in the .alp this sidecar accompanies
    source_total_points=source_count,     # the cloud those indexes address
)
```

Kept indexes must be unique and strictly less than the source count. Order may differ from source
order. Both `_total_points` and `_source_total_points` are required in the header, and this is the
one channel where the count field is mandatory.

## What the writers guarantee

- **Nothing partial is left behind.** A prior file at the path is unlinked _before_ validation, so a
  rejected re-run cannot leave the previous frame there for downstream globbing to pick up as if
  this run had succeeded.
- **A successful writer never emits a header its paired reader must refuse.** Writers apply the same
  nesting-depth bound, the same finite-binary64 metadata rule, and the same exact-spelling rule for
  geometry numbers that readers enforce.
- **Metadata keys must be strings at every depth.** Writers do not coerce an integer key `1` to
  `"1"`, because that can collide with an existing string key and produce the duplicate-member shape
  readers must refuse.
- **Caller-owned nested metadata is not mutated.**

## Verifying a writer

A self-round-trip proves nothing about interoperability. The gate that counts is another
implementation decoding your output:

```bash
python3 tests/conformance/encode_roundtrip.py \
  --encoder "python tests/conformance/runners/python_encode_roundtrip.py"
```

The encoder is handed a decoded typed request — values and application metadata, never an
already-valid ALF wire image or a sibling fixture it could copy — and the Python reference decodes
what comes back. Exact bytes may differ; canonical decoded meaning may not. See the
[conformance suite](../../reference/conformance.md).
