# Precision and the two codecs

ALF has two geometry codecs and they cost **exactly the same**: 6 bytes per point, three 16-bit
little-endian words in XYZ order. The choice between them is not a size trade — it is a question of
where you want the error to land.

## Version 1: float16

Each component is IEEE 754 binary16. The error scales with each component's magnitude — the
half-ULP is about `|c| × 2⁻¹¹` — so accuracy is excellent near the origin and degrades linearly
outward:

| Distance | float16 worst-case component error |
| -------- | ---------------------------------- |
| 8 m      | ~4 mm                              |
| 100 m    | ~3 cm                              |
| 512 m    | ~25 cm                             |

A typical automotive lidar is accurate to ±1–2 cm. So past roughly 30 m the _format_ dominates the
sensor, while near the origin it is spending bits on precision no sensor delivers. Both halves of
that trade are wrong, which is why version 3 exists.

## Version 3: int16 fixed point

Each component is a signed int16 reconstructed against two header fields:

```text
decoded[axis] = stored[axis] * _geom_scale + _geom_offset[axis]
```

`_geom_offset` is the frame's **per-axis bounding-box midpoint**, so the grid is built around the
data rather than around zero, and the full int16 range is spent on the frame's own span. The error
becomes uniform across the frame:

| Frame                | float16 worst | int16 worst | Improvement |
| -------------------- | ------------- | ----------- | ----------- |
| ±50 m                | 15.6 mm       | 0.76 mm     | **20×**     |
| ±100 m               | 31.3 mm       | 1.53 mm     | **20×**     |
| off-centre 100–200 m | 62.5 mm       | 0.76 mm     | **82×**     |

The off-centre row is what the midpoint offset buys. Without centring, a cloud spanning 100–200 m
would derive its step from 200 m instead of from its own 50 m half-span, and quantize four times
worse than it needed to. This matters less for ego-centric automotive lidar, which is roughly
origin-centred already — and a great deal for aerial tiles, which are exactly the offset, one-sided
frames the field exists for.

These numbers are **measured, not asserted**: `geom_error_report()` computes them and the test
suite runs it.

## Read the stated bound, not the scale

The most common mistake with v3 is deriving an accuracy claim from `_geom_scale`.

`_geom_scale / 2` is the **quantization** term only. Reconstruction happens in floating point and
its rounding lands on top of the step; for a far off-origin frame that second term dominates.
`_geom_max_err_m` is what the writer actually _measured_ for the grid it chose.

> Compare an accuracy threshold against `max_geom_error_m`. Never against the scale.

The field is optional for readers, so legacy and hand-written v3 files stay decodable — but current
writers must emit it, and a reader that does not find one must derive a conservative analytic bound
rather than assume the quantization term is the whole story.

## Which one to write

**Version 1 is the default, and version 3 is opt-in.** Selecting it also means declaring which
frame the points are in — see [The frame convention](#the-frame-convention) below:

```python
write_alp(path, xyz, codec="int16", frame_convention="world-aligned")
```

`{ codec: 'int16', frameConvention: 'world-aligned' }` in TypeScript,
`codec: AlfGeometryCodec.int16, frameConvention: alfFrameWorldAligned` in Dart.

That default is deliberate. The format has no skip-unknown mechanism, so an unrecognised version
byte is fatal:

- Writing v3 for a reader that rejects the version byte publishes **unreadable** geometry.
- Writing v3 for a reader that ignores the byte and decodes int16 as float16 publishes something
  worse: coordinates that look plausible and are wrong by metres.

So v3 cannot become the default until every consumer of a file can read it. All four
format-owned SDKs read v3. Follow the [adoption requirements](/spec/adoption) to verify the
application's complete read path, and keep its readiness inventory with that application.

## Choosing a grid yourself

By default a writer derives one shared scale from the largest absolute component after subtracting
the per-axis bounding-box midpoint. Passing your own scale and offset is conforming — not a private
variant — provided:

- the scale is positive and finite, and the offset has three finite components;
- every quantized sample fits in signed int16;
- reconstruction stays finite; and
- the emitted `_geom_max_err_m` records the error **measured for that grid**.

Python exposes the two default choices directly, so a caller who wants to inspect or reuse them
does not have to reimplement them:

```python
from alf import choose_geom_offset, choose_geom_scale

offset = choose_geom_offset(local_xyz)         # per-axis bounding-box midpoint
scale = choose_geom_scale(local_xyz - offset)  # expects ALREADY-CENTRED coordinates
```

`choose_geom_scale` derives the step from the largest absolute value it is handed, so passing raw
one-sided coordinates reproduces the exact waste the offset exists to remove.

## The precision rule on the numbers themselves

Geometry codec parameters must not claim numeric precision that disappears when decoded as
binary64. Readers keep enough of the raw JSON token to compare its decimal claim against the
decoded value, and refuse a mismatch as `geom-number-inexact` — because accepting one would
silently move the coordinate grid, not one sample.

Concretely:

- `9007199254740993`, `9007199254740993.0` and `9007199254740993.5` are refused. None of them
  survives binary64.
- `0.1` is fine. It is binary-inexact, but it is the shortest round-trip decimal for the value it
  decodes to.
- `18446744073709551616`, `18446744073709551616.0` and `18446744073709551616e0` are all fine and
  all mean 2⁶⁴, which is exactly representable.
- `1000000000000000100.0` is refused, but `1.0000000000000001e+18` is accepted — normalized
  scientific notation is a binary64-_preserving_ writer spelling when it is the decoded value's
  shortest round-trip decimal.
- `18446744073709552000` is refused even though some runtimes print it for 2⁶⁴, because as an
  expanded integer claim it differs from the exact decoded value.

The same rule governs `device_position` and `world_centroid_epsg3857` on a `.alp`, for the same
reason: an anchor that shifts under decoding shifts the whole frame. On any other channel those
member names are ordinary application metadata and follow the ordinary finite-binary64 rule.

## The frame convention

A `.alp`'s points are the world points with an origin subtracted. That is what makes them worth
6 bytes — and it is **half** of a rigid transform.

Producers disagreed about the other half. Some subtracted the origin and stopped, leaving points on
world axes. Others also rotated into the device frame. Both called the result "the frame", and
nothing in the bytes told them apart.

The two **agree exactly whenever the device heading is identity**. A consumer that omits the rotation can match a
producer that omitted it, the pair looked correct, and geometry drifted away from annotations only
once the vehicle turned, by the accumulated heading, with nothing raising.

Version 3 refuses to be ambiguous. Every v3 `.alp` states its convention:

| `_frame_convention` | Meaning                                            | World placement                 |
| ------------------- | -------------------------------------------------- | ------------------------------- |
| `world-aligned`     | only the origin was subtracted                     | `world = local + anchor`        |
| `device-aligned`    | the points were also rotated into the device frame | `world = R(q) · local + anchor` |

```python
# The points were only translated.
write_alp(path, local_xyz, codec="int16", frame_convention="world-aligned")

# The points were translated AND rotated into the device frame.
write_alp(
    path,
    local_xyz,
    codec="int16",
    frame_convention="device-aligned",
    frame_rotation={"w": 0.7071067811865476, "x": 0.0, "y": 0.0, "z": 0.7071067811865476},
)
```

`world_xyz()` then applies the rotation when the file says one was applied, so placement is decided
by the file rather than guessed by the caller.

The field is **required, not defaulted**. A default is the thing that failed: absence is exactly
what both producers had, and reading it as either convention re-creates the bug for the other one.
Making it required is only possible because v3 is opt-in and unadopted.

Four details worth knowing:

- **The rotation is an object, not an array.** Every array spelling of a quaternion must choose
  between `[x, y, z, w]` and `[w, x, y, z]`, both are in wide use, and the two are
  indistinguishable in the bytes. Naming the members removes the choice.
- **`R(q)` is Hamilton and active**, defined normatively in
  [`spec/format.md` §5.3](/spec/format). Naming the members settles which number is which, not what
  the four of them mean: the JPL convention agrees on every component and yields the transpose — a
  rotation the other way. The pinned vector is exact: `w=x=y=z=0.5` takes `(1, 2, 3)` to `(3, 1, 2)`.
- **It must be a unit quaternion.** Conjugation by a non-unit quaternion scales by |q|² as well as
  rotating — at |q| = 1.05 that is 10 m of error at 100 m, applied uniformly and silently. Readers
  refuse `frame-rotation-not-normalized`; **writers normalize**, because the writer knows what the
  caller meant and a reader does not.
- **Identity is a legal `device-aligned` rotation**, and is not the same as `world-aligned`. It is
  exactly the case where the two conventions agree, so a reader that treats identity as "no
  declaration" passes every straight-line frame and fails at the corner.

**Versions 1 and 2 are grandfathered.** They predate the field, both conventions were written, and
a deployed file cannot be told retroactively what it meant — so readers apply no rotation to them
and `world_xyz()` still asks the caller. What a v1/v2 file may _not_ do is carry the field:
honouring it is precisely what a conforming reader will not do.

## What about invalid returns?

Version 1 carries them natively as NaN. Version 3 reserves the word `-32768` in all three
components as a sentinel. The rules are strict enough — and important enough — to have their own
page: [Invalid returns](./invalid-returns.md).
