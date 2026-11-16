# Read in Python

`python/alf` is the reference implementation. Every rule in the specification is enforced here
first, and the other SDKs are held to the same corpus rather than to this code.

## Install

Registry publishing is not configured yet, so install from the repository:

```bash
pip install "avala-alf @ git+https://github.com/avala-ai/alf.git"
```

Or from a checkout, which is what you want if you are also running the conformance suite:

```bash
git clone https://github.com/avala-ai/alf.git
cd alf
pip install -e ".[dev]"
```

Python 3.11 or newer. The only runtime dependency is NumPy — and `import alf` does not load it. The
codecs need NumPy; answering "what channel is this file?" does not, and a caller who only wants the
latter should not pay for the former. A CI job enforces that in a clean environment.

## Read a frame

```python
from alf import read_alf

alf = read_alf("frame_000000.alp.gz")

alf.xyz                 # (N, 3) float64, LOCAL to device_position
alf.total_points        # header-declared count, when present
alf.geom_codec          # 'fp16' or 'int16'
alf.max_geom_error_m    # the frame's measured error bound, in metres
alf.device_position     # the frame's placement
alf.metadata            # the parsed JSON header
```

`read_alf` accepts a path, `bytes`/`bytearray`/`memoryview`, or any file-like object with `.read(n)`.
Gzip is detected from the `1f 8b` magic bytes, not from the filename.

The channel comes from the suffix. When you are reading from bytes or a stream there is no suffix
to read, so pass it:

```python
alf = read_alf(payload_bytes, extension="alp")
```

## The other channels

```python
read_alf("frame.ali.gz").intensity          # (N,) uint8
read_alf("frame.alc.gz").palette_indices    # (N,) uint8
read_alf("frame.alc.gz").palette            # ['ff0000', '00ff00', ...] or None
read_alf("frame.all.gz").categories         # (N,) uint8
read_alf("frame.all.gz").instance_ids       # (N,) uint16, or None for a v1 label file
read_alf("frame.all.gz").instance_map       # {'1': 'car', ...} or None
read_alf("frame.alx.gz").kept_indices       # (N,) uint32 into the source cloud
```

Accessors that do not belong to the channel you opened raise rather than returning something empty.
Entry _i_ of every one of these arrays describes point _i_ of the sibling `.alp` — that is the
invariant, and none of these readers compacts, reorders or filters.

## Geometry is local

`alf.xyz` is local to the frame. Placing it in a world frame is a separate, explicit call:

```python
alf.world_xyz(origin=(x, y, z))     # your frame, your origin
alf.world_xyz(sensor_pose=True)     # use the header's own anchor
```

It never guesses. A header can carry `device_position`, `world_centroid_epsg3857`, both, or a legacy
empty `device_position: {}` meaning the zero origin — and silently picking one is how a sequence
ends up half in one frame and half in another.

## Invalid returns

A NaN component means the ray did not come back. It is content, it keeps its index, and filtering
is your decision:

```python
import numpy as np

points = alf.xyz
finite = points[~np.isnan(points).any(axis=1)]
```

Do not filter before you are done with the sidecars — `.ali`, `.all`, `.alc` and `.alx` are
correlated with geometry by index. See [Invalid returns](../invalid-returns.md).

## Reading only the header

Scanning a sequence for `_total_points`, `device_position` or a `palette` should not cost a full
download and a full decompression per frame.

```python
from alf import read_alf_header

head = read_alf_header("frame_000000.alp.gz")   # or bytes, or any .read(n) stream

head.total_points
head.device_position
head.metadata
head.header_only        # True — this is what you are holding
head.xyz                # AlfError: the payload was never read
```

It reads the envelope and stops, never inflating the payload, and works on forward-only sources —
sockets, pipes, `botocore` streaming bodies — because it never seeks or tells.

**Its refusals are a strict subset of `read_alf`'s, and that split is the contract.** Every
header-scoped rule still applies; both entry points share one envelope parser, so they agree by
construction. The payload-scoped rules cannot apply, because their evidence is the bytes it declined
to read:

- the declared count is not cross-checked against the payload length;
- concatenated gzip members are not detected;
- for a gzipped input, the member's CRC32 and ISIZE go unverified — checking them means inflating to
  the end, which is the cost this function exists to avoid.

It is a separate function rather than `read_alf(..., header_only=True)` on purpose: a boolean lets
the weaker contract be selected by a variable at a call site that reads like the strong one.

Use `read_alf` when the points matter.

## Errors

```python
from alf import AlfError

try:
    alf = read_alf(path)
except AlfError as exc:
    log.warning("refused %s: %s", path, exc)
```

Every refusal names the rule it enforced; the identifiers are catalogued in
[Refusals](../../reference/refusals.md). A refusal is a deliberate answer, not a crash — see
[Why it fails closed](../failing-closed.md).

## Memory limits

Decompression and decoded-geometry ceilings are explicit and adjustable:

```python
from alf import memory_limits, set_memory_limits

memory_limits()                                   # the current ceilings, in bytes
set_memory_limits(decompressed_bytes=1 << 28)     # lower it on a constrained worker
set_memory_limits(decoded_geometry_bytes=1 << 30) # raise it for a batch job reading merged tiles
```

Both bound memory, not correctness — neither changes which files are well-formed, only which ones
this process will hold. They exist because a single frame is untrusted input: a 1 GB frame will OOM
a worker that reads it whole, which is the same pressure `read_alf_header` exists to relieve.

It is a function rather than two writable constants deliberately. `alf` re-exports lazily, so
`alf.MAX_DECOMPRESSED_BYTES = n` would bind the name on the _package_ while every reader kept
consulting the module's — a knob that reads as working and does nothing.

## Walking a sequence

```python
from alf import iter_frames

for frame in iter_frames(sorted(directory.glob("*.alp.gz"))):
    ...
```

## Next

- [Write in Python](./write-python.md) — every channel, both geometry codecs
- [Precision and the two codecs](../precision.md) — which codec to write, and why v3 is opt-in
- [`spec/format.md`](/spec/format) — the normative text
