# `avala-alf`

Rust reader for ALF — Arrays in Local Frames, a container format for point-cloud sequences in a
local coordinate frame. The library target is `alf`:

```bash
cargo add avala-alf --rename alf
```

It is a **reader**: every channel and wire layout decodes (`.alp` v1 float16 and v3 fixed-point
geometry, `.all` v1/v2 labels, `.ali` intensity, `.alc` colour palettes, `.alx` decimation
indexes), and there are no writers. It builds for `wasm32-unknown-unknown` — gzip uses flate2's
pure-Rust backend — and is consumed from WebAssembly, where a panic is not something a caller can
handle, so every malformed input produces a named `AlfError` rather than a panic and the repository
runs the whole invalid corpus through the crate to prove it.

```rust
use alf::{read_alf, read_alf_bytes, read_alf_header, Channel};

let file = read_alf("frame_000000.alp.gz", "")?;       // or read_alf_bytes(&bytes, Channel::Alp)
let xyz = &file.xyz;                // Vec<[f64; 3]>, local frame, NaN = invalid return
let bound = file.max_geom_error_m;  // the frame's measured error bound

let head = read_alf_header("frame_000000.alp.gz", "")?; // reads a bounded PREFIX of the file
(head.total_points, &head.metadata, head.version);
// head has no payload accessors — a header-only read cannot return points.
```

`AlfHeader.total_points` is `Option<f64>`, preserving the exact integral binary64
count even when it exceeds the runtime's integer range. Payload lengths and
allocation sizes come from the actual buffer after its count has been checked.
The shared count and geometry error checks execute both natively and as 32-bit WebAssembly in CI.
Geometry regressions cover the largest finite fp16 and binary64 values, invalid
returns, empty frames, and positive subnormal fixed-point scales. The subnormal
checks compare the API value directly because the corpus rounds its canonical
error field to twelve decimal places.

The channel is required because the wire header does not contain it: the path entry points take it
from the extension, the byte entry points as a `Channel`. Gzip is detected by magic number, not by
filename. `read_alf_header` never inflates the payload, so scanning a directory of gigabyte frames
costs kilobytes per frame; it returns an `AlfHeader`, not an `AlfFile`, so a payload accessor on a
header-only read does not compile.

The reader fails closed: where guessing would return plausible-but-wrong data, it refuses with a
stable `AlfError` code proven against the shared cross-language conformance corpus.

- Specification: <https://alf.dev/spec/format>
- Guides and refusal catalogue: <https://alf.dev>
- Source and conformance suite: <https://github.com/avala-ai/alf>

Apache-2.0, with the repository's NOTICE.
