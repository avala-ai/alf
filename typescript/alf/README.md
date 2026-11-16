# `@avala-ai/alf`

Pure TypeScript reading and writing for ALF — Arrays in Local Frames, a container format for
point-cloud sequences in a local coordinate frame.

The runtime has no Node.js or Bun imports: all byte inputs and outputs are `Uint8Array`, and the
package works in browsers, Node ESM, and Bun. Every ALF channel and current wire layout is
supported:

| Channel | Decoded value                                              | Writer     |
| ------- | ---------------------------------------------------------- | ---------- |
| `.alp`  | float64 XYZ triples from v1 fp16 or v3 fixed-point int16   | `writeAlp` |
| `.ali`  | normalized uint8 intensity                                 | `writeAli` |
| `.alc`  | uint8 palette indexes plus the header palette              | `writeAlc` |
| `.all`  | v1 semantic categories or v2 category + uint16 instance id | `writeAll` |
| `.alx`  | uint32 source indexes for a decimated `.alp`               | `writeAlx` |

## Usage

```ts
import { readAlf, readAlfHeader, writeAlp } from "@avala-ai/alf";

const decoded = await readAlf(bytes, { channel: "alp" });
if (decoded.channel === "alp") {
  console.log(decoded.codec, decoded.points, decoded.maxGeomErrorM);
}

const head = await readAlfHeader(bytes, { channel: "alp" }); // never inflates the payload
head.totalPoints; // head has no payload accessors — they do not compile on a header-only read

const encoded = await writeAlp(points, {
  codec: "int16",
  // Required on v3: says whether the points were rotated into the device
  // frame as well as translated. See spec/format.md §5.3.
  frameConvention: "world-aligned",
  gzip: true,
});
```

`readAlf` requires the channel because the wire header does not contain it. It detects gzip by
magic number, validates the channel/version/layout combination, and fails closed with an
`AlfError`: where guessing would return plausible-but-wrong data, it refuses. `AlfError.code` is
stable, and the portable refusal codes are proven by the repository's shared conformance corpus.

## Behaviour worth knowing

- **Bounded before allocation.** Gzip uses the pinned browser-safe `pako` runtime, incrementally
  enforces the 512 MiB decompression ceiling, and validates CRC32/ISIZE. Decoded float64 geometry
  is capped at 512 MiB and the combined category/instance arrays for v2 `.all` at 256 MiB; inputs
  past those ceilings fail with `input-too-large`.
- **`readAlfHeader`** parses the envelope and stops, inflating only a bounded gzip prefix. It
  returns an `AlfHeader`, not an `AlfFile`, so a payload accessor on a header-only read does not
  compile — and a gzip member's CRC32/ISIZE are never checked, because verifying them means
  inflating to the end, the cost the function exists to avoid.
- **Writers enforce the shared annotation-grade error budget** by default. Exceptional conversion
  tooling can opt out explicitly with `checkBudget: false` on `writeAlp` or `writeAlf`; this does
  not relax finite/range validation.
- **`writeAlf`** re-encodes a decoded file while preserving its channel, label layout, geometry
  codec, optional instance map, and metadata. Geometry bounds are always recomputed from the
  coordinates actually encoded, so copied metadata cannot publish a stale spatial extent. It exists
  primarily for converters and the cross-language encode gate.
- **Application-specific sidecars are outside this package.** It implements the five channels
  defined by the ALF specification.

## Links

- Specification: <https://alf.dev/spec/format>
- Guides and refusal catalogue: <https://alf.dev>
- Source and conformance suite: <https://github.com/avala-ai/alf>

Apache-2.0, with the repository's NOTICE.
