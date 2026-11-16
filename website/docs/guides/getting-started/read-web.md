# Read in a browser

`@avala-ai/alf` is pure TypeScript. The runtime has no Node.js or Bun imports, all byte inputs and
outputs are `Uint8Array`, and it works in browsers, Node ESM and Bun from the same build.

The package owns decoding. It does not own transport or persistence — you fetch, it decodes.

## Install

Registry publishing is not configured yet, and there is no working one-line install from Git: npm's
`github:` shorthand takes a committish after `#`, not a subdirectory, the repository root carries no
`package.json`, and `dist/` is neither committed nor produced by a `prepare` script. So build it from
a checkout and depend on the built package by path:

```bash
git clone https://github.com/avala-ai/alf.git
cd alf/typescript/alf
bun install && bun run build
```

```jsonc
// in your package.json
"dependencies": { "@avala-ai/alf": "file:../alf/typescript/alf" }
```

`bun run build` is not optional: `package.json` exports `./dist/index.js` only, so an unbuilt
checkout resolves to nothing.

## Read a frame

```ts
import { readAlf } from "@avala-ai/alf";

const response = await fetch("https://example.com/frame_000000.alp.gz");
const bytes = new Uint8Array(await response.arrayBuffer());

const decoded = await readAlf(bytes, { channel: "alp" });

if (decoded.channel === "alp") {
  decoded.points; // Float64Array, flat XYZ triples, LOCAL to the frame
  decoded.codec; // 'fp16' | 'int16'
  decoded.maxGeomErrorM; // the frame's measured error bound, in metres
  decoded.totalPoints; // number | null
  decoded.metadata; // the parsed JSON header
}
```

**`channel` is required.** The wire header deliberately does not duplicate the load-bearing
filename suffix, and the same payload bytes mean different things on different channels — so a
reader must be told rather than guess. Gzip is detected by magic bytes, not by filename.

`AlfFile` is a discriminated union on `channel`, so narrowing gives you exactly the fields that
channel has:

```ts
const labels = await readAlf(bytes, { channel: "all" });
if (labels.channel === "all") {
  labels.formatVersion; // 1 | 2
  labels.categories; // Uint8Array
  labels.instanceIds; // Uint16Array | null
  labels.instanceMap; // Record<string, string> | null
}
```

The other channels are `intensity` (`.ali`), `palette` + `paletteIndices` (`.alc`), and
`keptIndices` + `sourceTotalPoints` (`.alx`). Entry _i_ of each describes point _i_ of the sibling
`.alp`.

## A local file

```ts
const bytes = new Uint8Array(await fileInput.files[0].arrayBuffer());
const decoded = await readAlf(bytes, { channel: "alp" });
```

## Reading only the header

Scanning a sequence for `totalPoints`, an anchor or a palette should not cost a full decompression
per frame.

```ts
import { readAlfHeader } from "@avala-ai/alf";

const head = await readAlfHeader(bytes, { channel: "alp" });

head.totalPoints;
head.metadata;
head.version;
head.points; // does not COMPILE — a header-only read has no payload
```

It returns an `AlfHeader`, not an `AlfFile`. Python returns an `AlfFile` whose payload accessors
raise, because a dataclass cannot express "these fields are not present here"; TypeScript can, so
the weaker contract is a **type** rather than a runtime error.

The same caveats apply as in Python. Header-scoped refusals are identical — both entry points share
one envelope parser — but the payload-scoped ones cannot run: the declared count is not cross-checked
against the payload, concatenated gzip members are not detected, and for a gzipped input the
member's CRC32 and ISIZE go unverified, because verifying them means inflating to the end, which is
the cost this function exists to avoid.

Internally it is backed by `gunzipPrefix`, which inflates a bounded prefix and stops. That is a
different contract from the full `gunzip`, which still walks the member for CRC/ISIZE and the
concatenated-member refusal.

## Errors

```ts
import { AlfError, readAlf } from "@avala-ai/alf";

try {
  await readAlf(bytes, { channel: "alp" });
} catch (error) {
  if (error instanceof AlfError) {
    console.error(error.code); // 'geom-scale-not-positive', 'header-key-duplicate', ...
  }
}
```

`AlfError.code` is typed as `AlfErrorCode` and is **stable across runtimes and releases** — the same
string the Python, Dart and Rust readers produce for the same rule. The portable codes are proven by
the repository's shared conformance corpus; see the [refusal catalogue](../../reference/refusals.md).

## Writing

```ts
import { writeAlp, writeAli, writeAll, writeAlc, writeAlx } from "@avala-ai/alf";

const encoded = await writeAlp(points, {
  codec: "int16",
  frameConvention: "world-aligned", // required on v3
  gzip: true,
});
```

Every writer is async and returns a `Uint8Array`. `writeAlp` accepts `codec`, an explicit
`geomScale`/`geomOffset` pair, `metadata`, `gzip`, and `checkBudget`.

Geometry writers enforce the shared annotation-grade error budget by default. Conversion tooling can
opt out with `checkBudget: false`; that does not relax finite/range validation.

`writeAlf` re-encodes a decoded file, preserving channel, label layout, geometry codec, instance map
and metadata. Geometry bounds are always recomputed from the coordinates actually encoded, so copied
metadata cannot publish a stale spatial extent. It exists primarily for converters and the
cross-language encode gate.

## Bounds

Materialized typed results are bounded **before** allocation while the source envelope is still
live: decoded float64 geometry is capped at 512 MiB, and the combined category/instance arrays for a
v2 `.all` at 256 MiB. Inputs past those ceilings fail with `input-too-large`.

Gzip uses the pinned browser-safe `pako` runtime, enforces the 512 MiB decompression ceiling
incrementally, and validates CRC32 and ISIZE.

## Not in scope

Application-specific sidecars and label merging belong in the consuming application.
The package implements the five channels defined by the ALF specification.

## Next

- [Invalid returns](../invalid-returns.md) — `points` contains NaN, on purpose
- [Precision and the two codecs](../precision.md) — which codec to write
- [`spec/format.md`](/spec/format) — the normative text
