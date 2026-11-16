# Adoption: what has to be true before v3 becomes the default

v1 (float16 geometry) is what `write_alp` emits unless a caller passes
`codec="int16"`. All four SDKs read v3, but an SDK release does not establish
that an application's deployed readers can consume it.

## Why opt-in

The format has **no skip-unknown mechanism**. An unrecognised version byte is
fatal, by design: a reader cannot interpret the payload without knowing that
version's semantics.

**Every consumer of a file must support v3 before its producer writes v3.**
A reader that rejects the version byte makes the geometry unreadable. A reader
that ignores it and decodes int16 words as float16 returns plausible, wrong
coordinates.

Keep each deployment's consumer and producer inventory with that application.
The public [support matrix](support.md) records SDK evidence, not the state of
any particular deployment. SDK support alone does not establish readiness or
authorize a change to the default codec.

## Implementation evidence and consumer evidence

An implementation parses ALF itself. It must pass the shared conformance suite:

```bash
python3 tests/conformance/run.py --runner "<implementation runner>"
```

A consumer delegates parsing to an implementation. Its evidence is an
integration test and the absence of its own envelope parser, enforced in its
repository's CI. Running conformance against a thin wrapper alone only retests
the SDK; it does not prove that the application actually uses that wrapper.

Inventory every read path, including sequence loaders, batch jobs, fallback
workers, and WebAssembly modules. Record the supported codecs, the deployed
artifact version, and the integration test that demonstrates correct decoding.
Document any legacy recovery parser separately. A label-only recovery tool is
not a geometry-codec blocker, but it still needs a defined compatibility scope.

## Raw-buffer and WebAssembly boundaries

A compile-time check cannot prove agreement across a boundary that passes
geometry as `Uint16List` / `&[u16]`. Both codecs use three 16-bit words per
point; the buffer type does not distinguish them. A decoder that always calls
`half::f16::from_bits` can compile while interpreting int16 as float16.

Carry an explicit codec tag and the required grid parameters across that
boundary. Refuse missing or unknown tags. Verify the caller reads the actual
file's codec rather than filling every tag with fp16. The Rust crate exposes
`alf::geometry::GeometryDecoder` for this dispatch.

An audit can start with these searches in the consuming repository:

```bash
rg -n 'required Uint16List points'
rg -n 'f16::from_bits'
```

Inspect the hits: a verified decoder's fp16 branch is expected to use half-float
conversion. A function that assumes fp16 for a buffer that may hold int16 is
the migration gap. Compare decoded coordinates against the reference using
inputs that distinguish the codecs, and test refusal of missing tags.

Precompiled worker assets need the same check. Rebuild and exercise the shipped
JavaScript and WebAssembly bundles, including fallback paths. Updating source
code alone cannot change the decoder embedded in an older bundle.

## Producers must determine their frame convention

**A v3 file states which frame its points are in; a v1 file cannot.** See
[`format.md` §5.3](format.md). Subtracting an origin leaves world-aligned axes.
Subtracting the origin and applying the inverse device rotation produces
device-aligned coordinates. These agree when the rotation is identity, so
rotation tests must include a non-identity pose.

A converter cannot infer the convention from an array of local coordinates.
Read each producer's localization code and record which transform it applies.
The answer must come from the producer, not from a guessed default.

Do not stage `_frame_convention` into v1 files: readers refuse it as
`frame-convention-on-legacy-version`. Record the determination with the
producer until it starts writing v3. For device-aligned v3, supply the rotation
required by the specification.

Migrating geometry does not automatically align annotations or other external
coordinates. Test their transforms together in the consuming application.

## Order of work

1. Verify the implementations against reader conformance and, where they offer
   writers, the independent writer interoperability gate.
2. Migrate every consumer of the affected files. Replace local envelope parsers
   with SDK calls and enforce delegation in the application's CI.
3. Verify complete deployed paths, including codec tags, fallback workers, and
   rebuilt bundles. Record evidence for every row of the deployment inventory.
4. Determine each producer's frame convention by inspecting its localization.
   This investigation can happen alongside consumer migration.
5. Enable v3 writes only when every affected reader is ready, declaring each
   producer's established convention. Retain v1 reads for stored files.

The library default remains v1. Applications can explicitly enable v3 on paths
whose consumers satisfy these requirements; publishing ALF or its SDKs is not
itself evidence that those requirements have been met.
