# Changelog

## 0.1.0

- Version 3 `.alp` requires `_frame_convention`, and `writeAlp` gains
  `frameConvention`/`frameRotation`. See `spec/format.md` §5.3.
- Sample rotation getters once so validation, normalization, and the emitted
  v3 header use the same pose. Refuse a null runtime rotation with a named
  writer error.

- Add channel-aware readers and writers for `.alp`, `.ali`, `.alc`, `.all`, and
  `.alx`.
- Support v1 float16 geometry, v2 panoptic labels, and opt-in v3 fixed-point
  geometry.
- Add stable refusal codes, gzip handling, and shared-corpus conformance tools.
- Make gzip portable across browser, Node ESM, and Bun runtimes while bounding
  decompression and validating member trailers.
- Correct direct binary64-to-binary16 ties-to-even conversion and reject v3
  grid integer tokens that binary64 cannot represent exactly.
- Reject decimal and exponent v3 grid tokens whose stated value is not the
  binary64 shortest round-trip value, including expanded integral `.0` and
  `e0` spellings and normalized-looking scientific tokens that would otherwise
  collide after integer rounding.
- Keep v3 writer midpoint, scale, and reconstruction arithmetic finite across
  the full binary64 range.
- Accept and normalize one optional leading `#` in legacy palette colors.
- Bound input and validation memory, including reusing the `.alx` wire buffer
  for duplicate checks, and use a constant-size histogram for intensity
  percentiles.
- Bound decoded float64 geometry and v2 label typed arrays before allocation,
  while retaining a fixed-size non-refusing cache for valid DEFLATE tables.
- Keep a view over bounded payload bytes until producing the final typed result,
  reuse owned decompressed byte-channel storage, and bound caller-owned input
  plus exposed byte-channel or ALX result copies before allocation.
- Reject malformed runtime geometry values, sparse point arrays, unknown
  codecs, malformed explicit grid tuples, float16 overflow exponents, and
  lossy reserved integer tokens before JavaScript coercion.
- Accept exactly integral count spellings and wide integers that binary64
  represents exactly, emitting unsafe integer numbers in round-trip notation.
- Enforce the shared geometry quality budget by default, recompute decoded
  spatial bounds, and preserve absent legacy v2 instance maps on re-encode.
- Snapshot caller-controlled geometry and ALX count inputs, preflight palette
  cardinality before copying, and leave optional anchor precision checks to
  consumers that interpret those metadata fields.
- Run build, lint, unit, Node ESM, and conformance gates on Bun 1.2.14 and
  1.3.11 in CI. Python, Dart, TypeScript, and Rust decode every TypeScript
  writer variant in the shared interoperability gate.
- Declare the source repository and package directory required by npm trusted
  publishing, and link package consumers to alf.dev.
