# Changelog

## 0.1.0

- Export `decodeSingleGzipMember` and `gzipInputOverheadAllowance` from the
  public library, so byte-oriented consumers can reuse the strict, bounded
  single-member gzip decoder.

- Require `archive` `>=4.0.7 <4.6.0`. Later versions buffer inflate output
  before the sink sees it, which defeats the bounded prefix inflater.

- Preserve exact counts beyond signed 64-bit range. `totalPoints` and
  `sourceTotalPoints` return `num`; native-sized counts still return `int`, and
  wider counts remain integral doubles. `writeAlx` accepts a `num` source count
  and emits its exact integer spelling. Payload checks avoid integer overflow.

- Execute the shared v3 runtime checks on the VM, JavaScript and WebAssembly.
  Cover frame metadata, precision, error bounds and companion-channel index
  preservation; keep wide-integer smoke fixtures portable to all three runtimes.

- Preserve infinity invalid returns in JavaScript writers as on the Dart VM.
  Check non-finite content before integer narrowing; both geometry codecs are
  exercised by the platform regression gate.

- Add `readAlfHeader`, which parses an envelope without inflating the payload —
  about 64 KiB for a gzipped frame instead of the whole member. It returns an
  `AlfHeader` rather than an `AlfFile`, so a payload accessor on a header-only
  read does not compile. Its refusals are a strict subset of `readAlf`'s; both
  entry points share one envelope parser.

- Version 3 `.alp` requires `_frame_convention`, and `writeAlp` gains
  `frameConvention`/`frameRotation`. See `spec/format.md` §5.3.

- Add pure Dart readers and writers for every ALF channel.
- Add v1 float16 and v3 fixed-point geometry support.
- Add v1 semantic and v2 panoptic label support.
- Detect compressed input by magic and emit deterministic gzip envelopes.
- Bound hostile JSON nesting and reject non-finite metadata numbers before decode.
- Avoid full boxed geometry copies while writing large frames.
- Reject lossy integer intensity inputs and accept exactly representable wide geometry integers.
- Keep parsed payloads and eagerly validated decimation indices immutable.
- Enforce one shared 512 MiB reader/writer envelope ceiling.
- Validate panoptic payload identity whenever its instance map is requested.
- Validate optional gzip header checksums instead of skipping them.
- Deeply freeze parsed JSON metadata so validated state cannot be mutated.
- Link pub.dev users to the project site, source, issue tracker, and Dart guide.
