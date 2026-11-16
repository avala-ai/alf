# `alf`

Pure Dart reading and writing for ALF — Arrays in Local Frames, a container format for point-cloud
sequences in a local coordinate frame.

The package has no Flutter, renderer, HTTP, cache, or file-system dependency and is usable from the
Dart VM and Flutter web (JavaScript or Wasm). Supported channels are `.alp` v1 float16 and v3
fixed-point geometry, `.all` v1/v2 labels, `.ali` intensity, `.alc` colour palettes, and `.alx`
decimation indexes.

## Usage

```dart
import 'dart:typed_data';

import 'package:alf/alf.dart';

final AlfFile frame = readAlf(bytes, channel: AlfChannel.alp);
final Float64List localXyz = frame.xyz;

final AlfHeader head = readAlfHeader(bytes, channel: AlfChannel.alp); // never inflates the payload
head.totalPoints; // head has no payload accessors — they do not compile on a header-only read

final Uint8List encoded = writeAlp(
  localXyz,
  codec: AlfGeometryCodec.int16,
  // Required on v3: says whether the points were rotated into the device
  // frame as well as translated. See spec/format.md §5.3.
  frameConvention: alfFrameWorldAligned,
);
```

Count accessors return `num`: an `int` when it fits the runtime's integer range,
otherwise an integral `double` preserving the exact binary64 count declared by
the file. `writeAlx` accepts the same numeric domain for `sourceTotalPoints`.

The byte API is deliberate: consumers own transport and persistence. `readAlf` requires the channel
because the wire header does not contain it; gzip is detected by its magic bytes on read and is
deterministic on write. The reader fails closed — where guessing would return plausible-but-wrong
data, it refuses with a stable `AlfError` code proven against the repository's shared
cross-language conformance corpus.

`readAlfHeader` inflates about 64 KiB of a gzipped frame instead of the whole member, so scanning a
sequence for metadata does not pay for its payloads. It returns an `AlfHeader` rather than an
`AlfFile`, so a payload accessor on a header-only read does not compile — and a gzip member's
CRC32/ISIZE are never checked, because verifying them means inflating to the end, the cost the
function exists to avoid.

## Runtime verification

The same synthetic v3 checks execute on the VM, JavaScript and WebAssembly. They cover raw and gzip
round trips, precision and error bounds, frame conventions and quaternion metadata, invalid-return
index preservation across companion channels, and existing malformed-input refusals. They also
verify that the default writer remains v1 float16. Run from this package directory with Node 22:

```sh
dart pub get
dart test
dart run tool/platform_smoke.dart
dart compile js tool/platform_smoke.dart -o /tmp/alf-smoke.js
node /tmp/alf-smoke.js
dart compile wasm tool/platform_smoke.dart -o /tmp/alf-smoke.wasm
node tool/run_wasm.mjs /tmp/alf-smoke.wasm
```

The Wasm runner loads the generated `.mjs` file beside the `.wasm`, invokes Dart `main`, and lets
runtime errors fail the command. Merely compiling the artifact does not exercise these checks.

## Links

- Specification: <https://alf.dev/spec/format>
- Guides and refusal catalogue: <https://alf.dev>
- Source and conformance suite: <https://github.com/avala-ai/alf>

Apache-2.0, with the repository's NOTICE.
