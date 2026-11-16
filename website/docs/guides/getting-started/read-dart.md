# Read in Dart

`package:alf` is pure Dart. It has no Flutter, renderer, HTTP, cache or file-system dependency, and
runs on the Dart VM and on Flutter web — both the JavaScript and the Wasm compilation targets.

The byte API is deliberate: consumers own transport and persistence.

## Install

Registry publishing is not configured yet, so depend on it from the repository:

```yaml
dependencies:
  alf:
    git:
      url: https://github.com/avala-ai/alf.git
      path: dart/alf
```

Dart SDK 3.9 or newer.

## Read a frame

```dart
import 'dart:typed_data';

import 'package:alf/alf.dart';

final AlfFile frame = readAlf(bytes, channel: AlfChannel.alp);

final Float64List localXyz = frame.xyz;   // flat XYZ triples, LOCAL to the frame
frame.geometryCodec;                      // AlfGeometryCodec.fp16 | .int16
frame.totalPoints;                        // int?
frame.metadata;                           // Map<String, Object?>
frame.version;                            // wire version byte
```

`channel` is mandatory. The envelope deliberately does not duplicate its load-bearing filename
suffix, and the same payload bytes have different meanings by channel. Gzip is detected by its magic
bytes on read, and is deterministic on write.

The other channels:

```dart
readAlf(bytes, channel: AlfChannel.ali).intensity;        // Uint8List
readAlf(bytes, channel: AlfChannel.alc).paletteIndices;   // Uint8List
readAlf(bytes, channel: AlfChannel.all).categories;       // Uint8List
readAlf(bytes, channel: AlfChannel.all).instanceIds;      // Uint16List?
readAlf(bytes, channel: AlfChannel.alx).keptIndices;      // Uint32List
```

`frame.formatVersion` reports the `.all` layout — 1 for categories only, 2 for category plus
instance id. Accessors that do not belong to the channel you opened throw rather than answering
confidently: a mixed-channel scan that touches `formatVersion` on a v3 `.alp` should see the misuse,
not "label layout 1".

`frame.payload` is a read-only view of the decompressed bytes. The parser owns the backing storage,
so cached and future decoded values cannot be invalidated through it.

## Resolving a channel from a name

```dart
final channel = AlfChannel.fromExtension('alp');
```

## Errors

```dart
try {
  readAlf(bytes, channel: AlfChannel.alp);
} on AlfException catch (error) {
  print(error.refusal);   // 'geom-scale-not-positive', 'header-key-duplicate', ...
  print(error.message);   // the human-readable diagnostic, with file context
}
```

The identifier is the same string every other implementation produces for the same rule; see the
[refusal catalogue](../../reference/refusals.md).

## Writing

```dart
final Uint8List encoded = writeAlp(
  localXyz,
  codec: AlfGeometryCodec.int16,
  frameConvention: alfFrameWorldAligned,  // required on v3
  gzip: true,
);
```

`writeAli`, `writeAll`, `writeAlc` and `writeAlx` cover the sidecars. Writers refuse a coordinate
integer that cannot be represented exactly in binary64 (`geom-number-inexact`), and int16 refuses a
**partly** non-finite point (`geom-partial-no-return`) while encoding a wholly non-finite one as the
reserved sentinel.

## Platform reach

The package is verified on the Dart VM and compiled for both web targets in CI:

```bash
dart compile js tool/platform_smoke.dart -o /tmp/alf-smoke.js
dart compile wasm tool/platform_smoke.dart -o /tmp/alf-smoke.wasm
```

`io.dart` is out of scope on purpose — the probe is the durable part, and a file-system entry point
would be the thing that stops this package compiling for web.

## Reading only the header

Scanning a sequence for `_total_points`, `device_position` or a `palette` should not cost a full
decompression per frame.

```dart
final AlfHeader head = readAlfHeader(bytes, channel: AlfChannel.alp);

head.version;
head.metadata;
head.totalPoints;
head.xyz;        // does not COMPILE — a header-only read has no payload
```

It returns an `AlfHeader`, not an `AlfFile`. Python returns an `AlfFile` whose payload accessors
raise at runtime, because it cannot say "the same object, minus those" in its types. Dart can, so
reaching for geometry on a header-only read fails at compile time rather than on a user's machine.

For a gzipped file it inflates about 64 KiB, which holds essentially every real header; a longer one
costs a second, exactly-sized inflate rather than a full decompression.

**Its refusals are a strict subset of `readAlf`'s, and that split is the contract.** Every
header-scoped rule still applies — both entry points share one envelope parser, so they agree by
construction rather than by review. The payload-scoped rules cannot apply, because their evidence is
the bytes it declined to read:

- the declared count is not cross-checked against the payload length;
- a second gzip member goes undetected;
- the member's CRC32 and ISIZE go unverified — checking them means inflating to the end, which is
  the cost this exists to avoid.

So a header it accepts is a header `readAlf` would also accept. A **file** it accepts is not
necessarily a file `readAlf` would: a frame truncated mid-download still yields its header here.

## Next

- [Invalid returns](../invalid-returns.md) — `xyz` contains NaN, on purpose
- [Concepts](../concepts.md) — the envelope and the index-correlation invariant
- [`spec/format.md`](/spec/format) — the normative text
