import 'dart:typed_data';

import 'package:alf/alf.dart';
import 'package:test/test.dart';

// spec/format.md §2 (Envelope): gzip wraps the same uncompressed envelope.
// These public-API tests retain the existing strict whole-member and bounded
// decoder behavior exercised by codec_regression_test.dart and
// header_only_test.dart; this export adds no wire rule or conformance case.
void main() {
  test('public strict decoder honors the caller envelope budget', () {
    final raw = writeAli(List<int>.filled(128, 0), lo: 0, hi: 255, gzip: false);
    final compressed = writeAli(List<int>.filled(128, 0), lo: 0, hi: 255);
    expect(
      decodeSingleGzipMember(compressed, maxDecompressedBytes: raw.length),
      raw,
    );
    expect(
      () => decodeSingleGzipMember(
        compressed,
        maxDecompressedBytes: raw.length - 1,
      ),
      throwsA(isA<AlfException>()),
    );
    expect(gzipInputOverheadAllowance, 1024 * 1024);
  });

  test('public strict decoder rejects corrupt or concatenated members', () {
    final compressed = writeAli(const [0, 127, 255], lo: 0, hi: 255);
    for (final offset in [compressed.length - 8, compressed.length - 4]) {
      final damaged = Uint8List.fromList(compressed)..[offset] ^= 1;
      expect(
        () => decodeSingleGzipMember(damaged),
        throwsA(isA<AlfException>()),
      );
    }
    expect(
      () =>
          decodeSingleGzipMember(compressed.sublist(0, compressed.length - 1)),
      throwsA(isA<AlfException>()),
    );
    expect(
      () => decodeSingleGzipMember(
        Uint8List.fromList([...compressed, ...compressed]),
      ),
      throwsA(isA<AlfException>()),
    );
  });
}
