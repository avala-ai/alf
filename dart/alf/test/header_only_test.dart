import 'dart:convert';
import 'dart:typed_data';

import 'package:alf/alf.dart';
import 'package:alf/src/gzip.dart';
import 'package:alf/src/limits.dart';
import 'package:archive/archive.dart' show getCrc32;
import 'package:test/test.dart';

/// An uncompressed ALF envelope with an arbitrary header and payload.
Uint8List envelope(
  int version,
  Map<String, Object?> header,
  List<int> payload,
) {
  final headerBytes = utf8.encode(jsonEncode(header));
  final out = BytesBuilder();
  out.addByte(version);
  final length = ByteData(4)..setUint32(0, headerBytes.length, Endian.little);
  out.add(length.buffer.asUint8List());
  out.add(headerBytes);
  out.add(payload);
  return out.toBytes();
}

void main() {
  group('header-only reads agree with the full read', () {
    test('every channel returns the same version, metadata and count', () {
      final sources = <AlfChannel, Uint8List>{
        AlfChannel.alp: writeAlp(<num>[1.5, -2.5, 3.75], metadata: {'a': 1}),
        AlfChannel.ali: writeAli(<num>[1, 2, 3], lo: 0, hi: 3),
        AlfChannel.alc: writeAlc(<int>[0, 1], <Object>['ff0000', '00ff00']),
        AlfChannel.all: writeAll(<int>[1, 2, 3]),
        AlfChannel.alx: writeAlx(
          <int>[0, 2, 5],
          alpTotalPoints: 3,
          sourceTotalPoints: 6,
        ),
      };
      sources.forEach((channel, bytes) {
        final whole = readAlf(bytes, channel: channel);
        final head = readAlfHeader(bytes, channel: channel);
        expect(head.version, whole.version, reason: '${channel.name} version');
        expect(head.channel, channel);
        expect(
          head.metadata,
          whole.metadata,
          reason: '${channel.name} metadata',
        );
        expect(
          head.totalPoints,
          whole.totalPoints,
          reason: '${channel.name} count',
        );
      });
    });

    test('an uncompressed envelope reads the same as a gzipped one', () {
      final plain = writeAlp(<num>[1.0, 2.0, 3.0], gzip: false);
      final zipped = writeAlp(<num>[1.0, 2.0, 3.0], gzip: true);
      expect(
        readAlfHeader(plain, channel: AlfChannel.alp).metadata,
        readAlfHeader(zipped, channel: AlfChannel.alp).metadata,
      );
    });

    test('a header past the 64 KiB probe still reads, in a second inflate', () {
      // Deliberately over the probe: the first pass returns a header the
      // envelope parser would call truncated, and the retry is what makes it
      // whole. A reader that trusted one pass would refuse a legal file.
      final padding = 'x' * (80 * 1024);
      final bytes = writeAlp(<num>[1.0, 2.0, 3.0], metadata: {'pad': padding});
      final head = readAlfHeader(bytes, channel: AlfChannel.alp);
      expect(head.metadata['pad'], padding);
      expect(head.metadata, readAlf(bytes, channel: AlfChannel.alp).metadata);
    });
  });

  group('it really does stop at the header', () {
    // The resource property, asserted the only way that cannot be faked: on
    // files whose PAYLOAD is missing or wrong. A reader that inflated to the
    // end could not answer any of these.

    test('a member truncated after the header still yields the header', () {
      // A frame big enough that half its compressed member still decompresses
      // to far more than the probe wants. This is the partial-download case:
      // the header is at the front and complete, and the payload is not there
      // at all — which `readAlf` cannot survive and this does not need to.
      final points = <num>[
        for (var i = 0; i < 120000; i++) (i % 977) * 0.125 - 61.0,
      ];
      final whole = writeAlp(
        points,
        metadata: {'kept': true},
        checkBudget: false,
      );
      final cut = Uint8List.sublistView(whole, 0, whole.length ~/ 2);
      expect(
        () => readAlf(cut, channel: AlfChannel.alp),
        throwsA(isA<AlfException>()),
      );
      expect(
        readAlfHeader(cut, channel: AlfChannel.alp).metadata['kept'],
        true,
      );
    });

    test('a corrupt payload CRC32 is invisible here and fatal to readAlf', () {
      final bytes = Uint8List.fromList(writeAlp(<num>[1.0, 2.0, 3.0]));
      // Flip a bit inside the CRC32 in the trailer, which is the last 8 bytes.
      bytes[bytes.length - 8] ^= 0xff;
      expect(
        () => readAlf(bytes, channel: AlfChannel.alp),
        throwsA(
          isA<AlfException>().having(
            (e) => e.refusal,
            'refusal',
            'gzip-malformed',
          ),
        ),
      );
      expect(readAlfHeader(bytes, channel: AlfChannel.alp).version, 1);
    });

    test('a second gzip member is undetected here and refused by readAlf', () {
      final first = writeAlp(<num>[1.0, 2.0, 3.0]);
      final joined = Uint8List.fromList(<int>[...first, ...first]);
      expect(
        () => readAlf(joined, channel: AlfChannel.alp),
        throwsA(
          isA<AlfException>().having(
            (e) => e.refusal,
            'refusal',
            'multiple-gzip-members',
          ),
        ),
      );
      expect(readAlfHeader(joined, channel: AlfChannel.alp).version, 1);
    });

    test('the declared count is not cross-checked against the payload', () {
      // `_total_points: 4` over a one-point payload. That disagreement is a
      // payload-scoped rule and this never saw the payload.
      final bytes = envelope(1, {'_total_points': 4}, List<int>.filled(6, 0));
      expect(
        () => readAlf(bytes, channel: AlfChannel.alp),
        throwsA(isA<AlfException>()),
      );
      expect(readAlfHeader(bytes, channel: AlfChannel.alp).totalPoints, 4);
    });

    test('the prefix inflate does not materialize the whole member', () {
      // 4 MiB of compressible payload behind a small header. The bound is on
      // what came back: a full inflate would return every byte of it.
      final payload = Uint8List(4 * 1024 * 1024);
      for (var i = 0; i < payload.length; i++) {
        payload[i] = i & 0xff;
      }
      final bytes = encodeDeterministicGzip(
        envelope(1, {'_total_points': 0}, payload),
      );
      final probed = decodeGzipPrefix(bytes, atLeast: 64 * 1024);
      expect(probed.length, lessThan(payload.length ~/ 8));
      expect(readAlfHeader(bytes, channel: AlfChannel.alp).totalPoints, 0);
    });
  });

  group('every header-scoped rule still applies', () {
    // The dangerous drift is a header-only reader that validates LESS: it would
    // hand back a header the full reader refuses, and a caller scanning a
    // sequence would never learn the file was bad. Both entry points share one
    // envelope parser so this holds by construction — these pin it anyway.

    void bothRefuse(
      String name,
      Uint8List bytes,
      AlfChannel channel,
      String refusal,
    ) {
      test(name, () {
        for (final read in <void Function()>[
          () => readAlf(bytes, channel: channel),
          () => readAlfHeader(bytes, channel: channel),
        ]) {
          expect(
            read,
            throwsA(
              isA<AlfException>().having((e) => e.refusal, 'refusal', refusal),
            ),
            reason: name,
          );
        }
      });
    }

    bothRefuse(
      'a v3 .alp that does not say which frame its points are in',
      envelope(3, {
        '_total_points': 0,
        '_geom_scale': 1.0,
        '_geom_offset': [0, 0, 0],
      }, const []),
      AlfChannel.alp,
      'frame-convention-missing',
    );
    bothRefuse(
      'a v1 .alp carrying a version-3 frame declaration',
      envelope(1, {
        '_total_points': 0,
        '_frame_convention': 'world-aligned',
      }, const []),
      AlfChannel.alp,
      'frame-convention-on-legacy-version',
    );
    bothRefuse(
      'an unsupported version byte',
      envelope(9, {'_total_points': 0}, const []),
      AlfChannel.alp,
      'version-unsupported',
    );
    bothRefuse(
      'the label layout on a geometry channel',
      envelope(2, {'_total_points': 0}, const []),
      AlfChannel.alp,
      'label-version-on-non-label-channel',
    );
    bothRefuse(
      'fixed-point geometry on an intensity channel',
      envelope(3, {'_total_points': 0}, const []),
      AlfChannel.ali,
      'geometry-version-on-non-geometry-channel',
    );
    bothRefuse(
      'a header that is not a JSON object',
      Uint8List.fromList(<int>[1, 2, 0, 0, 0, 0x5b, 0x5d]),
      AlfChannel.alp,
      'header-not-an-object',
    );
    bothRefuse(
      'a present-null declared count',
      envelope(1, {'_total_points': null}, const []),
      AlfChannel.alp,
      'total-points-not-an-integer',
    );
    // CHANNEL-scoped rather than envelope-scoped: they depend on which channel
    // you opened, and they lived only inside `_validateEnvelope` and the
    // accessors — neither of which a header-only read reaches.
    bothRefuse(
      'a .alx that does not declare its point count',
      envelope(1, {'_source_total_points': 10}, List<int>.filled(4, 0)),
      AlfChannel.alx,
      'total-points-missing',
    );
    bothRefuse(
      'a .all naming a label layout that does not exist',
      envelope(1, {'_total_points': 1, '_format_version': 9}, const [1]),
      AlfChannel.all,
      'label-layout-unsupported',
    );
    bothRefuse(
      'a header declaring more bytes than the envelope holds',
      Uint8List.fromList(<int>[1, 0xff, 0xff, 0x00, 0x00, 0x7b, 0x7d]),
      AlfChannel.alp,
      'header-truncated',
    );

    test('a duplicate JSON member name', () {
      final header = utf8.encode('{"a":1,"a":2}');
      final bytes = BytesBuilder()
        ..addByte(1)
        ..add(
          (ByteData(
            4,
          )..setUint32(0, header.length, Endian.little)).buffer.asUint8List(),
        )
        ..add(header);
      final raw = bytes.toBytes();
      for (final read in <void Function()>[
        () => readAlf(raw, channel: AlfChannel.alp),
        () => readAlfHeader(raw, channel: AlfChannel.alp),
      ]) {
        expect(
          read,
          throwsA(
            isA<AlfException>().having(
              (e) => e.refusal,
              'refusal',
              'header-key-duplicate',
            ),
          ),
        );
      }
    });

    test('an uncompressed input past the envelope limit', () {
      expect(
        () => readAlfHeader(
          Uint8List(maxAlfEnvelopeBytes + 1),
          channel: AlfChannel.alp,
        ),
        throwsA(
          isA<AlfException>().having(
            (e) => e.refusal,
            'refusal',
            'input-too-large',
          ),
        ),
      );
    });
  });

  group('bounded work, without refusing what a full read accepts', () {
    /// A gzip member that spends `emptyBlocks * 5` compressed bytes producing
    /// NOTHING before its real content. Empty non-final stored blocks are legal
    /// DEFLATE; no encoder emits them, and an author can emit as many as they
    /// like.
    Uint8List lowYieldGzip(Uint8List inner, int emptyBlocks) {
      final out = BytesBuilder();
      out.add(<int>[0x1f, 0x8b, 0x08, 0, 0, 0, 0, 0, 0, 0xff]);
      for (var i = 0; i < emptyBlocks; i++) {
        out.add(<int>[0x00, 0x00, 0x00, 0xff, 0xff]);
      }
      final length = ByteData(4)
        ..setUint16(0, inner.length, Endian.little)
        ..setUint16(2, inner.length ^ 0xffff, Endian.little);
      out.add(<int>[0x01]);
      out.add(length.buffer.asUint8List());
      out.add(inner);
      final trailer = ByteData(8)
        ..setUint32(0, getCrc32(inner), Endian.little)
        ..setUint32(4, inner.length, Endian.little);
      out.add(trailer.buffer.asUint8List());
      return out.toBytes();
    }

    test('a low-yield member the full reader accepts is not refused here', () {
      // THE PROPERTY THIS NEARLY BROKE. A first attempt at bounding the work
      // used a tight per-request input budget, which made `readAlfHeader`
      // refuse a member `readAlf` reads without complaint — inverting the one
      // guarantee the entry point makes. The bound has to be the SAME one the
      // full reader uses, or the subset claim is false.
      final inner = envelope(1, {'_total_points': 0, 'seq': 's1'}, const []);
      final bytes = lowYieldGzip(inner, 40000); // 200 KB of nothing
      expect(readAlf(bytes, channel: AlfChannel.alp).metadata['seq'], 's1');
      expect(
        readAlfHeader(bytes, channel: AlfChannel.alp).metadata['seq'],
        's1',
      );
    });

    test('both readers refuse an input past the compressed ceiling', () {
      // The bound is on the INPUT, because bounding the output does not bound
      // the work: these blocks produce zero bytes each. Exercised through the
      // injectable ceiling rather than by allocating 513 MiB.
      final oversized = Uint8List(1024 + 1024 * 1024 + 1);
      oversized[0] = 0x1f;
      oversized[1] = 0x8b;
      for (final read in <void Function()>[
        () => decodeGzipPrefix(
          oversized,
          atLeast: 16,
          maxDecompressedBytes: 1024,
        ),
        () => decodeSingleGzipMember(oversized, maxDecompressedBytes: 1024),
      ]) {
        expect(
          read,
          throwsA(
            isA<AlfException>().having(
              (e) => e.refusal,
              'refusal',
              'gzip-malformed',
            ),
          ),
        );
      }
    });
  });

  group('the gzip prefix decoder', () {
    test('returns what exists when the member is shorter than asked for', () {
      final bytes = encodeDeterministicGzip(Uint8List.fromList(<int>[1, 2, 3]));
      expect(decodeGzipPrefix(bytes, atLeast: 4096).length, 3);
    });

    test('asking for nothing inflates nothing', () {
      final bytes = encodeDeterministicGzip(Uint8List.fromList(<int>[1, 2, 3]));
      expect(decodeGzipPrefix(bytes, atLeast: 0), isEmpty);
    });

    test('its prefix is byte-identical to the full decode', () {
      final source = Uint8List.fromList(
        List<int>.generate(200000, (i) => i & 0xff),
      );
      final bytes = encodeDeterministicGzip(source);
      final prefix = decodeGzipPrefix(bytes, atLeast: 1024);
      expect(prefix.length, greaterThanOrEqualTo(1024));
      expect(prefix, Uint8List.sublistView(source, 0, prefix.length));
      expect(
        getCrc32(prefix),
        getCrc32(Uint8List.sublistView(source, 0, prefix.length)),
      );
    });

    test('a malformed gzip header is refused, not silently truncated', () {
      expect(
        () => decodeGzipPrefix(
          Uint8List.fromList(<int>[0x1f, 0x8b, 0x09, 0, 0, 0, 0, 0, 0, 0]),
          atLeast: 10,
        ),
        throwsA(
          isA<AlfException>().having(
            (e) => e.refusal,
            'refusal',
            'gzip-malformed',
          ),
        ),
      );
    });

    test(
      'a bomb is stopped by the decompressed ceiling, not by the target',
      () {
        final bomb = encodeDeterministicGzip(Uint8List(1024 * 1024));
        expect(
          () => decodeGzipPrefix(
            bomb,
            atLeast: 1024 * 1024,
            maxDecompressedBytes: 4096,
          ),
          throwsA(
            isA<AlfException>().having(
              (e) => e.refusal,
              'refusal',
              'gzip-malformed',
            ),
          ),
        );
      },
    );
  });
}
