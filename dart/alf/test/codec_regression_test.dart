import 'dart:convert';
import 'dart:collection';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:archive/archive.dart' show getCrc32;
import 'package:alf/alf.dart';
import 'package:alf/src/gzip.dart';
import 'package:alf/src/half.dart';
import 'package:alf/src/limits.dart';
import 'package:test/test.dart';

void main() {
  test('wide counts survive public sidecar and header accessors', () {
    for (final token in <String>[
      '18446744073709551616',
      '18446744073709551616.0',
      '18446744073709551616e0',
    ]) {
      final expected = double.parse(token);
      final bytes = _rawEnvelope(
        1,
        '{"_total_points":1,"_source_total_points":$token}',
        <int>[0, 0, 0, 0],
      );
      expect(
        readAlf(bytes, channel: AlfChannel.alx).sourceTotalPoints,
        expected,
      );
      final header = _rawEnvelope(1, '{"_total_points":$token}', <int>[]);
      expect(
        readAlfHeader(header, channel: AlfChannel.ali).totalPoints,
        expected,
      );
    }
  });

  test('a large count cannot wrap its payload length to zero', () {
    final bytes = _rawEnvelope(
      1,
      '{"_total_points":4611686018427387904,"_source_total_points":1}',
      <int>[],
    );
    expect(
      () => readAlf(bytes, channel: AlfChannel.alx),
      throwsA(_refusal('payload-length-mismatch')),
    );
  });

  group('invalid returns are valid content', () {
    // A NaN component in float16 .alp geometry is a lidar's invalid return
    // (spec/format.md 5.1), not damage. Every invalid return must keep its
    // original index so companion channels stay aligned.
    test('float16 geometry keeps NaN and both infinities, in place', () {
      final source = <num>[
        1.5, -2.5, 3.75, //
        double.nan, double.nan, double.nan, //
        double.infinity, double.negativeInfinity, 0.0,
      ];
      final file = readAlf(
        writeAlp(source, gzip: false, checkBudget: false),
        channel: AlfChannel.alp,
      );
      final xyz = file.xyz;
      expect(
        xyz.length,
        source.length,
        reason: 'invalid returns keep their index',
      );
      expect(xyz[0], 1.5);
      expect(xyz[3].isNaN, isTrue);
      expect(xyz[6], double.infinity);
      expect(xyz[7], double.negativeInfinity);
    });

    test('the error bound ignores invalid returns instead of becoming NaN', () {
      final clean = readAlf(
        writeAlp(<num>[100.0, 0.0, 0.0], gzip: false, checkBudget: false),
        channel: AlfChannel.alp,
      ).maxGeometryErrorM;
      final withNaN = readAlf(
        writeAlp(
          <num>[100.0, 0.0, 0.0, double.nan, double.nan, double.nan],
          gzip: false,
          checkBudget: false,
        ),
        channel: AlfChannel.alp,
      ).maxGeometryErrorM;
      expect(withNaN, clean);
      expect(withNaN!.isFinite, isTrue);
    });

    test('the box describes the finite extent, and is omitted without one', () {
      final mixed = readAlf(
        writeAlp(
          <num>[
            1.0,
            2.0,
            3.0,
            double.nan,
            double.nan,
            double.nan,
            -1.0,
            0.0,
            5.0,
          ],
          gzip: false,
          checkBudget: false,
        ),
        channel: AlfChannel.alp,
      );
      expect(mixed.metadata['box'], '-1.0:0.0:3.0:1.0:2.0:5.0');

      final allInvalid = readAlf(
        writeAlp(
          <num>[double.nan, double.nan, double.nan],
          gzip: false,
          checkBudget: false,
        ),
        channel: AlfChannel.alp,
      );
      expect(allInvalid.metadata.containsKey('box'), isFalse);
      expect(allInvalid.metadata.containsKey('_box_frame'), isFalse);
    });

    test('int16 carries a whole invalid return through the sentinel', () {
      final file = readAlf(
        writeAlp(
          <num>[1.0, 2.0, 3.0, double.nan, double.nan, double.nan],
          codec: AlfGeometryCodec.int16,
          frameConvention: alfFrameWorldAligned,
          gzip: false,
          checkBudget: false,
        ),
        channel: AlfChannel.alp,
      );
      final xyz = file.xyz;
      // The index-correlation invariant is the whole point of the sentinel:
      // dropping the invalid point would slide every later channel entry.
      expect(xyz.length, 6, reason: 'invalid returns keep their index');
      expect(xyz[0], closeTo(1.0, 1e-3));
      expect(xyz[1], closeTo(2.0, 1e-3));
      expect(xyz[2], closeTo(3.0, 1e-3));
      expect(xyz[3].isNaN, isTrue);
      expect(xyz[4].isNaN, isTrue);
      expect(xyz[5].isNaN, isTrue);
    });

    test('int16 writes a frame of nothing BUT invalid returns', () {
      // The degenerate end of the range: a fully-occluded sweep. There is no
      // finite extent to centre the grid on, so a writer that derives its
      // offset from Infinity/-Infinity seeds produces a NaN origin and then
      // rejects its own header. Every word here is the sentinel, so any finite
      // origin decodes identically.
      final file = readAlf(
        writeAlp(
          <num>[
            double.nan, double.nan, double.nan, //
            double.nan, double.nan, double.nan,
          ],
          codec: AlfGeometryCodec.int16,
          frameConvention: alfFrameWorldAligned,
          gzip: false,
          checkBudget: false,
        ),
        channel: AlfChannel.alp,
      );
      expect(file.xyz.length, 6);
      expect(file.xyz.every((value) => value.isNaN), isTrue);
    });

    test('the error bound ignores sentinel rows instead of overflowing', () {
      // A file whose `xyz` decodes cannot simultaneously be "too large to
      // bound". Reconstructing -32768 against a large finite scale overflows,
      // which made this throw for a file that had just decoded to NaN.
      final payload = ByteData(6);
      for (var i = 0; i < 3; i++) {
        payload.setInt16(i * 2, -32768, Endian.little);
      }
      final file = readAlf(
        _envelope(3, <String, Object?>{
          '_total_points': 1,
          '_geom_codec': 'int16',
          '_frame_convention': 'world-aligned',
          '_geom_scale': 1e308,
          '_geom_offset': <double>[0, 0, 0],
        }, payload.buffer.asUint8List()),
        channel: AlfChannel.alp,
      );
      expect(file.xyz.every((value) => value.isNaN), isTrue);
      final bound = file.maxGeometryErrorM;
      expect(bound, isNotNull);
      expect(bound!.isFinite, isTrue);
    });

    test('a PARTIAL sentinel triple is refused, not decoded as a position', () {
      // Hand-built, because no conforming writer can emit this: -32768 is
      // reserved and the encoder clamps real coordinates to +/-32767. Decoding
      // it arithmetically gives -327.68 — finite, plausible, never measured.
      for (final words in <List<int>>[
        <int>[-32768, 0, 0],
        <int>[0, -32768, 0],
        <int>[0, 0, -32768],
      ]) {
        final payload = ByteData(6);
        for (var i = 0; i < 3; i++) {
          payload.setInt16(i * 2, words[i], Endian.little);
        }
        final bytes = _envelope(3, <String, Object?>{
          '_total_points': 1,
          '_geom_codec': 'int16',
          '_frame_convention': 'world-aligned',
          '_geom_scale': 0.01,
          '_geom_offset': <double>[0, 0, 0],
        }, payload.buffer.asUint8List());
        expect(
          () => readAlf(bytes, channel: AlfChannel.alp).xyz,
          throwsA(_refusal('geom-partial-no-return')),
        );
      }
    });

    test(
      'int16 refuses a PARTLY invalid point, which has no interpretation',
      () {
        expect(
          () => writeAlp(
            <num>[1.0, double.nan, 3.0],
            codec: AlfGeometryCodec.int16,
            frameConvention: alfFrameWorldAligned,
            gzip: false,
            checkBudget: false,
          ),
          throwsA(isA<AlfException>()),
        );
      },
    );

    test(
      'an int16 invalid return decodes to NaN, never to the sentinel value',
      () {
        // -32768 * scale + offset is a perfectly ordinary coordinate. A decoder
        // that forgets the sentinel does not crash — it silently reports a point
        // sitting at the far edge of the grid, which is why this is pinned.
        final file = readAlf(
          writeAlp(
            <num>[1000.0, 0.0, 0.0, double.nan, double.nan, double.nan],
            codec: AlfGeometryCodec.int16,
            frameConvention: alfFrameWorldAligned,
            gzip: false,
            checkBudget: false,
          ),
          channel: AlfChannel.alp,
        );
        expect(file.xyz[3].isNaN, isTrue);
        expect(file.xyz[4].isNaN, isTrue);
        expect(file.xyz[5].isNaN, isTrue);
      },
    );
  });

  group('bounded gzip', () {
    test('stops before accumulating past the decompressed limit', () {
      final source = Uint8List.fromList(List<int>.filled(1024, 0x61));
      final compressed = encodeDeterministicGzip(source);

      expect(
        () => decodeSingleGzipMember(compressed, maxDecompressedBytes: 64),
        throwsA(_refusal('gzip-malformed')),
      );
      expect(
        decodeSingleGzipMember(compressed, maxDecompressedBytes: source.length),
        orderedEquals(source),
      );
    });

    test('validates an optional gzip header checksum', () {
      final source = Uint8List.fromList(<int>[1, 2, 3]);
      final valid = _withHeaderCrc(encodeDeterministicGzip(source));
      expect(decodeSingleGzipMember(valid), orderedEquals(source));

      final corrupt = Uint8List.fromList(valid)..[10] ^= 0x01;
      expect(
        () => decodeSingleGzipMember(corrupt),
        throwsA(_refusal('gzip-malformed')),
      );
    });
  });

  test('writer envelope ceiling matches the reader ceiling', () {
    expect(
      () => validateWritableEnvelopeSize(
        headerLength: 0,
        payloadLength: maxAlfEnvelopeBytes - 5,
      ),
      returnsNormally,
    );
    expect(
      () => validateWritableEnvelopeSize(
        headerLength: 0,
        payloadLength: maxAlfEnvelopeBytes - 4,
      ),
      throwsA(_refusal('output-too-large')),
    );
  });

  group('binary16 conversion', () {
    test('rounds binary64 directly around a half-way point', () {
      const midpoint = 1.00048828125; // 1 + 2^-11
      final justAbove = _nextUp(midpoint);

      expect(doubleToHalf(midpoint), 0x3c00, reason: 'ties round to even');
      expect(doubleToHalf(justAbove), 0x3c01);
      expect(doubleToHalf(-justAbove), 0xbc01);
    });

    test('handles subnormal and exponent-carry boundaries', () {
      const zeroToSubnormalMidpoint = 2.98023223876953125e-8; // 2^-25

      expect(doubleToHalf(zeroToSubnormalMidpoint), 0);
      expect(doubleToHalf(_nextUp(zeroToSubnormalMidpoint)), 1);
      expect(doubleToHalf(65519), 0x7bff);
      expect(doubleToHalf(65520), 0x7c00);
    });

    test('round-trips every finite binary16 value', () {
      for (var bits = 0; bits <= 0xffff; bits++) {
        if (bits & 0x7c00 == 0x7c00) continue;
        expect(doubleToHalf(halfToDouble(bits)), bits);
      }
    });
  });

  test('intensity normalization handles opposite finite double extremes', () {
    final magnitude = double.maxFinite;
    final file = readAlf(
      writeAli(
        <double>[-magnitude, 0, magnitude],
        lo: -magnitude,
        hi: magnitude,
        gzip: false,
      ),
      channel: AlfChannel.ali,
    );

    expect(file.intensity, orderedEquals(<int>[0, 127, 255]));
  });

  test('parsed payload storage cannot be mutated through the public API', () {
    final source = _envelope(
      1,
      <String, Object?>{'_total_points': 2},
      <int>[7, 9],
    );
    final file = readAlf(source, channel: AlfChannel.ali);

    expect(() => file.payload[0] = 255, throwsA(isA<UnsupportedError>()));
    source[source.length - 2] = 255;

    expect(file.payload, orderedEquals(<int>[7, 9]));
    expect(file.intensity, orderedEquals(<int>[7, 9]));
  });

  test('parsed metadata is deeply immutable', () {
    final geometry = readAlf(
      _rawEnvelope(
        3,
        '{"_total_points":1,"_geom_codec":"int16","_frame_convention":"world-aligned","_geom_scale":1,'
        '"_geom_offset":[10,20,30],'
        '"custom":{"items":[{"value":"original"}]}}',
        List<int>.filled(6, 0),
      ),
      channel: AlfChannel.alp,
    );
    final offset = geometry.metadata['_geom_offset']! as List<Object?>;
    final custom = geometry.metadata['custom']! as Map<String, Object?>;
    final items = custom['items']! as List<Object?>;
    final item = items.single! as Map<String, Object?>;

    expect(() => offset[0] = 99, throwsA(isA<UnsupportedError>()));
    expect(() => item['value'] = 'changed', throwsA(isA<UnsupportedError>()));
    expect(geometry.xyz, orderedEquals(<double>[10, 20, 30]));

    final labels = readAlf(
      _envelope(
        2,
        <String, Object?>{
          '_total_points': 1,
          'instance_map': <String, Object?>{'1': 'original-id'},
        },
        <int>[7, 1, 0],
      ),
      channel: AlfChannel.all,
    );
    final instanceMap =
        labels.metadata['instance_map']! as Map<String, Object?>;
    expect(
      () => instanceMap['1'] = 'changed-id',
      throwsA(isA<UnsupportedError>()),
    );
    expect(labels.instanceMap, <String, String>{'1': 'original-id'});

    final colors = readAlf(
      _envelope(
        1,
        <String, Object?>{
          '_total_points': 1,
          'palette': <Object?>['ff0000'],
        },
        <int>[0],
      ),
      channel: AlfChannel.alc,
    );
    final palette = colors.metadata['palette']! as List<Object?>;
    expect(() => palette[0] = '00ff00', throwsA(isA<UnsupportedError>()));
    expect(colors.palette, <String>['ff0000']);
  });

  test('intensity writer refuses integers that binary64 would change', () {
    expect(
      () => writeAli(
        <num>[9007199254740992, 9007199254740993],
        lo: 9007199254740992,
        hi: 9007199254740993,
      ),
      throwsA(_refusal('intensity-number-inexact')),
    );
    expect(
      () => writeAli(
        <num>[9007199254740993],
        lo: 9007199254740992,
        hi: 9007199254740994,
      ),
      throwsA(_refusal('intensity-number-inexact')),
    );
  });

  test('header preflight bounds nesting and non-finite custom numbers', () {
    final nested =
        '${List<String>.filled(60000, '[').join()}0${List<String>.filled(60000, ']').join()}';
    expect(
      () => readAlf(
        _rawEnvelope(1, '{"_total_points":0,"custom":$nested}', const <int>[]),
        channel: AlfChannel.ali,
      ),
      throwsA(_refusal('header-nesting-too-deep')),
    );
    expect(
      () => readAlf(
        _rawEnvelope(
          1,
          '{"_total_points":0,"custom":{"unsafe":1e309}}',
          const <int>[],
        ),
        channel: AlfChannel.ali,
      ),
      throwsA(_refusal('header-not-json')),
    );
  });

  test('reader and writer share the 256 collection depth limit', () {
    Object? allowed = 0;
    for (var depth = 0; depth < 255; depth++) {
      allowed = <Object?>[allowed];
    }
    final encoded = writeAli(
      const <num>[],
      lo: 0,
      hi: 1,
      metadata: <String, Object?>{'custom': allowed},
      gzip: false,
    );
    expect(
      readAlf(encoded, channel: AlfChannel.ali).metadata['custom'],
      isNotNull,
    );

    final tooDeep = <Object?>[allowed];
    expect(
      () => writeAli(
        const <num>[],
        lo: 0,
        hi: 1,
        metadata: <String, Object?>{'custom': tooDeep},
        gzip: false,
      ),
      throwsA(_refusal('header-nesting-too-deep')),
    );
  });

  test('writer validates the JSON structure produced by custom toJson', () {
    Object? allowed = 0;
    for (var depth = 0; depth < 255; depth++) {
      allowed = <Object?>[allowed];
    }
    final encoded = writeAli(
      const <num>[],
      lo: 0,
      hi: 1,
      metadata: <String, Object?>{'custom': _CustomJsonValue(allowed)},
      gzip: false,
    );
    expect(
      readAlf(encoded, channel: AlfChannel.ali).metadata['custom'],
      isNotNull,
    );

    final tooDeep = <Object?>[allowed];
    expect(
      () => writeAli(
        const <num>[],
        lo: 0,
        hi: 1,
        metadata: <String, Object?>{'custom': _CustomJsonValue(tooDeep)},
        gzip: false,
      ),
      throwsA(_refusal('header-nesting-too-deep')),
    );
  });

  test('reserved counts ignore insignificant exponent padding', () {
    final file = readAlf(
      _rawEnvelope(1, '{"_total_points":1e00000000000}', const <int>[7]),
      channel: AlfChannel.ali,
    );

    expect(file.totalPoints, 1);
    expect(file.intensity, orderedEquals(<int>[7]));
  });

  test(
    'geometry writer uses indexed input without iterable materialization',
    () {
      final source = _IndexedOnlyNumbers(<num>[1.25, -2.5, 3.75]);
      for (final codec in AlfGeometryCodec.values) {
        final file = readAlf(
          writeAlp(
            source,
            codec: codec,
            // Only v3 may state a convention, and it must.
            frameConvention: codec == AlfGeometryCodec.int16
                ? alfFrameWorldAligned
                : null,
            gzip: false,
            checkBudget: false,
          ),
          channel: AlfChannel.alp,
        );
        expect(file.xyz, hasLength(3));
      }
    },
  );

  group('v3 fixed-point geometry', () {
    test('writes opposite finite double extremes without NaN or infinity', () {
      final magnitude = double.maxFinite;
      final source = <double>[
        -magnitude,
        -magnitude,
        -magnitude,
        magnitude,
        magnitude,
        magnitude,
      ];
      final file = readAlf(
        writeAlp(
          source,
          codec: AlfGeometryCodec.int16,
          frameConvention: alfFrameWorldAligned,
          gzip: false,
          checkBudget: false,
        ),
        channel: AlfChannel.alp,
      );

      expect(file.version, 3);
      expect(file.metadata['_geom_offset'], orderedEquals(<double>[0, 0, 0]));
      expect(
        (file.metadata['_geom_scale']! as num).toDouble().isFinite,
        isTrue,
      );
      expect(file.xyz.every((double value) => value.isFinite), isTrue);
      expect(file.xyz.first, lessThan(0));
      expect(file.xyz.last, greaterThan(0));
    });

    test('writer emits an exact binary64 token for a wide integer grid', () {
      final exact = math.pow(2, 64).toDouble();
      final file = readAlf(
        writeAlp(
          <double>[exact, 0, 0],
          codec: AlfGeometryCodec.int16,
          frameConvention: alfFrameWorldAligned,
          checkBudget: false,
        ),
        channel: AlfChannel.alp,
      );
      expect(file.xyz.first, exact);
    });

    test('refuses geometry number spellings binary64 changes', () {
      for (final header in <String>[
        '{"_total_points":1,"_geom_codec":"int16","_frame_convention":"world-aligned","_geom_scale":1,'
            '"_geom_offset":[9007199254740993,0,0]}',
        '{"_total_points":1,"_geom_codec":"int16","_frame_convention":"world-aligned","_geom_scale":1,'
            '"_geom_offset":[9007199254740993.5,0,0]}',
        '{"_total_points":1,"_geom_codec":"int16","_frame_convention":"world-aligned","_geom_scale":1,'
            '"_geom_offset":[18446744073709552000,0,0]}',
        '{"_total_points":1,"_geom_codec":"int16","_frame_convention":"world-aligned",'
            '"_geom_scale":9007199254740993.0,"_geom_offset":[0,0,0]}',
        '{"_total_points":1,"_geom_codec":"int16","_frame_convention":"world-aligned","_geom_scale":1,'
            '"_geom_offset":[1000000000000000100.0,0,0]}',
        '{"_total_points":1,"_geom_codec":"int16","_frame_convention":"world-aligned",'
            '"_geom_scale":1000000000000000100e0,"_geom_offset":[0,0,0]}',
      ]) {
        expect(
          () => readAlf(
            _rawEnvelope(3, header, List<int>.filled(6, 0)),
            channel: AlfChannel.alp,
          ),
          throwsA(_refusal('geom-number-inexact')),
        );
      }

      expect(
        () => readAlf(
          _rawEnvelope(
            3,
            '{"_total_points":1,"_geom_codec":"int16","_frame_convention":"world-aligned",'
            '"custom":{"_geom_scale":9007199254740993.0,'
            '"array":[true,null,{"text":"}"}]},'
            '"_geom_scale":0.1,"_geom_offset":[9007199254740991.0,0,0]}',
            List<int>.filled(6, 0),
          ),
          channel: AlfChannel.alp,
        ),
        returnsNormally,
      );

      for (final exact in <String>[
        '9007199254740992', // 2^53
        '9007199254740994', // first representable even integer above 2^53
        '18446744073709551616', // 2^64: wide decimal, exact binary64
        '18446744073709551616.0', // 2^64 in integral decimal form
        '18446744073709551616e0', // 2^64 in integral exponent form
      ]) {
        final file = readAlf(
          _rawEnvelope(
            3,
            '{"_total_points":1,"_geom_codec":"int16","_frame_convention":"world-aligned",'
            '"_geom_scale":$exact,"_geom_offset":[$exact,0,0]}',
            List<int>.filled(6, 0),
          ),
          channel: AlfChannel.alp,
        );
        expect(file.xyz.first, double.parse(exact));
        expect(file.metadata['_geom_scale'], double.parse(exact));
      }

      expect(
        () => readAlf(
          _rawEnvelope(
            3,
            '{"_total_points":1,"_geom_codec":"int16","_frame_convention":"world-aligned",'
            '"_geom_scale":1e400,"_geom_offset":[0,0,0]}',
            List<int>.filled(6, 0),
          ),
          channel: AlfChannel.alp,
        ),
        throwsA(_refusal('geom-number-inexact')),
      );
    });

    test('bounds hostile geometry token canonicalization', () {
      final excessiveSignificand = List<String>.filled(100000, '7').join();
      final excessiveExponent = List<String>.filled(100000, '9').join();
      for (final scale in <String>[
        excessiveSignificand,
        '1e$excessiveExponent',
      ]) {
        expect(
          () => readAlf(
            _rawEnvelope(
              3,
              '{"_total_points":1,"_geom_codec":"int16","_frame_convention":"world-aligned",'
              '"_geom_scale":$scale,"_geom_offset":[0,0,0]}',
              List<int>.filled(6, 0),
            ),
            channel: AlfChannel.alp,
          ),
          throwsA(_refusal('geom-number-inexact')),
        );
      }
    });

    test('ignores exponent leading zeros when checking token bounds', () {
      final file = readAlf(
        _rawEnvelope(
          3,
          '{"_total_points":1,"_geom_codec":"int16","_frame_convention":"world-aligned",'
          '"_geom_scale":1e00000000001,"_geom_offset":[0,0,0]}',
          <int>[1, 0, 0, 0, 0, 0],
        ),
        channel: AlfChannel.alp,
      );

      expect(file.xyz, orderedEquals(<double>[10, 0, 0]));
    });

    test('cancels decimal placement before bounding geometry exponents', () {
      final exactWide =
          '0.${List<String>.filled(9980, '0').join()}'
          '18446744073709551616e10000';
      final file = readAlf(
        _rawEnvelope(
          3,
          '{"_total_points":1,"_geom_codec":"int16","_frame_convention":"world-aligned",'
          '"_geom_scale":$exactWide,"_geom_offset":[0,0,0]}',
          <int>[1, 0, 0, 0, 0, 0],
        ),
        channel: AlfChannel.alp,
      );

      expect(file.xyz.first, double.parse('18446744073709551616'));
    });
  });

  test('reader validates geometry anchor shape before accessors', () {
    expect(
      () => readAlf(
        _rawEnvelope(
          1,
          '{"_total_points":1,"device_position":{"x":1,"y":2}}',
          List<int>.filled(6, 0),
        ),
        channel: AlfChannel.alp,
      ),
      throwsA(_refusal('geometry-anchor-malformed')),
    );
  });

  test('instanceMap validates the payload identity stream', () {
    final reserved = readAlf(
      _envelope(
        2,
        <String, Object?>{
          '_total_points': 1,
          'instance_map': <String, Object?>{'1': 'one'},
        },
        <int>[7, 255, 255],
      ),
      channel: AlfChannel.all,
    );
    expect(
      () => reserved.instanceMap,
      throwsA(_refusal('instance-id-reserved')),
    );

    final incomplete = readAlf(
      _envelope(
        2,
        <String, Object?>{
          '_total_points': 1,
          'instance_map': <String, Object?>{'1': 'one'},
        },
        <int>[7, 2, 0],
      ),
      channel: AlfChannel.all,
    );
    expect(
      () => incomplete.instanceMap,
      throwsA(_refusal('instance-map-incomplete')),
    );
    expect(incomplete.categories, orderedEquals(<int>[7]));
  });

  test('eagerly validated ALX indices are cached and immutable', () {
    final file = readAlf(
      _envelope(
        1,
        <String, Object?>{'_total_points': 2, '_source_total_points': 10},
        <int>[1, 0, 0, 0, 7, 0, 0, 0],
      ),
      channel: AlfChannel.alx,
    );

    final first = file.keptIndices;

    expect(identical(first, file.keptIndices), isTrue);
    expect(() => first[0] = 0, throwsA(isA<UnsupportedError>()));
    expect(first, orderedEquals(<int>[1, 7]));
  });

  test('legacy palette normalizes one optional hash but rejects two', () {
    final file = readAlf(
      _envelope(
        1,
        <String, Object?>{
          '_total_points': 2,
          'palette': <Object?>['#FF0000', '00ff00'],
        },
        <int>[0, 1],
      ),
      channel: AlfChannel.alc,
    );

    expect(file.palette, orderedEquals(<String>['ff0000', '00ff00']));
    expect(
      () => readAlf(
        _envelope(
          1,
          <String, Object?>{
            '_total_points': 1,
            'palette': <Object?>['##ff0000'],
          },
          <int>[0],
        ),
        channel: AlfChannel.alc,
      ).palette,
      throwsA(_refusal('palette-entry-malformed')),
    );
    expect(
      () => writeAlc(<int>[0], <Object>['##ff0000']),
      throwsA(_refusal('palette-entry-malformed')),
    );
  });

  test('refuses duplicate JSON member names at any header depth', () {
    expect(
      () => readAlf(
        _rawEnvelope(
          1,
          '{"_total_points":1,"custom":{"sensor":"left","sensor":"right"}}',
          <int>[7],
        ),
        channel: AlfChannel.ali,
      ),
      throwsA(_refusal('header-key-duplicate')),
    );
  });

  test('alx writer refuses a binary64-inexact source count', () {
    expect(
      () => writeAlx(
        <int>[0],
        alpTotalPoints: 1,
        sourceTotalPoints: 9007199254740993,
      ),
      throwsA(_refusal('total-points-not-an-integer')),
    );
  });

  test('alx writer refuses a fractional numeric source count', () {
    expect(
      () => writeAlx(<int>[0], alpTotalPoints: 1, sourceTotalPoints: 1.5),
      throwsA(_refusal('total-points-not-an-integer')),
    );
  });

  test('alx writer preserves an exactly representable wide source count', () {
    // Retain the existing int input path as well as the larger integral
    // doubles exercised by the VM/JavaScript/Wasm runtime checks.
    const exact = 9007199254740992;
    final bytes = writeAlx(
      <int>[0],
      alpTotalPoints: 1,
      sourceTotalPoints: exact,
      gzip: false,
    );
    final headerLength = ByteData.sublistView(
      bytes,
    ).getUint32(1, Endian.little);
    final header = utf8.decode(bytes.sublist(5, 5 + headerLength));
    expect(header, contains('"_source_total_points":9007199254740992'));
    expect(readAlf(bytes, channel: AlfChannel.alx).sourceTotalPoints, exact);
  });
}

double _nextUp(double value) {
  final data = ByteData(8)..setFloat64(0, value, Endian.big);
  final high = data.getUint32(0, Endian.big);
  final low = data.getUint32(4, Endian.big);
  if (low == 0xffffffff) {
    data
      ..setUint32(0, high + 1, Endian.big)
      ..setUint32(4, 0, Endian.big);
  } else {
    data.setUint32(4, low + 1, Endian.big);
  }
  return data.getFloat64(0, Endian.big);
}

Matcher _refusal(String identifier) => isA<AlfException>().having(
  (AlfException error) => error.refusal,
  'refusal',
  identifier,
);

Uint8List _envelope(
  int version,
  Map<String, Object?> metadata,
  List<int> payload,
) {
  return _rawEnvelope(version, jsonEncode(metadata), payload);
}

Uint8List _rawEnvelope(int version, String metadata, List<int> payload) {
  final header = utf8.encode(metadata);
  final builder = BytesBuilder(copy: false)
    ..addByte(version)
    ..add(
      (ByteData(
        4,
      )..setUint32(0, header.length, Endian.little)).buffer.asUint8List(),
    )
    ..add(header)
    ..add(payload);
  return builder.takeBytes();
}

Uint8List _withHeaderCrc(Uint8List gzip) {
  final header = Uint8List.fromList(gzip.sublist(0, 10));
  header[3] |= 0x02;
  final crc16 = getCrc32(header) & 0xffff;
  return (BytesBuilder(copy: false)
        ..add(header)
        ..add(<int>[crc16 & 0xff, crc16 >> 8])
        ..add(Uint8List.sublistView(gzip, 10)))
      .takeBytes();
}

final class _IndexedOnlyNumbers extends ListBase<num> {
  _IndexedOnlyNumbers(this._values);

  final List<num> _values;

  @override
  int get length => _values.length;

  @override
  set length(int value) => throw UnsupportedError('immutable');

  @override
  num operator [](int index) => _values[index];

  @override
  void operator []=(int index, num value) =>
      throw UnsupportedError('immutable');

  @override
  Iterator<num> get iterator =>
      throw StateError('writer must not materialize through iteration');
}

final class _CustomJsonValue {
  const _CustomJsonValue(this.value);

  final Object? value;

  Object? toJson() => value;
}
