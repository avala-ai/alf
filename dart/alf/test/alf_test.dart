import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:alf/alf.dart';
import 'package:test/test.dart';

void main() {
  group('geometry writers', () {
    const source = <double>[
      1.37,
      -2.11,
      0.43,
      87.62,
      41.19,
      -3.07,
      -0.31,
      12.73,
      5.88,
    ];

    test('v1 float16 writes a readable, deterministic envelope', () {
      final first = writeAlp(source, gzip: true, checkBudget: false);
      final second = writeAlp(source, gzip: true, checkBudget: false);
      expect(first, orderedEquals(second));
      expect(first.sublist(4, 8), orderedEquals(<int>[0, 0, 0, 0]));

      final file = readAlf(first, channel: AlfChannel.alp);
      expect(file.version, 1);
      expect(file.geometryCodec, AlfGeometryCodec.fp16);
      expect(file.xyz.length, source.length);
      expect(file.maxGeometryErrorM, 0.03125);
    });

    test('v3 fixed point writes midpoint metadata and a bounded decode', () {
      final bytes = writeAlp(
        source,
        codec: AlfGeometryCodec.int16,
        frameConvention: alfFrameWorldAligned,
        checkBudget: false,
      );
      final file = readAlf(bytes, channel: AlfChannel.alp);
      expect(file.version, 3);
      expect(file.geometryCodec, AlfGeometryCodec.int16);
      expect(file.metadata['_geom_offset'], isA<List<Object?>>());
      expect(file.metadata['_geom_scale'], isA<num>());
      expect(file.maxGeometryErrorM, lessThan(0.001));
      for (var i = 0; i < source.length; i++) {
        expect(
          (file.xyz[i] - source[i]).abs(),
          lessThanOrEqualTo(file.maxGeometryErrorM!),
        );
      }
    });

    test('refuses integer coordinates that narrow lossily to binary64', () {
      expect(
        () => writeAlp(
          <num>[9007199254740993, 0, 0],
          codec: AlfGeometryCodec.int16,
          frameConvention: alfFrameWorldAligned,
        ),
        throwsA(_refusal('geom-number-inexact')),
      );
      expect(
        () => writeAlp(
          <num>[9007199254740994, 0, 0],
          codec: AlfGeometryCodec.int16,
          frameConvention: alfFrameWorldAligned,
          checkBudget: false,
        ),
        returnsNormally,
      );
    });

    test('validates and normalizes inherited geometry anchors', () {
      final file = readAlf(
        writeAlp(
          const <num>[1, 2, 3],
          metadata: <String, Object?>{
            'device_position': <num>[4, 5, 6],
          },
        ),
        channel: AlfChannel.alp,
      );
      expect(file.metadata['device_position'], <String, double>{
        'x': 4,
        'y': 5,
        'z': 6,
      });
      for (final metadata in <Map<String, Object?>>[
        <String, Object?>{
          'device_position': <String, Object?>{'x': 1},
        },
        <String, Object?>{
          'world_centroid_epsg3857': <Object?>[1, double.infinity, 3],
        },
        <String, Object?>{
          'device_position': <Object?>[9007199254740993, 0, 0],
        },
      ]) {
        expect(
          () => writeAlp(const <num>[1, 2, 3], metadata: metadata),
          throwsA(_refusal('geometry-anchor-malformed')),
        );
      }
    });

    test('preserves anchor application members without mutating metadata', () {
      final metadata = <String, Object?>{
        'device_position': <String, Object?>{
          'z': 3,
          'note': <String, Object?>{
            'source': 'gps',
            'flags': <Object?>[
              'rtk',
              <String, Object?>{'fixed': true},
            ],
          },
          'x': 1,
          'quality': 0.875,
          'y': 2,
        },
      };
      final before = jsonEncode(metadata);

      final file = readAlf(
        writeAlp(const <num>[0, 0, 0], metadata: metadata),
        channel: AlfChannel.alp,
      );
      expect(file.metadata['device_position'], <String, Object?>{
        'note': <String, Object?>{
          'source': 'gps',
          'flags': <Object?>[
            'rtk',
            <String, Object?>{'fixed': true},
          ],
        },
        'quality': 0.875,
        'x': 1.0,
        'y': 2.0,
        'z': 3.0,
      });
      expect(jsonEncode(metadata), before);
    });

    test('writes exact wide anchor axes for both geometry codecs', () {
      final exact = double.parse('18446744073709551616');
      for (final codec in AlfGeometryCodec.values) {
        final bytes = writeAlp(
          const <num>[0, 0, 0],
          metadata: <String, Object?>{
            'device_position': <String, Object?>{
              'x': exact,
              'y': 0,
              'z': 0,
              'note': 'gps',
            },
          },
          codec: codec,
          // Only v3 may state a convention, and it must; this loop writes both.
          frameConvention: codec == AlfGeometryCodec.int16
              ? alfFrameWorldAligned
              : null,
          gzip: false,
          checkBudget: false,
        );
        final headerLength = ByteData.sublistView(
          bytes,
          1,
          5,
        ).getUint32(0, Endian.little);
        final headerText = utf8.decode(bytes.sublist(5, 5 + headerLength));
        expect(headerText, contains('1.8446744073709552e+19'));
        expect(headerText, isNot(contains('18446744073709552000')));

        final file = readAlf(bytes, channel: AlfChannel.alp);
        final anchor =
            file.metadata['device_position']! as Map<String, Object?>;
        expect(anchor['x'], exact);
        expect(anchor['note'], 'gps');
      }
    });
  });

  test('intensity writer normalizes and records deciles', () {
    final file = readAlf(
      writeAli(<double>[0, 5, 10], lo: 0, hi: 10),
      channel: AlfChannel.ali,
    );
    expect(file.intensity, orderedEquals(<int>[0, 127, 255]));
    expect(file.metadata['_intensities_every_10_percent'], hasLength(9));
  });

  test('intensity writer refuses malformed inherited deciles', () {
    for (final malformed in <Object?>[
      'not-a-list',
      <Object?>[1, 2, '3'],
      <int>[1, 2, 3],
    ]) {
      expect(
        () => writeAli(
          const <num>[0],
          lo: 0,
          hi: 1,
          metadata: <String, Object?>{
            '_intensities_every_10_percent': malformed,
          },
        ),
        throwsA(_refusal('intensity-deciles-malformed')),
      );
    }
  });

  group('label writers', () {
    test('v1 categories', () {
      final file = readAlf(writeAll(<int>[1, 2, 3]), channel: AlfChannel.all);
      expect(file.version, 1);
      expect(file.formatVersion, 1);
      expect(file.categories, orderedEquals(<int>[1, 2, 3]));
      expect(file.instanceIds, isNull);
    });

    test('v2 categories and instance ids', () {
      final file = readAlf(
        writeAll(
          <int>[3, 7, 255],
          instanceIds: <int>[0, 1, 65534],
          instanceMap: const <String, String>{
            '1': 'first-instance',
            '65534': 'last-instance',
          },
        ),
        channel: AlfChannel.all,
      );
      expect(file.version, 2);
      expect(file.formatVersion, 2);
      expect(file.categories, orderedEquals(<int>[3, 7, 255]));
      expect(file.instanceIds, orderedEquals(<int>[0, 1, 65534]));
      expect(file.instanceMap, const <String, String>{
        '1': 'first-instance',
        '65534': 'last-instance',
      });
    });

    test('version disagreement resolves only from a unique counted stride', () {
      final headerSaysV2 = readAlf(
        _envelope(
          1,
          <String, Object?>{
            '_total_points': 2,
            '_format_version': 2,
            'instance_map': <String, Object?>{'1': 'first'},
          },
          <int>[7, 1, 0, 8, 0, 0],
        ),
        channel: AlfChannel.all,
      );
      expect(headerSaysV2.formatVersion, 2);
      expect(headerSaysV2.categories, orderedEquals(<int>[7, 8]));
      expect(headerSaysV2.instanceIds, orderedEquals(<int>[1, 0]));

      final headerSaysV1 = readAlf(
        _envelope(
          2,
          <String, Object?>{'_total_points': 2, '_format_version': 1},
          <int>[7, 8],
        ),
        channel: AlfChannel.all,
      );
      expect(headerSaysV1.formatVersion, 1);
      expect(headerSaysV1.categories, orderedEquals(<int>[7, 8]));
    });

    test('ambiguous version disagreement fails closed', () {
      for (final bytes in <Uint8List>[
        _envelope(1, <String, Object?>{'_format_version': 2}, <int>[7, 1, 0]),
        _envelope(1, <String, Object?>{
          '_total_points': 0,
          '_format_version': 2,
        }, const <int>[]),
      ]) {
        expect(
          () => readAlf(bytes, channel: AlfChannel.all),
          throwsA(_refusal('label-layout-ambiguous')),
        );
      }
    });

    test('instance map keys must be canonical decimal ids in 1..65534', () {
      for (final badKey in <String>['0', '01', '65535', 'one']) {
        final metadata = <String, Object?>{
          '_total_points': 1,
          '_format_version': 2,
          'instance_map': <String, Object?>{
            '1': 'valid-emitted-id',
            badKey: 'invalid-extra-id',
          },
        };
        expect(
          () => readAlf(
            _envelope(2, metadata, <int>[7, 1, 0]),
            channel: AlfChannel.all,
          ).instanceIds,
          throwsA(_refusal('instance-map-malformed')),
          reason: 'reader accepted key $badKey',
        );
        expect(
          () => writeAll(
            <int>[7],
            instanceIds: <int>[1],
            instanceMap: <String, String>{
              '1': 'valid-emitted-id',
              badKey: 'invalid-extra-id',
            },
          ),
          throwsA(_refusal('instance-map-malformed')),
          reason: 'writer accepted key $badKey',
        );
      }
    });

    test(
      'semantic categories remain readable when panoptic identity fails',
      () {
        final file = readAlf(
          _envelope(
            2,
            <String, Object?>{'_total_points': 1},
            <int>[7, 255, 255],
          ),
          channel: AlfChannel.all,
        );
        expect(file.categories, orderedEquals(<int>[7]));
        expect(
          () => file.instanceIds,
          throwsA(_refusal('instance-id-reserved')),
        );
      },
    );
  });

  test('palette writer permits the no-colour sentinel', () {
    final file = readAlf(
      writeAlc(
        <int>[0, 1, 2],
        <Object>[
          '#ff0000',
          <int>[0, 255, 0],
        ],
      ),
      channel: AlfChannel.alc,
    );
    expect(file.palette, <String>['ff0000', '00ff00']);
    expect(file.paletteIndices, orderedEquals(<int>[0, 1, 2]));
  });

  test('decimation writer preserves reordered source indexes', () {
    final file = readAlf(
      writeAlx(<int>[7, 2, 19], alpTotalPoints: 3, sourceTotalPoints: 20),
      channel: AlfChannel.alx,
    );
    expect(file.keptIndices, orderedEquals(<int>[7, 2, 19]));
    expect(file.sourceTotalPoints, 20);
  });

  test('present-null total point counts are malformed on every channel', () {
    final cases = <(AlfChannel, Uint8List)>[
      (
        AlfChannel.alp,
        _envelope(1, <String, Object?>{
          '_total_points': null,
        }, List<int>.filled(6, 0)),
      ),
      (
        AlfChannel.ali,
        _envelope(1, <String, Object?>{'_total_points': null}, <int>[1]),
      ),
      (
        AlfChannel.all,
        _envelope(1, <String, Object?>{'_total_points': null}, <int>[1]),
      ),
      (
        AlfChannel.alc,
        _envelope(
          1,
          <String, Object?>{
            '_total_points': null,
            'palette': <Object?>['ff0000'],
          },
          <int>[0],
        ),
      ),
      (
        AlfChannel.alx,
        _envelope(1, <String, Object?>{
          '_total_points': null,
          '_source_total_points': 1,
        }, List<int>.filled(4, 0)),
      ),
    ];
    for (final (channel, bytes) in cases) {
      expect(
        () => readAlf(bytes, channel: channel),
        throwsA(_refusal('total-points-not-an-integer')),
        reason: '.${channel.name}',
      );
    }
  });

  test('alx requires a declared total point count', () {
    expect(
      () => readAlf(
        _envelope(1, <String, Object?>{
          '_source_total_points': 1,
        }, List<int>.filled(4, 0)),
        channel: AlfChannel.alx,
      ),
      throwsA(_refusal('total-points-missing')),
    );
  });

  test('v3 error bound includes far-offset reconstruction rounding', () {
    const base = -4.488e11;
    const source = <double>[base, 0, 0, base + 0.004, 0, 0, base + 0.008, 0, 0];
    final written = writeAlp(
      source,
      codec: AlfGeometryCodec.int16,
      frameConvention: alfFrameWorldAligned,
      gzip: false,
      checkBudget: false,
    );
    for (final recorded in <Object?>[null, 1e-12]) {
      final rewritten = _rewriteMetadata(written, (
        Map<String, Object?> header,
      ) {
        if (recorded == null) {
          header.remove('_geom_max_err_m');
        } else {
          header['_geom_max_err_m'] = recorded;
        }
      });
      final file = readAlf(rewritten, channel: AlfChannel.alp);
      final scale = (file.metadata['_geom_scale']! as num).toDouble();
      var achieved = 0.0;
      for (var i = 0; i < source.length; i++) {
        achieved = achieved > (file.xyz[i] - source[i]).abs()
            ? achieved
            : (file.xyz[i] - source[i]).abs();
      }
      expect(achieved, greaterThan(scale / 2));
      expect(file.maxGeometryErrorM, greaterThanOrEqualTo(achieved));
    }
  });

  test('read detects gzip by magic instead of a filename', () {
    final compressed = writeAll(<int>[1, 2]);
    final uncompressed = writeAll(<int>[1, 2], gzip: false);
    expect(
      readAlf(compressed, channel: AlfChannel.all).categories,
      orderedEquals(<int>[1, 2]),
    );
    expect(
      readAlf(uncompressed, channel: AlfChannel.all).categories,
      orderedEquals(<int>[1, 2]),
    );
  });

  test('read refuses a gzip member whose CRC was tampered with', () {
    final bytes = writeAll(<int>[1, 2]);
    // Trailer is CRC32 then ISIZE. Flip one CRC byte without changing the
    // compressed ALF stream, so successful inflate cannot mask corruption.
    bytes[bytes.length - 8] ^= 0xff;
    expect(
      () => readAlf(bytes, channel: AlfChannel.all),
      throwsA(
        isA<AlfException>().having(
          (AlfException error) => error.refusal,
          'refusal',
          'gzip-malformed',
        ),
      ),
    );
  });

  test('publishable library has no VM, Flutter, or consumer imports', () {
    final packageRoot = Directory.current;
    final forbidden = RegExp(
      r"(?:dart:io|package:flutter/|package:http/|package:three_js/)",
    );
    final violations = <String>[];
    for (final entity in Directory(
      '${packageRoot.path}/lib',
    ).listSync(recursive: true)) {
      if (entity is! File || !entity.path.endsWith('.dart')) continue;
      if (forbidden.hasMatch(entity.readAsStringSync())) {
        violations.add(entity.path);
      }
    }
    expect(
      violations,
      isEmpty,
      reason: 'The codec must compile on Flutter web JS/Wasm',
    );
  });

  test('writer rejects metadata that JSON cannot represent', () {
    expect(
      () => writeAll(<int>[1], metadata: <String, Object?>{'bad': double.nan}),
      throwsA(isA<AlfException>()),
    );
  });

  test('unknown versions retain their stable refusal id', () {
    final bytes = Uint8List.fromList(<int>[
      9,
      2,
      0,
      0,
      0,
      ...utf8.encode('{}'),
    ]);
    expect(
      () => readAlf(bytes, channel: AlfChannel.alp),
      throwsA(
        isA<AlfException>().having(
          (AlfException error) => error.refusal,
          'refusal',
          'version-unsupported',
        ),
      ),
    );
  });
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
  final header = utf8.encode(jsonEncode(metadata));
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

Uint8List _rewriteMetadata(
  Uint8List source,
  void Function(Map<String, Object?> metadata) mutate,
) {
  final headerLength = ByteData.sublistView(
    source,
    1,
    5,
  ).getUint32(0, Endian.little);
  final decoded = jsonDecode(utf8.decode(source.sublist(5, 5 + headerLength)));
  final metadata = Map<String, Object?>.from(
    (decoded! as Map<Object?, Object?>).cast<String, Object?>(),
  );
  mutate(metadata);
  return _envelope(source[0], metadata, source.sublist(5 + headerLength));
}
