import 'dart:convert';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:alf/alf.dart';
import 'package:alf/src/gzip.dart';

/// The same checks run as native Dart, compiled JavaScript and compiled Wasm.
/// Throw explicitly: compiler settings may disable Dart `assert` statements.
void runV3RuntimeChecks() {
  for (final gzip in <bool>[false, true]) {
    for (final convention in <String>[
      alfFrameWorldAligned,
      alfFrameDeviceAligned,
    ]) {
      _roundTrip(gzip, convention);
    }
    _edgeCases(gzip);
    _refusals(gzip);
  }
}

void _roundTrip(bool gzip, String convention) {
  final source = <double>[
    -400.123456, 29.876543, -0.012345, //
    double.nan, double.infinity, double.negativeInfinity, //
    400.654321, -31.234567, 8.765432, //
    0.234567, 1.345678, -2.456789, //
    double.nan, double.nan, double.nan,
  ];
  const anchor = <String, Object?>{
    'x': 1200.125,
    'y': -3400.5,
    'z': 75.25,
    'note': 'synthetic',
  };
  final bytes = writeAlp(
    source,
    metadata: <String, Object?>{
      'device_position': anchor,
      'name': 'synthetic \u03bb frame',
    },
    codec: AlfGeometryCodec.int16,
    frameConvention: convention,
    // Normalized, this quaternion cycles (x, y, z) to (z, x, y). A non-unit
    // input exercises the writer's documented near-unit normalization too.
    frameRotation: convention == alfFrameDeviceAligned
        ? <String, num>{'w': 0.525, 'x': 0.525, 'y': 0.525, 'z': 0.525}
        : null,
    gzip: gzip,
  );
  final file = readAlf(bytes, channel: AlfChannel.alp);
  final head = readAlfHeader(bytes, channel: AlfChannel.alp);
  _check(file.version == 3 && head.version == 3, 'v3 envelope');
  _check(file.geometryCodec == AlfGeometryCodec.int16, 'v3 codec');
  _check(file.frameConvention == convention, 'frame convention');
  _check(head.totalPoints == 5 && file.totalPoints == 5, 'point count');
  _check(file.xyz.length == source.length, 'geometry index preservation');
  _check(head.metadata['name'] == 'synthetic \u03bb frame', 'UTF-8 header');
  final storedAnchor =
      file.metadata['device_position']! as Map<String, Object?>;
  _check(
    anchor.entries.every((entry) => storedAnchor[entry.key] == entry.value),
    'anchor metadata preservation',
  );
  if (convention == alfFrameDeviceAligned) {
    _check(
      file.frameRotation!.values.every((value) => value == 0.5),
      'quaternion normalization',
    );
  } else {
    _check(file.frameRotation == null, 'world-aligned rotation');
  }

  final legacy = readAlf(writeAlp(source, gzip: gzip), channel: AlfChannel.alp);
  _check(
    legacy.version == 1 && legacy.geometryCodec == AlfGeometryCodec.fp16,
    'default remains legacy fp16',
  );
  final error = _maxError(source, file.xyz);
  final legacyError = _maxError(source, legacy.xyz);
  final bound = file.maxGeometryErrorM!;
  _check(bound.isFinite && error <= bound, 'measured error is bounded');
  _check(error > 1e-8, 'fixture exercises actual quantization');
  _check(error * 10 < legacyError, 'v3 improves nontrivial fp16 precision');
  _check(bound < 0.01, 'v3 sub-centimetre bound');

  const categories = <int>[7, 254, 19, 0, 23];
  const instances = <int>[257, 0, 513, 65534, 0];
  const instanceMap = <String, String>{
    '257': 'synthetic-a',
    '513': 'synthetic-b',
    '65534': 'synthetic-c',
  };
  const intensities = <int>[0, 42, 128, 254, 255];
  const colors = <int>[0, 1, 2, 1, 0];
  const kept = <int>[2, 7, 19, 41, 95];
  final labels = readAlf(
    writeAll(
      categories,
      instanceIds: instances,
      instanceMap: instanceMap,
      gzip: gzip,
    ),
    channel: AlfChannel.all,
  );
  final intensity = readAlf(
    writeAli(intensities, lo: 0, hi: 255, gzip: gzip),
    channel: AlfChannel.ali,
  );
  final color = readAlf(
    writeAlc(colors, <String>['336699', 'f0a020'], gzip: gzip),
    channel: AlfChannel.alc,
  );
  final index = readAlf(
    writeAlx(kept, alpTotalPoints: 5, sourceTotalPoints: 128, gzip: gzip),
    channel: AlfChannel.alx,
  );
  _check(labels.version == 2 && labels.formatVersion == 2, 'panoptic layout');
  _check(
    instanceMap.entries.every(
      (entry) => labels.instanceMap![entry.key] == entry.value,
    ),
    'instance identity',
  );
  _check(index.sourceTotalPoints == 128, 'source cloud size');
  for (final sidecar in <AlfFile>[labels, intensity, color, index]) {
    _check(
      sidecar.totalPoints == file.totalPoints,
      'channel count correlation',
    );
  }
  for (var i = 0; i < 5; i++) {
    _check(labels.categories[i] == categories[i], 'category index $i');
    _check(labels.instanceIds![i] == instances[i], 'instance index $i');
    _check(intensity.intensity[i] == intensities[i], 'intensity index $i');
    _check(color.paletteIndices[i] == colors[i], 'colour index $i');
    _check(index.keptIndices[i] == kept[i], 'source index $i');
    final local = file.xyz.sublist(i * 3, i * 3 + 3);
    if (i == 1 || i == 4) {
      _check(local.every((value) => value.isNaN), 'invalid return index $i');
      continue;
    }
    final world = _toWorld(local, storedAnchor, file.frameRotation);
    final original = source.sublist(i * 3, i * 3 + 3);
    final expectedAxes = convention == alfFrameDeviceAligned
        ? <double>[original[2], original[0], original[1]]
        : original;
    for (var axis = 0; axis < 3; axis++) {
      final origin = (anchor[<String>['x', 'y', 'z'][axis]]! as num).toDouble();
      _check(
        (world[axis] - (expectedAxes[axis] + origin)).abs() <= bound + 1e-10,
        'world reconstruction point $i axis $axis',
      );
    }
  }
}

// Reconstruction uses only the SDK's public local coordinates and metadata.
// The known cyclic rotation above provides an independent expected world point.
List<double> _toWorld(
  List<double> local,
  Map<String, Object?> anchor,
  Map<String, double>? rotation,
) {
  final result = List<double>.of(local);
  if (rotation != null) {
    final w = rotation['w']!;
    final x = rotation['x']!;
    final y = rotation['y']!;
    final z = rotation['z']!;
    final tx = 2 * (y * local[2] - z * local[1]);
    final ty = 2 * (z * local[0] - x * local[2]);
    final tz = 2 * (x * local[1] - y * local[0]);
    result[0] += w * tx + y * tz - z * ty;
    result[1] += w * ty + z * tx - x * tz;
    result[2] += w * tz + x * ty - y * tx;
  }
  for (var axis = 0; axis < 3; axis++) {
    result[axis] += (anchor[<String>['x', 'y', 'z'][axis]]! as num).toDouble();
  }
  return result;
}

void _edgeCases(bool gzip) {
  for (final source in <List<double>>[
    <double>[],
    <double>[double.nan, double.nan, double.nan],
  ]) {
    final file = readAlf(
      writeAlp(
        source,
        codec: AlfGeometryCodec.int16,
        frameConvention: alfFrameWorldAligned,
        gzip: gzip,
      ),
      channel: AlfChannel.alp,
    );
    _check(file.xyz.length == source.length, 'degenerate frame point count');
    _check(file.xyz.every((value) => value.isNaN), 'all-invalid frame');
    _check(file.maxGeometryErrorM!.isFinite, 'degenerate frame error bound');
    if (source.isEmpty) {
      _check(file.maxGeometryErrorM == 0, 'empty frame bound');
    }
  }

  // A span whose half-step exceeds the default 0.25 m budget must be opt-in.
  const wide = <double>[-20000.123456, 0, 0, 20000.654321, 0, 0];
  _expectRefusal('geometry-budget-exceeded', () {
    writeAlp(
      wide,
      codec: AlfGeometryCodec.int16,
      frameConvention: alfFrameWorldAligned,
      gzip: gzip,
    );
  });
  final unbudgeted = readAlf(
    writeAlp(
      wide,
      codec: AlfGeometryCodec.int16,
      frameConvention: alfFrameWorldAligned,
      gzip: gzip,
      checkBudget: false,
    ),
    channel: AlfChannel.alp,
  );
  _check(
    _maxError(wide, unbudgeted.xyz) <= unbudgeted.maxGeometryErrorM!,
    'explicit budget override retains error bound',
  );
}

void _refusals(bool gzip) {
  const header = <String, Object?>{
    '_total_points': 1,
    '_geom_codec': 'int16',
    '_geom_scale': 0.001,
    '_geom_offset': <int>[0, 0, 0],
    '_frame_convention': alfFrameWorldAligned,
  };
  // Existing format refusals must survive all three compilation targets.
  for (final (key, value, refusal) in <(String, Object?, String)>[
    ('_geom_codec', 'synthetic-unknown', 'codec-unsupported'),
    ('_geom_codec', null, 'codec-unsupported'),
    ('_geom_codec', 'fp16', 'codec-version-disagreement'),
    ('_geom_scale', 0, 'geom-scale-not-positive'),
    ('_geom_scale', '0.001', 'geom-scale-not-positive'),
    ('_geom_offset', <int>[0, 0], 'geom-offset-malformed'),
    ('_geom_offset', <Object>[0, '0', 0], 'geom-offset-malformed'),
    ('_frame_convention', 'synthetic-unknown', 'frame-convention-unsupported'),
    ('_frame_convention', alfFrameDeviceAligned, 'frame-rotation-missing'),
    ('_total_points', 2, 'payload-length-mismatch'),
  ]) {
    _expectRefusal(refusal, () {
      readAlf(
        _envelope(<String, Object?>{...header, key: value}, gzip: gzip),
        channel: AlfChannel.alp,
      );
    });
  }
  for (final (version, channel, refusal) in <(int, AlfChannel, String)>[
    (4, AlfChannel.alp, 'version-unsupported'),
    (3, AlfChannel.all, 'geometry-version-on-non-geometry-channel'),
  ]) {
    _expectRefusal(refusal, () {
      readAlf(
        _envelope(header, version: version, gzip: gzip),
        channel: channel,
      );
    });
  }
  for (final (rotation, refusal) in <(Object, String)>[
    (<int>[1, 0, 0, 0], 'frame-rotation-malformed'),
    (<String, int>{'w': 1, 'x': 0, 'y': 0}, 'frame-rotation-malformed'),
    (
      <String, int>{'w': 2, 'x': 0, 'y': 0, 'z': 0},
      'frame-rotation-not-normalized',
    ),
  ]) {
    _expectRefusal(refusal, () {
      readAlf(
        _envelope(<String, Object?>{
          ...header,
          '_frame_convention': alfFrameDeviceAligned,
          '_frame_rotation': rotation,
        }, gzip: gzip),
        channel: AlfChannel.alp,
      );
    });
  }
  final partial = ByteData(6)..setInt16(0, -32768, Endian.little);
  for (final (payload, refusal) in <(Uint8List, String)>[
    (Uint8List(5), 'payload-not-whole-points'),
    (Uint8List(12), 'payload-length-mismatch'),
    (partial.buffer.asUint8List(), 'geom-partial-no-return'),
  ]) {
    _expectRefusal(refusal, () {
      readAlf(
        _envelope(header, payload: payload, gzip: gzip),
        channel: AlfChannel.alp,
      );
    });
  }
  _expectRefusal('writer-input-invalid', () {
    writeAlp(<double>[0, 0, 0], codec: AlfGeometryCodec.int16, gzip: gzip);
  });
  _expectRefusal('writer-input-invalid', () {
    writeAlp(
      <double>[0, double.nan, 0],
      codec: AlfGeometryCodec.int16,
      frameConvention: alfFrameWorldAligned,
      gzip: gzip,
    );
  });
}

Uint8List _envelope(
  Map<String, Object?> metadata, {
  required bool gzip,
  int version = 3,
  Uint8List? payload,
}) {
  final header = utf8.encode(jsonEncode(metadata));
  final raw =
      (BytesBuilder(copy: false)
            ..addByte(version)
            ..add(
              (ByteData(4)..setUint32(0, header.length, Endian.little)).buffer
                  .asUint8List(),
            )
            ..add(header)
            ..add(payload ?? Uint8List(6)))
          .takeBytes();
  return gzip ? encodeDeterministicGzip(raw) : raw;
}

double _maxError(List<double> source, List<double> decoded) {
  _check(source.length == decoded.length, 'geometry length before error check');
  var error = 0.0;
  for (var i = 0; i < source.length; i++) {
    if (!source[i].isFinite) continue;
    _check(decoded[i].isFinite, 'finite geometry point $i');
    error = math.max(error, (decoded[i] - source[i]).abs());
  }
  return error;
}

void _expectRefusal(String refusal, void Function() action) {
  try {
    action();
  } on AlfException catch (error) {
    _check(error.refusal == refusal, '$refusal instead of ${error.refusal}');
    return;
  }
  throw StateError('Accepted malformed ALF: expected $refusal');
}

void _check(bool condition, String message) {
  if (!condition) throw StateError('ALF v3 runtime check failed: $message');
}
