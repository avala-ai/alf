import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:alf/alf.dart';

void main(List<String> arguments) {
  if (arguments.isEmpty) {
    stderr.writeln('usage: conformance.dart <path> [extension]');
    exitCode = 64;
    return;
  }
  final path = arguments[0];
  final extension = arguments.length > 1 ? arguments[1] : path;
  try {
    final file = readAlf(
      Uint8List.fromList(File(path).readAsBytesSync()),
      channel: AlfChannel.fromExtension(extension),
    );
    stdout.writeln(jsonEncode(_canonical(file)));
  } on AlfException catch (error) {
    stdout.writeln(jsonEncode(<String, Object?>{'refused': error.refusal}));
  }
}

/// One coordinate, in the spelling every language must produce.
///
/// JSON has no NaN: `jsonEncode(double.nan)` throws here, and JavaScript's
/// `JSON.stringify` silently writes `null` — so three conforming readers would
/// disagree on the text while agreeing on the bytes. Non-finite components are
/// valid `.alp` content (spec/format.md §5.1), and the three spellings stay
/// distinct because a reader that turned +Infinity into NaN would have changed
/// the file's meaning.
Object _component(double value) {
  if (value.isNaN) return 'NaN';
  if (value.isInfinite) return value > 0 ? 'Infinity' : '-Infinity';
  return _round(value, 6);
}

Map<String, Object?> _canonical(AlfFile file) {
  final answer = <String, Object?>{
    'version': file.version,
    'extension': file.channel.name,
  };
  switch (file.channel) {
    case AlfChannel.alp:
      answer['codec'] = file.geometryCodec.name;
      answer['points'] = <List<Object>>[
        for (var i = 0; i < file.xyz.length; i += 3)
          <Object>[
            _component(file.xyz[i]),
            _component(file.xyz[i + 1]),
            _component(file.xyz[i + 2]),
          ],
      ];
      final bound = file.maxGeometryErrorM;
      answer['max_geom_error_m'] = bound == null ? null : _round(bound, 12);
      // Surfaced, not merely validated: the bug this field exists for was a
      // field that was present and ignored.
      answer['frame_convention'] = file.frameConvention;
      final rotation = file.frameRotation;
      if (rotation != null) {
        answer['frame_rotation'] = <String, Object?>{
          for (final axis in const <String>['w', 'x', 'y', 'z'])
            axis: _component(rotation[axis]!),
        };
      }
      final anchors = <String, Object?>{
        for (final key in <String>[
          'device_position',
          'world_centroid_epsg3857',
        ])
          if (file.metadata.containsKey(key)) key: file.metadata[key],
      };
      if (anchors.isNotEmpty) answer['anchors'] = anchors;
    case AlfChannel.ali:
      answer['intensity'] = file.intensity.toList();
    case AlfChannel.all:
      answer['format_version'] = file.formatVersion;
      answer['categories'] = file.categories.toList();
      answer['instance_ids'] = file.instanceIds?.toList();
      answer['instance_map'] = file.instanceMap;
    case AlfChannel.alc:
      answer['palette'] = file.palette;
      answer['palette_indices'] = file.paletteIndices.toList();
    case AlfChannel.alx:
      answer['kept_indices'] = file.keptIndices.toList();
      answer['source_total_points'] = file.sourceTotalPoints;
  }
  return answer;
}

double _round(double value, int places) {
  final factor = math.pow(10, places).toDouble();
  // At this magnitude binary64 has no fractional scaled unit left to round.
  // Returning the original also avoids turning a finite 1e308 into infinity.
  if (!factor.isFinite || value.abs() >= 9007199254740992 / factor) {
    return value;
  }

  // Scale the exact binary64 rational, not a rounded binary64 multiplication.
  // Multiplying 0.0000025 by 1e6 rounds the intermediate to exactly 2.5 and
  // loses the fact that the original binary64 is just above the decimal tie.
  final bytes = ByteData(8)..setFloat64(0, value, Endian.big);
  final high = bytes.getUint32(0, Endian.big);
  final low = bytes.getUint32(4, Endian.big);
  final negative = high & 0x80000000 != 0;
  final exponentBits = (high >>> 20) & 0x7ff;
  final fraction = (BigInt.from(high & 0x000fffff) << 32) | BigInt.from(low);
  final significand = exponentBits == 0
      ? fraction
      : (BigInt.one << 52) | fraction;
  final exponent = exponentBits == 0 ? -1074 : exponentBits - 1023 - 52;
  var numerator = significand * BigInt.from(10).pow(places);
  var denominator = BigInt.one;
  if (exponent >= 0) {
    numerator <<= exponent;
  } else {
    denominator <<= -exponent;
  }
  var rounded = numerator ~/ denominator;
  final remainderTwice = (numerator.remainder(denominator)) * BigInt.two;
  if (remainderTwice > denominator ||
      (remainderTwice == denominator && rounded.isOdd)) {
    rounded += BigInt.one;
  }
  final result = rounded.toDouble() / factor;
  return negative ? -result : result;
}
