import 'dart:convert';
import 'dart:io';

import 'package:alf/alf.dart';

/// Parse coordinates from the request's canonical spelling.
///
/// JSON has no NaN, so a non-finite component travels as a string. A candidate
/// writer that ignores this and coerces the string to a number writes garbage
/// rather than an invalid return, which is what this gate exists to catch.
List<num> _components(List<Object?> raw) {
  return <num>[
    for (final Object? value in raw)
      if (value is String)
        switch (value) {
          'NaN' => double.nan,
          'Infinity' => double.infinity,
          '-Infinity' => double.negativeInfinity,
          _ => throw ArgumentError(
            'Unknown non-finite coordinate spelling $value',
          ),
        }
      else
        value! as num,
  ];
}

void main(List<String> arguments) {
  if (arguments.length != 3) {
    stderr.writeln(
      'usage: encode_roundtrip.dart <request.json> <channel> <output>',
    );
    exitCode = 64;
    return;
  }
  final input = arguments[0];
  final channel = AlfChannel.fromExtension(arguments[1]);
  final output = arguments[2];
  final request =
      jsonDecode(File(input).readAsStringSync()) as Map<String, Object?>;
  if (request['channel'] != arguments[1]) {
    throw const FormatException(
      'Request channel disagrees with command channel',
    );
  }
  final metadata = Map<String, Object?>.from(request['metadata']! as Map);
  final gzip = output.endsWith('.gz');
  final bytes = switch (channel) {
    AlfChannel.alp => writeAlp(
      _components(request['points']! as List<Object?>),
      metadata: metadata,
      codec: request['codec'] == 'int16'
          ? AlfGeometryCodec.int16
          : AlfGeometryCodec.fp16,
      gzip: gzip,
      checkBudget: false,
      // A writer INPUT: the coordinates of a world-aligned and a
      // device-aligned frame are identical, so an encoder cannot derive it.
      frameConvention: request['frame_convention'] as String?,
      frameRotation: (request['frame_rotation'] as Map<String, Object?>?)?.map(
        (String axis, Object? value) =>
            MapEntry<String, num>(axis, value! as num),
      ),
    ),
    AlfChannel.ali => writeAli(
      List<num>.from(request['intensity']! as List),
      lo: 0,
      hi: 255,
      metadata: metadata,
      gzip: gzip,
    ),
    AlfChannel.all => writeAll(
      List<num>.from(request['categories']! as List),
      instanceIds: request['instance_ids'] == null
          ? null
          : List<num>.from(request['instance_ids']! as List),
      instanceMap: request['instance_map'] == null
          ? null
          : Map<String, String>.from(request['instance_map']! as Map),
      metadata: metadata,
      gzip: gzip,
    ),
    AlfChannel.alc => writeAlc(
      List<num>.from(request['palette_indices']! as List),
      List<Object>.from(request['palette']! as List),
      metadata: metadata,
      gzip: gzip,
    ),
    AlfChannel.alx => writeAlx(
      List<num>.from(request['kept_indices']! as List),
      alpTotalPoints: (request['alp_total_points']! as num).toInt(),
      sourceTotalPoints: request['source_total_points']! as num,
      metadata: metadata,
      gzip: gzip,
    ),
  };
  File(output).writeAsBytesSync(bytes, flush: true);
}
