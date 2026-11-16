import 'dart:convert';
import 'dart:typed_data';

import 'package:archive/archive.dart' show getCrc32;
import 'package:alf/alf.dart';
import 'package:alf/src/gzip.dart';

import 'v3_runtime_checks.dart';

void main() {
  runV3RuntimeChecks();

  // On JavaScript, infinity also satisfies `is int`. Handle invalid returns
  // before integer-narrowing checks so the writer agrees with the Dart VM.
  for (final codec in AlfGeometryCodec.values) {
    final missing = readAlf(
      writeAlp(
        <num>[
          1.37,
          -2.11,
          0.43,
          double.infinity,
          double.negativeInfinity,
          double.nan,
          4.13,
          -5.27,
          6.39,
        ],
        codec: codec,
        frameConvention: codec == AlfGeometryCodec.int16
            ? alfFrameWorldAligned
            : null,
        gzip: false,
      ),
      channel: AlfChannel.alp,
    );
    final xyz = missing.xyz;
    if (missing.totalPoints != 3 || xyz.length != 9) {
      throw StateError('Missing returns changed point indices');
    }
    if (codec == AlfGeometryCodec.fp16) {
      if (xyz[3] != double.infinity ||
          xyz[4] != double.negativeInfinity ||
          !xyz[5].isNaN) {
        throw StateError('Binary16 non-finite components were changed');
      }
    } else if (!xyz.sublist(3, 6).every((value) => value.isNaN)) {
      throw StateError('Fixed-point missing return was not preserved');
    }
    if (!missing.maxGeometryErrorM!.isFinite ||
        (xyz.first - 1.37).abs() > missing.maxGeometryErrorM! ||
        (xyz.last - 6.39).abs() > missing.maxGeometryErrorM!) {
      throw StateError('Missing returns corrupted the finite geometry bound');
    }
  }

  final exactWideAnchor = double.parse('18446744073709551616');
  for (final codec in AlfGeometryCodec.values) {
    final wideAnchor = readAlf(
      writeAlp(
        const <num>[0, 0, 0],
        metadata: <String, Object?>{
          'device_position': <String, Object?>{
            'x': exactWideAnchor,
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
      ),
      channel: AlfChannel.alp,
    );
    final anchor =
        wideAnchor.metadata['device_position']! as Map<String, Object?>;
    if (anchor['x'] != exactWideAnchor || anchor['note'] != 'gps') {
      throw StateError('Wide anchor changed in ${codec.name} writer');
    }
  }

  // Valid counts can exceed the VM/Wasm native integer range. Keep this exact
  // binary64 value through both the writer and the public reader accessor.
  for (final count in <double>[
    double.parse('18446744073709551616'),
    double.maxFinite,
  ]) {
    final bytes = writeAlx(
      const <int>[0],
      alpTotalPoints: 1,
      sourceTotalPoints: count,
      gzip: false,
    );
    if (readAlf(bytes, channel: AlfChannel.alx).sourceTotalPoints != count) {
      throw StateError('Wide ALX source count changed during serialization');
    }
    final header = _envelope(
      1,
      '{"_total_points":${BigInt.from(count)}}',
      Uint8List(0),
    );
    if (readAlfHeader(header, channel: AlfChannel.ali).totalPoints != count) {
      throw StateError('Wide count changed during a header scan');
    }
  }

  final wrappedCount = _envelope(
    1,
    '{"_total_points":4611686018427387904,"_source_total_points":1}',
    Uint8List(0),
  );
  try {
    readAlf(wrappedCount, channel: AlfChannel.alx);
    throw StateError('A large declared count wrapped to an empty payload');
  } on AlfException catch (error) {
    if (error.refusal != 'payload-length-mismatch') rethrow;
  }

  // JavaScript's JSON parser rounds these values before Dart sees them. The
  // reader must validate the file's raw spelling, not only the decoded number.
  for (final json in <String>[
    '{"_total_points":1,"_geom_codec":"int16","_frame_convention":"world-aligned","_geom_scale":1,'
        '"_geom_offset":[9007199254740993,0,0]}',
    '{"_total_points":1,"_geom_codec":"int16","_frame_convention":"world-aligned","_geom_scale":1,'
        '"_geom_offset":[9007199254740993.5,0,0]}',
    '{"_total_points":1,"_geom_codec":"int16","_frame_convention":"world-aligned",'
        '"_geom_scale":9007199254740993.0,"_geom_offset":[0,0,0]}',
  ]) {
    try {
      readAlf(_v3Envelope(json), channel: AlfChannel.alp);
    } on AlfException catch (error) {
      if (error.refusal == 'geom-number-inexact') continue;
      throw StateError('Unexpected inexact geometry refusal: ${error.refusal}');
    }
    throw StateError('Inexact geometry number was accepted: $json');
  }

  for (final exact in <String>[
    '1e00000000001',
    '9007199254740991.0',
    '9007199254740992',
    '9007199254740994',
  ]) {
    final file = readAlf(
      _v3Envelope(
        '{"_total_points":1,"_geom_codec":"int16","_frame_convention":"world-aligned",'
        '"_geom_scale":0.1,"_geom_offset":[$exact,0,0]}',
      ),
      channel: AlfChannel.alp,
    );
    if (file.xyz.first != double.parse(exact)) {
      throw StateError('Exact geometry number changed: $exact');
    }
  }

  final cancelledWide =
      '0.${List<String>.filled(9980, '0').join()}'
      '18446744073709551616e10000';
  final widePayload = ByteData(6)..setInt16(0, 1, Endian.little);
  final wideGrid = readAlf(
    _v3Envelope(
      '{"_total_points":1,"_geom_codec":"int16","_frame_convention":"world-aligned",'
      '"_geom_scale":$cancelledWide,"_geom_offset":[0,0,0]}',
      payload: widePayload.buffer.asUint8List(),
    ),
    channel: AlfChannel.alp,
  );
  if (wideGrid.xyz.first != double.parse('18446744073709551616')) {
    throw StateError('Cancellable wide geometry exponent changed');
  }

  // Exercise adjacent-float error bounds under compiled JavaScript. Uint64
  // bit arithmetic cannot preserve the low bits in that runtime.
  final payload = ByteData(18);
  for (final (index, value) in <int>[0, 0, 0, 40, 0, 0, 80, 0, 0].indexed) {
    payload.setInt16(index * 2, value, Endian.little);
  }
  final bounded = readAlf(
    _v3Envelope(
      '{"_total_points":3,"_geom_codec":"int16","_frame_convention":"world-aligned",'
      '"_geom_scale":0.0001,"_geom_offset":[-448800000000,0,0]}',
      payload: payload.buffer.asUint8List(),
    ),
    channel: AlfChannel.alp,
  );
  if (bounded.maxGeometryErrorM! <= 0.00005) {
    throw StateError('JavaScript geometry bound omitted float64 rounding');
  }

  final frozen = readAlf(
    _v3Envelope(
      '{"_total_points":1,"_geom_codec":"int16","_frame_convention":"world-aligned",'
      '"_geom_scale":1,"_geom_offset":[10,20,30]}',
    ),
    channel: AlfChannel.alp,
  );
  try {
    (frozen.metadata['_geom_offset']! as List<Object?>)[0] = 99;
  } on UnsupportedError {
    // Expected: nested parsed metadata stays immutable on web runtimes too.
  }
  if ((frozen.metadata['_geom_offset']! as List<Object?>).first != 10) {
    throw StateError('Nested parsed metadata was mutable');
  }

  // The header-only read, on both web targets. Its stop condition is an
  // exception thrown out of the inflater's output sink, and "a throw unwinds
  // the way it does on the VM" is exactly the assumption compiled JavaScript
  // and Wasm are entitled to break.
  final scanned = writeAlp(
    <num>[for (var i = 0; i < 39999; i++) (i % 733) * 0.25 - 91.0],
    metadata: <String, Object?>{'seq': 's1'},
    checkBudget: false,
  );
  final head = readAlfHeader(scanned, channel: AlfChannel.alp);
  if (head.metadata['seq'] != 's1' || head.totalPoints != 39999 ~/ 3) {
    throw StateError('Header-only read disagreed with the header it parsed');
  }
  if (head.version != readAlf(scanned, channel: AlfChannel.alp).version) {
    throw StateError('Header-only read disagreed with the full read');
  }
  final probed = decodeGzipPrefix(scanned, atLeast: 1024);
  if (probed.length >= 39999 * 2) {
    throw StateError('Prefix inflate ran past its target on this runtime');
  }

  final gzipSource = Uint8List.fromList(<int>[1, 2, 3]);
  final validHeaderCrc = _withHeaderCrc(encodeDeterministicGzip(gzipSource));
  if (!_sameBytes(decodeSingleGzipMember(validHeaderCrc), gzipSource)) {
    throw StateError('Valid gzip header CRC was not preserved');
  }
  final corruptHeaderCrc = Uint8List.fromList(validHeaderCrc)..[10] ^= 0x01;
  try {
    decodeSingleGzipMember(corruptHeaderCrc);
  } on AlfException catch (error) {
    if (error.refusal == 'gzip-malformed') {
      print('ALF platform checks passed');
      return;
    }
  }
  throw StateError('Corrupt gzip header CRC was accepted');
}

bool _sameBytes(Uint8List left, Uint8List right) {
  if (left.length != right.length) return false;
  for (var i = 0; i < left.length; i++) {
    if (left[i] != right[i]) return false;
  }
  return true;
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

Uint8List _v3Envelope(String json, {Uint8List? payload}) {
  return _envelope(3, json, payload ?? Uint8List(6));
}

Uint8List _envelope(int version, String json, Uint8List payload) {
  final header = utf8.encode(json);
  return (BytesBuilder(copy: false)
        ..addByte(version)
        ..add(
          (ByteData(
            4,
          )..setUint32(0, header.length, Endian.little)).buffer.asUint8List(),
        )
        ..add(header)
        ..add(payload))
      .takeBytes();
}
