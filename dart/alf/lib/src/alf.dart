import 'dart:convert';
import 'dart:math' as math;
import 'dart:typed_data';

import 'error.dart';
import 'gzip.dart';
import 'half.dart';
import 'limits.dart';

const _maxJsonDepth = 256;
const _minGeometryScaleM = 0.0001;
const _int16Max = 32767;

/// The reserved word for "this beam produced no return", in v3 fixed point.
///
/// Free by construction: the scale is derived as `extent / _int16Max`, so a real
/// coordinate quantises into [-32767, 32767] and -32768 is unreachable. See
/// spec/format.md §5.2.
const _int16NoReturn = -32768;

/// True when all three components of the point starting at [base] are finite.
///
/// A partial sentinel is not a shape any writer may emit, so "invalid return"
/// is a whole-point property (spec/format.md §5.2).
bool _alfPointIsFinite(List<num> xyz, int base) =>
    xyz[base].toDouble().isFinite &&
    xyz[base + 1].toDouble().isFinite &&
    xyz[base + 2].toDouble().isFinite;

/// Whether ALL three components at [base] are non-finite.
///
/// Distinct from `!_alfPointIsFinite`, which is true for a PARTLY invalid point
/// too. The int16 writer needs the difference: a whole invalid point encodes as
/// the sentinel, a partial one is refused.
bool _alfPointIsWhollyNonFinite(List<num> xyz, int base) =>
    !xyz[base].toDouble().isFinite &&
    !xyz[base + 1].toDouble().isFinite &&
    !xyz[base + 2].toDouble().isFinite;

const _fp16SubnormalHalfUlp = 2.9802322387695312e-8; // 2^-25

/// The channel selected by an ALF filename's load-bearing suffix.
enum AlfChannel {
  alp,
  ali,
  all,
  alc,
  alx;

  /// Parse `alp`, `.alp`, `frame.alp`, or `frame.alp.gz`.
  static AlfChannel fromExtension(String value) {
    var normalized = value.toLowerCase();
    if (normalized.endsWith('.gz')) {
      normalized = normalized.substring(0, normalized.length - 3);
    }
    final dot = normalized.lastIndexOf('.');
    if (dot >= 0) normalized = normalized.substring(dot + 1);
    return switch (normalized) {
      'alp' => alp,
      'ali' => ali,
      'all' => all,
      'alc' => alc,
      'alx' => alx,
      _ => throw AlfException(
        'channel-unsupported',
        'Unknown ALF channel $value',
      ),
    };
  }
}

/// The two geometry encodings. Both occupy six payload bytes per point.
enum AlfGeometryCodec { fp16, int16 }

/// How a v3 `.alp`'s local points relate to world axes. See `spec/format.md` §5.3.
const String alfFrameWorldAligned = 'world-aligned';
const String alfFrameDeviceAligned = 'device-aligned';

// The tolerance is on the SQUARED norm because that is the factor a non-unit
// quaternion actually applies, and 1e-9 is far looser than the rounding of a
// genuinely normalized quaternion and far tighter than any error worth a metre.
const double _frameRotationNormTolerance = 1e-9;

/// One parsed ALF channel.
final class AlfFile {
  /// Wire version byte.
  final int version;

  /// Payload channel supplied by the filename or its caller.
  final AlfChannel channel;

  /// Parsed JSON header.
  final Map<String, Object?> metadata;

  /// Owned, decompressed payload bytes after the ALF envelope.
  final Uint8List _payload;
  final Uint8List _payloadView;

  Float64List? _xyz;
  Uint8List? _categories;
  Uint16List? _instanceIds;
  Uint32List? _keptIndices;
  Uint32List? _keptIndicesView;

  AlfFile._(this.version, this.channel, this.metadata, Uint8List payload)
    : _payload = payload,
      _payloadView = payload.asUnmodifiableView();

  /// Read-only raw, decompressed payload bytes.
  ///
  /// The parser owns the backing storage so cached and future decoded values
  /// cannot be invalidated through this view.
  Uint8List get payload => _payloadView;

  /// Exact header count, when present. Counts outside the native integer range
  /// remain integral doubles rather than being clamped by `toInt()`.
  num? get totalPoints => _checkedCount(metadata, '_total_points');

  /// How this `.alp`'s local points relate to world axes, or null.
  ///
  /// `'world-aligned'` — only the origin was subtracted. `'device-aligned'` —
  /// the points were also rotated into the device frame, and [frameRotation]
  /// carries the rotation that undoes it. Null for versions 1 and 2, which do
  /// not say and cannot. See `spec/format.md` §5.3.
  String? get frameConvention {
    _require(AlfChannel.alp, 'frameConvention');
    final declared = metadata['_frame_convention'];
    return declared is String ? declared : null;
  }

  /// The device rotation as `{w, x, y, z}`, present only on a device-aligned
  /// frame. Validated while the envelope was read.
  Map<String, double>? get frameRotation {
    _require(AlfChannel.alp, 'frameRotation');
    final rotation = metadata['_frame_rotation'];
    if (rotation is! Map<String, Object?>) return null;
    return <String, double>{
      for (final axis in const <String>['w', 'x', 'y', 'z'])
        axis: (rotation[axis]! as num).toDouble(),
    };
  }

  /// Original cloud size declared by a decimation `.alx` sidecar.
  num? get sourceTotalPoints {
    _require(AlfChannel.alx, 'sourceTotalPoints');
    return _checkedCount(metadata, '_source_total_points');
  }

  /// Geometry codec selected jointly by the version and header.
  AlfGeometryCodec get geometryCodec {
    _require(AlfChannel.alp, 'geometryCodec');
    final raw = metadata.containsKey('_geom_codec')
        ? metadata['_geom_codec']
        : 'fp16';
    if (raw == 'fp16') return AlfGeometryCodec.fp16;
    if (raw == 'int16') return AlfGeometryCodec.int16;
    throw AlfException('codec-unsupported', 'Unknown .alp geometry codec $raw');
  }

  /// `.all` layout: 1 for semantic categories or 2 for panoptic labels.
  int get formatVersion {
    _require(AlfChannel.all, 'formatVersion');
    final byteSays = version == 2 ? 2 : 1;
    final declared = _checkedLabelLayout(metadata);
    if (declared == null) return byteSays;
    if (declared == byteSays) return declared;

    // Neither version field wins by assertion. The point count and payload
    // stride are objective evidence, but only when exactly one layout fits.
    final count = totalPoints;
    if (count != null) {
      final fitsV1 = _payload.lengthInBytes == count;
      final fitsV2 = _payload.lengthInBytes == count.toDouble() * 3;
      if (fitsV1 != fitsV2) return fitsV2 ? 2 : 1;
      if (!fitsV1) {
        throw const AlfException(
          'payload-length-mismatch',
          '.all version fields disagree and payload matches neither label layout',
        );
      }
      // Empty payload/count fits both strides, so it proves neither assertion.
      throw const AlfException(
        'label-layout-ambiguous',
        '.all version fields disagree and both label layouts fit the payload',
      );
    }
    // Without a declared count v1 always fits. A non-v2 stride can still rule
    // v2 out; a multiple of three remains genuinely ambiguous.
    if (_payload.lengthInBytes % 3 != 0) return 1;
    throw const AlfException(
      'label-layout-ambiguous',
      '.all version fields disagree and payload cannot resolve the layout',
    );
  }

  /// Flattened local XYZ coordinates as float64: x, y, z, x, y, z, ...
  Float64List get xyz {
    _require(AlfChannel.alp, 'xyz');
    return _xyz ??= _decodeXyz();
  }

  /// Worst per-component geometry loss in metres.
  ///
  /// For v3 this respects the writer's measured `_geom_max_err_m` and never
  /// falls below the quantization half-step. For v1, whose legacy header has no
  /// error key, it derives the binary16 half-ULP bound from the decoded frame.
  double? get maxGeometryErrorM {
    _require(AlfChannel.alp, 'maxGeometryErrorM');
    final decoded = xyz;
    if (decoded.isEmpty) return 0;
    if (geometryCodec == AlfGeometryCodec.int16) {
      final scale = _geometryScale();
      final reach = _reconstructionRounding(scale, _geometryOffset());
      var analytic = scale / 2 + reach;
      if (scale < 1e-300) {
        analytic = math.max(analytic, _nextUp(scale / 2));
      }
      if (analytic <= 0) analytic = double.minPositive;
      final recorded = metadata['_geom_max_err_m'];
      return math.max(
        analytic,
        recorded is num && recorded.toDouble().isFinite
            ? recorded.toDouble()
            : 0,
      );
    }
    // FINITE components only. `max` over a frame carrying one invalid return is
    // NaN, and NaN compares false against every threshold a caller checks this
    // bound with — so the documented accuracy preflight silently stopped
    // answering on the frames it is most used for. A frame that is entirely
    // invalid returns has no magnitude, and falls through to the same floor the
    // all-zero frame gets.
    var largest = 0.0;
    for (final value in decoded) {
      if (value.isFinite) largest = math.max(largest, value.abs());
    }
    if (largest == 0) return _fp16SubnormalHalfUlp;
    final exponent = (math.log(largest) / math.ln2).floor();
    return math.max(
      math.pow(2, exponent - 11).toDouble(),
      _fp16SubnormalHalfUlp,
    );
  }

  /// Normalized uint8 intensity values from `.ali`.
  Uint8List get intensity {
    _require(AlfChannel.ali, 'intensity');
    return Uint8List.fromList(_payload);
  }

  /// Semantic category for each point from either `.all` layout.
  Uint8List get categories {
    _require(AlfChannel.all, 'categories');
    _decodeLabels();
    return Uint8List.fromList(_categories!);
  }

  /// Panoptic instance ids, or null for v1 semantic-only labels.
  Uint16List? get instanceIds {
    _require(AlfChannel.all, 'instanceIds');
    if (formatVersion == 1) return null;
    _decodeLabels();
    final ids = _instanceIds!;
    _validateReservedInstanceIds(ids);
    final map = metadata.containsKey('instance_map')
        ? _checkedInstanceMap(metadata['instance_map'])
        : null;
    if (map != null) _validateInstanceMapCoverage(ids, map);
    return Uint16List.fromList(ids);
  }

  /// Persistent instance map from local uint16 id to UUID.
  Map<String, String>? get instanceMap {
    _require(AlfChannel.all, 'instanceMap');
    if (formatVersion != 2) return null;
    if (!metadata.containsKey('instance_map')) return null;
    final map = _checkedInstanceMap(metadata['instance_map']);
    _decodeLabels();
    _validateReservedInstanceIds(_instanceIds!);
    _validateInstanceMapCoverage(_instanceIds!, map);
    return Map<String, String>.unmodifiable(map);
  }

  /// Palette strings from an `.alc` header.
  List<String>? get palette {
    _require(AlfChannel.alc, 'palette');
    if (!metadata.containsKey('palette')) return null;
    final raw = metadata['palette'];
    if (raw is! List<Object?> || raw.length > 255) {
      throw const AlfException(
        'palette-entry-malformed',
        'palette must be a list with at most 255 entries',
      );
    }
    final colors = <String>[];
    final hex = RegExp(r'^[0-9a-fA-F]{6}$');
    for (final value in raw) {
      final color = value is String && value.startsWith('#')
          ? value.substring(1)
          : value;
      if (color is! String || !hex.hasMatch(color)) {
        throw AlfException(
          'palette-entry-malformed',
          'Palette entry $value is not six hexadecimal digits',
        );
      }
      colors.add(color.toLowerCase());
    }
    return List<String>.unmodifiable(colors);
  }

  /// Uint8 indices into [palette]. `palette.length` is the no-colour sentinel.
  Uint8List get paletteIndices {
    _require(AlfChannel.alc, 'paletteIndices');
    final colors = palette;
    if (colors == null || colors.isEmpty) {
      throw const AlfException(
        'palette-entry-malformed',
        '.alc requires a non-empty palette',
      );
    }
    for (final value in _payload) {
      if (value > colors.length) {
        throw AlfException(
          'palette-index-out-of-range',
          'Palette index $value is out of range for ${colors.length} entries',
        );
      }
    }
    return Uint8List.fromList(_payload);
  }

  /// Original point indices from a decimation `.alx` sidecar.
  Uint32List get keptIndices {
    _require(AlfChannel.alx, 'keptIndices');
    final values = _validatedKeptIndices();
    return _keptIndicesView ??= values.asUnmodifiableView();
  }

  Uint32List _validatedKeptIndices() {
    final cached = _keptIndices;
    if (cached != null) return cached;
    if (_payload.lengthInBytes % 4 != 0) {
      throw const AlfException(
        'payload-not-whole-points',
        '.alx payload is not a whole number of uint32 values',
      );
    }
    final data = ByteData.sublistView(_payload);
    final values = Uint32List(_payload.lengthInBytes ~/ 4);
    final seen = <int>{};
    final sourceCount = sourceTotalPoints;
    if (sourceCount == null) {
      throw const AlfException(
        'source-total-points-missing',
        '.alx is missing _source_total_points',
      );
    }
    for (var i = 0; i < values.length; i++) {
      final value = data.getUint32(i * 4, Endian.little);
      if (!seen.add(value)) {
        throw const AlfException(
          'kept-index-duplicate',
          '.alx kept indices contain duplicates',
        );
      }
      if (value >= sourceCount) {
        throw AlfException(
          'kept-index-out-of-range',
          '.alx index $value is outside a $sourceCount-point source cloud',
        );
      }
      values[i] = value;
    }
    _keptIndices = values;
    return values;
  }

  Float64List _decodeXyz() {
    if (_payload.lengthInBytes % 6 != 0) {
      throw AlfException(
        'payload-not-whole-points',
        '.alp payload is ${_payload.lengthInBytes} bytes, not whole 6-byte XYZ points',
      );
    }
    final data = ByteData.sublistView(_payload);
    final result = Float64List(_payload.lengthInBytes ~/ 2);
    if (geometryCodec == AlfGeometryCodec.int16) {
      if (version != 3) {
        throw AlfException(
          'codec-version-disagreement',
          'int16 geometry requires wire version 3, not $version',
        );
      }
      final scale = _geometryScale();
      final offset = _geometryOffset();
      for (var i = 0; i < result.length; i += 3) {
        // A WHOLE point of sentinel words is an invalid return, not a position
        // (spec/format.md §5.2). Checked per point rather than per component
        // because a partial sentinel is not a shape any writer may emit.
        final wx = data.getInt16(i * 2, Endian.little);
        final wy = data.getInt16((i + 1) * 2, Endian.little);
        final wz = data.getInt16((i + 2) * 2, Endian.little);
        if (wx == _int16NoReturn &&
            wy == _int16NoReturn &&
            wz == _int16NoReturn) {
          result[i] = double.nan;
          result[i + 1] = double.nan;
          result[i + 2] = double.nan;
          continue;
        }
        // A PARTIAL marker is corruption, and the one kind nothing downstream
        // can catch: -32768 is reserved, so no conforming writer emits it as a
        // coordinate (the encoder clamps to +/-32767). Reconstructing it yields
        // an ordinary-looking position — [-32768, 0, 0] at scale 0.01 decodes
        // to -327.68 — which no later check can tell from real data.
        if (wx == _int16NoReturn ||
            wy == _int16NoReturn ||
            wz == _int16NoReturn) {
          throw const AlfException(
            'geom-partial-no-return',
            'int16 .alp contains a point with SOME but not all components equal to the '
                'reserved invalid-return marker (-32768). No conforming writer can emit '
                'that word as a coordinate, so this payload is corrupt; decoding it would '
                'report a plausible position that was never measured.',
          );
        }
        for (var component = 0; component < 3; component++) {
          final word = component == 0 ? wx : (component == 1 ? wy : wz);
          final value = word * scale + offset[component];
          if (!value.isFinite) {
            throw const AlfException(
              'coordinate-non-finite',
              'int16 geometry decoded to a non-finite coordinate',
            );
          }
          result[i + component] = value;
        }
      }
      return result;
    }
    if (version == 3) {
      throw const AlfException(
        'codec-version-disagreement',
        'Wire version 3 requires int16 geometry',
      );
    }
    for (var i = 0; i < result.length; i++) {
      // NON-FINITE COMPONENTS ARE KEPT. A NaN in float16 `.alp` geometry is the
      // invalid-return marker (spec/format.md 5.1), not damage, and it is kept
      // IN PLACE because every other channel is correlated with geometry by
      // index. Fixed-point above uses a sentinel for whole invalid points;
      // a non-finite reconstruction outside a sentinel row is overflow.
      result[i] = halfToDouble(data.getUint16(i * 2, Endian.little));
    }
    return result;
  }

  double _geometryScale() {
    if (!metadata.containsKey('_geom_scale')) {
      throw const AlfException(
        'geom-scale-missing',
        'int16 geometry is missing _geom_scale',
      );
    }
    final raw = metadata['_geom_scale'];
    _rejectInexactGeometryInteger(raw, '_geom_scale');
    if (raw is! num || !raw.toDouble().isFinite || raw <= 0) {
      throw AlfException(
        'geom-scale-not-positive',
        '_geom_scale must be a positive finite number, got $raw',
      );
    }
    return raw.toDouble();
  }

  List<double> _geometryOffset() {
    final raw = metadata['_geom_offset'];
    if (raw is List<Object?>) {
      for (final component in raw) {
        _rejectInexactGeometryInteger(component, '_geom_offset');
      }
    }
    if (raw is! List<Object?> ||
        raw.length != 3 ||
        raw.any(
          (Object? value) => value is! num || !value.toDouble().isFinite,
        )) {
      throw AlfException(
        'geom-offset-malformed',
        '_geom_offset must have exactly 3 finite numeric components, got $raw',
      );
    }
    return raw
        .cast<num>()
        .map((num value) => value.toDouble())
        .toList(growable: false);
  }

  double _reconstructionRounding(double scale, List<double> offset) {
    if (_payload.isEmpty || _payload.lengthInBytes % 6 != 0) return 0;
    final data = ByteData.sublistView(_payload);
    var worst = 0.0;
    for (var i = 0; i < _payload.lengthInBytes ~/ 2; i++) {
      // SENTINEL ROWS CARRY NO ERROR. -32768 is a marker, not a coordinate, so
      // reconstructing it states the rounding error of a position that was
      // never measured — and with a large `_geom_scale` the product overflows,
      // making this throw "cannot be bounded" for a file whose `xyz` decodes
      // perfectly to NaN. The reader read it; the bound called it unreadable.
      // Checked at the point boundary, since the marker is a whole-point shape.
      final base = i - (i % 3);
      if (data.getInt16(base * 2, Endian.little) == _int16NoReturn &&
          data.getInt16((base + 1) * 2, Endian.little) == _int16NoReturn &&
          data.getInt16((base + 2) * 2, Endian.little) == _int16NoReturn) {
        continue;
      }
      final quantized = data.getInt16(i * 2, Endian.little).toDouble();
      final product = quantized * scale;
      final axisOffset = offset[i % 3];
      final total = product + axisOffset;
      // Knuth TwoSum recovers the addition's exact rounding error. The
      // multiplication error is bounded by half the larger adjacent ULP.
      final near = total - product;
      final addError = (product - (total - near)) + (axisOffset - near);
      final rounding = addError.abs() + _adjacentUlp(product) / 2;
      if (!rounding.isFinite) {
        throw const AlfException(
          'coordinate-non-finite',
          'Geometry reconstruction rounding cannot be bounded in float64',
        );
      }
      worst = math.max(worst, rounding);
    }
    return worst;
  }

  void _decodeLabels() {
    if (_categories != null) return;
    if (formatVersion == 1) {
      _categories = Uint8List.fromList(_payload);
      return;
    }
    if (_payload.lengthInBytes % 3 != 0) {
      throw const AlfException(
        'payload-not-whole-points',
        '.all v2 payload is not a whole number of 3-byte rows',
      );
    }
    final categories = Uint8List(_payload.lengthInBytes ~/ 3);
    final instances = Uint16List(categories.length);
    for (var i = 0; i < categories.length; i++) {
      categories[i] = _payload[i * 3];
      instances[i] = _payload[i * 3 + 1] | (_payload[i * 3 + 2] << 8);
    }
    _categories = categories;
    _instanceIds = instances;
  }

  void _validateReservedInstanceIds(Uint16List ids) {
    for (final id in ids) {
      if (id == 65535) {
        throw const AlfException(
          'instance-id-reserved',
          'The .all payload contains reserved instance id 65535',
        );
      }
    }
  }

  void _validateInstanceMapCoverage(
    Uint16List ids,
    Map<String, String> instanceMap,
  ) {
    final missing =
        ids
            .where((int id) => id != 0 && !instanceMap.containsKey('$id'))
            .toSet()
            .toList()
          ..sort();
    if (missing.isNotEmpty) {
      throw AlfException(
        'instance-map-incomplete',
        'instance_map is missing entries for $missing',
      );
    }
  }

  void _require(AlfChannel expected, String accessor) {
    if (channel != expected) {
      throw AlfException(
        'wrong-channel',
        '$accessor belongs to .${expected.name}, not .${channel.name}',
      );
    }
  }
}

/// Read one compressed or uncompressed ALF envelope.
///
/// Gzip is detected by magic bytes, not by filename. [channel] is mandatory
/// because the envelope deliberately does not duplicate its load-bearing
/// suffix, and the same payload bytes have different meanings by channel.
AlfFile readAlf(Uint8List bytes, {required AlfChannel channel}) {
  final isGzip = bytes.length >= 2 && bytes[0] == 0x1f && bytes[1] == 0x8b;
  if (!isGzip && bytes.lengthInBytes > maxAlfEnvelopeBytes) {
    throw const AlfException(
      'input-too-large',
      'ALF input exceeds the 512 MiB envelope limit',
    );
  }
  // The gzip decoder owns its slightly larger compressed-input allowance and
  // the decompressed limit. Applying the raw-envelope cap first would reject a
  // valid 512 MiB member solely because its gzip framing is a few bytes larger.
  final raw = isGzip ? decodeSingleGzipMember(bytes) : bytes;
  final envelope = _parseEnvelope(raw, channel);
  // Take exactly one owned payload copy. `sublist()` followed by `fromList()`
  // copied the largest part of the envelope twice while both temporaries were
  // live; a view avoids the intermediate allocation without retaining caller
  // storage in the parsed object.
  final payload = Uint8List.fromList(
    Uint8List.sublistView(raw, 5 + envelope.headerLength),
  );
  final file = AlfFile._(envelope.version, channel, envelope.metadata, payload);
  _validateEnvelope(file);
  return file;
}

/// The declared `.all` layout, or null when the header omits it.
///
/// HEADER-SCOPED, all of it — which is why it is a function rather than the
/// first half of [AlfFile.formatVersion]. The rest of that getter picks between
/// the wire byte and this field using the payload's length, and a header-only
/// read has no payload to pick with; this part needs none.
int? _checkedLabelLayout(Map<String, Object?> metadata) {
  if (!metadata.containsKey('_format_version')) return null;
  final raw = metadata['_format_version'];
  if (raw is! num ||
      !raw.toDouble().isFinite ||
      raw.toDouble() != raw.toDouble().truncateToDouble() ||
      (raw != 1 && raw != 2)) {
    throw AlfException(
      'label-layout-unsupported',
      'Unsupported .all _format_version $raw',
    );
  }
  return raw.toInt();
}

/// Channel rules a header-only read must ALSO enforce.
///
/// The envelope rules apply to every channel; these depend on which one you
/// opened, and they lived only inside [_validateEnvelope] and the channel
/// accessors — so [readAlfHeader], which reaches neither, accepted files
/// [readAlf] refuses. That is the "validates less" drift the header-only
/// contract exists to rule out, and it was present in every implementation.
///
/// Only rules the full reader enforces EAGERLY belong here. A `.alx` missing
/// `_source_total_points` is refused at the accessor rather than at the read,
/// so adding it would make a header scan stricter than the read it stands in
/// for.
void _checkHeaderScopedChannelRules(
  Map<String, Object?> metadata,
  AlfChannel channel,
) {
  if (channel == AlfChannel.alx && !metadata.containsKey('_total_points')) {
    throw const AlfException(
      'total-points-missing',
      '.alx requires _total_points',
    );
  }
  if (channel == AlfChannel.all) _checkedLabelLayout(metadata);
}

/// Bytes inflated on the first probe of a gzipped file.
///
/// Large enough that essentially every real header lands in one pass, small
/// enough that a 1 GiB frame costs kilobytes. A header longer than this costs
/// one more, exactly-sized inflate — not a full decompression.
const _headerProbeBytes = 64 * 1024;

/// An ALF envelope read WITHOUT its payload.
///
/// A distinct type from [AlfFile], deliberately, and this is the one place the
/// Dart API differs from the Python reference: Python returns an `AlfFile`
/// whose payload accessors raise at runtime, because Python cannot say
/// "the same object, minus those" in its types. Dart can. Reaching for
/// geometry on a header-only read does not compile rather than failing on a
/// caller's machine.
final class AlfHeader {
  /// Wire version byte.
  final int version;

  /// Payload channel supplied by the caller.
  final AlfChannel channel;

  /// Parsed JSON header.
  final Map<String, Object?> metadata;

  AlfHeader._(this.version, this.channel, this.metadata);

  /// Header-declared point count, when present.
  ///
  /// Through the same validated accessor [AlfFile] uses, so a header-only read
  /// refuses a malformed `_total_points` exactly where [readAlf] would.
  num? get totalPoints => _checkedCount(metadata, '_total_points');
}

/// Parse ONLY an ALF file's envelope, without inflating or reading its payload.
///
/// For the scan-many-frames case: a prefetch that needs `_total_points`,
/// `device_position`, `_geom_scale` or a `palette` across a thousand frames
/// should not pay a full decompression for each. [readAlf] inflates everything
/// before it returns anything; this stops after the header.
///
/// ITS REFUSALS ARE A STRICT SUBSET OF [readAlf]'s, AND THAT SPLIT IS THE
/// CONTRACT. Every header-scoped rule still applies — both entry points share
/// [_parseEnvelope], so they agree by construction rather than by review. The
/// payload-scoped rules cannot apply, because their evidence is the bytes this
/// declined to read:
///
///   * the declared count is not cross-checked against the payload length;
///   * a second gzip member goes undetected;
///   * for a gzipped input the member's CRC32 and ISIZE go unverified —
///     checking them means inflating to the end, which is the cost this exists
///     to avoid.
///
/// So a header this accepts is a header [readAlf] would also accept. A FILE it
/// accepts is not necessarily a file [readAlf] would. Use [readAlf] when the
/// points matter, or when the file must be proven intact.
///
/// A separate function rather than `readAlf(headerOnly: true)` on purpose: a
/// boolean lets the weaker contract be selected by a variable at a call site
/// that reads exactly like the strong one.
AlfHeader readAlfHeader(Uint8List bytes, {required AlfChannel channel}) {
  final isGzip = bytes.length >= 2 && bytes[0] == 0x1f && bytes[1] == 0x8b;
  if (!isGzip && bytes.lengthInBytes > maxAlfEnvelopeBytes) {
    throw const AlfException(
      'input-too-large',
      'ALF input exceeds the 512 MiB envelope limit',
    );
  }
  Uint8List raw;
  if (!isGzip) {
    raw = bytes;
  } else {
    raw = decodeGzipPrefix(bytes, atLeast: _headerProbeBytes);
    // Re-inflate only when the DECLARED header genuinely outran the probe. The
    // length is read here rather than by catching `_parseEnvelope`'s truncation
    // refusal, because a genuinely truncated file would otherwise be retried
    // and then reported as though the probe size were at fault.
    if (raw.lengthInBytes >= 5) {
      final declared = ByteData.sublistView(
        raw,
        1,
        5,
      ).getUint32(0, Endian.little);
      if (declared <= maxAlfHeaderBytes && 5 + declared > raw.lengthInBytes) {
        raw = decodeGzipPrefix(bytes, atLeast: 5 + declared);
      }
    }
  }
  final envelope = _parseEnvelope(raw, channel);
  // EAGERLY, here rather than when someone asks for [AlfHeader.totalPoints].
  // The count is decidable from the header alone, so refusing a malformed one
  // is a header-scoped rule and this function enforces every one of those.
  // Leaving it to the getter would make a scan hand back a header the full
  // reader refuses, and the caller would learn that from an accessor on
  // whichever frame they happened to look at closely.
  _checkedCount(envelope.metadata, '_total_points');
  _checkHeaderScopedChannelRules(envelope.metadata, channel);
  return AlfHeader._(envelope.version, channel, envelope.metadata);
}

/// Validate an INFLATED ALF envelope and parse its header.
///
/// Split out of [readAlf] so the full read and the header-only read share ONE
/// implementation of every envelope rule. Two copies would drift, and the
/// direction they would drift in is the dangerous one: a header-only reader
/// that validated less would hand back a header the full reader refuses, and
/// the caller scanning a sequence would never learn the file was bad.
///
/// Every rule here is HEADER-SCOPED — decidable from the envelope alone.
/// Payload-scoped rules live in [_validateEnvelope], which only [readAlf]
/// reaches, because their evidence is bytes a header-only read never got.
({int version, Map<String, Object?> metadata, int headerLength}) _parseEnvelope(
  Uint8List raw,
  AlfChannel channel,
) {
  if (raw.lengthInBytes < 5) {
    throw const AlfException('header-truncated', 'Too short to be ALF');
  }
  final version = raw[0];
  if (version != 1 && version != 2 && version != 3) {
    throw AlfException(
      'version-unsupported',
      'Unsupported ALF version byte $version',
    );
  }
  if (version == 2 && channel != AlfChannel.all) {
    throw AlfException(
      'label-version-on-non-label-channel',
      'Version 2 is the .all label layout, not .${channel.name}',
    );
  }
  if (version == 3 && channel != AlfChannel.alp) {
    throw AlfException(
      'geometry-version-on-non-geometry-channel',
      'Version 3 is fixed-point .alp geometry, not .${channel.name}',
    );
  }
  final headerLength = ByteData.sublistView(
    raw,
    1,
    5,
  ).getUint32(0, Endian.little);
  if (headerLength > maxAlfHeaderBytes) {
    throw AlfException(
      'header-too-large',
      'ALF header exceeds the $maxAlfHeaderBytes-byte limit',
    );
  }
  if (5 + headerLength > raw.lengthInBytes) {
    throw AlfException(
      'header-truncated',
      'Truncated header: declared $headerLength bytes',
    );
  }
  Object? decoded;
  String? headerText;
  Map<String, List<String>>? geometryNumberTokens;
  Map<String, String>? integralNumberTokens;
  try {
    if (headerLength == 0) {
      decoded = <String, Object?>{};
    } else {
      headerText = utf8.decode(
        Uint8List.sublistView(raw, 5, 5 + headerLength),
        allowMalformed: false,
      );
      _preflightJson(headerText, validateNumbers: false);
      _JsonDuplicateKeys(headerText).validate();
      // Geometry tokens must be classified before `jsonDecode`: Dart refuses
      // an integer outside binary64 at the JSON layer, which would otherwise
      // hide the more specific portable `geom-number-inexact` rule. Only an
      // object can own those fields; non-object roots retain the dedicated
      // `header-not-an-object` refusal after decoding below.
      if (channel == AlfChannel.alp && headerText.trimLeft().startsWith('{')) {
        geometryNumberTokens = _geometryParameterTokens(headerText);
        _rejectOversizedGeometryTokens(
          geometryNumberTokens,
          includeGrid: version == 3,
        );
      }
      // Inspect reserved count tokens before the generic finite-number pass.
      // Otherwise a value such as 10^400 is classified as malformed metadata
      // instead of the portable count-domain refusal used by Python and
      // TypeScript. Keep non-object roots for the dedicated refusal below.
      if (headerText.trimLeft().startsWith('{')) {
        integralNumberTokens = _JsonNumberTokens(headerText)
            .topLevelNumbers(<String>{
              '_total_points',
              if (channel == AlfChannel.alx) '_source_total_points',
              if (channel == AlfChannel.all) '_format_version',
            });
        _rejectNonFiniteIntegralNumberTokens(integralNumberTokens);
      }
      // Reserved integral tokens have their own exact raw-token validator.
      // Dart's binary64 parser may overflow a long-but-cancellable significand
      // before applying its negative exponent, so validate generic numbers
      // after restoring those exact reserved values below.
      _preflightJson(headerText, validateNumbers: false);
      decoded = jsonDecode(headerText);
      if (decoded is Map<String, Object?>) {
        _restoreExactIntegralNumbers(
          integralNumberTokens ?? const <String, String>{},
          decoded,
        );
        if (channel == AlfChannel.alp) {
          _restoreExactGeometryNumbers(
            geometryNumberTokens ?? const <String, List<String>>{},
            decoded,
            includeGrid: version == 3,
          );
        }
      }
      _validateFiniteJsonNumbers(decoded);
      // The integral-token scanner implements reserved OBJECT fields; do not
      // run it against an array/scalar root before the portable
      // `header-not-an-object` refusal below has a chance to name that rule.
      if (decoded is Map<String, Object?> && integralNumberTokens == null) {
        integralNumberTokens = _JsonNumberTokens(headerText)
            .topLevelNumbers(<String>{
              '_total_points',
              if (channel == AlfChannel.alx) '_source_total_points',
              if (channel == AlfChannel.all) '_format_version',
            });
      }
    }
  } on FormatException catch (error) {
    final headerBytes = Uint8List.sublistView(raw, 5, 5 + headerLength);
    try {
      utf8.decode(headerBytes, allowMalformed: false);
    } on FormatException {
      throw const AlfException('header-not-utf8', 'Header is not valid UTF-8');
    }
    throw AlfException('header-not-json', 'Header is not valid JSON: $error');
  }
  if (decoded is! Map<String, Object?>) {
    throw const AlfException(
      'header-not-an-object',
      'Header must be a JSON object',
    );
  }
  _validateIntegralNumberTokens(integralNumberTokens ?? const {}, decoded);
  if (channel == AlfChannel.alp) {
    _validateGeometryNumberTokens(
      geometryNumberTokens ?? const {},
      decoded,
      includeGrid: version == 3,
    );
    _validateGeometryAnchorShapes(decoded);
    // Before the grid parameters: a file that does not say where its points are
    // is refused for that, which is decidable first and from less.
    _validateFrameConvention(decoded, version);
  }
  return (
    version: version,
    metadata: _deepFreezeMetadata(decoded),
    headerLength: headerLength,
  );
}

/// Bound work before Dart's recursive JSON decoder sees an attacker-controlled
/// header, and reject numeric tokens that cannot become finite binary64.
///
/// This scanner deliberately leaves full grammar validation to [jsonDecode].
/// It only recognizes strings, nesting, and syntactically complete number
/// prefixes, so its work is linear and independent of the decoder's stack.
void _preflightJson(String source, {required bool validateNumbers}) {
  var depth = 0;
  var inString = false;
  var escaped = false;
  var index = 0;
  while (index < source.length) {
    final code = source.codeUnitAt(index);
    if (inString) {
      if (escaped) {
        escaped = false;
      } else if (code == 0x5c) {
        escaped = true;
      } else if (code == 0x22) {
        inString = false;
      }
      index++;
      continue;
    }
    if (code == 0x22) {
      inString = true;
      index++;
      continue;
    }
    if (code == 0x7b || code == 0x5b) {
      depth++;
      if (depth > _maxJsonDepth) {
        throw const AlfException(
          'header-nesting-too-deep',
          'Header JSON exceeds the maximum nesting depth',
        );
      }
      index++;
      continue;
    }
    if (code == 0x7d || code == 0x5d) {
      depth--;
      index++;
      continue;
    }
    if (code == 0x2d || (code >= 0x30 && code <= 0x39)) {
      final end = _jsonNumberEnd(source, index);
      if (end > index) {
        if (validateNumbers) {
          final parsed = double.tryParse(source.substring(index, end));
          if (parsed != null && !parsed.isFinite) {
            throw const AlfException(
              'header-not-json',
              'Header JSON numbers must be finite binary64 values',
            );
          }
        }
        index = end;
        continue;
      }
    }
    index++;
  }
}

int _jsonNumberEnd(String source, int start) {
  var index = start;
  if (source.codeUnitAt(index) == 0x2d) index++;
  if (index >= source.length) return start;
  var code = source.codeUnitAt(index);
  if (code == 0x30) {
    index++;
  } else if (code >= 0x31 && code <= 0x39) {
    do {
      index++;
      if (index >= source.length) break;
      code = source.codeUnitAt(index);
    } while (code >= 0x30 && code <= 0x39);
  } else {
    return start;
  }
  if (index < source.length && source.codeUnitAt(index) == 0x2e) {
    final fractionStart = ++index;
    while (index < source.length) {
      code = source.codeUnitAt(index);
      if (code < 0x30 || code > 0x39) break;
      index++;
    }
    if (index == fractionStart) return start;
  }
  if (index < source.length) {
    code = source.codeUnitAt(index);
    if (code == 0x65 || code == 0x45) {
      index++;
      if (index < source.length) {
        code = source.codeUnitAt(index);
        if (code == 0x2b || code == 0x2d) index++;
      }
      final exponentStart = index;
      while (index < source.length) {
        code = source.codeUnitAt(index);
        if (code < 0x30 || code > 0x39) break;
        index++;
      }
      if (index == exponentStart) return start;
    }
  }
  return index;
}

void _validateFiniteJsonNumbers(Object? value) {
  if (value is num) {
    if (!value.toDouble().isFinite) {
      throw const AlfException(
        'header-not-json',
        'Header JSON numbers must be finite binary64 values',
      );
    }
    return;
  }
  if (value is List<Object?>) {
    for (final element in value) {
      _validateFiniteJsonNumbers(element);
    }
    return;
  }
  if (value is Map<String, Object?>) {
    for (final element in value.values) {
      _validateFiniteJsonNumbers(element);
    }
  }
}

Map<String, Object?> _deepFreezeMetadata(Map<String, Object?> metadata) {
  return Map<String, Object?>.unmodifiable(<String, Object?>{
    for (final entry in metadata.entries)
      entry.key: _deepFreezeJson(entry.value),
  });
}

Object? _deepFreezeJson(Object? value) {
  if (value is Map<String, Object?>) {
    return _deepFreezeMetadata(value);
  }
  if (value is List<Object?>) {
    return List<Object?>.unmodifiable(value.map(_deepFreezeJson));
  }
  return value;
}

void _validateEnvelope(AlfFile file) {
  final declared = file.totalPoints;
  if (declared == null && file.channel == AlfChannel.alx) {
    throw const AlfException(
      'total-points-missing',
      '.alx requires _total_points',
    );
  }
  int bytesPerPoint;
  switch (file.channel) {
    case AlfChannel.alp:
      bytesPerPoint = 6;
      // Validate the codec contract at read time, before a caller can trust the
      // header while postponing access to the payload.
      final codec = file.geometryCodec;
      if ((file.version == 3) != (codec == AlfGeometryCodec.int16)) {
        throw const AlfException(
          'codec-version-disagreement',
          'Geometry codec and wire version disagree',
        );
      }
      if (codec == AlfGeometryCodec.int16) {
        file._geometryScale();
        file._geometryOffset();
      }
      if (file._payload.lengthInBytes % 6 != 0) {
        throw const AlfException(
          'payload-not-whole-points',
          '.alp payload is not whole XYZ triples',
        );
      }
      break;
    case AlfChannel.all:
      bytesPerPoint = file.formatVersion == 2 ? 3 : 1;
      if (file.formatVersion == 2 && file._payload.lengthInBytes % 3 != 0) {
        throw const AlfException(
          'payload-not-whole-points',
          '.all v2 payload is not whole rows',
        );
      }
      break;
    case AlfChannel.alx:
      bytesPerPoint = 4;
      break;
    case AlfChannel.ali:
    case AlfChannel.alc:
      bytesPerPoint = 1;
  }
  if (declared != null &&
      file._payload.lengthInBytes != declared.toDouble() * bytesPerPoint) {
    throw AlfException(
      'payload-length-mismatch',
      'Header declares $declared points but payload holds ${file._payload.lengthInBytes} bytes',
    );
  }
  // These accessors carry format validation in addition to decoding.
  if (file.channel == AlfChannel.alp) file.xyz;
  if (file.channel == AlfChannel.alc) file.paletteIndices;
  // Panoptic validation is intentionally lazy. Semantic callers can still use
  // valid categories; identity accessors fail closed when they are requested.
  if (file.channel == AlfChannel.alx) file._validatedKeptIndices();
}

num? _checkedCount(Map<String, Object?> metadata, String key) {
  if (!metadata.containsKey(key)) return null;
  final raw = metadata[key];
  if (raw is! num ||
      !raw.toDouble().isFinite ||
      raw < 0 ||
      raw.toDouble() != raw.toDouble().truncateToDouble()) {
    throw AlfException(
      'total-points-not-an-integer',
      '$key must be a non-negative integer, got $raw',
    );
  }
  // VM/Wasm ints are signed 64-bit; JavaScript ints use binary64. Keep the
  // familiar int result only when conversion preserves the exact count.
  final integer = raw.toInt();
  return BigInt.from(integer) == BigInt.from(raw) ? integer : raw;
}

void _validateIntegralNumberTokens(
  Map<String, String> tokens,
  Map<String, Object?> metadata,
) {
  for (final entry in tokens.entries) {
    final exact = _exactJsonInteger(entry.value);
    final decoded = metadata[entry.key];
    if (exact == null || decoded is! num || !decoded.toDouble().isFinite) {
      throw AlfException(
        'total-points-not-an-integer',
        '${entry.key} must state an exact non-negative integer',
      );
    }
    final narrowed = decoded.toDouble();
    if (narrowed != narrowed.truncateToDouble() ||
        BigInt.from(narrowed) != exact) {
      throw AlfException(
        'total-points-not-an-integer',
        '${entry.key} contains ${entry.value}, which binary64 cannot represent exactly',
      );
    }
  }
}

void _rejectNonFiniteIntegralNumberTokens(Map<String, String> tokens) {
  for (final entry in tokens.entries) {
    final exact = _exactJsonInteger(entry.value);
    if (exact != null && exact.toDouble().isFinite) continue;
    final decoded = double.tryParse(entry.value);
    if (decoded == null || !decoded.isFinite) {
      throw AlfException(
        'total-points-not-an-integer',
        '${entry.key} must state a finite binary64 integer',
      );
    }
  }
}

void _restoreExactIntegralNumbers(
  Map<String, String> tokens,
  Map<String, Object?> metadata,
) {
  for (final entry in tokens.entries) {
    final exact = _exactJsonInteger(entry.value);
    final narrowed = exact?.toDouble();
    if (narrowed != null && narrowed.isFinite) {
      metadata[entry.key] = narrowed;
    }
  }
}

void _restoreExactGeometryNumbers(
  Map<String, List<String>> tokens,
  Map<String, Object?> metadata, {
  required bool includeGrid,
}) {
  double? exactDouble(String token) {
    final exact = _exactJsonInteger(token);
    final narrowed = exact?.toDouble();
    return narrowed != null && narrowed.isFinite ? narrowed : null;
  }

  if (includeGrid) {
    final scaleTokens = tokens['_geom_scale'];
    if (scaleTokens?.length == 1) {
      final restored = exactDouble(scaleTokens!.single);
      if (restored != null) metadata['_geom_scale'] = restored;
    }
    final offset = metadata['_geom_offset'];
    final offsetTokens = tokens['_geom_offset'];
    if (offset is List<Object?> && offsetTokens?.length == offset.length) {
      for (var i = 0; i < offset.length; i++) {
        final restored = exactDouble(offsetTokens![i]);
        if (restored != null) offset[i] = restored;
      }
    }
  }

  for (final key in <String>['device_position', 'world_centroid_epsg3857']) {
    final anchor = metadata[key];
    final anchorTokens = tokens[key];
    if (anchorTokens?.length != 3) continue;
    if (anchor is List<Object?> && anchor.length == 3) {
      for (var i = 0; i < 3; i++) {
        final restored = exactDouble(anchorTokens![i]);
        if (restored != null) anchor[i] = restored;
      }
    } else if (anchor is Map<String, Object?>) {
      for (var i = 0; i < 3; i++) {
        final restored = exactDouble(anchorTokens![i]);
        if (restored != null) {
          anchor[const <String>['x', 'y', 'z'][i]] = restored;
        }
      }
    }
  }
}

BigInt? _exactJsonInteger(String token) {
  final match = RegExp(
    r'^(-?)(0|[1-9][0-9]*)(?:\.([0-9]+))?(?:[eE]([+-]?[0-9]+))?$',
  ).firstMatch(token);
  if (match == null) return null;
  final fraction = match.group(3) ?? '';
  final digits = '${match.group(2)}$fraction';
  if (!RegExp(r'[1-9]').hasMatch(digits)) return BigInt.zero;
  final rawExponent = match.group(4) ?? '0';
  final negativeExponent = rawExponent.startsWith('-');
  final unsignedExponent = rawExponent.replaceFirst(RegExp(r'^[+-]'), '');
  final exponentDigits = unsignedExponent.replaceFirst(RegExp(r'^0+'), '');
  final normalizedExponent = exponentDigits.isEmpty ? '0' : exponentDigits;
  if (normalizedExponent.length > 10) return null;
  final exponent = int.tryParse(
    negativeExponent ? '-$normalizedExponent' : normalizedExponent,
  );
  if (exponent == null || exponent.abs() > maxAlfHeaderBytes) return null;
  final power = exponent - fraction.length;
  late BigInt value;
  if (power >= 0) {
    final normalizedDigits = digits.replaceFirst(RegExp(r'^0+'), '');
    // No finite binary64 integer has more than 309 decimal digits. Refuse
    // before materializing a power such as 10^9,999,999 merely to discover
    // that its decoded value is infinite.
    if (normalizedDigits.length + power > 309) return null;
    value = BigInt.parse(normalizedDigits);
    value *= BigInt.from(10).pow(power);
  } else {
    // A negative decimal exponent can be integral only when the significand
    // supplies at least that many trailing zeros. Establish that from the
    // bounded token string before constructing either BigInt: a compact token
    // such as 1e-10000000 must not allocate a ten-million-digit divisor merely
    // to prove that it is fractional.
    final requiredTrailingZeros = -power;
    var availableTrailingZeros = 0;
    for (
      var i = digits.length - 1;
      i >= 0 && digits.codeUnitAt(i) == 0x30;
      i--
    ) {
      availableTrailingZeros++;
    }
    if (requiredTrailingZeros > availableTrailingZeros) return null;
    final integerDigits = digits
        .substring(0, digits.length - requiredTrailingZeros)
        .replaceFirst(RegExp(r'^0+'), '');
    if (integerDigits.length > 309) return null;
    value = BigInt.parse(integerDigits);
  }
  return match.group(1) == '-' ? -value : value;
}

void _rejectInexactGeometryInteger(Object? value, String key) {
  // Large integers are valid whenever binary64 represents that particular
  // value exactly: 2^53 and even integers immediately above it are examples.
  // On JavaScript runtimes a lossy JSON integer is rounded before it reaches
  // this check, so the raw-token comparison above remains the decisive guard.
  if (value is int &&
      value.toDouble().isFinite &&
      value.toDouble().toInt() != value) {
    throw AlfException(
      'geom-number-inexact',
      '$key contains an integer binary64 cannot represent exactly',
    );
  }
}

void _validateGeometryNumberTokens(
  Map<String, List<String>> tokens,
  Map<String, Object?> metadata, {
  required bool includeGrid,
}) {
  if (includeGrid) {
    final rawScale = metadata['_geom_scale'];
    final scaleToken = tokens['_geom_scale'];
    if (rawScale is num && scaleToken?.length == 1) {
      _rejectLossyGeometryToken(scaleToken!.single, rawScale, '_geom_scale');
    }

    final rawOffset = metadata['_geom_offset'];
    final offsetTokens = tokens['_geom_offset'];
    if (rawOffset is List<Object?> &&
        rawOffset.length == 3 &&
        offsetTokens?.length == rawOffset.length) {
      for (var i = 0; i < rawOffset.length; i++) {
        final component = rawOffset[i];
        if (component is num) {
          _rejectLossyGeometryToken(
            offsetTokens![i],
            component,
            '_geom_offset',
          );
        }
      }
    }

    // The rotation is under the same rule, and NOTHING else can see a lossy
    // component: `0.99999999999999999` decodes to exactly 1.0, so the shape
    // check passes and the quaternion is a perfect unit. Only the raw token
    // says the header stated something the reader did not use.
    final rawRotation = metadata['_frame_rotation'];
    final rotationTokens = tokens['_frame_rotation'];
    if (rawRotation is Map<String, Object?> && rotationTokens?.length == 4) {
      const axes = <String>['w', 'x', 'y', 'z'];
      for (var i = 0; i < axes.length; i++) {
        final component = rawRotation[axes[i]];
        if (component is num) {
          _rejectLossyGeometryToken(
            rotationTokens![i],
            component,
            '_frame_rotation',
          );
        }
      }
    }
  }
  for (final key in <String>['device_position', 'world_centroid_epsg3857']) {
    final anchorTokens = tokens[key];
    if (anchorTokens == null) continue;
    final rawAnchor = metadata[key];
    final components = <num>[
      if (rawAnchor is List<Object?>)
        for (final value in rawAnchor)
          if (value is num) value,
      if (rawAnchor is Map<String, Object?>)
        for (final axis in <String>['x', 'y', 'z'])
          if (rawAnchor[axis] case final num value) value,
    ];
    if (components.length != anchorTokens.length) continue;
    for (var i = 0; i < components.length; i++) {
      _rejectLossyGeometryToken(anchorTokens[i], components[i], key);
    }
  }
}

Map<String, List<String>> _geometryParameterTokens(String header) {
  late final Map<String, List<String>> tokens;
  try {
    tokens = _JsonNumberTokens(header).geometryParameters();
  } on FormatException catch (error) {
    throw AlfException(
      'header-not-json',
      'Could not inspect geometry number tokens: $error',
    );
  }
  return tokens;
}

void _rejectOversizedGeometryTokens(
  Map<String, List<String>> tokens, {
  required bool includeGrid,
}) {
  for (final key in <String>[
    // `_frame_rotation` is here for the same reason the grid parameters are: a
    // component whose stated decimal changes when binary64 decodes it turns the
    // WHOLE frame, not one sample.
    if (includeGrid) ...<String>[
      '_geom_scale',
      '_geom_offset',
      '_frame_rotation',
    ],
    'device_position',
    'world_centroid_epsg3857',
  ]) {
    for (final token in tokens[key] ?? const <String>[]) {
      final decoded = double.tryParse(token);
      if (decoded == null || !decoded.isFinite) {
        throw AlfException(
          'geom-number-inexact',
          '$key contains a numeric token outside finite binary64',
        );
      }
      if (_exactIntegerTokenMatchesDouble(token, decoded)) continue;
      if (_exactIntegralTokenMatchesDouble(token, decoded)) continue;
      if (_canonicalDecimal(token) == _decimalTooPrecise) {
        throw AlfException(
          'geom-number-inexact',
          '$key contains a numeric token too precise for binary64',
        );
      }
    }
  }
}

void _rejectLossyGeometryToken(String token, num value, String key) {
  if (_isJsonIntegerToken(token)) {
    if (_exactIntegerTokenMatchesDouble(token, value.toDouble())) return;
    throw AlfException(
      'geom-number-inexact',
      '$key contains $token, which binary64 cannot represent exactly',
    );
  }
  final exactIntegral = _exactJsonInteger(token);
  if (exactIntegral != null) {
    if (_exactIntegralTokenMatchesDouble(token, value.toDouble())) return;
    if (!_canonicalScientificGeometryToken.hasMatch(token)) {
      throw AlfException(
        'geom-number-inexact',
        '$key contains $token, which binary64 cannot represent exactly',
      );
    }
  }
  final stated = _canonicalDecimal(token);
  if (stated == _decimalTooPrecise) {
    throw AlfException(
      'geom-number-inexact',
      '$key contains a numeric token too precise for binary64',
    );
  }
  final decoded = _canonicalDecimal(value.toDouble().toString());
  if (stated != null && decoded != null && stated != decoded) {
    throw AlfException(
      'geom-number-inexact',
      '$key contains $token, which binary64 reads as ${value.toDouble()}',
    );
  }
}

final _jsonIntegerToken = RegExp(r'^-?(?:0|[1-9][0-9]*)$');
final _canonicalScientificGeometryToken = RegExp(
  r'^-?[0-9](?:\.[0-9]*[1-9])?[eE][+-]?[1-9][0-9]*$',
);

bool _isJsonIntegerToken(String token) => _jsonIntegerToken.hasMatch(token);

bool _exactIntegerTokenMatchesDouble(String token, double decoded) {
  if (!_isJsonIntegerToken(token) || !decoded.isFinite) return false;
  final exact = BigInt.tryParse(token);
  if (exact == null) return false;
  return BigInt.from(decoded) == exact;
}

bool _exactIntegralTokenMatchesDouble(String token, double decoded) {
  if (!decoded.isFinite || decoded != decoded.truncateToDouble()) return false;
  final match = RegExp(
    r'^(-?)(0|[1-9][0-9]*)(?:\.([0-9]+))?(?:[eE]([+-]?[0-9]+))?$',
  ).firstMatch(token);
  if (match == null) return false;

  var digits = '${match.group(2)!}${match.group(3) ?? ''}'.replaceFirst(
    RegExp(r'^0+'),
    '',
  );
  if (digits.isEmpty) return decoded == 0;

  var trailingZeros = 0;
  while (trailingZeros < digits.length &&
      digits.codeUnitAt(digits.length - trailingZeros - 1) == 0x30) {
    trailingZeros++;
  }
  if (trailingZeros > 0) {
    digits = digits.substring(0, digits.length - trailingZeros);
  }

  // The exponent by itself is not the amount of power-of-ten work required.
  // A 10,000-digit fractional significand followed by e10000 can cancel to an
  // ordinary integer. Strip that decimal placement before applying the hostile
  // exponent bound; the complete header cap gives us a small, safe integer
  // ceiling without ever constructing an attacker-sized BigInt power.
  final exponentToken = match.group(4) ?? '0';
  final exponentNegative = exponentToken.startsWith('-');
  final unsignedExponent = exponentToken.startsWith('+') || exponentNegative
      ? exponentToken.substring(1)
      : exponentToken;
  final normalizedExponent = unsignedExponent.replaceFirst(RegExp(r'^0+'), '');
  final exponentDigits = normalizedExponent.isEmpty ? '0' : normalizedExponent;
  final maximumRelevantExponent = maxAlfHeaderBytes + 309;
  if (exponentDigits.length > maximumRelevantExponent.toString().length) {
    return false;
  }
  final exponentMagnitude = int.tryParse(exponentDigits);
  if (exponentMagnitude == null ||
      exponentMagnitude > maximumRelevantExponent) {
    return false;
  }
  final power =
      (exponentNegative ? -exponentMagnitude : exponentMagnitude) -
      (match.group(3)?.length ?? 0) +
      trailingZeros;
  if (power < 0 || digits.length + power > 309) return false;

  var exact = BigInt.parse(digits) * BigInt.from(10).pow(power);
  if (match.group(1) == '-') exact = -exact;
  return exact == BigInt.from(decoded);
}

/// Validate the caller's frame declaration and return the header it becomes.
///
/// NORMALIZATION LIVES HERE, and only here. A pose quaternion that has been
/// through a few compositions is a unit quaternion to about 1e-7, not to the
/// 1e-9 the reader requires, so a writer that passed it through untouched would
/// emit files its own reader refuses. The writer knows what the caller meant; a
/// reader knows only what it was handed, and normalizing there would turn a
/// producer's bug into a silent scale on every coordinate.
Map<String, Object?> _writerFrameFields(
  AlfGeometryCodec codec,
  String? frameConvention,
  Map<String, num>? frameRotation,
) {
  if (codec != AlfGeometryCodec.int16) {
    if (frameConvention != null || frameRotation != null) {
      throw const AlfException(
        'writer-input-invalid',
        'frameConvention/frameRotation are version-3 fields and this write is fp16; readers '
            'apply no rotation to v1 geometry, so recording one would be a promise the format '
            'does not keep',
      );
    }
    return const <String, Object?>{};
  }
  if (frameConvention == null) {
    throw const AlfException(
      'writer-input-invalid',
      'codec int16 requires frameConvention: the points are the world points with an origin '
          'subtracted, and the file must say whether they were rotated into the device frame too',
    );
  }
  if (frameConvention != alfFrameWorldAligned &&
      frameConvention != alfFrameDeviceAligned) {
    throw AlfException(
      'writer-input-invalid',
      'Unknown frameConvention $frameConvention',
    );
  }
  if (frameConvention == alfFrameWorldAligned) {
    if (frameRotation != null) {
      throw const AlfException(
        'writer-input-invalid',
        'frameConvention world-aligned means no rotation was applied, so a frameRotation '
            'cannot also be recorded',
      );
    }
    return <String, Object?>{'_frame_convention': alfFrameWorldAligned};
  }
  if (frameRotation == null) {
    throw const AlfException(
      'writer-input-invalid',
      'frameConvention device-aligned requires frameRotation',
    );
  }
  final values = <String, double>{};
  var normSquared = 0.0;
  for (final axis in const <String>['w', 'x', 'y', 'z']) {
    final component = frameRotation[axis];
    if (component == null || !component.toDouble().isFinite) {
      throw AlfException(
        'writer-input-invalid',
        'frameRotation.$axis must be a finite number',
      );
    }
    final value = component.toDouble();
    values[axis] = value;
    normSquared += value * value;
  }
  final norm = math.sqrt(normSquared);
  if (norm == 0.0) {
    throw const AlfException(
      'writer-input-invalid',
      'frameRotation is the zero quaternion and describes no rotation',
    );
  }
  // Drift is a small number. |q| = 2 is a units or ordering mistake, and
  // scaling it to length 1 would manufacture a plausible rotation out of it.
  if (norm < 0.9 || norm > 1.1) {
    throw AlfException(
      'writer-input-invalid',
      'frameRotation has |q| = $norm, too far from 1 to be accumulated drift',
    );
  }
  return <String, Object?>{
    '_frame_convention': alfFrameDeviceAligned,
    '_frame_rotation': <String, double>{
      for (final entry in values.entries) entry.key: entry.value / norm,
    },
  };
}

/// The v3 frame convention: how a `.alp`'s local points reach the world.
///
/// A `.alp`'s points are the world points with an origin subtracted, which is
/// HALF of a rigid transform. Producers disagreed about the other half — some
/// translated and stopped, leaving points on world axes, some also rotated into
/// the device frame — and nothing in the bytes told them apart. The two agree
/// exactly whenever the device heading is identity, so a consumer that omitted
/// the rotation matched a producer that omitted it, and the pair looked right
/// until the vehicle turned. See `spec/format.md` §5.3.
void _validateFrameConvention(Map<String, Object?> metadata, int version) {
  final declared = metadata['_frame_convention'];
  final hasDeclared = metadata.containsKey('_frame_convention');
  final hasRotation = metadata.containsKey('_frame_rotation');

  // Versions 1 and 2 are grandfathered: they predate the field and a deployed
  // file cannot be told retroactively what it meant. What they may not do is
  // CARRY it — this reader applies no rotation to v1/v2 geometry, so the
  // declaration would be a promise the format does not keep.
  if (version != 3) {
    if (hasDeclared || hasRotation) {
      final named = hasDeclared ? '_frame_convention' : '_frame_rotation';
      throw AlfException(
        'frame-convention-on-legacy-version',
        '`$named` is a version-3 field, but this .alp is version $version. Readers apply no '
            'rotation to v1/v2 geometry, so honouring it is exactly what they will not do',
      );
    }
    return;
  }

  if (!hasDeclared) {
    throw const AlfException(
      'frame-convention-missing',
      'version-3 .alp is missing `_frame_convention`; its points are the world points with an '
          'origin subtracted, and the file must say whether the device rotation was applied too',
    );
  }
  if (declared != alfFrameWorldAligned && declared != alfFrameDeviceAligned) {
    throw AlfException(
      'frame-convention-unsupported',
      'Unsupported `_frame_convention` $declared; expected '
          '"$alfFrameWorldAligned" or "$alfFrameDeviceAligned"',
    );
  }

  if (declared == alfFrameWorldAligned) {
    if (hasRotation) {
      throw const AlfException(
        'frame-rotation-unexpected',
        '`_frame_convention: world-aligned` declares that no rotation was applied, but '
            '`_frame_rotation` is present',
      );
    }
    return;
  }

  if (!hasRotation) {
    throw const AlfException(
      'frame-rotation-missing',
      '`_frame_convention: device-aligned` requires `_frame_rotation`; without it the points '
          'cannot be returned to the world at all',
    );
  }
  final rotation = metadata['_frame_rotation'];
  if (rotation is! Map<String, Object?>) {
    throw const AlfException(
      'frame-rotation-malformed',
      '`_frame_rotation` must be an object with members w, x, y and z',
    );
  }
  // EXACTLY the four members: a stray `qw` beside `w` is a producer mixing two
  // spellings, not application data, and only one of them would be read.
  if (rotation.length != 4) {
    throw const AlfException(
      'frame-rotation-malformed',
      '`_frame_rotation` must carry exactly w, x, y and z',
    );
  }
  var normSquared = 0.0;
  for (final axis in const <String>['w', 'x', 'y', 'z']) {
    final component = rotation[axis];
    if (component is! num || !component.toDouble().isFinite) {
      throw AlfException(
        'frame-rotation-malformed',
        '`_frame_rotation.$axis` must be a finite JSON number',
      );
    }
    final value = component.toDouble();
    normSquared += value * value;
  }
  // Conjugation by a non-unit quaternion scales by |q|^2 as well as rotating,
  // uniformly across the frame and nowhere else in the file. Refused rather
  // than normalized: normalizing would hide the producer's fault, not fix it.
  if ((normSquared - 1.0).abs() > _frameRotationNormTolerance) {
    throw AlfException(
      'frame-rotation-not-normalized',
      '`_frame_rotation` is not a unit quaternion (|q|^2 = $normSquared)',
    );
  }
}

void _validateGeometryAnchorShapes(Map<String, Object?> metadata) {
  for (final key in <String>['device_position', 'world_centroid_epsg3857']) {
    if (!metadata.containsKey(key)) continue;
    final raw = metadata[key];
    if (key == 'device_position' && raw is Map && raw.isEmpty) continue;
    final components = switch (raw) {
      List<Object?> value when value.length == 3 => value,
      Map<String, Object?> value
          when value.containsKey('x') &&
              value.containsKey('y') &&
              value.containsKey('z') =>
        <Object?>[value['x'], value['y'], value['z']],
      _ => throw AlfException(
        'geometry-anchor-malformed',
        '$key must contain x, y, and z or be a three-element list',
      ),
    };
    if (components.any(
      (Object? component) =>
          component is! num || !component.toDouble().isFinite,
    )) {
      throw AlfException(
        'geometry-anchor-malformed',
        '$key components must be finite JSON numbers',
      );
    }
  }
}

const _decimalTooPrecise = '#too-precise';

String? _canonicalDecimal(String token) {
  final match = RegExp(
    r'^(-?)([0-9]+)(?:\.([0-9]*))?(?:[eE]([+-]?[0-9]+))?$',
  ).firstMatch(token);
  if (match == null) return null;
  final integer = match.group(2)!;
  final fraction = match.group(3) ?? '';
  final totalDigits = integer.length + fraction.length;
  var firstNonZero = -1;
  var lastNonZero = -1;
  for (var i = 0; i < totalDigits; i++) {
    final digit = i < integer.length
        ? integer.codeUnitAt(i)
        : fraction.codeUnitAt(i - integer.length);
    if (digit != 0x30) {
      firstNonZero = firstNonZero < 0 ? i : firstNonZero;
      lastNonZero = i;
    }
  }
  if (firstNonZero < 0) return '0:0';
  final exponentToken = match.group(4) ?? '0';
  final rawExponentDigits =
      exponentToken.startsWith('+') || exponentToken.startsWith('-')
      ? exponentToken.substring(1)
      : exponentToken;
  var firstExponentDigit = 0;
  while (firstExponentDigit < rawExponentDigits.length &&
      rawExponentDigits.codeUnitAt(firstExponentDigit) == 0x30) {
    firstExponentDigit++;
  }
  final exponentDigits = firstExponentDigit == rawExponentDigits.length
      ? '0'
      : rawExponentDigits.substring(firstExponentDigit);
  if (exponentDigits.length > 10) return _decimalTooPrecise;
  final parsedExponent = BigInt.tryParse(
    exponentToken.startsWith('-') ? '-$exponentDigits' : exponentDigits,
  );
  if (parsedExponent == null) return null;
  final significantDigits = lastNonZero - firstNonZero + 1;
  if (significantDigits > 17) return _decimalTooPrecise;
  final digits = String.fromCharCodes(<int>[
    for (var i = firstNonZero; i <= lastNonZero; i++)
      i < integer.length
          ? integer.codeUnitAt(i)
          : fraction.codeUnitAt(i - integer.length),
  ]);
  final trailingZeros = totalDigits - lastNonZero - 1;
  final exponent =
      parsedExponent -
      BigInt.from(fraction.length) +
      BigInt.from(trailingZeros);
  return '${match.group(1) ?? ''}$digits:$exponent';
}

final class _JsonNumberTokens {
  final String source;
  int index = 0;

  _JsonNumberTokens(this.source);

  Map<String, String> topLevelNumbers(Set<String> keys) {
    final result = <String, String>{};
    _whitespace();
    _expect('{');
    _whitespace();
    if (_consume('}')) return result;
    while (true) {
      final key = _string();
      _whitespace();
      _expect(':');
      _whitespace();
      if (keys.contains(key) && _atNumber) {
        result[key] = _number();
      } else {
        _value();
      }
      _whitespace();
      if (_consume('}')) return result;
      _expect(',');
      _whitespace();
    }
  }

  Map<String, List<String>> geometryParameters() {
    final result = <String, List<String>>{};
    _whitespace();
    _expect('{');
    _whitespace();
    if (_consume('}')) return result;
    while (true) {
      final key = _string();
      _whitespace();
      _expect(':');
      _whitespace();
      if (key == '_geom_scale' && _atNumber) {
        result[key] = <String>[_number()];
      } else if (key == '_geom_offset' && _peek == '[') {
        result[key] = _numberArray();
      } else if ((key == 'device_position' ||
              key == 'world_centroid_epsg3857') &&
          (_peek == '[' || _peek == '{')) {
        result[key] = _peek == '[' ? _numberArray() : _anchorObjectNumbers();
      } else if (key == '_frame_rotation' && _peek == '{') {
        result[key] = _rotationObjectNumbers();
      } else {
        _value();
      }
      _whitespace();
      if (_consume('}')) return result;
      _expect(',');
      _whitespace();
    }
  }

  List<String> _anchorObjectNumbers() {
    final byAxis = <String, String>{};
    _expect('{');
    _whitespace();
    if (_consume('}')) return <String>[];
    while (true) {
      final member = _string();
      _whitespace();
      _expect(':');
      _whitespace();
      if ((member == 'x' || member == 'y' || member == 'z') && _atNumber) {
        byAxis[member] = _number();
      } else {
        // Non-axis members are application metadata. The generic finite-number
        // preflight validates them as header-not-json; geometry precision rules
        // apply only to the three values that place the frame.
        _value();
      }
      _whitespace();
      if (_consume('}')) {
        return <String>[
          for (final axis in <String>['x', 'y', 'z']) ?byAxis[axis],
        ];
      }
      _expect(',');
      _whitespace();
    }
  }

  /// The four rotation tokens, in w/x/y/z order.
  ///
  /// Parallel to [_anchorObjectNumbers] but over a different member set, and
  /// without its tolerance for extra members: `_frame_rotation` accepts exactly
  /// four, so anything else is refused by the shape check rather than skipped
  /// here.
  List<String> _rotationObjectNumbers() {
    final byAxis = <String, String>{};
    _expect('{');
    _whitespace();
    if (_consume('}')) return <String>[];
    while (true) {
      final member = _string();
      _whitespace();
      _expect(':');
      _whitespace();
      if ((member == 'w' || member == 'x' || member == 'y' || member == 'z') &&
          _atNumber) {
        byAxis[member] = _number();
      } else {
        _value();
      }
      _whitespace();
      if (_consume('}')) {
        return <String>[
          for (final axis in <String>['w', 'x', 'y', 'z']) ?byAxis[axis],
        ];
      }
      _expect(',');
      _whitespace();
    }
  }

  List<String> _numberArray() {
    final result = <String>[];
    _expect('[');
    _whitespace();
    if (_consume(']')) return result;
    while (true) {
      if (_atNumber) {
        result.add(_number());
      } else {
        _value();
      }
      _whitespace();
      if (_consume(']')) return result;
      _expect(',');
      _whitespace();
    }
  }

  void _value() {
    _whitespace();
    switch (_peek) {
      case '"':
        _string();
      case '{' || '[':
        _compound();
      case 't':
        _literal('true');
      case 'f':
        _literal('false');
      case 'n':
        _literal('null');
      default:
        _number();
    }
  }

  void _compound() {
    var depth = 0;
    var inString = false;
    var escaped = false;
    while (index < source.length) {
      final character = source[index++];
      if (inString) {
        if (escaped) {
          escaped = false;
        } else if (character == r'\') {
          escaped = true;
        } else if (character == '"') {
          inString = false;
        }
        continue;
      }
      switch (character) {
        case '"':
          inString = true;
        case '{' || '[':
          depth++;
        case '}' || ']':
          depth--;
          if (depth == 0) return;
      }
    }
    throw const FormatException('Unterminated JSON collection');
  }

  String _string() {
    final start = index;
    _expect('"');
    while (index < source.length) {
      final character = source[index++];
      if (character == '"') {
        return jsonDecode(source.substring(start, index))! as String;
      }
      if (character == r'\') index++;
    }
    throw const FormatException('Unterminated JSON string');
  }

  String _number() {
    final start = index;
    if (_peek == '-') index++;
    while (_digit) {
      index++;
    }
    if (_peek == '.') {
      index++;
      while (_digit) {
        index++;
      }
    }
    if (_peek == 'e' || _peek == 'E') {
      index++;
      if (_peek == '+' || _peek == '-') index++;
      while (_digit) {
        index++;
      }
    }
    return source.substring(start, index);
  }

  void _literal(String value) {
    if (!source.startsWith(value, index)) {
      throw const FormatException('Malformed JSON literal');
    }
    index += value.length;
  }

  bool get _atNumber => _peek == '-' || _digit;

  bool get _digit {
    final code = index < source.length ? source.codeUnitAt(index) : -1;
    return code >= 0x30 && code <= 0x39;
  }

  String get _peek => index < source.length ? source[index] : '';

  bool _consume(String character) {
    if (_peek != character) return false;
    index++;
    return true;
  }

  void _expect(String character) {
    if (!_consume(character)) {
      throw FormatException('Expected $character at JSON byte $index');
    }
  }

  void _whitespace() {
    while (_peek == ' ' || _peek == '\n' || _peek == '\r' || _peek == '\t') {
      index++;
    }
  }
}

/// Token-only JSON walk that preserves member names before jsonDecode can
/// silently apply its last-value-wins duplicate-key behavior.
final class _JsonDuplicateKeys {
  final String source;
  int index = 0;

  _JsonDuplicateKeys(this.source);

  void validate() {
    _whitespace();
    _value();
    _whitespace();
    if (index != source.length) {
      throw FormatException('Unexpected JSON data at byte $index');
    }
  }

  void _value() {
    _whitespace();
    switch (_peek) {
      case '"':
        _string();
      case '{':
        _object();
      case '[':
        _array();
      case 't':
        _literal('true');
      case 'f':
        _literal('false');
      case 'n':
        _literal('null');
      default:
        _number();
    }
  }

  void _object() {
    final keys = <String>{};
    _expect('{');
    _whitespace();
    if (_consume('}')) return;
    while (true) {
      final key = _string();
      if (!keys.add(key)) {
        throw AlfException(
          'header-key-duplicate',
          'Header contains duplicate JSON object key $key',
        );
      }
      _whitespace();
      _expect(':');
      _value();
      _whitespace();
      if (_consume('}')) return;
      _expect(',');
      _whitespace();
    }
  }

  void _array() {
    _expect('[');
    _whitespace();
    if (_consume(']')) return;
    while (true) {
      _value();
      _whitespace();
      if (_consume(']')) return;
      _expect(',');
      _whitespace();
    }
  }

  String _string() {
    final start = index;
    _expect('"');
    while (index < source.length) {
      final character = source[index++];
      if (character == '"') {
        return jsonDecode(source.substring(start, index))! as String;
      }
      if (character == r'\') index++;
    }
    throw const FormatException('Unterminated JSON string');
  }

  void _number() {
    final start = index;
    if (_peek == '-') index++;
    while (_digit) {
      index++;
    }
    if (_peek == '.') {
      index++;
      while (_digit) {
        index++;
      }
    }
    if (_peek == 'e' || _peek == 'E') {
      index++;
      if (_peek == '+' || _peek == '-') index++;
      while (_digit) {
        index++;
      }
    }
    if (start == index) {
      throw FormatException('Expected JSON value at byte $index');
    }
  }

  void _literal(String value) {
    if (!source.startsWith(value, index)) {
      throw const FormatException('Malformed JSON literal');
    }
    index += value.length;
  }

  bool get _digit {
    final code = index < source.length ? source.codeUnitAt(index) : -1;
    return code >= 0x30 && code <= 0x39;
  }

  String get _peek => index < source.length ? source[index] : '';

  bool _consume(String character) {
    if (_peek != character) return false;
    index++;
    return true;
  }

  void _expect(String character) {
    if (!_consume(character)) {
      throw FormatException('Expected $character at JSON byte $index');
    }
  }

  void _whitespace() {
    while (_peek == ' ' || _peek == '\n' || _peek == '\r' || _peek == '\t') {
      index++;
    }
  }
}

double _adjacentUlp(double value) {
  final magnitude = value.abs();
  if (magnitude == 0) return double.minPositive;
  final below = magnitude - _nextDown(magnitude);
  final aboveValue = _nextUp(magnitude);
  final above = aboveValue - magnitude;
  return above.isFinite ? math.max(below, above) : below;
}

double _nextUp(double value) {
  if (value.isNaN || value == double.infinity) return value;
  if (value == 0) return double.minPositive;
  return _stepFloat(value, upward: true);
}

double _nextDown(double value) {
  if (value.isNaN || value == double.negativeInfinity) return value;
  if (value == 0) return -double.minPositive;
  return _stepFloat(value, upward: false);
}

double _stepFloat(double value, {required bool upward}) {
  final data = ByteData(8)..setFloat64(0, value, Endian.big);
  var high = data.getUint32(0, Endian.big);
  var low = data.getUint32(4, Endian.big);
  if (upward != value.isNegative) {
    if (low == 0xffffffff) {
      low = 0;
      high = (high + 1) & 0xffffffff;
    } else {
      low++;
    }
  } else if (low == 0) {
    low = 0xffffffff;
    high = (high - 1) & 0xffffffff;
  } else {
    low--;
  }
  data
    ..setUint32(0, high, Endian.big)
    ..setUint32(4, low, Endian.big);
  return data.getFloat64(0, Endian.big);
}

Map<String, String> _checkedInstanceMap(Object? raw) {
  if (raw is! Map<String, Object?>) {
    throw const AlfException(
      'instance-map-malformed',
      'instance_map must be a JSON object',
    );
  }
  final result = <String, String>{};
  final canonicalId = RegExp(r'^[1-9][0-9]*$');
  for (final entry in raw.entries) {
    final id = canonicalId.hasMatch(entry.key) ? int.tryParse(entry.key) : null;
    if (id == null ||
        id > 65534 ||
        entry.value is! String ||
        (entry.value! as String).trim().isEmpty) {
      throw AlfException(
        'instance-map-malformed',
        'instance_map entry ${entry.key} is not a canonical id in 1..65534 with a non-empty value',
      );
    }
    result[entry.key] = entry.value! as String;
  }
  return result;
}

/// Write flattened local XYZ coordinates to a complete ALF envelope.
Uint8List writeAlp(
  List<num> xyz, {
  Map<String, Object?> metadata = const <String, Object?>{},
  AlfGeometryCodec codec = AlfGeometryCodec.fp16,
  bool gzip = true,
  bool checkBudget = true,
  String? frameConvention,
  Map<String, num>? frameRotation,
}) {
  final frameFields = _writerFrameFields(codec, frameConvention, frameRotation);
  if (xyz.length % 3 != 0) {
    throw const AlfException(
      'payload-not-whole-points',
      'XYZ must contain complete triples',
    );
  }
  final coordinateCount = xyz.length;
  final sourceLo = List<double>.filled(3, double.infinity);
  final sourceHi = List<double>.filled(3, double.negativeInfinity);
  var largest = 0.0;
  for (var i = 0; i < coordinateCount; i++) {
    final value = xyz[i];
    final narrowed = value.toDouble();
    // Both codecs can now carry an invalid return: binary16 has NaN natively,
    // and int16 reserves `_int16NoReturn` (-32768) for it (spec/format.md §5.2).
    //
    // What int16 still refuses is a PARTLY non-finite point, matching the
    // Python reference: the sentinel marks a whole point as having no return,
    // so half a coordinate has no encoding and no meaning. Refusing beats
    // inventing a rule for data no sensor produces.
    if (!narrowed.isFinite) {
      if (codec == AlfGeometryCodec.int16 &&
          !_alfPointIsWhollyNonFinite(xyz, i - (i % 3))) {
        throw const AlfException(
          'writer-input-invalid',
          'codec int16 found a point with SOME but not all components non-finite. '
              'An invalid return is a whole point — a partial one has no interpretation, '
              'and encoding it would have to guess. Make each such point entirely '
              'non-finite, or filter it out.',
        );
      }
      continue;
    }
    // JavaScript represents infinity as an `int` too. Classify non-finite
    // content above before checking finite integers for lossy narrowing.
    if (value is int && narrowed.toInt() != value) {
      throw const AlfException(
        'geom-number-inexact',
        'Coordinate integer cannot be represented exactly in binary64',
      );
    }
    // FINITE components only. `math.max` against NaN is NaN, and every
    // comparison against NaN is false — so one invalid return silently disabled
    // the fp16 budget guard below.
    final axis = i % 3;
    sourceLo[axis] = math.min(sourceLo[axis], narrowed);
    sourceHi[axis] = math.max(sourceHi[axis], narrowed);
    largest = math.max(largest, narrowed.abs());
  }
  final header = Map<String, Object?>.from(metadata)
    ..['_total_points'] = coordinateCount ~/ 3
    // The explicit arguments OWN these two keys. Metadata read back from
    // another .alp carries the previous file's declaration, and merging over it
    // would leave a stale `_frame_rotation` beside a world-aligned write, or a
    // `_frame_convention` on a v1 envelope — headers this writer's own reader
    // refuses. Cleared here, then set from `frameFields` below.
    ..remove('_frame_convention')
    ..remove('_frame_rotation');
  _validateAndNormalizeGeometryAnchors(header);
  late final ByteData payload;
  final decodedLo = List<double>.filled(3, double.infinity);
  final decodedHi = List<double>.filled(3, double.negativeInfinity);
  int version;
  if (codec == AlfGeometryCodec.fp16) {
    const reserved = <String>[
      '_geom_codec',
      '_geom_scale',
      '_geom_offset',
      '_geom_max_err_m',
    ];
    for (final key in reserved) {
      header.remove(key);
    }
    if (checkBudget && largest >= 512) {
      throw AlfException(
        'geometry-budget-exceeded',
        'float16 coordinate reaches $largest m; localize or tile the frame',
      );
    }
    payload = ByteData(coordinateCount * 2);
    for (var i = 0; i < coordinateCount; i++) {
      final value = xyz[i].toDouble();
      final bits = doubleToHalf(value);
      final back = halfToDouble(bits);
      // Only where a FINITE input overflowed. A non-finite input is an invalid
      // return being stored faithfully — binary16 round-trips all three — and
      // this check exists for a representability failure, not for the marker.
      if (value.isFinite && !back.isFinite) {
        throw AlfException(
          'coordinate-overflow',
          'Coordinate $value overflows float16',
        );
      }
      payload.setUint16(i * 2, bits, Endian.little);
      // The box is the FINITE extent: an invalid return has no position, and a
      // box string containing NaN or Infinity is unparseable to every consumer
      // — the server turns this header straight into item-metadata bounds.
      if (back.isFinite) {
        final axis = i % 3;
        decodedLo[axis] = math.min(decodedLo[axis], back);
        decodedHi[axis] = math.max(decodedHi[axis], back);
      }
    }
    version = 1;
  } else {
    final offset = List<double>.filled(3, 0);
    // `sourceLo`/`sourceHi` only ever saw FINITE components, so a frame of
    // nothing but invalid returns leaves them at their Infinity/-Infinity
    // seeds. Their midpoint is NaN, which would become the header origin and
    // make every decoded coordinate NaN for the wrong reason. A fully-occluded
    // sweep is legal content; every word is the sentinel, so any finite origin
    // decodes identically and zero is the same one the empty-frame case uses.
    final hasFiniteExtent = sourceLo[0].isFinite;
    if (coordinateCount != 0 && hasFiniteExtent) {
      for (var axis = 0; axis < 3; axis++) {
        final lo = sourceLo[axis];
        final hi = sourceHi[axis];
        final span = hi - lo;
        // Opposite finite binary64 extremes have an infinite difference even
        // though their midpoint is representable. Split the operands only for
        // that overflow case to preserve the usual midpoint rounding.
        offset[axis] = span.isFinite ? lo + span / 2 : lo / 2 + hi / 2;
      }
    }
    // REAL coordinates only. `math.max` with NaN yields NaN, which then makes
    // the derived scale NaN and every stored word garbage — the invalid returns
    // must be skipped when sizing the grid (spec/format.md §5.2).
    var extent = 0.0;
    for (var i = 0; i < coordinateCount; i += 3) {
      if (!_alfPointIsFinite(xyz, i)) continue;
      for (var component = 0; component < 3; component++) {
        extent = math.max(
          extent,
          (xyz[i + component].toDouble() - offset[component]).abs(),
        );
      }
    }
    var scale = math.max(extent / _int16Max, _minGeometryScaleM);
    // Division can round the scale upward just enough that reconstructing the
    // int16 endpoint overflows, even though the source extent itself is finite.
    if (!(scale * _int16Max).isFinite) scale = _nextDown(scale);
    payload = ByteData(coordinateCount * 2);
    var measured = 0.0;
    for (var i = 0; i < coordinateCount; i += 3) {
      if (!_alfPointIsFinite(xyz, i)) {
        // Written EXPLICITLY, never left to the cast: rounding a NaN and
        // narrowing it to int16 is platform-defined, so the marker would become
        // an arbitrary coordinate that looks like data.
        payload.setInt16(i * 2, _int16NoReturn, Endian.little);
        payload.setInt16((i + 1) * 2, _int16NoReturn, Endian.little);
        payload.setInt16((i + 2) * 2, _int16NoReturn, Endian.little);
        continue;
      }
      for (var component = 0; component < 3; component++) {
        final value = xyz[i + component].toDouble();
        final stored = _roundTiesToEven(
          (value - offset[component]) / scale,
        ).clamp(-_int16Max, _int16Max);
        payload.setInt16((i + component) * 2, stored, Endian.little);
        final back = stored * scale + offset[component];
        if (!back.isFinite) {
          throw const AlfException(
            'coordinate-overflow',
            'Fixed-point reconstruction produced a non-finite coordinate',
          );
        }
        decodedLo[component] = math.min(decodedLo[component], back);
        decodedHi[component] = math.max(decodedHi[component], back);
        measured = math.max(measured, (back - value).abs());
      }
    }
    measured = coordinateCount == 0 ? 0 : math.max(measured, scale / 2);
    if (checkBudget && measured >= 0.25) {
      throw AlfException(
        'geometry-budget-exceeded',
        'int16 reconstruction error $measured m exceeds the 0.25 m budget',
      );
    }
    header['_geom_codec'] = 'int16';
    header['_geom_scale'] = scale;
    header['_geom_offset'] = offset;
    header['_geom_max_err_m'] = measured;
    header.addAll(frameFields);
    version = 3;
  }
  _setDecodedBox(header, coordinateCount, decodedLo, decodedHi);
  return _writeEnvelope(
    version,
    header,
    payload.buffer.asUint8List(),
    gzip: gzip,
    geometryHeader: true,
  );
}

/// Normalize raw intensity into uint8 and write an `.ali` envelope.
Uint8List writeAli(
  List<num> intensity, {
  required num lo,
  required num hi,
  List<int>? deciles,
  Map<String, Object?> metadata = const <String, Object?>{},
  bool gzip = true,
}) {
  final lower = _exactIntensityNumber(lo, 'lower bound');
  final upper = _exactIntensityNumber(hi, 'upper bound');
  if (!lower.isFinite || !upper.isFinite || upper < lower) {
    throw const AlfException(
      'intensity-range-malformed',
      'Intensity bounds must be finite and ordered',
    );
  }
  final payload = Uint8List(intensity.length);
  for (var i = 0; i < intensity.length; i++) {
    final value = _exactIntensityNumber(intensity[i], 'sample');
    if (!value.isFinite || value < lower || value > upper) {
      throw AlfException(
        'intensity-out-of-range',
        'Intensity $value is outside [$lower, $upper]',
      );
    }
    if (upper == lower) {
      payload[i] = 0;
      continue;
    }
    final span = upper - lower;
    final normalized = span.isFinite
        ? (value - lower) / span
        : (value / 2 - lower / 2) / (upper / 2 - lower / 2);
    if (!normalized.isFinite) {
      throw const AlfException(
        'intensity-range-malformed',
        'Intensity normalization produced a non-finite value',
      );
    }
    payload[i] = (normalized * 255).clamp(0, 255).toInt();
  }
  final header = Map<String, Object?>.from(metadata)
    ..['_total_points'] = payload.length
    ..['_min_intensity'] = 0
    ..['_max_intensity'] = 255;
  final inheritedDeciles = deciles == null ? _metadataDeciles(header) : null;
  header['_intensities_every_10_percent'] = _intensityDeciles(
    payload,
    deciles ?? inheritedDeciles,
  );
  return _writeEnvelope(1, header, payload, gzip: gzip);
}

/// Write v1 semantic labels or v2 interleaved category/instance labels.
Uint8List writeAll(
  List<num> categories, {
  List<num>? instanceIds,
  Map<String, String>? instanceMap,
  Map<String, Object?> metadata = const <String, Object?>{},
  bool gzip = true,
}) {
  final cats = _checkedIntegerStream(categories, 0, 255, 'category');
  final header = Map<String, Object?>.from(metadata)
    ..['_total_points'] = cats.length;
  header.remove('_format_version');
  if (instanceIds == null) {
    if (instanceMap != null) {
      throw const AlfException(
        'instance-map-malformed',
        'instanceMap requires instanceIds',
      );
    }
    header.remove('instance_map');
    return _writeEnvelope(1, header, Uint8List.fromList(cats), gzip: gzip);
  }
  final ids = _checkedIntegerStream(instanceIds, 0, 65534, 'instance id');
  if (ids.length != cats.length) {
    throw const AlfException(
      'payload-length-mismatch',
      'categories and instanceIds must have the same length',
    );
  }
  final inherited = header['instance_map'];
  final map = _checkedInstanceMap(
    instanceMap ??
        (header.containsKey('instance_map') ? inherited : <String, Object?>{}),
  );
  final missing = ids
      .where((int id) => id != 0 && !map.containsKey('$id'))
      .toSet()
      .toList();
  if (missing.isNotEmpty) {
    throw AlfException(
      'instance-map-incomplete',
      'instanceMap is missing $missing',
    );
  }
  final payload = Uint8List(cats.length * 3);
  for (var i = 0; i < cats.length; i++) {
    payload[i * 3] = cats[i];
    payload[i * 3 + 1] = ids[i] & 0xff;
    payload[i * 3 + 2] = ids[i] >> 8;
  }
  header['_format_version'] = 2;
  header['instance_map'] = map;
  return _writeEnvelope(2, header, payload, gzip: gzip);
}

/// Write palette indexes and their `rrggbb` palette to an `.alc` envelope.
Uint8List writeAlc(
  List<num> paletteIndices,
  List<Object> palette, {
  Map<String, Object?> metadata = const <String, Object?>{},
  bool gzip = true,
}) {
  final colors = palette.map(_normalizeColor).toList(growable: false);
  if (colors.isEmpty || colors.length > 255) {
    throw const AlfException(
      'palette-entry-malformed',
      '.alc needs 1..255 palette entries',
    );
  }
  final indices = _checkedIntegerStream(
    paletteIndices,
    0,
    colors.length,
    'palette index',
  );
  final header = Map<String, Object?>.from(metadata)
    ..['_total_points'] = indices.length
    ..['palette'] = colors;
  return _writeEnvelope(1, header, Uint8List.fromList(indices), gzip: gzip);
}

/// Write a uint32 decimation sidecar mapping each `.alp` point to its source.
Uint8List writeAlx(
  List<num> keptIndices, {
  required int alpTotalPoints,
  required num sourceTotalPoints,
  Map<String, Object?> metadata = const <String, Object?>{},
  bool gzip = true,
}) {
  final indices = _checkedIntegerStream(
    keptIndices,
    0,
    0xffffffff,
    'kept index',
  );
  if (alpTotalPoints != indices.length) {
    throw const AlfException(
      'payload-length-mismatch',
      'keptIndices must match alpTotalPoints',
    );
  }
  final sourceTotalAsDouble = sourceTotalPoints.toDouble();
  if (!sourceTotalAsDouble.isFinite ||
      sourceTotalAsDouble != sourceTotalAsDouble.truncateToDouble() ||
      BigInt.from(sourceTotalAsDouble) != BigInt.from(sourceTotalPoints)) {
    throw const AlfException(
      'total-points-not-an-integer',
      'sourceTotalPoints must be exactly representable as finite binary64',
    );
  }
  if (sourceTotalPoints < 0 ||
      indices.any((int value) => value >= sourceTotalPoints)) {
    throw const AlfException(
      'kept-index-out-of-range',
      'kept index is outside the source cloud',
    );
  }
  if (indices.toSet().length != indices.length) {
    throw const AlfException(
      'kept-index-duplicate',
      'kept indices must be unique',
    );
  }
  final payload = ByteData(indices.length * 4);
  for (var i = 0; i < indices.length; i++) {
    payload.setUint32(i * 4, indices[i], Endian.little);
  }
  final header = Map<String, Object?>.from(metadata)
    ..['_total_points'] = indices.length
    ..['_source_total_points'] = sourceTotalPoints;
  return _writeEnvelope(
    1,
    header,
    payload.buffer.asUint8List(),
    gzip: gzip,
    exactIntegerKeys: const <String>{'_source_total_points'},
  );
}

Uint8List _writeEnvelope(
  int version,
  Map<String, Object?> metadata,
  Uint8List payload, {
  required bool gzip,
  bool geometryHeader = false,
  Set<String> exactIntegerKeys = const <String>{},
}) {
  _validateWriterJsonDepth(metadata);
  late final String headerText;
  try {
    if (geometryHeader) {
      headerText = _encodeAlpHeader(metadata, includeGrid: version == 3);
    } else if (exactIntegerKeys.isNotEmpty) {
      headerText = _encodeExactIntegerHeader(metadata, exactIntegerKeys);
    } else {
      headerText = jsonEncode(metadata);
    }
  } catch (error) {
    throw AlfException(
      'header-not-json',
      'Metadata is not JSON-encodable: $error',
    );
  }
  // `jsonEncode` may expand custom objects through `toJson`. Validate the
  // structure actually destined for the wire so the paired reader accepts it.
  _preflightJson(headerText, validateNumbers: true);
  final header = Uint8List.fromList(utf8.encode(headerText));
  if (header.length > maxAlfHeaderBytes) {
    throw const AlfException(
      'header-too-large',
      'Serialized header exceeds the ALF limit',
    );
  }
  validateWritableEnvelopeSize(
    headerLength: header.length,
    payloadLength: payload.lengthInBytes,
  );
  final builder = BytesBuilder(copy: false)
    ..addByte(version)
    ..add(
      (ByteData(
        4,
      )..setUint32(0, header.length, Endian.little)).buffer.asUint8List(),
    )
    ..add(header)
    ..add(payload);
  final envelope = builder.takeBytes();
  return gzip ? encodeDeterministicGzip(envelope) : envelope;
}

String _encodeExactIntegerHeader(
  Map<String, Object?> metadata,
  Set<String> exactIntegerKeys,
) {
  return '{${metadata.entries.map((MapEntry<String, Object?> entry) {
    final value = entry.value;
    final encoded = exactIntegerKeys.contains(entry.key) && value is num ? BigInt.from(value).toString() : jsonEncode(value);
    return '${jsonEncode(entry.key)}:$encoded';
  }).join(',')}}';
}

void _validateWriterJsonDepth(Map<String, Object?> metadata) {
  final stack = <({Object? value, int depth, bool leaving})>[
    (value: metadata, depth: 1, leaving: false),
  ];
  final active = Set<Object>.identity();
  while (stack.isNotEmpty) {
    final frame = stack.removeLast();
    final value = frame.value;
    if (value is! Map && value is! List) continue;
    if (frame.leaving) {
      active.remove(value!);
      continue;
    }
    if (frame.depth > _maxJsonDepth) {
      throw const AlfException(
        'header-nesting-too-deep',
        'Metadata exceeds the maximum JSON nesting depth',
      );
    }
    if (!active.add(value!)) {
      throw const AlfException(
        'header-not-json',
        'Metadata contains a circular JSON collection',
      );
    }
    stack.add((value: value, depth: frame.depth, leaving: true));
    final children = value is Map ? value.values : (value as List);
    for (final child in children) {
      stack.add((value: child, depth: frame.depth + 1, leaving: false));
    }
  }
}

String _encodeAlpHeader(
  Map<String, Object?> metadata, {
  required bool includeGrid,
}) {
  String geometryNumber(Object? value) {
    if (value is double &&
        value.isFinite &&
        value == value.truncateToDouble() &&
        !value.isSafeInteger) {
      return value.toStringAsExponential();
    }
    return jsonEncode(value);
  }

  String anchorObject(Object? value) {
    if (value is! Map<String, Object?>) return jsonEncode(value);
    return '{${value.entries.map((MapEntry<String, Object?> entry) {
      final encoded = switch (entry.key) {
        'x' || 'y' || 'z' => geometryNumber(entry.value),
        _ => jsonEncode(entry.value),
      };
      return '${jsonEncode(entry.key)}:$encoded';
    }).join(',')}}';
  }

  return '{${metadata.entries.map((MapEntry<String, Object?> entry) {
    var encoded = jsonEncode(entry.value);
    if (entry.key == 'device_position' || entry.key == 'world_centroid_epsg3857') {
      encoded = anchorObject(entry.value);
    }
    if (includeGrid && entry.key == '_geom_scale') {
      encoded = geometryNumber(entry.value);
    }
    if (includeGrid && entry.key == '_geom_offset' && entry.value is List<Object?>) {
      encoded = '[${(entry.value! as List<Object?>).map(geometryNumber).join(',')}]';
    }
    return '${jsonEncode(entry.key)}:$encoded';
  }).join(',')}}';
}

extension on double {
  bool get isSafeInteger => abs() <= 9007199254740991;
}

List<int> _checkedIntegerStream(
  List<num> values,
  int minimum,
  int maximum,
  String name,
) {
  final result = <int>[];
  for (final raw in values) {
    final value = raw.toDouble();
    if (!value.isFinite ||
        value != value.truncateToDouble() ||
        value < minimum ||
        value > maximum) {
      throw AlfException(
        'integer-stream-malformed',
        '$name $raw is not a whole value in $minimum..$maximum',
      );
    }
    result.add(raw.toInt());
  }
  return result;
}

String _normalizeColor(Object value) {
  if (value is String) {
    final color = value.startsWith('#') ? value.substring(1) : value;
    if (RegExp(r'^[0-9a-fA-F]{6}$').hasMatch(color)) return color.toLowerCase();
  } else if (value is List<int> &&
      value.length == 3 &&
      value.every((int component) => component >= 0 && component <= 255)) {
    return value
        .map((int component) => component.toRadixString(16).padLeft(2, '0'))
        .join();
  }
  throw AlfException(
    'palette-entry-malformed',
    'Palette entry $value is not rrggbb or an RGB triple',
  );
}

List<int>? _metadataDeciles(Map<String, Object?> header) {
  if (!header.containsKey('_intensities_every_10_percent')) return null;
  final raw = header['_intensities_every_10_percent'];
  if (raw is! List<Object?> || !raw.every((Object? value) => value is int)) {
    throw const AlfException(
      'intensity-deciles-malformed',
      'Inherited intensity deciles must be a list of integers',
    );
  }
  return raw.cast<int>();
}

void _validateAndNormalizeGeometryAnchors(Map<String, Object?> header) {
  for (final key in <String>['device_position', 'world_centroid_epsg3857']) {
    if (!header.containsKey(key)) continue;
    final raw = header[key];
    if (key == 'device_position' && raw is Map && raw.isEmpty) continue;
    late final List<Object?> components;
    final preserved = <String, Object?>{};
    if (raw is Map<String, Object?> &&
        raw.containsKey('x') &&
        raw.containsKey('y') &&
        raw.containsKey('z')) {
      components = <Object?>[raw['x'], raw['y'], raw['z']];
      for (final MapEntry<String, Object?> entry in raw.entries) {
        if (entry.key != 'x' && entry.key != 'y' && entry.key != 'z') {
          preserved[entry.key] = entry.value;
        }
      }
    } else if (raw is List<Object?> && raw.length == 3) {
      components = raw;
    } else {
      throw AlfException(
        'geometry-anchor-malformed',
        '$key must contain x, y, and z or be a three-element list',
      );
    }
    final normalized = <double>[];
    for (final component in components) {
      if (component is! num || !component.toDouble().isFinite) {
        throw AlfException(
          'geometry-anchor-malformed',
          '$key components must be finite JSON numbers',
        );
      }
      if (component is int && component.toDouble().toInt() != component) {
        throw AlfException(
          'geometry-anchor-malformed',
          '$key contains an integer binary64 cannot represent exactly',
        );
      }
      normalized.add(component.toDouble());
    }
    header[key] = <String, Object?>{
      ...preserved,
      'x': normalized[0],
      'y': normalized[1],
      'z': normalized[2],
    };
  }
}

List<int> _intensityDeciles(Uint8List values, List<int>? supplied) {
  if (supplied != null) {
    if (supplied.length != 9 ||
        supplied.any((int value) => value < 0 || value > 255)) {
      throw const AlfException(
        'intensity-deciles-malformed',
        'Intensity deciles must be nine uint8 values',
      );
    }
    for (var i = 1; i < supplied.length; i++) {
      if (supplied[i] < supplied[i - 1]) {
        throw const AlfException(
          'intensity-deciles-malformed',
          'Intensity deciles must be non-decreasing',
        );
      }
    }
    return List<int>.from(supplied);
  }
  if (values.isEmpty) return List<int>.filled(9, 0);
  final ordered = values.toList()..sort();
  return <int>[
    for (var i = 1; i < 10; i++)
      ordered[math.min((ordered.length / 10 * i).toInt(), ordered.length - 1)],
  ];
}

int _roundTiesToEven(double value) {
  final floor = value.floor();
  final fraction = value - floor;
  if (fraction < 0.5) return floor;
  if (fraction > 0.5) return floor + 1;
  return floor.isEven ? floor : floor + 1;
}

double _exactIntensityNumber(num value, String role) {
  final narrowed = value.toDouble();
  if (value is int && (!narrowed.isFinite || narrowed.toInt() != value)) {
    throw AlfException(
      'intensity-number-inexact',
      'Intensity $role integer cannot be represented exactly in binary64',
    );
  }
  return narrowed;
}

void _setDecodedBox(
  Map<String, Object?> header,
  int coordinateCount,
  List<double> lo,
  List<double> hi,
) {
  // A frame with no finite component on some axis has no extent to publish
  // there, so it publishes NO box rather than one holding NaN or Infinity.
  final hasExtent =
      coordinateCount != 0 &&
      lo.every((final double value) => value.isFinite) &&
      hi.every((final double value) => value.isFinite);
  if (!hasExtent) {
    header.remove('box');
    header.remove('_box_frame');
    return;
  }
  header['box'] = '${lo[0]}:${lo[1]}:${lo[2]}:${hi[0]}:${hi[1]}:${hi[2]}';
  header['_box_frame'] = 'local';
}
