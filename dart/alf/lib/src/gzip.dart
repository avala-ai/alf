import 'dart:typed_data';

import 'package:archive/archive.dart';

import 'error.dart';
import 'limits.dart';

const _gzipSignature = 0x8b1f;

/// Compressed input may exceed the caller's decoded budget by this amount.
const gzipInputOverheadAllowance = 1024 * 1024;

/// Decode one complete gzip member within [maxDecompressedBytes].
///
/// Validates the header, CRC32 and ISIZE, and rejects additional members;
/// trailing zero padding remains allowed. Compressed input is also bounded by
/// [maxDecompressedBytes] plus [gzipInputOverheadAllowance]. The returned bytes
/// still require channel-specific ALF envelope and payload validation.
Uint8List decodeSingleGzipMember(
  Uint8List bytes, {
  int maxDecompressedBytes = maxAlfEnvelopeBytes,
}) {
  if (maxDecompressedBytes < 0) {
    throw ArgumentError.value(
      maxDecompressedBytes,
      'maxDecompressedBytes',
      'must be non-negative',
    );
  }
  if (bytes.lengthInBytes > maxDecompressedBytes + gzipInputOverheadAllowance) {
    throw AlfException(
      'gzip-malformed',
      'Compressed ALF input is ${bytes.lengthInBytes} bytes, past the conservative input limit',
    );
  }
  final input = InputMemoryStream(bytes, byteOrder: ByteOrder.littleEndian);
  final output = _CappedMemoryOutput(maxDecompressedBytes);
  try {
    _readMemberHeader(input, bytes);
    Inflate.stream(input, output: output);
    if (input.length < 8) {
      throw const AlfException('gzip-malformed', 'Truncated gzip trailer');
    }
    final expectedCrc = input.readUint32();
    final expectedSize = input.readUint32();
    final decoded = output.getBytes();
    final actualCrc = getCrc32(decoded);
    if (actualCrc != expectedCrc) {
      throw AlfException(
        'gzip-malformed',
        'Gzip CRC32 mismatch: trailer says $expectedCrc, decoded bytes give $actualCrc',
      );
    }
    if (decoded.lengthInBytes & 0xffffffff != expectedSize) {
      throw AlfException(
        'gzip-malformed',
        'Gzip size mismatch: trailer says $expectedSize, decoded ${decoded.lengthInBytes} bytes',
      );
    }
    while (!input.isEOS && input.peekBytes(1).readByte() == 0) {
      input.skip(1); // RFC 1952 permits trailing zero padding.
    }
    if (!input.isEOS) {
      throw const AlfException(
        'multiple-gzip-members',
        'Compressed stream holds more than one gzip member',
      );
    }
    return decoded;
  } on AlfException {
    rethrow;
  } catch (error) {
    throw AlfException('gzip-malformed', 'Corrupt gzip stream: $error');
  }
}

/// Validate one gzip member's header and leave [input] positioned at its
/// deflate stream.
///
/// Shared by the full decoder and the prefix decoder so the two cannot drift
/// on which framing they accept — a prefix read that was laxer here would be a
/// second, weaker parser for the same bytes.
void _readMemberHeader(InputMemoryStream input, Uint8List bytes) {
  if (input.length < 10 || input.readUint16() != _gzipSignature) {
    throw const AlfException('gzip-malformed', 'Invalid gzip signature');
  }
  if (input.readByte() != 8) {
    throw const AlfException(
      'gzip-malformed',
      'Unsupported gzip compression method',
    );
  }
  final flags = input.readByte();
  if ((flags & 0xe0) != 0) {
    throw const AlfException('gzip-malformed', 'Gzip reserved flags are set');
  }
  input.skip(6); // mtime, extra flags, OS
  if ((flags & 0x04) != 0) {
    if (input.length < 2) {
      throw const AlfException('gzip-malformed', 'Truncated gzip extra field');
    }
    final length = input.readUint16();
    if (input.length < length) {
      throw const AlfException('gzip-malformed', 'Truncated gzip extra field');
    }
    input.skip(length);
  }
  if ((flags & 0x08) != 0) _skipZeroTerminated(input, 'filename');
  if ((flags & 0x10) != 0) _skipZeroTerminated(input, 'comment');
  if ((flags & 0x02) != 0) {
    if (input.length < 2) {
      throw const AlfException(
        'gzip-malformed',
        'Truncated gzip header checksum',
      );
    }
    final headerEnd = input.position;
    final expectedHeaderCrc = input.readUint16();
    final actualHeaderCrc =
        getCrc32(Uint8List.sublistView(bytes, 0, headerEnd)) & 0xffff;
    if (actualHeaderCrc != expectedHeaderCrc) {
      throw AlfException(
        'gzip-malformed',
        'Gzip header CRC16 mismatch: header says $expectedHeaderCrc, decoded header gives $actualHeaderCrc',
      );
    }
  }
}

/// Inflate only as much of one gzip member as the caller needs.
///
/// For the header-only read: an ALF header is a few hundred bytes at the front
/// of a member whose payload may be hundreds of megabytes, and
/// [decodeSingleGzipMember] has to reach the trailer before it can return
/// anything at all. This stops as soon as [atLeast] bytes exist.
///
/// WHAT IT GIVES UP, and it is not small: the member's CRC32 and ISIZE go
/// unverified, because checking either means inflating to the end — the exact
/// cost this exists to avoid. Trailing-member detection goes with them. So this
/// establishes that the bytes it returned decompressed correctly and NOTHING
/// about the bytes it did not read. Callers that must prove a file intact use
/// [decodeSingleGzipMember].
///
/// Returns fewer than [atLeast] bytes when the member itself is shorter, which
/// is not an error here — the envelope parser is what decides whether what came
/// back is a whole header.
Uint8List decodeGzipPrefix(
  Uint8List bytes, {
  required int atLeast,
  int maxDecompressedBytes = maxAlfEnvelopeBytes,
}) {
  if (atLeast < 0) {
    throw ArgumentError.value(atLeast, 'atLeast', 'must be non-negative');
  }
  // Asking for nothing gets nothing. The stop condition is checked AFTER a
  // write, so without this a zero-byte request would still inflate one block.
  if (atLeast == 0) return Uint8List(0);
  // The SAME compressed-input ceiling the full decoder applies, and for a
  // reason specific to this function: bounding a prefix read's OUTPUT does not
  // bound its work. DEFLATE can emit nothing at all for arbitrarily much input
  // — a member of empty non-final stored blocks costs five bytes each and
  // produces zero — so an output target is never reached and the inflater walks
  // the whole file. Measured here at ~10 ms per compressed MiB, which made a
  // 100 MiB member take a full second to answer a question its first 35 bytes
  // settled. Python's header reader caps consumed compressed input for exactly
  // this case and TypeScript's caps the input length; this is that same bound,
  // so the three readers refuse the same files.
  if (bytes.lengthInBytes > maxDecompressedBytes + gzipInputOverheadAllowance) {
    throw AlfException(
      'gzip-malformed',
      'Compressed ALF input is ${bytes.lengthInBytes} bytes, past the conservative input limit',
    );
  }
  final input = InputMemoryStream(bytes, byteOrder: ByteOrder.littleEndian);
  // Still capped: a prefix read is a bounded read, and the bound that matters
  // for a hostile file is the one on what a short request can be made to
  // allocate. `atLeast` is the target, `maxDecompressedBytes` is the ceiling.
  final output = _PrefixOutput(atLeast, maxDecompressedBytes);
  try {
    _readMemberHeader(input, bytes);
    Inflate.stream(input, output: output);
  } on _EnoughBytes {
    // The only non-error way out of a bounded inflate. `Inflate.stream` has no
    // "stop here" input, so the output signals completion by refusing to take
    // more — the alternative being to decompress the whole member and discard
    // it, which is the cost this function exists to avoid.
  } on AlfException {
    rethrow;
  } catch (error) {
    throw AlfException('gzip-malformed', 'Corrupt gzip stream: $error');
  }
  return output.getBytes();
}

/// Accumulates until [target] bytes exist, then stops the inflater by throwing.
final class _PrefixOutput extends OutputStream {
  final int target;
  final int limit;
  final OutputMemoryStream _output = OutputMemoryStream();

  _PrefixOutput(this.target, this.limit)
    : super(byteOrder: ByteOrder.littleEndian);

  @override
  int get length => _output.length;

  @override
  Uint8List getBytes() => _output.getBytes();

  void _afterWrite() {
    if (_output.length > limit) {
      throw AlfException(
        'gzip-malformed',
        'Compressed ALF expands past the $limit-byte decompressed limit',
      );
    }
    if (_output.length >= target) throw const _EnoughBytes();
  }

  @override
  void clear() => _output.clear();

  @override
  void flush() => _output.flush();

  @override
  void writeByte(int value) {
    _output.writeByte(value);
    _afterWrite();
  }

  @override
  void writeBytes(List<int> bytes, {int? length}) {
    _output.writeBytes(bytes, length: length ?? bytes.length);
    _afterWrite();
  }

  @override
  void writeStream(InputStream stream) {
    _output.writeStream(stream);
    _afterWrite();
  }

  @override
  Uint8List subset(int start, [int? end]) => _output.subset(start, end);
}

/// Control flow, not a fault: [_PrefixOutput] has what it was asked for.
final class _EnoughBytes implements Exception {
  const _EnoughBytes();
}

final class _CappedMemoryOutput extends OutputStream {
  final int limit;
  final OutputMemoryStream _output = OutputMemoryStream();

  _CappedMemoryOutput(this.limit) : super(byteOrder: ByteOrder.littleEndian);

  @override
  int get length => _output.length;

  void _reserve(int count) {
    if (count < 0 || length > limit - count) {
      throw AlfException(
        'gzip-malformed',
        'Compressed ALF expands past the $limit-byte decompressed limit',
      );
    }
  }

  @override
  void clear() => _output.clear();

  @override
  void flush() => _output.flush();

  @override
  void writeByte(int value) {
    _reserve(1);
    _output.writeByte(value);
  }

  @override
  void writeBytes(List<int> bytes, {int? length}) {
    final count = length ?? bytes.length;
    _reserve(count);
    _output.writeBytes(bytes, length: count);
  }

  @override
  void writeStream(InputStream stream) {
    _reserve(stream.length);
    _output.writeStream(stream);
  }

  @override
  Uint8List subset(int start, [int? end]) => _output.subset(start, end);
}

void _skipZeroTerminated(InputStream input, String field) {
  while (!input.isEOS) {
    if (input.readByte() == 0) return;
  }
  throw AlfException('gzip-malformed', 'Truncated gzip $field');
}

Uint8List encodeDeterministicGzip(Uint8List bytes) {
  final encoded = GZipEncoderWeb().encodeBytes(bytes);
  // archive's pure-Dart encoder stamps DateTime.now(). ALF gzip bytes must be a
  // pure function of their content, matching Python's mtime=0 envelope.
  encoded.fillRange(4, 8, 0);
  return encoded;
}
