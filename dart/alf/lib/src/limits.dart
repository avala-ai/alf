import 'error.dart';

/// Maximum serialized JSON header accepted by the ALF reader.
const int maxAlfHeaderBytes = 10000000;

/// Maximum complete, decompressed ALF envelope accepted by the reader.
const int maxAlfEnvelopeBytes = 512 * 1024 * 1024;

/// Refuse a writer result that its paired reader would reject for size.
///
/// This lives in an internal library so tests can exercise the boundary using
/// lengths alone instead of allocating a hostile 512 MiB payload.
void validateWritableEnvelopeSize({
  required int headerLength,
  required int payloadLength,
}) {
  if (headerLength < 0 || payloadLength < 0) {
    throw ArgumentError('ALF envelope lengths must be non-negative');
  }
  const prefixLength = 5;
  if (headerLength > maxAlfEnvelopeBytes - prefixLength ||
      payloadLength > maxAlfEnvelopeBytes - prefixLength - headerLength) {
    throw const AlfException(
      'output-too-large',
      'Serialized ALF envelope exceeds the 512 MiB reader limit',
    );
  }
}
