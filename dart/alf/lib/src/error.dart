/// A malformed ALF file or invalid ALF writer input.
final class AlfException implements Exception {
  /// Stable rule identifier used by the cross-language conformance suite.
  final String refusal;

  /// Human-readable diagnostic with the file-specific context.
  final String message;

  const AlfException(this.refusal, this.message);

  @override
  String toString() => message;
}
