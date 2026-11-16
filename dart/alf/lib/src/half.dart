import 'dart:typed_data';

double halfToDouble(int bits) {
  final sign = (bits & 0x8000) << 16;
  var exponent = (bits >> 10) & 0x1f;
  var mantissa = bits & 0x03ff;
  int floatBits;
  if (exponent == 0) {
    if (mantissa == 0) {
      floatBits = sign;
    } else {
      exponent = 1;
      while ((mantissa & 0x0400) == 0) {
        mantissa <<= 1;
        exponent--;
      }
      mantissa &= 0x03ff;
      floatBits = sign | ((exponent + 112) << 23) | (mantissa << 13);
    }
  } else if (exponent == 0x1f) {
    floatBits = sign | 0x7f800000 | (mantissa << 13);
  } else {
    floatBits = sign | ((exponent + 112) << 23) | (mantissa << 13);
  }
  final data = ByteData(4)..setUint32(0, floatBits, Endian.little);
  return data.getFloat32(0, Endian.little);
}

int doubleToHalf(double value) {
  // Read the binary64 fields directly. Narrowing through float32 first creates
  // a double-rounding hole: a binary64 value just above a half-way point can
  // become the exact midpoint in float32 and then tie toward the wrong half.
  final data = ByteData(8)..setFloat64(0, value, Endian.big);
  final high = data.getUint32(0, Endian.big);
  final low = data.getUint32(4, Endian.big);
  final sign = (high >> 16) & 0x8000;
  final rawExponent = (high >> 20) & 0x7ff;
  final fractionHigh = high & 0x000fffff;

  if (rawExponent == 0x7ff) {
    return sign | 0x7c00 | (fractionHigh == 0 && low == 0 ? 0 : 0x0200);
  }
  if (rawExponent == 0) return sign; // binary64 subnormals are below half range

  final exponent = rawExponent - 1023;
  if (exponent > 15) return sign | 0x7c00;
  if (exponent < -25) return sign;

  // The 53-bit significand is held as a 21-bit high word plus a 32-bit low
  // word so this remains exact when compiled to JavaScript and Wasm too.
  final significandHigh = 0x00100000 | fractionHigh;
  if (exponent >= -14) {
    final rounded = _roundSignificand(
      significandHigh,
      low,
      42,
    ); // 11 significant bits
    // Addition intentionally carries a rounded 0x800 significand into the
    // exponent field (including the largest-finite-to-infinity boundary).
    return sign | (((exponent + 15) << 10) + (rounded - 0x400));
  }

  // Half subnormals use a fixed 2^-24 step. E=-25 is the exact half-way
  // boundary between signed zero and the minimum subnormal, so ties-to-even
  // correctly selects zero there.
  final rounded = _roundSignificand(significandHigh, low, 28 - exponent);
  return sign | rounded;
}

int _roundSignificand(int high, int low, int shift) {
  final highShift = shift - 32;
  final kept = high >> highShift;
  final remainderMask = (1 << highShift) - 1;
  final remainderHigh = high & remainderMask;
  final halfwayHigh = 1 << (highShift - 1);
  final roundUp =
      remainderHigh > halfwayHigh ||
      (remainderHigh == halfwayHigh && (low != 0 || kept.isOdd));
  return kept + (roundUp ? 1 : 0);
}
