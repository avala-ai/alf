/** Python-compatible decimal rounding for canonical conformance output. */
export function roundCanonical(value: number, places: number): number {
  if (!Number.isFinite(value) || !Number.isInteger(places) || places < 0) return value;
  const factor = 10 ** places;
  // At this magnitude binary64 has no fractional scaled unit left to round.
  // Returning the original also avoids turning a finite 1e308 into Infinity.
  if (!Number.isFinite(factor) || Math.abs(value) >= Number.MAX_SAFE_INTEGER / factor) return value;

  const view = new DataView(new ArrayBuffer(8));
  view.setFloat64(0, value, false);
  const bits = view.getBigUint64(0, false);
  const negative = (bits >> 63n) !== 0n;
  const exponentBits = Number((bits >> 52n) & 0x7ffn);
  const fraction = bits & ((1n << 52n) - 1n);
  const significand = exponentBits === 0 ? fraction : (1n << 52n) | fraction;
  const exponent = exponentBits === 0 ? -1074 : exponentBits - 1023 - 52;
  let numerator = significand * 10n ** BigInt(places);
  let denominator = 1n;
  if (exponent >= 0) numerator <<= BigInt(exponent);
  else denominator <<= BigInt(-exponent);

  let rounded = numerator / denominator;
  const remainderTwice = (numerator % denominator) * 2n;
  if (remainderTwice > denominator || (remainderTwice === denominator && rounded % 2n !== 0n)) rounded += 1n;
  const result = Number(rounded) / factor;
  return negative ? -result : result;
}
