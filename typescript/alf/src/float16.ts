import { refuse } from './errors.js';

const scratch = new DataView(new ArrayBuffer(8));
const DOUBLE_FRACTION_MASK = (1n << 52n) - 1n;
const DOUBLE_HIDDEN_BIT = 1n << 52n;

export function decodeFloat16(bits: number): number {
  const sign = (bits & 0x8000) === 0 ? 1 : -1;
  const exponent = (bits >>> 10) & 0x1f;
  const fraction = bits & 0x03ff;
  if (exponent === 0) {
    return sign * 2 ** -14 * (fraction / 1024);
  }
  if (exponent === 0x1f) {
    return fraction === 0 ? sign * Infinity : NaN;
  }
  return sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
}

/** IEEE-754 binary64 -> binary16, round-to-nearest ties-to-even. */
export function encodeFloat16(value: number): number {
  // binary16 HAS NaN and both infinities, and `.alp` geometry uses NaN as the
  // invalid-return marker (spec/format.md 5.1), so refusing them here made this
  // writer unable to express a shape the format explicitly permits.
  // 0x7e00 is the quiet NaN every other implementation emits; `decodeFloat16`
  // above already reads all three back.
  if (Number.isNaN(value)) return 0x7e00;
  if (value === Number.POSITIVE_INFINITY) return 0x7c00;
  if (value === Number.NEGATIVE_INFINITY) return 0xfc00;
  scratch.setFloat64(0, value, false);
  const word = scratch.getBigUint64(0, false);
  const sign = Number((word >> 48n) & 0x8000n);
  const encodedExponent = Number((word >> 52n) & 0x7ffn);
  const fraction = word & DOUBLE_FRACTION_MASK;

  // A binary64 subnormal is many orders of magnitude below half of the
  // smallest binary16 subnormal. Preserve signed zero without routing through
  // a float32 value that has already discarded the original rounding side.
  if (encodedExponent === 0) return sign;

  const exponent = encodedExponent - 1023;
  const significand = DOUBLE_HIDDEN_BIT | fraction;
  if (exponent < -25) return sign;
  if (exponent < -14) {
    // In units of 2^-24, value is significand * 2^(exponent - 28).
    // At exponent -25 this also distinguishes the exact half-way value from
    // the upper half of the smallest subnormal interval.
    return sign | Number(roundRightTiesToEven(significand, 28 - exponent));
  }

  let halfExponent = exponent + 15;
  if (halfExponent >= 0x1f) refuse('writer-input-invalid', `Coordinate ${value} overflows float16`);
  const roundedSignificand = roundRightTiesToEven(significand, 42);
  if (roundedSignificand === 0x800n) {
    halfExponent += 1;
    if (halfExponent >= 0x1f) refuse('writer-input-invalid', `Coordinate ${value} overflows float16`);
    return sign | (halfExponent << 10);
  }
  return sign | (halfExponent << 10) | Number(roundedSignificand - 0x400n);
}

function roundRightTiesToEven(value: bigint, shift: number): bigint {
  const bits = BigInt(shift);
  const roundedDown = value >> bits;
  const remainder = value & ((1n << bits) - 1n);
  const halfway = 1n << (bits - 1n);
  return remainder > halfway || (remainder === halfway && (roundedDown & 1n) === 1n)
    ? roundedDown + 1n
    : roundedDown;
}

export function roundTiesToEven(value: number): number {
  const floor = Math.floor(value);
  const fraction = value - floor;
  if (fraction < 0.5) return floor;
  if (fraction > 0.5) return floor + 1;
  return floor % 2 === 0 ? floor : floor + 1;
}
