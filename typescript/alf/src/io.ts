import { AlfError, refuse } from './errors.js';
import { decodeFloat16, encodeFloat16, roundTiesToEven } from './float16.js';
import { gunzip, gunzipPrefix, gzip, isGzip, MAX_COMPRESSED_INPUT_BYTES } from './gzip.js';
import {
  ALF_VERSION_V1,
  ALF_VERSION_V2,
  ALF_VERSION_V3,
  type AlcFile,
  type AlfChannel,
  type AlfFile,
  type AlfHeader,
  type AlfMetadata,
  type AlfVersion,
  type AliFile,
  type AllFile,
  type AlpFile,
  type AlxFile,
  FRAME_DEVICE_ALIGNED,
  FRAME_WORLD_ALIGNED,
  type FrameConvention,
  type FrameRotation,
  type GeometryCodec,
  type ReadAlfOptions,
  type WriteAliOptions,
  type WriteAllOptions,
  type WriteAlpOptions,
  type WriteAlxOptions,
  type WriteOptions,
} from './types.js';

const MAX_HEADER_BYTES = 10_000_000;
const MAX_ALF_INPUT_BYTES = 512 * 1024 * 1024;
const MAX_DECODED_GEOMETRY_BYTES = 512 * 1024 * 1024;
const MAX_DECODED_GEOMETRY_COMPONENTS = Math.floor(
  MAX_DECODED_GEOMETRY_BYTES / Float64Array.BYTES_PER_ELEMENT,
);
const MAX_DECODED_V2_LABEL_BYTES = 256 * 1024 * 1024;
const MAX_DECODED_V2_LABEL_ROWS = Math.floor(MAX_DECODED_V2_LABEL_BYTES / 3);
const MAX_JSON_DEPTH = 256;
const FP16_SUBNORMAL_HALF_ULP = 2 ** -25;
/**
 * The reserved word for "this beam produced no return", in v3 fixed point.
 *
 * Free by construction: the scale is derived as `extent / 32767`, so a real
 * coordinate quantises into [-32767, 32767] and -32768 is unreachable.
 * See spec/format.md §5.2.
 */
const INT16_NO_RETURN = -32768;

/** True when all three components of the point at `base` are finite. */
function pointIsFinite(points: ArrayLike<number>, base: number): boolean {
  return (
    Number.isFinite(points[base] ?? NaN) &&
    Number.isFinite(points[base + 1] ?? NaN) &&
    Number.isFinite(points[base + 2] ?? NaN)
  );
}

/**
 * True when ALL three components of the point at `base` are non-finite.
 *
 * Not the negation of `pointIsFinite`, which is also true for a PARTLY invalid
 * point. The int16 writer needs to tell those apart: a whole invalid point
 * encodes as the sentinel, a partial one is refused.
 */
function pointIsWhollyNonFinite(points: ArrayLike<number>, base: number): boolean {
  return (
    !Number.isFinite(points[base] ?? NaN) &&
    !Number.isFinite(points[base + 1] ?? NaN) &&
    !Number.isFinite(points[base + 2] ?? NaN)
  );
}

const MIN_GEOM_SCALE_M = 1e-4;
const FP16_COMPONENT_BUDGET_M = 512;
const INT16_ERROR_BUDGET_M = 0.25;
const textEncoder = new TextEncoder();
const textDecoder = new TextDecoder('utf-8', { fatal: true });
const JSON_NUMBER = /-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?/y;
// `_frame_rotation` is here for the same reason the grid parameters are: a
// component whose stated decimal changes when binary64 decodes it turns the
// WHOLE frame, not one sample. The scanner walks the raw span, so an object
// value works the same way `_geom_offset`'s array does.
const GEOMETRY_NUMBER_KEYS = new Set(['_geom_scale', '_geom_offset', '_frame_rotation']);
const GEOMETRY_ANCHOR_KEYS = new Set(['device_position', 'world_centroid_epsg3857']);

// See `spec/format.md` §5.3. The tolerance is on the SQUARED norm because that
// is the factor a non-unit quaternion actually applies, and 1e-9 is far looser
// than the rounding of a genuinely normalized quaternion and far tighter than
// any error worth a metre.
const FRAME_ROTATION_NORM_TOLERANCE = 1e-9;
const INTEGRAL_NUMBER_KEYS = new Set(['_total_points', '_source_total_points', '_format_version']);
const PLATFORM_LITTLE_ENDIAN = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;

interface Envelope {
  version: AlfVersion;
  metadata: AlfMetadata;
  payload: Uint8Array;
  payloadOwned: boolean;
  retainedInputBytes: number;
}

function own(metadata: AlfMetadata, key: string): boolean {
  return Object.prototype.hasOwnProperty.call(metadata, key);
}

function checkedCount(
  metadata: AlfMetadata,
  key: string,
  missingCode?: 'source-total-points-missing' | 'total-points-missing',
): number | null {
  if (!own(metadata, key)) {
    if (missingCode) refuse(missingCode, `ALF header is missing required ${key}`);
    return null;
  }
  const value = metadata[key];
  if (typeof value !== 'number' || !Number.isInteger(value) || value < 0) {
    refuse('total-points-not-an-integer', `${key} must be a non-negative integer, got ${String(value)}`);
  }
  return value === 0 ? 0 : value;
}

function checkPayloadLength(channel: AlfChannel, count: number | null, payload: Uint8Array, bytesPerPoint: number): void {
  if (count === null) return;
  const expected = count * bytesPerPoint;
  if (!Number.isSafeInteger(expected) || payload.length !== expected) {
    refuse(
      'payload-length-mismatch',
      `.${channel} header declares ${count} points (${expected} bytes), but payload has ${payload.length}`,
    );
  }
}

function cloneMetadata(metadata: AlfMetadata | undefined): AlfMetadata {
  if (metadata === undefined) return {};
  if (!isPlainObject(metadata)) refuse('writer-input-invalid', 'ALF metadata must be a plain JSON object');
  assertJsonValue(metadata, 'metadata');
  // JSON serialization is now a validation proof, not a normalizer: the walk
  // above refuses every value JSON would drop, coerce, or rewrite.
  return JSON.parse(JSON.stringify(metadata)) as AlfMetadata;
}

function normalizeDecodedMetadata(metadata: Readonly<AlfMetadata>): AlfMetadata {
  // Reader metadata is already proven to be bounded JSON. A JSON round-trip is
  // therefore a safe converter-only clone, and canonicalizes JSON's `-0` to
  // `0` before the stricter direct-writer metadata preflight sees it.
  assertJsonValue(metadata, 'metadata', new WeakSet(), 1, true);
  return JSON.parse(JSON.stringify(metadata)) as AlfMetadata;
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) return false;
  const prototype = Object.getPrototypeOf(value) as object | null;
  return prototype === Object.prototype || prototype === null;
}

function assertJsonValue(
  value: unknown,
  path: string,
  seen: WeakSet<object> = new WeakSet(),
  depth = 1,
  allowNegativeZero = false,
): void {
  if (value === null || typeof value === 'string' || typeof value === 'boolean') return;
  if (typeof value === 'number') {
    if (!Number.isFinite(value) || (!allowNegativeZero && Object.is(value, -0))) {
      refuse('writer-input-invalid', `${path} contains a number JSON would rewrite`);
    }
    return;
  }
  if (Array.isArray(value)) {
    if (depth > MAX_JSON_DEPTH) refuse('writer-input-invalid', `ALF metadata exceeds the maximum JSON nesting depth of ${MAX_JSON_DEPTH}`);
    if (seen.has(value)) refuse('writer-input-invalid', `${path} contains a cycle`);
    seen.add(value);
    const arrayKeys = Reflect.ownKeys(value);
    if (
      arrayKeys.some(
        (key) =>
          typeof key === 'symbol' ||
          (key !== 'length' && (!/^(0|[1-9][0-9]*)$/.test(key) || Number(key) >= value.length)),
      )
    ) {
      refuse('writer-input-invalid', `${path} has array properties JSON would drop`);
    }
    for (let index = 0; index < value.length; index += 1) {
      if (!Object.prototype.hasOwnProperty.call(value, index)) {
        refuse('writer-input-invalid', `${path} contains an array hole that JSON would rewrite as null`);
      }
      const descriptor = Object.getOwnPropertyDescriptor(value, String(index));
      if (descriptor === undefined || !descriptor.enumerable || descriptor.get !== undefined || descriptor.set !== undefined) {
        refuse('writer-input-invalid', `${path}[${index}] is not an ordinary JSON value`);
      }
      assertJsonValue(value[index], `${path}[${index}]`, seen, depth + 1, allowNegativeZero);
    }
    seen.delete(value);
    return;
  }
  if (!isPlainObject(value)) refuse('writer-input-invalid', `${path} contains a non-JSON object`);
  if (depth > MAX_JSON_DEPTH) refuse('writer-input-invalid', `ALF metadata exceeds the maximum JSON nesting depth of ${MAX_JSON_DEPTH}`);
  if (seen.has(value)) refuse('writer-input-invalid', `${path} contains a cycle`);
  seen.add(value);
  const ownKeys = Reflect.ownKeys(value);
  if (ownKeys.some((key) => typeof key === 'symbol')) {
    refuse('writer-input-invalid', `${path} has symbol keys JSON would drop`);
  }
  for (const key of Object.keys(value)) {
    const descriptor = Object.getOwnPropertyDescriptor(value, key);
    if (descriptor === undefined || !descriptor.enumerable || descriptor.get !== undefined || descriptor.set !== undefined) {
      refuse('writer-input-invalid', `${path}.${key} is not an ordinary JSON property`);
    }
    const member = value[key];
    if (member === undefined || typeof member === 'function' || typeof member === 'symbol' || typeof member === 'bigint') {
      refuse('writer-input-invalid', `${path}.${key} would be dropped or rewritten by JSON serialization`);
    }
    assertJsonValue(member, `${path}.${key}`, seen, depth + 1, allowNegativeZero);
  }
  if (ownKeys.filter((key): key is string => typeof key === 'string').some((key) => !Object.prototype.propertyIsEnumerable.call(value, key))) {
    refuse('writer-input-invalid', `${path} has non-enumerable properties JSON would drop`);
  }
  seen.delete(value);
}

function readonlyMetadata(metadata: AlfMetadata): Readonly<AlfMetadata> {
  return Object.freeze(metadata);
}

function assertFiniteDecodedNumbers(root: unknown): void {
  const pending: unknown[] = [root];
  while (pending.length > 0) {
    const value = pending.pop();
    if (typeof value === 'number') {
      if (!Number.isFinite(value)) {
        refuse('header-not-json', 'ALF header contains a JSON number outside the finite binary64 range');
      }
    } else if (Array.isArray(value)) {
      for (const member of value) pending.push(member);
    } else if (value !== null && typeof value === 'object') {
      for (const member of Object.values(value)) pending.push(member);
    }
  }
}

function stringifyJson(value: unknown, key?: string): string {
  if (typeof value === 'number') {
    if (Number.isInteger(value) && !Number.isSafeInteger(value)) {
      return key !== undefined && INTEGRAL_NUMBER_KEYS.has(key) ? BigInt(value).toString() : value.toExponential();
    }
    return JSON.stringify(value);
  }
  if (Array.isArray(value)) return `[${value.map((member) => stringifyJson(member)).join(',')}]`;
  if (value !== null && typeof value === 'object') {
    return `{${Object.entries(value)
      .map(([memberKey, member]) => `${JSON.stringify(memberKey)}:${stringifyJson(member, memberKey)}`)
      .join(',')}}`;
  }
  const encoded = JSON.stringify(value);
  if (encoded === undefined) refuse('writer-input-invalid', 'ALF metadata contains an unencodable JSON value');
  return encoded;
}

function skipJsonWhitespace(text: string, start: number): number {
  let cursor = start;
  while (cursor < text.length && /\s/.test(text[cursor] ?? '')) cursor += 1;
  return cursor;
}

function jsonStringEnd(text: string, start: number): number {
  let cursor = start + 1;
  while (cursor < text.length) {
    const current = text[cursor];
    if (current === '\\') {
      cursor += 2;
    } else {
      cursor += 1;
      if (current === '"') return cursor;
    }
  }
  return text.length;
}

function jsonValueEnd(text: string, start: number): number {
  const first = text[start];
  if (first === '"') return jsonStringEnd(text, start);
  if (first !== '{' && first !== '[') {
    let cursor = start;
    while (cursor < text.length && !/[\s,}\]]/.test(text[cursor] ?? '')) cursor += 1;
    return cursor;
  }
  const stack: string[] = [first];
  let cursor = start + 1;
  while (cursor < text.length && stack.length > 0) {
    const current = text[cursor];
    if (current === '"') {
      cursor = jsonStringEnd(text, cursor);
      continue;
    }
    if (current === '{' || current === '[') stack.push(current);
    if (current === '}' || current === ']') stack.pop();
    cursor += 1;
  }
  return cursor;
}

function checkJsonDepth(text: string): void {
  let depth = 0;
  let inString = false;
  let escaped = false;
  for (let index = 0; index < text.length; index += 1) {
    const current = text[index];
    if (inString) {
      if (escaped) escaped = false;
      else if (current === '\\') escaped = true;
      else if (current === '"') inString = false;
      continue;
    }
    if (current === '"') inString = true;
    else if (current === '{' || current === '[') {
      depth += 1;
      if (depth > MAX_JSON_DEPTH) {
        refuse('header-nesting-too-deep', `ALF header exceeds the maximum JSON nesting depth of ${MAX_JSON_DEPTH}`);
      }
    } else if (current === '}' || current === ']') depth -= 1;
  }
}

/** Refuse duplicate member names in every object, including nested metadata. */
function checkDuplicateJsonKeys(text: string): void {
  let cursor = 0;
  const whitespace = (): void => {
    cursor = skipJsonWhitespace(text, cursor);
  };
  const value = (): void => {
    whitespace();
    const current = text[cursor];
    if (current === '{') {
      object();
      return;
    }
    if (current === '[') {
      cursor += 1;
      whitespace();
      if (text[cursor] === ']') {
        cursor += 1;
        return;
      }
      for (;;) {
        value();
        whitespace();
        if (text[cursor] === ']') {
          cursor += 1;
          return;
        }
        cursor += 1; // JSON validity was already established: this is a comma.
      }
    }
    cursor = jsonValueEnd(text, cursor);
  };
  const object = (): void => {
    cursor += 1;
    const keys = new Set<string>();
    whitespace();
    if (text[cursor] === '}') {
      cursor += 1;
      return;
    }
    for (;;) {
      const keyStart = cursor;
      const keyEnd = jsonStringEnd(text, keyStart);
      const key = JSON.parse(text.slice(keyStart, keyEnd)) as string;
      if (keys.has(key)) refuse('header-key-duplicate', `ALF header contains duplicate JSON object key ${key}`);
      keys.add(key);
      cursor = keyEnd;
      whitespace();
      cursor += 1; // colon
      value();
      whitespace();
      if (text[cursor] === '}') {
        cursor += 1;
        return;
      }
      cursor += 1; // comma
      whitespace();
    }
  };
  value();
}

function canonicalDecimal(token: string): string {
  const exponentAt = token.search(/[eE]/);
  const mantissaEnd = exponentAt === -1 ? token.length : exponentAt;
  const dotAt = token.indexOf('.');
  const fractionDigits = dotAt === -1 ? 0 : mantissaEnd - dotAt - 1;
  let digits = '';
  for (let index = token[0] === '-' ? 1 : 0; index < mantissaEnd; index += 1) {
    const character = token[index] ?? '';
    if (character !== '.') digits += character;
  }
  digits = digits.replace(/^0+/, '');
  if (digits === '') return '0e0';
  const trailingZeros = digits.length - digits.replace(/0+$/, '').length;
  if (trailingZeros > 0) digits = digits.slice(0, -trailingZeros);
  // A finite binary64 shortest representation has at most 17 significant
  // digits. Avoid constructing a hostile multi-megabyte BigInt for a token
  // that therefore cannot possibly round-trip unchanged.
  if (digits.length > 17) return `inexact:${token.length}`;
  const rawExponent = exponentAt === -1 ? '0' : token.slice(exponentAt + 1);
  const exponentSign = rawExponent[0] === '-' ? '-' : '';
  const exponentDigits = rawExponent.replace(/^[+-]?0*/, '') || '0';
  if (exponentDigits.length > 10) return `inexact-exponent:${token.length}`;
  const exponent = BigInt(`${exponentSign}${exponentDigits}`) - BigInt(fractionDigits) + BigInt(trailingZeros);
  return `${token[0] === '-' ? '-' : ''}${digits}e${exponent}`;
}

const JSON_INTEGER_TOKEN = /^-?(?:0|[1-9][0-9]*)$/;

function exactIntegerTokenMatchesNumber(token: string, numeric: number): boolean {
  if (!JSON_INTEGER_TOKEN.test(token) || !Number.isFinite(numeric) || !Number.isInteger(numeric)) return false;
  return BigInt(token) === BigInt(numeric);
}

function exactIntegralTokenMatchesNumber(token: string, numeric: number): boolean {
  if (!Number.isFinite(numeric) || !Number.isInteger(numeric)) return false;
  if (exactIntegerTokenMatchesNumber(token, numeric)) return true;
  const match = /^(-?)(0|[1-9][0-9]*)(?:\.([0-9]+))?(?:[eE]([+-]?[0-9]+))?$/.exec(token);
  if (match === null) return false;
  const sign = match[1] === '-' ? -1n : 1n;
  let digits = `${match[2] ?? ''}${match[3] ?? ''}`.replace(/^0+/, '');
  if (digits === '') return numeric === 0;
  let trailingZeros = 0;
  while (digits.charCodeAt(digits.length - trailingZeros - 1) === 0x30) trailingZeros += 1;
  if (trailingZeros > 0) digits = digits.slice(0, -trailingZeros);
  const exponentToken = match[4] ?? '0';
  const exponentDigits = exponentToken.replace(/^[+-]?0*/, '') || '0';
  const maximumExponentDigits = String(MAX_HEADER_BYTES).length;
  if (exponentDigits.length > maximumExponentDigits) return false;
  const exponentMagnitude = Number(exponentDigits);
  if (!Number.isSafeInteger(exponentMagnitude) || exponentMagnitude > MAX_HEADER_BYTES) return false;
  const exponent = exponentToken.startsWith('-') ? -exponentMagnitude : exponentMagnitude;
  const power = exponent - (match[3]?.length ?? 0) + trailingZeros;
  if (!Number.isSafeInteger(power) || power < 0 || digits.length + power > 309) return false;
  const exact = sign * BigInt(digits) * 10n ** BigInt(power);
  return exact === BigInt(numeric);
}

function isMathematicallyIntegralNumberToken(token: string): boolean {
  const match = /^-?(0|[1-9][0-9]*)(?:\.([0-9]+))?(?:[eE]([+-]?[0-9]+))?$/.exec(token);
  if (match === null) return false;
  const fractionLength = match[2]?.length ?? 0;
  const digits = `${match[1] ?? ''}${match[2] ?? ''}`.replace(/^0+/, '');
  if (digits === '') return true;
  const trailingZeros = digits.length - digits.replace(/0+$/, '').length;
  const minimumExponent = fractionLength - trailingZeros;
  const exponentToken = match[3] ?? '0';
  const exponentDigits = exponentToken.replace(/^[+-]?0*/, '') || '0';
  const maximumExponentDigits = String(MAX_HEADER_BYTES).length;
  if (exponentDigits.length > maximumExponentDigits) return !exponentToken.startsWith('-');
  const exponentMagnitude = Number(exponentDigits);
  if (!Number.isSafeInteger(exponentMagnitude) || exponentMagnitude > MAX_HEADER_BYTES) {
    return !exponentToken.startsWith('-');
  }
  const exponent = exponentToken.startsWith('-') ? -exponentMagnitude : exponentMagnitude;
  return exponent >= minimumExponent;
}

function isExactBinary64ScientificSpelling(token: string, numeric: number): boolean {
  // Writers use normalized scientific notation to preserve exact binary64
  // integers whose ordinary Number string is itself a lossy integer claim.
  // Only that exact spelling gets the compatibility escape: merely looking
  // normalized is insufficient because 9.007199254740993e15 rounds to a
  // different integer before JSON.parse returns it.
  return Number.isFinite(numeric) && token === numeric.toExponential();
}

function exactNonnegativeIntegerTokenMatchesNumber(token: string, numeric: number): boolean {
  return numeric >= 0 && exactIntegralTokenMatchesNumber(token, numeric);
}

function rejectLossyNumberTokens(text: string, start: number, end: number, key: string): void {
  let cursor = start;
  while (cursor < end) {
    if (text[cursor] === '"') {
      cursor = jsonStringEnd(text, cursor);
      continue;
    }
    JSON_NUMBER.lastIndex = cursor;
    const match = JSON_NUMBER.exec(text);
    const candidate = match?.[0];
    if (candidate === undefined || JSON_NUMBER.lastIndex > end) {
      cursor += 1;
      continue;
    }
    const numeric = Number(candidate);
    const integralToken = isMathematicallyIntegralNumberToken(candidate);
    const exactIntegralToken = integralToken && exactIntegralTokenMatchesNumber(candidate, numeric);
    if (
      (integralToken && !exactIntegralToken && !isExactBinary64ScientificSpelling(candidate, numeric)) ||
      (!integralToken &&
        !exactIntegralToken &&
        (!Number.isFinite(numeric) || canonicalDecimal(candidate) !== canonicalDecimal(String(numeric))))
    ) {
      refuse(
        'geom-number-inexact',
        `${key} contains numeric token ${candidate} that binary64 would read as ${String(numeric)}`,
      );
    }
    cursor += candidate.length;
  }
}

function rejectLossyAnchorNumberTokens(text: string, start: number, end: number, key: string): void {
  if (text[start] !== '{') {
    rejectLossyNumberTokens(text, start, end, key);
    return;
  }
  let cursor = skipJsonWhitespace(text, start + 1);
  while (cursor < end && text[cursor] !== '}') {
    const memberStart = cursor;
    const memberEnd = jsonStringEnd(text, memberStart);
    const member = JSON.parse(text.slice(memberStart, memberEnd)) as string;
    cursor = skipJsonWhitespace(text, memberEnd);
    cursor = skipJsonWhitespace(text, cursor + 1);
    const valueStart = cursor;
    const valueEnd = jsonValueEnd(text, valueStart);
    if (member === 'x' || member === 'y' || member === 'z') {
      rejectLossyNumberTokens(text, valueStart, valueEnd, key);
    }
    cursor = skipJsonWhitespace(text, valueEnd);
    if (text[cursor] === ',') cursor = skipJsonWhitespace(text, cursor + 1);
  }
}

/** Validate reserved numeric lexemes before JSON.parse rounding is trusted. */
function checkHeaderNumberTokens(headerText: string, version: AlfVersion, channel: AlfChannel): void {
  let cursor = skipJsonWhitespace(headerText, 0) + 1;
  const retainedValues = new Map<string, readonly [start: number, end: number]>();
  for (;;) {
    cursor = skipJsonWhitespace(headerText, cursor);
    if (headerText[cursor] === '}' || cursor >= headerText.length) break;
    const keyStart = cursor;
    const keyEnd = jsonStringEnd(headerText, keyStart);
    const key = JSON.parse(headerText.slice(keyStart, keyEnd)) as string;
    cursor = skipJsonWhitespace(headerText, keyEnd);
    cursor = skipJsonWhitespace(headerText, cursor + 1);
    const valueStart = cursor;
    const valueEnd = jsonValueEnd(headerText, valueStart);
    if (
      (version === ALF_VERSION_V3 && GEOMETRY_NUMBER_KEYS.has(key)) ||
      (channel === 'alp' && GEOMETRY_ANCHOR_KEYS.has(key)) ||
      key === '_total_points' ||
      (key === '_format_version' && channel === 'all') ||
      (key === '_source_total_points' && channel === 'alx')
    ) {
      // JSON.parse retains the final occurrence, so token validation must
      // inspect the same value. Any duplicate-key check runs before this pass.
      retainedValues.set(key, [valueStart, valueEnd]);
    }
    cursor = skipJsonWhitespace(headerText, valueEnd);
    if (headerText[cursor] === ',') cursor += 1;
  }
  for (const [key, [valueStart, valueEnd]] of retainedValues) {
    if (version === ALF_VERSION_V3 && GEOMETRY_NUMBER_KEYS.has(key)) {
      rejectLossyNumberTokens(headerText, valueStart, valueEnd, key);
    }
    if (channel === 'alp' && GEOMETRY_ANCHOR_KEYS.has(key)) {
      rejectLossyAnchorNumberTokens(headerText, valueStart, valueEnd, key);
    }
    if (
      key === '_total_points' ||
      (key === '_format_version' && channel === 'all') ||
      (key === '_source_total_points' && channel === 'alx')
    ) {
      const token = headerText.slice(valueStart, valueEnd);
      const numeric = Number(token);
      const exact = key === '_format_version'
        ? exactIntegralTokenMatchesNumber(token, numeric)
        : exactNonnegativeIntegerTokenMatchesNumber(token, numeric);
      if (!exact) {
        const domain = key === '_format_version' ? 'integer' : 'non-negative integer';
        refuse('total-points-not-an-integer', `${key} must state an exact ${domain}`);
      }
    }
  }
}

function validateGeometryAnchorShapes(metadata: AlfMetadata): void {
  for (const key of GEOMETRY_ANCHOR_KEYS) {
    if (!own(metadata, key)) continue;
    const raw = metadata[key];
    if (key === 'device_position' && isPlainObject(raw) && Object.keys(raw).length === 0) continue;
    let components: readonly unknown[];
    if (Array.isArray(raw) && raw.length === 3) {
      components = raw;
    } else if (isPlainObject(raw) && own(raw, 'x') && own(raw, 'y') && own(raw, 'z')) {
      components = [raw['x'], raw['y'], raw['z']];
    } else {
      refuse('geometry-anchor-malformed', `${key} must contain x, y, and z or be a three-element array`);
    }
    if (components.some((component) => typeof component !== 'number' || !Number.isFinite(component))) {
      refuse('geometry-anchor-malformed', `${key} components must be finite JSON numbers`);
    }
  }
}

/**
 * The v3 frame convention: how a `.alp`'s local points reach the world.
 *
 * A `.alp`'s points are the world points with an origin subtracted, which is
 * HALF of a rigid transform. Producers disagreed about the other half — some
 * translated and stopped, leaving points on world axes, some also rotated into
 * the device frame — and nothing in the bytes told them apart. The two agree
 * exactly whenever the device heading is identity, so a consumer that omitted
 * the rotation matched a producer that omitted it, and the pair looked right
 * until the vehicle turned. See `spec/format.md` §5.3.
 */
function validateFrameConvention(metadata: AlfMetadata, version: AlfVersion): void {
  const declared = own(metadata, '_frame_convention') ? metadata['_frame_convention'] : undefined;
  const rotation = own(metadata, '_frame_rotation') ? metadata['_frame_rotation'] : undefined;

  // Versions 1 and 2 are grandfathered: they predate the field and a deployed
  // file cannot be told retroactively what it meant. What they may not do is
  // CARRY it — this reader applies no rotation to v1/v2 geometry, so the
  // declaration would be a promise the format does not keep.
  if (version !== ALF_VERSION_V3) {
    if (declared !== undefined || rotation !== undefined) {
      const named = declared !== undefined ? '_frame_convention' : '_frame_rotation';
      refuse(
        'frame-convention-on-legacy-version',
        `${named} is a version-3 field, but this .alp is version ${version}`,
      );
    }
    return;
  }

  if (declared === undefined) {
    refuse(
      'frame-convention-missing',
      'version-3 .alp is missing _frame_convention; its points are the world points with an ' +
        'origin subtracted, and the file must say whether the device rotation was applied too',
    );
  }
  if (declared !== FRAME_WORLD_ALIGNED && declared !== FRAME_DEVICE_ALIGNED) {
    refuse(
      'frame-convention-unsupported',
      `Unsupported _frame_convention; expected "${FRAME_WORLD_ALIGNED}" or "${FRAME_DEVICE_ALIGNED}"`,
    );
  }

  if (declared === FRAME_WORLD_ALIGNED) {
    if (rotation !== undefined) {
      refuse(
        'frame-rotation-unexpected',
        '_frame_convention world-aligned declares that no rotation was applied, but ' +
          '_frame_rotation is present',
      );
    }
    return;
  }

  if (rotation === undefined) {
    refuse(
      'frame-rotation-missing',
      '_frame_convention device-aligned requires _frame_rotation; without it the points cannot ' +
        'be returned to the world at all',
    );
  }
  if (!isPlainObject(rotation)) {
    refuse('frame-rotation-malformed', '_frame_rotation must be an object with w, x, y and z');
  }
  // EXACTLY the four members: a stray `qw` beside `w` is a producer mixing two
  // spellings, not application data, and only one of them would be read.
  if (Object.keys(rotation).length !== 4) {
    refuse('frame-rotation-malformed', '_frame_rotation must carry exactly w, x, y and z');
  }
  let normSquared = 0;
  for (const axis of ['w', 'x', 'y', 'z'] as const) {
    if (!own(rotation, axis)) {
      refuse('frame-rotation-malformed', `_frame_rotation is missing ${axis}`);
    }
    const component = rotation[axis];
    if (typeof component !== 'number' || !Number.isFinite(component)) {
      refuse('frame-rotation-malformed', `_frame_rotation.${axis} must be a finite JSON number`);
    }
    normSquared += component * component;
  }
  // Conjugation by a non-unit quaternion scales by |q|^2 as well as rotating,
  // uniformly across the frame and nowhere else in the file. Refused rather
  // than normalized: normalizing would hide the producer's fault, not fix it.
  if (Math.abs(normSquared - 1) > FRAME_ROTATION_NORM_TOLERANCE) {
    refuse(
      'frame-rotation-not-normalized',
      `_frame_rotation is not a unit quaternion (|q|^2 = ${normSquared})`,
    );
  }
}

/**
 * Validate an INFLATED ALF envelope and parse its header.
 *
 * Split out of `readEnvelope` so the full read and the header-only read share
 * one implementation of every envelope rule. Two copies of this would drift, and
 * a header-only path that validated less than the full path is precisely the bug
 * a caller could not see — it would accept a file `readAlf` refuses.
 */
function parseEnvelopeHeader(
  raw: Uint8Array,
  channel: AlfChannel,
): { version: AlfVersion; metadata: AlfMetadata; payloadOffset: number } {
  if (raw.length < 5) refuse('header-truncated', `ALF input is ${raw.length} bytes; at least 5 are required`);

  const versionByte = raw[0] ?? 0;
  if (versionByte !== ALF_VERSION_V1 && versionByte !== ALF_VERSION_V2 && versionByte !== ALF_VERSION_V3) {
    refuse('version-unsupported', `Unsupported ALF version byte ${versionByte}`);
  }
  if (versionByte === ALF_VERSION_V2 && channel !== 'all') {
    refuse('label-version-on-non-label-channel', `Version 2 is the .all interleaved label layout, not .${channel}`);
  }
  if (versionByte === ALF_VERSION_V3 && channel !== 'alp') {
    refuse('geometry-version-on-non-geometry-channel', `Version 3 is fixed-point .alp geometry, not .${channel}`);
  }

  const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
  const headerLength = view.getUint32(1, true);
  if (headerLength > MAX_HEADER_BYTES) refuse('header-too-large', `ALF header exceeds ${MAX_HEADER_BYTES} bytes`);
  const payloadOffset = 5 + headerLength;
  if (payloadOffset > raw.length) {
    refuse('header-truncated', `ALF header declares ${headerLength} bytes, but only ${raw.length - 5} remain`);
  }

  let headerText: string;
  try {
    headerText = textDecoder.decode(raw.subarray(5, payloadOffset));
  } catch (cause) {
    throw new AlfError('header-not-utf8', 'ALF header is not valid UTF-8', { cause });
  }
  let decoded: unknown;
  try {
    checkJsonDepth(headerText);
    decoded = headerText.length === 0 ? {} : JSON.parse(headerText);
  } catch (cause) {
    if (cause instanceof AlfError) throw cause;
    throw new AlfError('header-not-json', 'ALF header is not valid JSON', { cause });
  }
  if (decoded === null || typeof decoded !== 'object' || Array.isArray(decoded)) {
    refuse('header-not-an-object', 'ALF header must be a JSON object');
  }
  checkDuplicateJsonKeys(headerText);
  checkHeaderNumberTokens(headerText, versionByte, channel);
  assertFiniteDecodedNumbers(decoded);
  if (channel === 'alp') {
    validateGeometryAnchorShapes(decoded as AlfMetadata);
    // Before the grid parameters: a file that does not say where its points are
    // is refused for that, which is decidable first and from less.
    validateFrameConvention(decoded as AlfMetadata, versionByte);
  }
  return { version: versionByte, metadata: decoded as AlfMetadata, payloadOffset };
}

function assertInputWithinLimit(inputLength: number, compressed: boolean): void {
  const inputLimit = compressed ? MAX_COMPRESSED_INPUT_BYTES : MAX_ALF_INPUT_BYTES;
  if (inputLength > inputLimit) {
    refuse('input-too-large', `ALF input exceeds the ${inputLimit}-byte limit`);
  }
}

async function readEnvelope(input: Uint8Array | ArrayBuffer, channel: AlfChannel): Promise<Envelope> {
  const inputLength = input.byteLength;
  let raw: Uint8Array = input instanceof Uint8Array ? input : new Uint8Array(input);
  const compressed = isGzip(raw);
  assertInputWithinLimit(inputLength, compressed);
  if (compressed) raw = await gunzip(raw);

  const { version, metadata, payloadOffset } = parseEnvelopeHeader(raw, channel);
  return {
    version,
    metadata,
    payload: raw.subarray(payloadOffset),
    payloadOwned: compressed,
    retainedInputBytes: raw.byteLength,
  };
}

function decodedByteChannelPayload(envelope: Envelope): Uint8Array {
  if (envelope.payloadOwned) return envelope.payload;
  assertDecodedByteChannelCopyFits(envelope.retainedInputBytes, envelope.payload.byteLength);
  return envelope.payload.slice();
}

function geometryCodec(version: AlfVersion, metadata: AlfMetadata): GeometryCodec {
  const raw = own(metadata, '_geom_codec') ? metadata['_geom_codec'] : undefined;
  const codec = raw === undefined ? 'fp16' : raw;
  if (codec !== 'fp16' && codec !== 'int16') {
    refuse('codec-unsupported', `Unknown .alp geometry codec ${String(codec)}`);
  }
  if ((version === ALF_VERSION_V3) !== (codec === 'int16')) {
    refuse('codec-version-disagreement', `ALF version ${version} disagrees with geometry codec ${codec}`);
  }
  return codec;
}

function checkedGeomScale(metadata: AlfMetadata): number {
  if (!own(metadata, '_geom_scale')) refuse('geom-scale-missing', 'Fixed-point geometry is missing _geom_scale');
  const scale = metadata['_geom_scale'];
  if (typeof scale !== 'number' || !Number.isFinite(scale) || scale <= 0) {
    refuse('geom-scale-not-positive', `_geom_scale must be a positive finite number, got ${String(scale)}`);
  }
  return scale;
}

function checkedGeomOffset(metadata: AlfMetadata): readonly [number, number, number] {
  const offset = metadata['_geom_offset'];
  if (
    !Array.isArray(offset) ||
    offset.length !== 3 ||
    offset.some((component) => typeof component !== 'number' || !Number.isFinite(component))
  ) {
    refuse('geom-offset-malformed', '_geom_offset must contain exactly three finite numbers');
  }
  return [offset[0] as number, offset[1] as number, offset[2] as number];
}

const floatScratch = new DataView(new ArrayBuffer(8));

function nextUp(value: number): number {
  if (Number.isNaN(value) || value === Infinity) return value;
  if (Object.is(value, -0) || value === 0) return Number.MIN_VALUE;
  floatScratch.setFloat64(0, value, false);
  let bits = floatScratch.getBigUint64(0, false);
  bits = value > 0 ? bits + 1n : bits - 1n;
  floatScratch.setBigUint64(0, bits, false);
  return floatScratch.getFloat64(0, false);
}

function nextDown(value: number): number {
  return -nextUp(-value);
}

function adjacentSpacing(value: number): number {
  const magnitude = Math.abs(value);
  const below = Math.abs(magnitude - nextDown(magnitude));
  const above = Math.abs(nextUp(magnitude) - magnitude);
  return Number.isFinite(above) ? Math.max(below, above) : below;
}

function reconstructionRounding(payload: Uint8Array, scale: number, offset: readonly number[]): number {
  if (payload.length === 0 || payload.length % 6 !== 0) return 0;
  const view = new DataView(payload.buffer, payload.byteOffset, payload.byteLength);
  let worst = 0;
  for (let index = 0; index < payload.length / 2; index += 1) {
    // A sentinel row carries no position, so it has no reconstruction error.
    if (
      index % 3 === 0 &&
      view.getInt16(index * 2, true) === INT16_NO_RETURN &&
      view.getInt16((index + 1) * 2, true) === INT16_NO_RETURN &&
      view.getInt16((index + 2) * 2, true) === INT16_NO_RETURN
    ) {
      index += 2;
      continue;
    }
    const product = view.getInt16(index * 2, true) * scale;
    const axisOffset = offset[index % 3] ?? 0;
    const total = product + axisOffset;
    const near = total - product;
    // Knuth TwoSum recovers the addition's exact rounding residual. Half the
    // larger adjacent float spacing conservatively bounds the multiply.
    const addError = (product - (total - near)) + (axisOffset - near);
    const component = Math.abs(addError) + adjacentSpacing(product) / 2;
    if (!Number.isFinite(component)) refuse('coordinate-non-finite', 'Fixed-point reconstruction has no finite error bound');
    worst = Math.max(worst, component);
  }
  return worst;
}

function decodeAlp(envelope: Envelope): AlpFile {
  const { version, metadata, payload } = envelope;
  const totalPoints = checkedCount(metadata, '_total_points');
  if (totalPoints !== null) assertDecodedAlpFits(totalPoints * 3);
  const codec = geometryCodec(version, metadata);
  let scale = 1;
  let offset: readonly [number, number, number] = [0, 0, 0];
  if (codec === 'int16') {
    scale = checkedGeomScale(metadata);
    offset = checkedGeomOffset(metadata);
  }
  if (payload.length % 6 !== 0) {
    refuse('payload-not-whole-points', `.${'alp'} payload has ${payload.length} bytes, not whole XYZ triples`);
  }
  checkPayloadLength('alp', totalPoints, payload, 6);
  const count = payload.length / 2;
  assertDecodedAlpFits(count);
  const points = new Float64Array(count);
  const view = new DataView(payload.buffer, payload.byteOffset, payload.byteLength);
  let largest = 0;
  for (let index = 0; index < count; index += 1) {
    const axis = index % 3;
    // A WHOLE point of sentinel words is an invalid return, in either codec
    // (spec/format.md 5.2). Checked at the point boundary; a partial sentinel is
    // not a shape a conforming writer emits.
    if (codec === 'int16' && axis === 0 && index + 2 < count) {
      const isNoReturn =
        view.getInt16(index * 2, true) === INT16_NO_RETURN &&
        view.getInt16((index + 1) * 2, true) === INT16_NO_RETURN &&
        view.getInt16((index + 2) * 2, true) === INT16_NO_RETURN;
      if (isNoReturn) {
        points[index] = NaN;
        points[index + 1] = NaN;
        points[index + 2] = NaN;
        index += 2;
        continue;
      }
      // A PARTIAL marker is corruption, and the one kind nothing downstream can
      // catch: -32768 is reserved, so no conforming writer emits it as a
      // coordinate (the encoder clamps to +/-32767). Reconstructing it yields an
      // ordinary-looking position — [-32768, 0, 0] at scale 0.01 decodes to
      // -327.68 — which no later check can tell from real data.
      const touchesMarker =
        view.getInt16(index * 2, true) === INT16_NO_RETURN ||
        view.getInt16((index + 1) * 2, true) === INT16_NO_RETURN ||
        view.getInt16((index + 2) * 2, true) === INT16_NO_RETURN;
      if (touchesMarker) {
        refuse(
          'geom-partial-no-return',
          'int16 .alp contains a point with SOME but not all components equal to the reserved invalid-return marker (-32768). No conforming writer can emit that word as a coordinate, so this payload is corrupt; decoding it would report a plausible position that was never measured.',
        );
      }
    }
    const value =
      codec === 'int16'
        ? view.getInt16(index * 2, true) * scale + (offset[axis] ?? 0)
        : decodeFloat16(view.getUint16(index * 2, true));
    // Fixed-point only, and OUTSIDE a sentinel row. int16 cannot encode a
    // non-finite value, so one here is `_geom_scale`/`_geom_offset` overflow. In
    // float16 it is the invalid-return marker (spec/format.md 5.1) and is kept
    // in place, because every other channel is correlated with geometry by index.
    if (codec === 'int16' && !Number.isFinite(value)) {
      refuse('coordinate-non-finite', 'int16 .alp decodes to a non-finite coordinate');
    }
    points[index] = value;
    // FINITE components only: `largest` feeds the fp16 error bound, and NaN
    // there compares false against every threshold a caller checks it with.
    if (Number.isFinite(value)) largest = Math.max(largest, Math.abs(value));
  }
  let maxGeomErrorM: number;
  if (points.length === 0) {
    maxGeomErrorM = 0;
  } else if (codec === 'int16') {
    const recorded = metadata['_geom_max_err_m'];
    const reach = reconstructionRounding(payload, scale, offset);
    let analytic = scale / 2 + reach;
    if (scale < 1e-300) analytic = Math.max(analytic, nextUp(scale / 2));
    if (analytic <= 0) analytic = Number.MIN_VALUE;
    maxGeomErrorM =
      typeof recorded === 'number' && Number.isFinite(recorded) ? Math.max(recorded, analytic) : analytic;
  } else if (largest === 0) {
    maxGeomErrorM = FP16_SUBNORMAL_HALF_ULP;
  } else {
    maxGeomErrorM = Math.max(2 ** (Math.floor(Math.log2(largest)) - 11), FP16_SUBNORMAL_HALF_ULP);
  }
  return {
    channel: 'alp',
    version,
    metadata: readonlyMetadata(metadata),
    totalPoints,
    codec,
    points,
    maxGeomErrorM,
    // Surfaced, not merely validated: the bug this field exists for was a field
    // that was present and ignored.
    frameConvention: frameConventionOf(metadata),
    frameRotation: frameRotationOf(metadata),
  };
}

/** The validated declaration, as a value a caller can act on. */
function frameConventionOf(metadata: AlfMetadata): FrameConvention | null {
  const declared = metadata['_frame_convention'];
  return declared === FRAME_WORLD_ALIGNED || declared === FRAME_DEVICE_ALIGNED ? declared : null;
}

function frameRotationOf(metadata: AlfMetadata): FrameRotation | null {
  const rotation = metadata['_frame_rotation'];
  if (!isPlainObject(rotation)) return null;
  return {
    w: rotation['w'] as number,
    x: rotation['x'] as number,
    y: rotation['y'] as number,
    z: rotation['z'] as number,
  };
}

function decodeAli(envelope: Envelope): AliFile {
  const totalPoints = checkedCount(envelope.metadata, '_total_points');
  checkPayloadLength('ali', totalPoints, envelope.payload, 1);
  return {
    channel: 'ali',
    version: envelope.version,
    metadata: readonlyMetadata(envelope.metadata),
    totalPoints,
    intensity: decodedByteChannelPayload(envelope),
  };
}

function checkedPalette(metadata: AlfMetadata): readonly string[] {
  const palette = metadata['palette'];
  if (!Array.isArray(palette) || palette.length === 0 || palette.length > 255) {
    refuse('palette-entry-malformed', '.alc palette must be a non-empty list of at most 255 colors');
  }
  const colors: string[] = [];
  for (const entry of palette) {
    const normalized = typeof entry === 'string' && entry.startsWith('#') ? entry.slice(1) : entry;
    if (typeof normalized !== 'string' || !/^[0-9a-fA-F]{6}$/.test(normalized)) {
      refuse('palette-entry-malformed', `.alc palette entry ${String(entry)} is not six hexadecimal digits`);
    }
    colors.push(normalized.toLowerCase());
  }
  return Object.freeze(colors);
}

function decodeAlc(envelope: Envelope): AlcFile {
  const totalPoints = checkedCount(envelope.metadata, '_total_points');
  checkPayloadLength('alc', totalPoints, envelope.payload, 1);
  const palette = checkedPalette(envelope.metadata);
  for (const index of envelope.payload) {
    if (index > palette.length) {
      refuse('palette-index-out-of-range', `.alc palette index ${index} exceeds sentinel ${palette.length}`);
    }
  }
  return {
    channel: 'alc',
    version: envelope.version,
    metadata: readonlyMetadata(envelope.metadata),
    totalPoints,
    palette,
    paletteIndices: decodedByteChannelPayload(envelope),
  };
}

function declaredLabelLayout(metadata: AlfMetadata): 1 | 2 | null {
  if (!own(metadata, '_format_version')) return null;
  const value = metadata['_format_version'];
  if (value !== 1 && value !== 2) {
    refuse('label-layout-unsupported', `Unsupported .all _format_version ${String(value)}`);
  }
  return value;
}

function resolvedLabelLayout(envelope: Envelope, totalPoints: number | null): 1 | 2 {
  const fromByte: 1 | 2 = envelope.version === ALF_VERSION_V2 ? 2 : 1;
  const declared = declaredLabelLayout(envelope.metadata);
  if (declared === null || declared === fromByte) return declared ?? fromByte;
  if (totalPoints !== null) {
    const matchesV1 = envelope.payload.length === totalPoints;
    const matchesV2 = envelope.payload.length === totalPoints * 3;
    if (matchesV1 && matchesV2) {
      refuse('label-layout-ambiguous', '.all version fields disagree and the empty payload matches both layouts');
    }
    if (matchesV1) return 1;
    if (matchesV2) return 2;
    refuse('payload-length-mismatch', '.all version fields disagree and payload matches neither label layout');
  }
  if (envelope.payload.length % 3 !== 0) return 1;
  refuse('label-layout-ambiguous', '.all version fields disagree and payload cannot resolve the layout');
}

function checkedInstanceMap(metadata: AlfMetadata, ids: Uint16Array): Readonly<Record<string, string>> | null {
  if (!own(metadata, 'instance_map')) return null;
  const raw = metadata['instance_map'];
  if (!isPlainObject(raw)) {
    refuse('instance-map-malformed', 'instance_map must be an object');
  }
  const map: Record<string, string> = {};
  for (const [key, value] of Object.entries(raw)) {
    if (!/^[1-9][0-9]*$/.test(key) || Number(key) > 65534 || typeof value !== 'string' || value.trim() === '') {
      refuse('instance-map-malformed', `instance_map entry ${key} is unusable`);
    }
    map[key] = value;
  }
  const missing = new Set<number>();
  for (const id of ids) if (id !== 0 && map[String(id)] === undefined) missing.add(id);
  if (missing.size > 0) {
    refuse('instance-map-incomplete', `instance_map is missing emitted id(s) ${[...missing].slice(0, 10).join(', ')}`);
  }
  return Object.freeze(map);
}

function decodeAll(envelope: Envelope): AllFile {
  const totalPoints = checkedCount(envelope.metadata, '_total_points');
  const formatVersion = resolvedLabelLayout(envelope, totalPoints);
  checkPayloadLength('all', totalPoints, envelope.payload, formatVersion === 2 ? 3 : 1);
  if (formatVersion === 1) {
    return {
      channel: 'all',
      version: envelope.version,
      metadata: readonlyMetadata(envelope.metadata),
      totalPoints,
      formatVersion,
      categories: decodedByteChannelPayload(envelope),
      instanceIds: null,
      instanceMap: null,
    };
  }
  if (envelope.payload.length % 3 !== 0) {
    refuse('payload-not-whole-points', '.all v2 payload is not a whole number of 3-byte rows');
  }
  const rows = envelope.payload.length / 3;
  assertDecodedV2LabelsFit(rows);
  const categories = new Uint8Array(rows);
  const instanceIds = new Uint16Array(rows);
  for (let index = 0; index < rows; index += 1) {
    categories[index] = envelope.payload[index * 3] ?? 0;
    const id = (envelope.payload[index * 3 + 1] ?? 0) | ((envelope.payload[index * 3 + 2] ?? 0) << 8);
    if (id === 65535) refuse('instance-id-reserved', '.all payload contains reserved instance id 65535');
    instanceIds[index] = id;
  }
  const instanceMap = checkedInstanceMap(envelope.metadata, instanceIds);
  return {
    channel: 'all',
    version: envelope.version,
    metadata: readonlyMetadata(envelope.metadata),
    totalPoints,
    formatVersion,
    categories,
    instanceIds,
    instanceMap,
  };
}

function decodeAlx(envelope: Envelope): AlxFile {
  const totalPoints = checkedCount(envelope.metadata, '_total_points', 'total-points-missing');
  const sourceTotalPoints = checkedCount(envelope.metadata, '_source_total_points', 'source-total-points-missing');
  if (sourceTotalPoints === null) refuse('source-total-points-missing', '.alx is missing _source_total_points');
  if (envelope.payload.length % 4 !== 0) {
    refuse('payload-not-whole-points', '.alx payload is not a whole number of uint32 indices');
  }
  checkPayloadLength('alx', totalPoints, envelope.payload, 4);
  assertDecodedAlxCopyFits(envelope.retainedInputBytes, envelope.payload.byteLength);
  const keptIndices = new Uint32Array(envelope.payload.length / 4);
  const view = new DataView(envelope.payload.buffer, envelope.payload.byteOffset, envelope.payload.byteLength);
  for (let index = 0; index < keptIndices.length; index += 1) {
    const kept = view.getUint32(index * 4, true);
    if (kept >= sourceTotalPoints) {
      refuse('kept-index-out-of-range', `.alx kept index ${kept} is outside source cloud of ${sourceTotalPoints}`);
    }
    keptIndices[index] = kept;
  }
  keptIndices.sort();
  for (let index = 1; index < keptIndices.length; index += 1) {
    if (keptIndices[index] === keptIndices[index - 1]) {
      refuse('kept-index-duplicate', `.alx repeats kept index ${keptIndices[index]}`);
    }
  }
  // Sorting the result buffer validates uniqueness with constant auxiliary
  // memory. Restore wire order from the retained payload before exposing it.
  for (let index = 0; index < keptIndices.length; index += 1) {
    keptIndices[index] = view.getUint32(index * 4, true);
  }
  return {
    channel: 'alx',
    version: envelope.version,
    metadata: readonlyMetadata(envelope.metadata),
    totalPoints,
    keptIndices,
    sourceTotalPoints,
  };
}

/** Decode and validate one ALF channel. The channel is mandatory because it is not stored in the header. */
/**
 * Bytes inflated on the first probe. Large enough that essentially every real
 * header lands in one pass, small enough that a 1 GB frame costs kilobytes.
 * A header longer than this simply costs a second, exactly-sized inflate.
 */
const HEADER_PROBE_BYTES = 64 * 1024;

/**
 * Parse ONLY an ALF file's envelope, without inflating or reading its payload.
 *
 * For the scan-many-frames case: a batched prefetch that needs `_total_points`,
 * `device_position`, `_geom_scale` or a `palette` across a thousand frames should
 * not pay a full decompression for each. `readAlf` inflates everything before it
 * returns anything; this stops after the header.
 *
 * Returns an {@link AlfHeader}, NOT an {@link AlfFile}. That is deliberate and is
 * the one place this differs from the Python reference, which returns an
 * `AlfFile` whose payload accessors raise at runtime. TypeScript can express the
 * weaker contract in the type system instead, so a caller cannot reach for
 * `.points` on a header-only read and find out at runtime — it does not compile.
 *
 * WHAT IT GIVES UP: for a gzipped input the member's CRC32 and ISIZE are never
 * checked, because verifying them means inflating to the end, which is the cost
 * this function exists to avoid. Header corruption is still caught. Payload
 * corruption is invisible here BY CONSTRUCTION — use `readAlf` when the whole
 * file must be proven intact. `gunzipPrefix` must not locate the trailer first
 * (`firstGzipMemberEnd` Huffman-decodes the whole member). See `gunzipPrefix`.
 */
export async function readAlfHeader(
  input: Uint8Array | ArrayBuffer,
  options?: ReadAlfOptions,
): Promise<AlfHeader> {
  const channel = options?.channel;
  if (channel === undefined) refuse('channel-required', 'readAlfHeader requires the ALF channel');
  if (!['alp', 'ali', 'alc', 'all', 'alx'].includes(channel)) {
    refuse('channel-unknown', `Unknown ALF channel ${String(channel)}`);
  }

  const source: Uint8Array = input instanceof Uint8Array ? input : new Uint8Array(input);
  const compressed = isGzip(source);
  assertInputWithinLimit(source.byteLength, compressed);

  let raw: Uint8Array;
  if (!compressed) {
    raw = source;
  } else {
    raw = await gunzipPrefix(source, HEADER_PROBE_BYTES);
    // Re-inflate only when the declared header genuinely outran the probe. Read
    // the length ourselves rather than catching `parseEnvelopeHeader`'s
    // truncation refusal: a GENUINELY truncated file would otherwise be retried
    // and reported as though the probe were at fault.
    if (raw.length >= 5) {
      const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
      const declared = view.getUint32(1, true);
      if (declared <= MAX_HEADER_BYTES && 5 + declared > raw.length) {
        raw = await gunzipPrefix(source, 5 + declared);
      }
    }
  }

  const { version, metadata } = parseEnvelopeHeader(raw, channel);
  // Channel rules a header-only read must ALSO enforce. The envelope rules
  // apply to every channel; these depend on which one you opened, and they
  // lived only inside the channel decoders — so this, which reaches no decoder,
  // accepted files `readAlf` refuses. That is the "validates less" drift the
  // header-only contract exists to rule out, and it was in every
  // implementation. Only rules the full reader enforces EAGERLY belong here.
  if (channel === 'alx' && !own(metadata, '_total_points')) {
    refuse('total-points-missing', '.alx is missing _total_points');
  }
  if (channel === 'all') declaredLabelLayout(metadata);
  return {
    channel,
    version,
    metadata,
    // The same validated accessor the full read uses, so a header-only read
    // refuses a malformed `_total_points` exactly where `readAlf` would.
    totalPoints: checkedCount(metadata, '_total_points'),
  };
}

export async function readAlf(input: Uint8Array | ArrayBuffer, options?: ReadAlfOptions): Promise<AlfFile> {
  const channel = options?.channel;
  if (channel === undefined) refuse('channel-required', 'readAlf requires the ALF channel');
  if (!['alp', 'ali', 'alc', 'all', 'alx'].includes(channel)) {
    refuse('channel-unknown', `Unknown ALF channel ${String(channel)}`);
  }
  const envelope = await readEnvelope(input, channel);
  switch (channel) {
    case 'alp':
      return decodeAlp(envelope);
    case 'ali':
      return decodeAli(envelope);
    case 'alc':
      return decodeAlc(envelope);
    case 'all':
      return decodeAll(envelope);
    case 'alx':
      return decodeAlx(envelope);
  }
}

function encodeHeader(metadata: AlfMetadata): Uint8Array {
  if (!isPlainObject(metadata)) refuse('writer-input-invalid', 'ALF metadata must remain a plain JSON object');
  assertJsonValue(metadata, 'metadata');
  try {
    const header = textEncoder.encode(stringifyJson(metadata));
    if (header.length > MAX_HEADER_BYTES) refuse('writer-input-invalid', `ALF header exceeds ${MAX_HEADER_BYTES} bytes`);
    return header;
  } catch (cause) {
    if (cause instanceof AlfError) throw cause;
    throw new AlfError('writer-input-invalid', 'ALF metadata cannot be encoded as JSON', { cause });
  }
}

function preflightEnvelope(metadata: AlfMetadata, payloadLength: number): Uint8Array {
  const header = encodeHeader(metadata);
  assertEnvelopeFits(header.length, payloadLength);
  return header;
}

async function encodePreparedEnvelope(
  version: AlfVersion,
  header: Uint8Array,
  payload: Uint8Array,
  compress: boolean,
): Promise<Uint8Array> {
  assertEnvelopeFits(header.length, payload.length);
  const bytes = new Uint8Array(5 + header.length + payload.length);
  bytes[0] = version;
  new DataView(bytes.buffer).setUint32(1, header.length, true);
  bytes.set(header, 5);
  bytes.set(payload, 5 + header.length);
  return compress ? gzip(bytes) : bytes;
}

/** Enforce the reader's uncompressed ceiling before allocating an envelope. */
export function assertEnvelopeFits(headerLength: number, payloadLength: number): void {
  if (
    !Number.isSafeInteger(headerLength) ||
    headerLength < 0 ||
    !Number.isSafeInteger(payloadLength) ||
    payloadLength < 0 ||
    5 + headerLength + payloadLength > MAX_ALF_INPUT_BYTES
  ) {
    refuse('writer-input-invalid', `ALF envelope exceeds the ${MAX_ALF_INPUT_BYTES}-byte reader limit`);
  }
}

/** Reject v2 rows the paired reader cannot materialize before writer allocation. */
export function assertV2LabelPayloadFits(pointCount: number): void {
  if (!Number.isSafeInteger(pointCount) || pointCount < 0) {
    refuse('writer-input-invalid', 'v2 label point count must be a non-negative safe integer');
  }
  if (pointCount > MAX_DECODED_V2_LABEL_ROWS) {
    refuse(
      'writer-input-invalid',
      `v2 labels exceed the paired reader's ${MAX_DECODED_V2_LABEL_BYTES}-byte decoded storage limit`,
    );
  }
  assertEnvelopeFits(0, pointCount * 3);
}

/** Bound the uint8 category plus uint16 instance arrays materialized by the v2 reader. */
export function assertDecodedV2LabelsFit(pointCount: number): void {
  if (!Number.isSafeInteger(pointCount) || pointCount < 0 || pointCount > MAX_DECODED_V2_LABEL_ROWS) {
    refuse(
      'input-too-large',
      `decoded v2 labels exceed the ${MAX_DECODED_V2_LABEL_BYTES}-byte typed-array storage limit`,
    );
  }
}

/** Bound a byte-channel result copy while its caller-owned envelope remains live. */
export function assertDecodedByteChannelCopyFits(retainedInputBytes: number, payloadBytes: number): void {
  if (
    !Number.isSafeInteger(retainedInputBytes) ||
    retainedInputBytes < 0 ||
    !Number.isSafeInteger(payloadBytes) ||
    payloadBytes < 0 ||
    retainedInputBytes + payloadBytes > MAX_ALF_INPUT_BYTES
  ) {
    refuse(
      'input-too-large',
      `decoded byte channel plus retained input exceed the ${MAX_ALF_INPUT_BYTES}-byte storage limit`,
    );
  }
}

/** Bound the uint32 ALX result while its decoded envelope remains live. */
export function assertDecodedAlxCopyFits(retainedInputBytes: number, payloadBytes: number): void {
  if (
    !Number.isSafeInteger(retainedInputBytes) ||
    retainedInputBytes < 0 ||
    !Number.isSafeInteger(payloadBytes) ||
    payloadBytes < 0 ||
    retainedInputBytes + payloadBytes > MAX_ALF_INPUT_BYTES
  ) {
    refuse(
      'input-too-large',
      `decoded ALX indices plus retained input exceed the ${MAX_ALF_INPUT_BYTES}-byte storage limit`,
    );
  }
}

/** Reject impossible geometry payloads before cloning coordinates to float64. */
export function assertAlpPayloadFits(coordinateCount: number): void {
  if (!Number.isSafeInteger(coordinateCount) || coordinateCount < 0) {
    refuse('writer-input-invalid', 'geometry coordinate count must be a non-negative safe integer');
  }
  if (coordinateCount > MAX_DECODED_GEOMETRY_COMPONENTS) {
    refuse(
      'writer-input-invalid',
      `decoded geometry exceeds the ${MAX_DECODED_GEOMETRY_BYTES}-byte Float64 storage limit`,
    );
  }
  assertEnvelopeFits(0, coordinateCount * 2);
}

/** Bound the float64 coordinate array materialized by the geometry reader. */
export function assertDecodedAlpFits(coordinateCount: number): void {
  if (
    !Number.isSafeInteger(coordinateCount) ||
    coordinateCount < 0 ||
    coordinateCount > MAX_DECODED_GEOMETRY_COMPONENTS
  ) {
    refuse(
      'input-too-large',
      `decoded geometry exceeds the ${MAX_DECODED_GEOMETRY_BYTES}-byte Float64 storage limit`,
    );
  }
}

/** Reject impossible decimation payloads before allocating uint32 buffers. */
export function assertAlxPayloadFits(pointCount: number): void {
  if (!Number.isSafeInteger(pointCount) || pointCount < 0) {
    refuse('writer-input-invalid', 'ALX point count must be a non-negative safe integer');
  }
  assertEnvelopeFits(0, pointCount * 4);
}

function checkedArrayLikeLength(values: ArrayLike<unknown>, name: string): number {
  const length = values.length;
  if (!Number.isSafeInteger(length) || length < 0) {
    refuse('writer-input-invalid', `${name} length must be a non-negative safe integer`);
  }
  return length;
}

function flattenPoints(
  points: ArrayLike<number> | ReadonlyArray<readonly [number, number, number]>,
  inputLength: number,
  nestedPoints: boolean,
): Float64Array {
  if (nestedPoints) {
    const nested = points as ReadonlyArray<readonly [number, number, number]>;
    const flat = new Float64Array(inputLength * 3);
    for (let index = 0; index < inputLength; index += 1) {
      if (!Object.prototype.hasOwnProperty.call(nested, index)) {
        refuse('writer-input-invalid', 'Nested .alp points must not contain array holes');
      }
      const point = nested[index];
      if (!Array.isArray(point) || point.length !== 3) {
        refuse('writer-input-invalid', 'Every .alp point must have exactly three coordinates');
      }
      for (let axis = 0; axis < 3; axis += 1) {
        if (!Object.prototype.hasOwnProperty.call(point, axis)) {
          refuse('writer-input-invalid', 'Every .alp point must contain three coordinates');
        }
        const value = point[axis];
        // NUMBERS, not necessarily FINITE ones — the same rule the flat branch
        // below applies. A non-finite component is a lidar's invalid return and
        // valid .alp content in both codecs; refusing it only here meant the
        // SAME frame was accepted flat and rejected nested, which is an API
        // shape deciding whether data is representable.
        if (typeof value !== 'number') {
          refuse('writer-input-invalid', 'points must contain numbers');
        }
        flat[index * 3 + axis] = value;
      }
    }
    return flat;
  }
  const flat = new Float64Array(inputLength);
  for (let index = 0; index < inputLength; index += 1) {
    const value = (points as ArrayLike<unknown>)[index];
    // NUMBERS, not necessarily FINITE ones. A NaN component is a lidar's
    // invalid return and valid .alp content (spec/format.md 5.1). The int16
    // branch encodes whole invalid points with its reserved sentinel and
    // refuses partly non-finite points.
    if (typeof value !== 'number') {
      refuse('writer-input-invalid', 'points must contain numbers');
    }
    flat[index] = value;
  }
  if (flat.length % 3 !== 0) refuse('writer-input-invalid', '.alp coordinates must be XYZ triples');
  return flat;
}

function finiteArray(values: ArrayLike<number>, name: string): void {
  for (let index = 0; index < values.length; index += 1) {
    if (!Number.isFinite(values[index])) refuse('writer-input-invalid', `${name} must contain finite numbers`);
  }
}

function finiteMidpoint(low: number, high: number): number {
  const span = high - low;
  return Number.isFinite(span) ? low + span / 2 : low / 2 + high / 2;
}

function dividedDifference(value: number, origin: number, divisor: number): number {
  const difference = value - origin;
  return Number.isFinite(difference) ? difference / divisor : value / divisor - origin / divisor;
}

function validateAndNormalizeGeometryAnchors(metadata: AlfMetadata): void {
  for (const key of GEOMETRY_ANCHOR_KEYS) {
    if (!own(metadata, key)) continue;
    const raw = metadata[key];
    if (key === 'device_position' && isPlainObject(raw) && Object.keys(raw).length === 0) continue;
    let components: readonly unknown[];
    let preserved: Record<string, unknown> = {};
    if (Array.isArray(raw) && raw.length === 3) {
      components = raw;
    } else if (
      isPlainObject(raw) &&
      own(raw, 'x') &&
      own(raw, 'y') &&
      own(raw, 'z')
    ) {
      components = [raw['x'], raw['y'], raw['z']];
      preserved = Object.fromEntries(
        Object.entries(raw).filter(([member]) => member !== 'x' && member !== 'y' && member !== 'z'),
      );
    } else {
      refuse('writer-input-invalid', `${key} must contain x, y, and z or be a three-element array`);
    }
    if (components.some((component) => typeof component !== 'number' || !Number.isFinite(component))) {
      refuse('writer-input-invalid', `${key} components must be finite JSON numbers`);
    }
    metadata[key] = {
      ...preserved,
      x: components[0] as number,
      y: components[1] as number,
      z: components[2] as number,
    };
  }
}

function preflightAlpEnvelope(
  metadata: AlfMetadata,
  coordinateCount: number,
  codec: GeometryCodec,
  frameFields: AlfMetadata,
): void {
  // Every finite binary64 string is at most 24 characters. Reserve the largest
  // generated geometry header before cloning or encoding the caller's point
  // storage; the exact header is serialized again after quantization.
  const longestNumber = -Number.MAX_VALUE;
  const pointCount = coordinateCount / 3;
  metadata['_total_points'] = pointCount;
  if (codec === 'fp16') {
    for (const key of ['_geom_codec', '_geom_scale', '_geom_offset', '_geom_max_err_m']) delete metadata[key];
  } else {
    metadata['_geom_codec'] = 'int16';
    metadata['_geom_scale'] = longestNumber;
    metadata['_geom_offset'] = [longestNumber, longestNumber, longestNumber];
    metadata['_geom_max_err_m'] = longestNumber;
  }
  // The explicit options OWN these two keys. Metadata read back from another
  // .alp carries the previous file's declaration, and merging over it would
  // leave a stale `_frame_rotation` beside a `world-aligned` write, or a
  // `_frame_convention` on a v1 envelope — headers this writer's own reader
  // refuses. Cleared first, then set from the arguments.
  delete metadata['_frame_convention'];
  delete metadata['_frame_rotation'];
  Object.assign(metadata, frameFields);
  if (coordinateCount === 0) {
    delete metadata['box'];
    delete metadata['_box_frame'];
  } else {
    metadata['box'] = '0'.repeat(6 * 24 + 5);
    metadata['_box_frame'] = 'local';
  }
  preflightEnvelope(metadata, coordinateCount * 2);
}

/**
 * Validate the caller's frame declaration and return the header it becomes.
 *
 * NORMALIZATION LIVES HERE, and only here. A pose quaternion that has been
 * through a few compositions is a unit quaternion to about 1e-7, not to the
 * 1e-9 the reader requires, so a writer that passed it through untouched would
 * emit files its own reader refuses. The writer knows what the caller meant; a
 * reader knows only what it was handed, and normalizing there would turn a
 * producer's bug into a silent scale on every coordinate.
 */
function writerFrameFields(options: WriteAlpOptions, codec: GeometryCodec): AlfMetadata {
  const { frameConvention, frameRotation } = options;
  if (codec !== 'int16') {
    if (frameConvention !== undefined || frameRotation !== undefined) {
      refuse(
        'writer-input-invalid',
        'frameConvention/frameRotation are version-3 fields and this write is fp16; readers ' +
          'apply no rotation to v1 geometry, so recording one would be a promise the format ' +
          'does not keep',
      );
    }
    return {};
  }
  if (frameConvention === undefined) {
    refuse(
      'writer-input-invalid',
      'codec "int16" requires frameConvention: the points are the world points with an origin ' +
        'subtracted, and the file must say whether they were rotated into the device frame too',
    );
  }
  if (frameConvention !== FRAME_WORLD_ALIGNED && frameConvention !== FRAME_DEVICE_ALIGNED) {
    refuse('writer-input-invalid', `Unknown frameConvention ${String(frameConvention)}`);
  }
  if (frameConvention === FRAME_WORLD_ALIGNED) {
    if (frameRotation !== undefined) {
      refuse(
        'writer-input-invalid',
        'frameConvention world-aligned means no rotation was applied, so a frameRotation cannot ' +
          'also be recorded',
      );
    }
    return { _frame_convention: FRAME_WORLD_ALIGNED };
  }
  if (frameRotation === undefined) {
    refuse('writer-input-invalid', 'frameConvention device-aligned requires frameRotation');
  }
  if (frameRotation === null) {
    refuse('writer-input-invalid', 'frameRotation must be an object with w, x, y and z');
  }
  // Pose objects can expose getters. Validation, normalization and emission
  // must all use the same sampled components, as they do for the geometry grid.
  const rotation = {
    w: frameRotation.w,
    x: frameRotation.x,
    y: frameRotation.y,
    z: frameRotation.z,
  };
  let normSquared = 0;
  for (const axis of ['w', 'x', 'y', 'z'] as const) {
    const component = rotation[axis];
    if (typeof component !== 'number' || !Number.isFinite(component)) {
      refuse('writer-input-invalid', `frameRotation.${axis} must be a finite number`);
    }
    normSquared += component * component;
  }
  const norm = Math.sqrt(normSquared);
  if (norm === 0) {
    refuse('writer-input-invalid', 'frameRotation is the zero quaternion and describes no rotation');
  }
  // Drift is a small number. |q| = 2 is a units or ordering mistake, and scaling
  // it to length 1 would manufacture a plausible rotation out of it.
  if (norm < 0.9 || norm > 1.1) {
    refuse(
      'writer-input-invalid',
      `frameRotation has |q| = ${norm}, too far from 1 to be accumulated drift`,
    );
  }
  return {
    _frame_convention: FRAME_DEVICE_ALIGNED,
    _frame_rotation: {
      w: rotation.w / norm,
      x: rotation.x / norm,
      y: rotation.y / norm,
      z: rotation.z / norm,
    },
  };
}

export async function writeAlp(
  pointsInput: ArrayLike<number> | ReadonlyArray<readonly [number, number, number]>,
  options: WriteAlpOptions = {},
): Promise<Uint8Array> {
  return writeAlpInternal(pointsInput, options);
}

async function writeAlpInternal(
  pointsInput: ArrayLike<number> | ReadonlyArray<readonly [number, number, number]>,
  options: WriteAlpOptions,
  minimumGeomErrorM?: number,
): Promise<Uint8Array> {
  if (minimumGeomErrorM !== undefined && (!Number.isFinite(minimumGeomErrorM) || minimumGeomErrorM < 0)) {
    refuse('writer-input-invalid', 'minimum geometry error bound must be finite and non-negative');
  }
  const inputLength = checkedArrayLikeLength(pointsInput, 'points');
  const nestedPoints = Array.isArray(pointsInput) && inputLength > 0 && Array.isArray(pointsInput[0]);
  const coordinateCount = nestedPoints ? inputLength * 3 : inputLength;
  assertAlpPayloadFits(coordinateCount);
  const frameFields = writerFrameFields(options, options.codec ?? 'fp16');
  if (coordinateCount % 3 !== 0) refuse('writer-input-invalid', '.alp coordinates must be XYZ triples');
  const codec = options.codec ?? 'fp16';
  if (codec !== 'fp16' && codec !== 'int16') {
    refuse('writer-input-invalid', `Unknown geometry codec ${String(codec)}`);
  }
  const metadata = cloneMetadata(options.metadata);
  validateAndNormalizeGeometryAnchors(metadata);
  preflightAlpEnvelope(metadata, coordinateCount, codec, frameFields);
  const points = flattenPoints(pointsInput, inputLength, nestedPoints);
  metadata['_total_points'] = points.length / 3;
  const decodedLo = [Infinity, Infinity, Infinity];
  const decodedHi = [-Infinity, -Infinity, -Infinity];
  let payload: Uint8Array;
  let version: AlfVersion;
  if (codec === 'fp16') {
    // FINITE components only: Math.max against NaN is NaN, and NaN >= budget is
    // false, so one invalid return silently disabled the budget guard entirely.
    let largest = 0;
    for (const point of points) if (Number.isFinite(point)) largest = Math.max(largest, Math.abs(point));
    if ((options.checkBudget ?? true) && largest >= FP16_COMPONENT_BUDGET_M) {
      refuse('writer-input-invalid', `float16 coordinate reaches ${largest} m; localize or tile the frame`);
    }
    for (const key of ['_geom_codec', '_geom_scale', '_geom_offset', '_geom_max_err_m']) delete metadata[key];
    payload = new Uint8Array(points.length * 2);
    const view = new DataView(payload.buffer);
    for (let index = 0; index < points.length; index += 1) {
      const bits = encodeFloat16(points[index] ?? 0);
      view.setUint16(index * 2, bits, true);
      const decoded = decodeFloat16(bits);
      const axis = index % 3;
      // The box is the FINITE extent. A single invalid return would otherwise
      // make every axis NaN, and the server turns this header straight into
      // item-metadata spatial bounds — so the frame drops out of every spatial
      // query. Inf is excluded from the other end for the same reason: "inf" in
      // a six-number box string is unparseable to every consumer.
      if (Number.isFinite(decoded)) {
        decodedLo[axis] = Math.min(decodedLo[axis] ?? Infinity, decoded);
        decodedHi[axis] = Math.max(decodedHi[axis] ?? -Infinity, decoded);
      }
    }
    version = ALF_VERSION_V1;
  } else {
    // Fixed-point encodes an invalid return as INT16_NO_RETURN (spec/format.md
    // §5.2), so a whole non-finite point is ordinary content here.
    //
    // A PARTLY non-finite point is still refused, matching the Python
    // reference: the sentinel marks an entire point as having no return, so
    // there is no encoding for "x is invalid but y is 3.0", and inventing one
    // would guess at data no sensor produces.
    for (let index = 0; index < points.length; index += 3) {
      if (pointIsFinite(points, index)) continue;
      if (!pointIsWhollyNonFinite(points, index)) {
        refuse(
          'writer-input-invalid',
          "codec 'int16' found a point with SOME but not all components non-finite. An invalid return is a whole point — a partial one has no interpretation, and encoding it would have to guess. Make each such point entirely non-finite, or filter it out.",
        );
      }
    }
    let offset: readonly [number, number, number];
    const suppliedOffset = options.geomOffset;
    if (suppliedOffset !== undefined) {
      if (suppliedOffset.length !== 3) {
        refuse('writer-input-invalid', 'geomOffset must contain exactly three components');
      }
      // A caller can supply an accessor-backed tuple. Snapshot every component
      // once so validation, quantization, reconstruction, and header emission
      // cannot observe different coordinate grids.
      offset = [suppliedOffset[0], suppliedOffset[1], suppliedOffset[2]];
    } else if (points.length === 0) {
      offset = [0, 0, 0];
    } else {
      // REAL coordinates only, exactly as the scale derivation below. `Math.min`
      // against NaN is NaN, so one invalid return used to poison the midpoint,
      // and `finiteArray` then rejected the writer's OWN derived offset — a
      // frame the format explicitly permits could not be written at all.
      const minima = [Infinity, Infinity, Infinity];
      const maxima = [-Infinity, -Infinity, -Infinity];
      let sawFinitePoint = false;
      for (let index = 0; index < points.length; index += 3) {
        if (!pointIsFinite(points, index)) continue;
        sawFinitePoint = true;
        for (let axis = 0; axis < 3; axis += 1) {
          const value = points[index + axis] ?? 0;
          minima[axis] = Math.min(minima[axis] ?? Infinity, value);
          maxima[axis] = Math.max(maxima[axis] ?? -Infinity, value);
        }
      }
      // A frame of nothing but invalid returns has no extent to centre on. Any
      // finite origin decodes identically — every word is the sentinel — so
      // take the same zero the empty-frame branch above takes, rather than the
      // Infinity/-Infinity seeds, whose midpoint is NaN.
      offset = sawFinitePoint
        ? ([0, 1, 2].map((axis) => finiteMidpoint(minima[axis] ?? 0, maxima[axis] ?? 0)) as [
            number,
            number,
            number,
          ])
        : [0, 0, 0];
    }
    finiteArray(offset, 'geomOffset');
    let scale = options.geomScale;
    if (scale === undefined) {
      scale = MIN_GEOM_SCALE_M;
      // REAL coordinates only. An invalid return holds no position, so sizing
      // the grid from it gives a wrong grid (spec/format.md §5.2).
      for (let index = 0; index < points.length; index += 3) {
        if (!pointIsFinite(points, index)) continue;
        for (let component = 0; component < 3; component += 1) {
          scale = Math.max(
            scale,
            Math.abs(dividedDifference(points[index + component] ?? 0, offset[component] ?? 0, 32767)),
          );
        }
      }
      // Number.MAX_VALUE / 32767 rounds upward: multiplying that quotient by
      // 32767 becomes Infinity even though the source endpoint is finite.
      // One ULP downward still rounds the endpoint to the same int16 code and
      // makes the mandated `stored * scale + offset` reconstruction finite.
      if (!Number.isFinite(scale * 32767)) scale = nextDown(scale);
    }
    if (!Number.isFinite(scale) || scale <= 0) refuse('writer-input-invalid', 'geomScale must be positive and finite');
    payload = new Uint8Array(points.length * 2);
    const view = new DataView(payload.buffer);
    let maxError = 0;
    for (let index = 0; index < points.length; index += 3) {
      if (!pointIsFinite(points, index)) {
        // Written EXPLICITLY. Rounding a NaN and narrowing it to int16 is
        // implementation-defined, so the marker would become an arbitrary
        // coordinate that looks like data (spec/format.md §5.2).
        view.setInt16(index * 2, INT16_NO_RETURN, true);
        view.setInt16((index + 1) * 2, INT16_NO_RETURN, true);
        view.setInt16((index + 2) * 2, INT16_NO_RETURN, true);
        continue;
      }
      for (let component = 0; component < 3; component += 1) {
      const original = points[index + component] ?? 0;
      const quantized = roundTiesToEven(dividedDifference(original, offset[component] ?? 0, scale));
      if (quantized <= INT16_NO_RETURN || quantized > 32767) refuse('writer-input-invalid', 'Geometry exceeds the selected int16 grid');
      view.setInt16((index + component) * 2, quantized, true);
      const reconstructed = quantized * scale + (offset[component] ?? 0);
      const error = Math.abs(original - reconstructed);
      if (!Number.isFinite(reconstructed) || !Number.isFinite(error)) {
        refuse('writer-input-invalid', 'Selected int16 grid reconstructs outside finite binary64 coordinates');
      }
      maxError = Math.max(maxError, error);
      decodedLo[component] = Math.min(decodedLo[component] ?? Infinity, reconstructed);
      decodedHi[component] = Math.max(decodedHi[component] ?? -Infinity, reconstructed);
      }
    }
    if (points.length > 0) {
      let halfScale = scale / 2;
      if (scale < 1e-300) halfScale = Math.max(halfScale, nextUp(halfScale));
      if (halfScale <= 0) halfScale = Number.MIN_VALUE;
      maxError = Math.max(maxError, halfScale);
    }
    if (minimumGeomErrorM !== undefined) {
      maxError = Math.max(maxError, minimumGeomErrorM);
    }
    if ((options.checkBudget ?? true) && maxError >= INT16_ERROR_BUDGET_M) {
      refuse('writer-input-invalid', `int16 reconstruction error ${maxError} m exceeds the 0.25 m budget`);
    }
    // The explicit options OWN these two keys. Metadata read back from another
  // .alp carries the previous file's declaration, and merging over it would
  // leave a stale `_frame_rotation` beside a `world-aligned` write, or a
  // `_frame_convention` on a v1 envelope — headers this writer's own reader
  // refuses. Cleared first, then set from the arguments.
  delete metadata['_frame_convention'];
  delete metadata['_frame_rotation'];
  Object.assign(metadata, frameFields);
    metadata['_geom_codec'] = 'int16';
    metadata['_geom_scale'] = scale;
    metadata['_geom_offset'] = [...offset];
    metadata['_geom_max_err_m'] = maxError;
    version = ALF_VERSION_V3;
  }
  // A frame with no finite component on some axis has no extent to publish
  // there, so it publishes NO box rather than one containing NaN or Infinity.
  const hasExtent =
    points.length > 0 && decodedLo.every((value) => Number.isFinite(value)) && decodedHi.every((value) => Number.isFinite(value));
  if (!hasExtent) {
    delete metadata['box'];
    delete metadata['_box_frame'];
  } else {
    metadata['box'] = `${decodedLo[0]}:${decodedLo[1]}:${decodedLo[2]}:${decodedHi[0]}:${decodedHi[1]}:${decodedHi[2]}`;
    metadata['_box_frame'] = 'local';
  }
  return encodePreparedEnvelope(
    version,
    preflightEnvelope(metadata, payload.length),
    payload,
    options.gzip ?? false,
  );
}

function byteValues(values: ArrayLike<number>, name: string, knownLength?: number): Uint8Array {
  const length = knownLength ?? checkedArrayLikeLength(values, name);
  const output = new Uint8Array(length);
  for (let index = 0; index < length; index += 1) {
    const value = values[index];
    if (!Number.isInteger(value) || (value ?? -1) < 0 || (value ?? 256) > 255) {
      refuse('writer-input-invalid', `${name} must contain uint8 values`);
    }
    output[index] = value ?? 0;
  }
  return output;
}

function checkedIntensityDeciles(inherited: unknown): number[] | null {
  if (inherited === undefined) return null;
  if (!Array.isArray(inherited) || inherited.length !== 9) {
    refuse('writer-input-invalid', 'Intensity deciles must contain the 10th through 90th percentiles');
  }
  const deciles = inherited.map((value) => {
    if (!Number.isInteger(value) || (value as number) < 0 || (value as number) > 255) {
      refuse('writer-input-invalid', 'Intensity deciles must be uint8 values');
    }
    return value as number;
  });
  if (deciles.some((value, index) => index > 0 && value < (deciles[index - 1] ?? 0))) {
    refuse('writer-input-invalid', 'Intensity deciles must be non-decreasing');
  }
  return deciles;
}

function decilesFromHistogram(histogram: Float64Array, count: number): number[] {
  if (count === 0) return new Array<number>(9).fill(0);
  return Array.from({ length: 9 }, (_, index) => {
    const position = Math.min(Math.trunc((count / 10) * (index + 1)), count - 1);
    let cumulative = 0;
    for (let value = 0; value < histogram.length; value += 1) {
      cumulative += histogram[value] ?? 0;
      if (cumulative > position) return value;
    }
    return 255;
  });
}

function scaledIntensity(value: number, low: number, high: number): number {
  if (high === low) return 0;
  const span = high - low;
  const scaled = Number.isFinite(span)
    ? ((value - low) / span) * 255
    : ((value / 2 - low / 2) / (high / 2 - low / 2)) * 255;
  // Samples are separately proved in-range. This clip covers only floating
  // arithmetic at the endpoints; it cannot hide stale caller bounds.
  return Math.max(0, Math.min(255, Math.trunc(scaled)));
}

export async function writeAli(values: ArrayLike<number>, options: WriteAliOptions = {}): Promise<Uint8Array> {
  const valueCount = checkedArrayLikeLength(values, 'intensity');
  const range = options.inputRange;
  let low: number | undefined;
  let high: number | undefined;
  if (range !== undefined) {
    if (!Array.isArray(range) || range.length !== 2) {
      refuse('writer-input-invalid', 'inputRange must contain exactly two bounds');
    }
    low = range[0];
    high = range[1];
    if (!Number.isFinite(low) || !Number.isFinite(high) || (high ?? -Infinity) < (low ?? Infinity)) {
      refuse('writer-input-invalid', 'inputRange must be finite and non-decreasing');
    }
    if (
      (Number.isInteger(low) && Math.abs(low ?? 0) >= 2 ** 53) ||
      (Number.isInteger(high) && Math.abs(high ?? 0) >= 2 ** 53)
    ) {
      refuse(
        'writer-input-invalid',
        'Integer intensity samples and bounds must stay below 2**53 so consecutive values remain distinguishable',
      );
    }
  }
  const metadata = cloneMetadata(options.metadata);
  metadata['_total_points'] = valueCount;
  metadata['_min_intensity'] = 0;
  metadata['_max_intensity'] = 255;
  const inherited = options.deciles ?? metadata['_intensities_every_10_percent'];
  let deciles = checkedIntensityDeciles(inherited);
  const histogram = deciles === null ? new Float64Array(256) : null;

  // Auto-generated deciles can use at most three digits apiece. Reserve that
  // exact worst-case serialized shape before scanning a near-limit ArrayLike;
  // the actual deciles are serialized and checked again before payload allocation.
  metadata['_intensities_every_10_percent'] = deciles ?? new Array<number>(9).fill(255);
  const reservedHeader = preflightEnvelope(metadata, valueCount);

  const intensity = new Uint8Array(valueCount);
  if (range !== undefined) {
    let minimum = Infinity;
    let maximum = -Infinity;
    for (let index = 0; index < valueCount; index += 1) {
      const value = values[index];
      if (!Number.isFinite(value)) refuse('writer-input-invalid', 'Intensity values must be finite');
      if (Number.isInteger(value) && Math.abs(value ?? 0) >= 2 ** 53) {
        refuse(
          'writer-input-invalid',
          'Integer intensity samples and bounds must stay below 2**53 so consecutive values remain distinguishable',
        );
      }
      minimum = Math.min(minimum, value ?? Infinity);
      maximum = Math.max(maximum, value ?? -Infinity);
      const encoded = scaledIntensity(value ?? low ?? 0, low ?? 0, high ?? 0);
      intensity[index] = encoded;
      if (histogram !== null) histogram[encoded] = (histogram[encoded] ?? 0) + 1;
    }
    if (minimum < (low ?? Infinity) || maximum > (high ?? -Infinity)) {
      refuse('writer-input-invalid', `Intensity samples [${minimum}, ${maximum}] fall outside inputRange [${low}, ${high}]`);
    }
  } else {
    for (let index = 0; index < valueCount; index += 1) {
      const value = values[index];
      if (!Number.isInteger(value) || (value ?? -1) < 0 || (value ?? 256) > 255) {
        refuse('writer-input-invalid', 'intensity must contain uint8 values');
      }
      intensity[index] = value ?? 0;
      if (histogram !== null) histogram[value ?? 0] = (histogram[value ?? 0] ?? 0) + 1;
    }
  }
  if (deciles === null) deciles = decilesFromHistogram(histogram ?? new Float64Array(256), valueCount);
  metadata['_intensities_every_10_percent'] = deciles;
  const header = inherited === undefined ? preflightEnvelope(metadata, valueCount) : reservedHeader;
  return encodePreparedEnvelope(ALF_VERSION_V1, header, intensity, options.gzip ?? false);
}

export async function writeAlc(
  paletteIndicesInput: ArrayLike<number>,
  paletteInput: readonly string[],
  options: WriteOptions = {},
): Promise<Uint8Array> {
  const paletteIndexCount = checkedArrayLikeLength(paletteIndicesInput, 'palette indices');
  if (!Array.isArray(paletteInput)) {
    refuse('writer-input-invalid', 'palette must be an array');
  }
  const paletteLength = paletteInput.length;
  if (paletteLength === 0 || paletteLength > 255) {
    refuse('writer-input-invalid', 'palette must contain between 1 and 255 entries');
  }
  const paletteSnapshot = new Array<unknown>(paletteLength);
  for (let index = 0; index < paletteLength; index += 1) {
    paletteSnapshot[index] = paletteInput[index];
  }
  const metadata = cloneMetadata(options.metadata);
  metadata['palette'] = paletteSnapshot;
  const palette = checkedPalette(metadata);
  metadata['_total_points'] = paletteIndexCount;
  metadata['palette'] = [...palette];
  const header = preflightEnvelope(metadata, paletteIndexCount);
  const paletteIndices = byteValues(paletteIndicesInput, 'palette indices', paletteIndexCount);
  for (const index of paletteIndices) {
    if (index > palette.length) refuse('writer-input-invalid', `Palette index ${index} exceeds sentinel ${palette.length}`);
  }
  return encodePreparedEnvelope(ALF_VERSION_V1, header, paletteIndices, options.gzip ?? false);
}

async function writeAllInternal(
  categoriesInput: ArrayLike<number>,
  options: WriteAllOptions = {},
  preserveMissingInstanceMap = false,
): Promise<Uint8Array> {
  const categoryCount = checkedArrayLikeLength(categoriesInput, 'categories');
  const instanceIdsInput = options.instanceIds;
  if (instanceIdsInput !== undefined && instanceIdsInput !== null) {
    const instanceCount = checkedArrayLikeLength(instanceIdsInput, 'instanceIds');
    if (instanceCount !== categoryCount) {
      refuse('writer-input-invalid', 'categories and instanceIds must have equal lengths');
    }
    assertV2LabelPayloadFits(categoryCount);
    const metadata = cloneMetadata(options.metadata);
    metadata['_total_points'] = categoryCount;
    metadata['_format_version'] = 2;
    if (options.instanceMap === null) delete metadata['instance_map'];
    else if (options.instanceMap !== undefined) metadata['instance_map'] = { ...options.instanceMap };
    const header = preflightEnvelope(metadata, categoryCount * 3);
    const categories = byteValues(categoriesInput, 'categories', categoryCount);
    const payload = new Uint8Array(categories.length * 3);
    const ids = new Uint16Array(categories.length);
    for (let index = 0; index < categories.length; index += 1) {
      const id = instanceIdsInput[index];
      if (!Number.isInteger(id) || (id ?? -1) < 0 || (id ?? 65535) > 65534) {
        refuse('writer-input-invalid', 'instanceIds must be integers in 0..65534');
      }
      ids[index] = id ?? 0;
      payload[index * 3] = categories[index] ?? 0;
      payload[index * 3 + 1] = (id ?? 0) & 0xff;
      payload[index * 3 + 2] = (id ?? 0) >>> 8;
    }
    const instanceMap = checkedInstanceMap(metadata, ids);
    if (!preserveMissingInstanceMap && instanceMap === null && ids.some((id) => id !== 0)) {
      refuse('instance-map-incomplete', 'instance_map is required for newly authored nonzero instance IDs');
    }
    return encodePreparedEnvelope(ALF_VERSION_V2, header, payload, options.gzip ?? false);
  } else {
    const metadata = cloneMetadata(options.metadata);
    metadata['_total_points'] = categoryCount;
    if (options.instanceMap !== undefined && options.instanceMap !== null) {
      refuse('writer-input-invalid', 'instanceMap requires instanceIds');
    }
    delete metadata['_format_version'];
    delete metadata['instance_map'];
    const header = preflightEnvelope(metadata, categoryCount);
    const categories = byteValues(categoriesInput, 'categories', categoryCount);
    return encodePreparedEnvelope(ALF_VERSION_V1, header, categories, options.gzip ?? false);
  }
}

export async function writeAll(
  categoriesInput: ArrayLike<number>,
  options: WriteAllOptions = {},
): Promise<Uint8Array> {
  return writeAllInternal(categoriesInput, options);
}

export async function writeAlx(keptInput: ArrayLike<number>, options: WriteAlxOptions): Promise<Uint8Array> {
  const keptCount = checkedArrayLikeLength(keptInput, 'keptIndices');
  if (!Number.isSafeInteger(options.alpTotalPoints) || options.alpTotalPoints < 0 || options.alpTotalPoints !== keptCount) {
    refuse('writer-input-invalid', 'alpTotalPoints must equal the number of kept indices');
  }
  const sourceTotalPoints = options.sourceTotalPoints;
  if (!Number.isFinite(sourceTotalPoints) || !Number.isInteger(sourceTotalPoints) || sourceTotalPoints < 0) {
    refuse('writer-input-invalid', 'sourceTotalPoints must be a non-negative integer');
  }
  assertAlxPayloadFits(keptCount);
  const metadata = cloneMetadata(options.metadata);
  metadata['_total_points'] = keptCount;
  metadata['_source_total_points'] = sourceTotalPoints;
  const header = preflightEnvelope(metadata, keptCount * 4);
  const payload = new Uint8Array(keptCount * 4);
  const normalized = new Uint32Array(payload.buffer, payload.byteOffset, keptCount);
  const view = new DataView(payload.buffer);
  for (let index = 0; index < keptCount; index += 1) {
    const kept = keptInput[index];
    if (!Number.isInteger(kept) || (kept ?? -1) < 0 || (kept ?? 2 ** 32) > 2 ** 32 - 1) {
      refuse('writer-input-invalid', 'keptIndices must contain uint32 values');
    }
    if ((kept ?? 0) >= sourceTotalPoints) refuse('writer-input-invalid', `kept index ${kept} is out of source range`);
    view.setUint32(index * 4, kept ?? 0, true);
  }
  // Preserve caller order on the wire before sorting this same backing buffer
  // for duplicate detection. This keeps one four-byte-per-index temporary
  // instead of retaining separate normalized and payload allocations.
  const encoded = await encodePreparedEnvelope(ALF_VERSION_V1, header, payload, options.gzip ?? false);
  // Uint32Array follows platform byte order; convert the explicit little-endian
  // wire values in-place only on a hypothetical big-endian JavaScript host.
  if (!PLATFORM_LITTLE_ENDIAN) {
    for (let index = 0; index < normalized.length; index += 1) {
      normalized[index] = view.getUint32(index * 4, true);
    }
  }
  normalized.sort();
  for (let index = 1; index < normalized.length; index += 1) {
    if (normalized[index] === normalized[index - 1]) {
      refuse('writer-input-invalid', `keptIndices repeats ${normalized[index]}`);
    }
  }
  return encoded;
}

/** Re-encode a decoded file while preserving its channel, layout, codec, and metadata. */
export async function writeAlf(file: AlfFile, options: { gzip?: boolean; checkBudget?: boolean } = {}): Promise<Uint8Array> {
  const metadata = normalizeDecodedMetadata(file.metadata);
  const shared = { metadata, gzip: options.gzip ?? false };
  switch (file.channel) {
    case 'alp':
      return writeAlpInternal(
        file.points,
        {
          ...shared,
          codec: file.codec,
          ...(options.checkBudget === undefined ? {} : { checkBudget: options.checkBudget }),
          ...(file.codec === 'int16'
            ? {
                geomScale: metadata['_geom_scale'] as number,
                geomOffset: metadata['_geom_offset'] as [number, number, number],
                // Carried through rather than re-derived: a re-encode does not
                // move the points, so it cannot change which frame they are in,
                // and inventing a convention here would be the original bug
                // arriving through a converter.
                ...(file.frameConvention === null ? {} : { frameConvention: file.frameConvention }),
                ...(file.frameRotation === null ? {} : { frameRotation: file.frameRotation }),
              }
            : {}),
        },
        file.codec === 'int16' ? file.maxGeomErrorM : undefined,
      );
    case 'ali':
      return writeAli(file.intensity, shared);
    case 'alc':
      return writeAlc(file.paletteIndices, file.palette, shared);
    case 'all':
      return writeAllInternal(
        file.categories,
        {
          ...shared,
          ...(file.formatVersion === 2
            ? {
                instanceIds: file.instanceIds,
                ...(file.instanceMap === null ? {} : { instanceMap: file.instanceMap }),
              }
            : {}),
        },
        true,
      );
    case 'alx':
      return writeAlx(file.keptIndices, {
        ...shared,
        alpTotalPoints: file.keptIndices.length,
        sourceTotalPoints: file.sourceTotalPoints,
      });
  }
}
