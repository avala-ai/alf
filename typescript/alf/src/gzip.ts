import { Inflate, gzip as pakoGzip } from 'pako';

import { AlfError, refuse } from './errors.js';

const GZIP_MAGIC_0 = 0x1f;
const GZIP_MAGIC_1 = 0x8b;
const MAX_DECOMPRESSED_BYTES = 512 * 1024 * 1024;
export const MAX_COMPRESSED_INPUT_BYTES = MAX_DECOMPRESSED_BYTES + 1024 * 1024;
const COMPRESSED_CHUNK_BYTES = 64 * 1024;
const DECOMPRESSED_CHUNK_BYTES = 64 * 1024;
const MAX_HUFFMAN_CACHE_ENTRIES = 64;

export function isGzip(bytes: Uint8Array): boolean {
  return bytes.length >= 2 && bytes[0] === GZIP_MAGIC_0 && bytes[1] === GZIP_MAGIC_1;
}

/** Safe-integer cursor helpers, exported only through this non-root module for boundary regression tests. */
export function bitCursorByteIndex(bit: number): number {
  return Math.floor(bit / 8);
}

export function bitCursorBitOffset(bit: number): number {
  return bit % 8;
}

export function alignedBitCursor(bit: number): number {
  return Math.ceil(bit / 8) * 8;
}

class BitReader {
  private bit = 0;

  constructor(private readonly bytes: Uint8Array, byteOffset: number) {
    this.bit = byteOffset * 8;
  }

  read(count: number): number {
    if (count < 0 || count > 24 || this.bit + count > this.bytes.length * 8) {
      refuse('gzip-invalid', 'Truncated DEFLATE stream');
    }
    let result = 0;
    for (let i = 0; i < count; i += 1) {
      const byte = this.bytes[bitCursorByteIndex(this.bit)];
      result |= (((byte ?? 0) >>> bitCursorBitOffset(this.bit)) & 1) << i;
      this.bit += 1;
    }
    return result;
  }

  alignByte(): void {
    this.bit = alignedBitCursor(this.bit);
  }

  skipBytes(count: number): void {
    this.alignByte();
    const bits = count * 8;
    if (!Number.isSafeInteger(bits) || bits < 0 || this.bit + bits > this.bytes.length * 8) {
      refuse('gzip-invalid', 'Truncated DEFLATE stored block');
    }
    this.bit += bits;
  }

  get byteOffset(): number {
    return Math.ceil(this.bit / 8);
  }
}

interface HuffmanTable {
  readonly byLength: readonly Map<number, number>[];
  readonly maxLength: number;
}

let huffmanTableConstructions = 0;

function reverseBits(value: number, count: number): number {
  let result = 0;
  for (let i = 0; i < count; i += 1) result = (result << 1) | ((value >>> i) & 1);
  return result;
}

function huffman(lengths: readonly number[]): HuffmanTable {
  huffmanTableConstructions += 1;
  const maxLength = Math.max(0, ...lengths);
  const counts = new Array<number>(maxLength + 1).fill(0);
  for (const length of lengths) if (length > 0) counts[length] = (counts[length] ?? 0) + 1;
  const next = new Array<number>(maxLength + 1).fill(0);
  let code = 0;
  for (let bits = 1; bits <= maxLength; bits += 1) {
    code = (code + (counts[bits - 1] ?? 0)) << 1;
    next[bits] = code;
  }
  const byLength = Array.from({ length: maxLength + 1 }, () => new Map<number, number>());
  lengths.forEach((length, symbol) => {
    if (length === 0) return;
    const canonical = next[length] ?? 0;
    next[length] = canonical + 1;
    byLength[length]?.set(reverseBits(canonical, length), symbol);
  });
  return { byLength, maxLength };
}

class HuffmanCache {
  private readonly entries = new Map<string, HuffmanTable>();

  get(lengths: readonly number[]): HuffmanTable {
    // DEFLATE code lengths are values in 0..15, so one UTF-16 code unit per
    // entry is a collision-free compact key. Keep the first definitions when
    // the cache fills: an attacker cycling a larger set cannot evict the common
    // definitions and force them to allocate Maps again on every block.
    const key = String.fromCharCode(...lengths);
    const cached = this.entries.get(key);
    if (cached !== undefined) return cached;
    const table = huffman(lengths);
    if (this.entries.size < MAX_HUFFMAN_CACHE_ENTRIES) this.entries.set(key, table);
    return table;
  }
}

function decodeSymbol(reader: BitReader, table: HuffmanTable): number {
  let code = 0;
  for (let length = 1; length <= table.maxLength; length += 1) {
    code |= reader.read(1) << (length - 1);
    const symbol = table.byLength[length]?.get(code);
    if (symbol !== undefined) return symbol;
  }
  refuse('gzip-invalid', 'Invalid DEFLATE Huffman code');
}

const FIXED_LITERAL_LENGTHS = Array.from({ length: 288 }, (_, symbol) =>
  symbol <= 143 ? 8 : symbol <= 255 ? 9 : symbol <= 279 ? 7 : 8,
);
const FIXED_DISTANCE_LENGTHS = new Array<number>(32).fill(5);
const FIXED_LITERAL_TABLE = huffman(FIXED_LITERAL_LENGTHS);
const FIXED_DISTANCE_TABLE = huffman(FIXED_DISTANCE_LENGTHS);

/** Internal regression seam; this module is not part of the package root exports. */
export function huffmanTableBuildCount(): number {
  return huffmanTableConstructions;
}

const CODE_LENGTH_ORDER = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15] as const;
const LENGTH_EXTRA = [
  0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
] as const;
const DISTANCE_EXTRA = [
  0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13,
] as const;

function dynamicTables(reader: BitReader, cache: HuffmanCache): readonly [HuffmanTable, HuffmanTable] {
  const literalCount = reader.read(5) + 257;
  const distanceCount = reader.read(5) + 1;
  const codeLengthCount = reader.read(4) + 4;
  const codeLengths = new Array<number>(19).fill(0);
  for (let i = 0; i < codeLengthCount; i += 1) codeLengths[CODE_LENGTH_ORDER[i] ?? 0] = reader.read(3);
  const codeLengthTable = cache.get(codeLengths);
  const lengths: number[] = [];
  while (lengths.length < literalCount + distanceCount) {
    const symbol = decodeSymbol(reader, codeLengthTable);
    if (symbol <= 15) {
      lengths.push(symbol);
    } else if (symbol === 16) {
      if (lengths.length === 0) refuse('gzip-invalid', 'DEFLATE repeats a missing code length');
      const previous = lengths[lengths.length - 1] ?? 0;
      const repeat = reader.read(2) + 3;
      for (let i = 0; i < repeat; i += 1) lengths.push(previous);
    } else if (symbol === 17) {
      const repeat = reader.read(3) + 3;
      for (let i = 0; i < repeat; i += 1) lengths.push(0);
    } else if (symbol === 18) {
      const repeat = reader.read(7) + 11;
      for (let i = 0; i < repeat; i += 1) lengths.push(0);
    } else {
      refuse('gzip-invalid', 'Invalid DEFLATE code-length symbol');
    }
    if (lengths.length > literalCount + distanceCount) refuse('gzip-invalid', 'DEFLATE code lengths overflow');
  }
  return [cache.get(lengths.slice(0, literalCount)), cache.get(lengths.slice(literalCount))];
}

function skipCompressedBlock(reader: BitReader, literal: HuffmanTable, distance: HuffmanTable): void {
  for (;;) {
    const symbol = decodeSymbol(reader, literal);
    if (symbol < 256) continue;
    if (symbol === 256) return;
    if (symbol < 257 || symbol > 285) refuse('gzip-invalid', 'Invalid DEFLATE length symbol');
    const lengthIndex = symbol - 257;
    reader.read(LENGTH_EXTRA[lengthIndex] ?? 0);
    const distanceSymbol = decodeSymbol(reader, distance);
    if (distanceSymbol > 29) refuse('gzip-invalid', 'Invalid DEFLATE distance symbol');
    reader.read(DISTANCE_EXTRA[distanceSymbol] ?? 0);
  }
}

function gzipHeaderEnd(bytes: Uint8Array): number {
  if (bytes.length < 10 || !isGzip(bytes) || bytes[2] !== 8) refuse('gzip-invalid', 'Invalid gzip header');
  const flags = bytes[3] ?? 0;
  if ((flags & 0xe0) !== 0) refuse('gzip-invalid', 'Reserved gzip flags are set');
  let offset = 10;
  if ((flags & 4) !== 0) {
    if (offset + 2 > bytes.length) refuse('gzip-invalid', 'Truncated gzip extra header');
    const size = (bytes[offset] ?? 0) | ((bytes[offset + 1] ?? 0) << 8);
    offset += 2 + size;
  }
  for (const flag of [8, 16]) {
    if ((flags & flag) !== 0) {
      while (offset < bytes.length && bytes[offset] !== 0) offset += 1;
      offset += 1;
    }
  }
  if ((flags & 2) !== 0) offset += 2;
  if (offset > bytes.length) refuse('gzip-invalid', 'Truncated gzip header');
  return offset;
}

let gzipMemberBoundaryWalks = 0;

/** Internal regression seam; this module is not part of the package root exports. */
export function gzipMemberBoundaryWalkCount(): number {
  return gzipMemberBoundaryWalks;
}

function firstGzipMemberEnd(bytes: Uint8Array): number {
  // Full-member contract only (`gunzip` needs the trailer and the
  // concatenated-member check). `gunzipPrefix` must not call this: the walk
  // Huffman-decodes every DEFLATE symbol, which is the payload cost a header
  // scan exists to avoid.
  gzipMemberBoundaryWalks += 1;
  const reader = new BitReader(bytes, gzipHeaderEnd(bytes));
  // The fixed-size cache bounds retained tables. Cache misses remain linear in
  // the bounded compressed input; they must not invent a TypeScript-only
  // refusal for a valid DEFLATE member that other ALF readers accept.
  const huffmanCache = new HuffmanCache();
  let final = 0;
  while (final === 0) {
    final = reader.read(1);
    const type = reader.read(2);
    if (type === 0) {
      reader.alignByte();
      const length = reader.read(16);
      const complement = reader.read(16);
      if (((length ^ 0xffff) & 0xffff) !== complement) refuse('gzip-invalid', 'Invalid stored DEFLATE block length');
      reader.skipBytes(length);
    } else if (type === 1) {
      skipCompressedBlock(reader, FIXED_LITERAL_TABLE, FIXED_DISTANCE_TABLE);
    } else if (type === 2) {
      const [literal, distance] = dynamicTables(reader, huffmanCache);
      skipCompressedBlock(reader, literal, distance);
    } else {
      refuse('gzip-invalid', 'Reserved DEFLATE block type');
    }
  }
  reader.alignByte();
  return reader.byteOffset + 8;
}

export async function gunzip(bytes: Uint8Array, maxOutputBytes = MAX_DECOMPRESSED_BYTES): Promise<Uint8Array> {
  const maxInputBytes = maxOutputBytes + 1024 * 1024;
  if (!Number.isSafeInteger(maxInputBytes) || bytes.length > maxInputBytes) {
    refuse('gzip-invalid', `Gzip input exceeds the ${maxInputBytes}-byte compressed-input limit`);
  }
  const end = firstGzipMemberEnd(bytes);
  if (end > bytes.length) refuse('gzip-invalid', 'Truncated gzip trailer');
  // RFC 1952 permits trailing zero padding. Feed only the first member into the
  // inflater (some runtimes reject even permitted padding), while
  // still refusing another member or arbitrary nonzero junk.
  for (let index = end; index < bytes.length; index += 1) {
    if (bytes[index] !== 0) refuse('multiple-gzip-members', 'ALF gzip input contains bytes after its first member');
  }

  if (!Number.isSafeInteger(maxOutputBytes) || maxOutputBytes < 0) {
    refuse('gzip-invalid', 'Invalid gzip decompression limit');
  }
  const trailer = new DataView(bytes.buffer, bytes.byteOffset + end - 8, 8);
  const declaredSize = trailer.getUint32(4, true);
  if (declaredSize > maxOutputBytes) {
    refuse('gzip-invalid', `Gzip member declares ${declaredSize} bytes, past the decompression limit`);
  }

  // First validate and measure while discarding output. ISIZE is attacker
  // controlled, so allocating its claimed size before CRC/length validation
  // lets a tiny corrupt member reserve the entire limit. A second pass writes
  // the already-proven size into one contiguous Uint8Array without the 2x peak
  // of retaining chunks and joining them.
  let measuredSize = 0;
  try {
    inflateMember(bytes, end, (chunk) => {
      const nextSize = measuredSize + chunk.length;
      if (nextSize > maxOutputBytes || nextSize > declaredSize) {
        refuse('gzip-invalid', `Gzip member expands past the ${maxOutputBytes}-byte decompression limit`);
      }
      measuredSize = nextSize;
    });
    if (measuredSize !== declaredSize) refuse('gzip-malformed', 'Gzip output does not match its declared size');

    const output = new Uint8Array(measuredSize);
    let outputOffset = 0;
    inflateMember(bytes, end, (chunk) => {
      const nextOffset = outputOffset + chunk.length;
      if (nextOffset > output.length) refuse('gzip-invalid', 'Gzip output changed between validation passes');
      output.set(chunk, outputOffset);
      outputOffset = nextOffset;
    });
    if (outputOffset !== measuredSize) refuse('gzip-invalid', 'Gzip output changed between validation passes');
    return output;
  } catch (cause) {
    if (cause instanceof AlfError) throw cause;
    throw new AlfError('gzip-malformed', 'Invalid gzip stream', { cause });
  }
}

function inflateMember(bytes: Uint8Array, end: number, onData: (chunk: Uint8Array) => void): void {
  const inflater = new Inflate({ chunkSize: DECOMPRESSED_CHUNK_BYTES });
  inflater.onData = (data): void => {
    onData(data instanceof Uint8Array ? data : new Uint8Array(data));
  };
  for (let offset = 0; offset < end; offset += COMPRESSED_CHUNK_BYTES) {
    const next = Math.min(offset + COMPRESSED_CHUNK_BYTES, end);
    const final = next === end;
    if (!inflater.push(bytes.subarray(offset, next), final)) {
      throw new Error(inflater.msg || `pako inflate error ${inflater.err}`);
    }
  }
  if (inflater.err !== 0) throw new Error(inflater.msg || `pako inflate error ${inflater.err}`);
}

/** Thrown internally to stop inflation once enough output exists. Never escapes. */
class EnoughOutput extends Error {}

/**
 * Inflate only the first `wantedBytes` of a gzip member, then stop.
 *
 * `gunzip` REFUSES a member that expands past its limit; this one stops and
 * returns the prefix, which is a different contract for a different job. It
 * exists for `readAlfHeader`, where the caller wants an envelope out of a file
 * whose payload may be hundreds of megabytes and must not pay to inflate it.
 *
 * It must not call `firstGzipMemberEnd`. That helper fully Huffman-decodes
 * every DEFLATE symbol to find the trailer — a whole-member walk, paid again
 * when `readAlfHeader` retries a header that outran the probe. Prefix inflate
 * feeds pako until it has enough output (or the member ends on its trailer)
 * and stops; trailing RFC 1952 zero padding is never pushed.
 *
 * WHAT THIS GIVES UP, and why that is the honest trade rather than an oversight:
 * a prefix read never reaches the member's trailer, so **the CRC32 and ISIZE are
 * not verified** — `gunzip`'s two-pass measure-then-write cannot run when the
 * point is not to read to the end. Corruption inside the header bytes is still
 * caught (the inflate itself fails, or the header does not parse), but
 * corruption LATER in the payload is invisible here by construction. A caller
 * that needs the whole file checked must call `gunzip`.
 *
 * Returns fewer than `wantedBytes` only when the member genuinely ends first.
 */
export async function gunzipPrefix(bytes: Uint8Array, wantedBytes: number): Promise<Uint8Array> {
  if (!Number.isSafeInteger(wantedBytes) || wantedBytes < 0) {
    refuse('gzip-invalid', 'Invalid gzip prefix length');
  }
  if (wantedBytes === 0) return new Uint8Array(0);

  const chunks: Uint8Array[] = [];
  let collected = 0;
  // Cap the inflater's output buffer at the request so one `onData` is enough
  // to stop. A 64 KiB default chunk would expand a highly compressible prefix
  // far past the envelope before the callback could notice.
  const inflater = new Inflate({ chunkSize: Math.min(DECOMPRESSED_CHUNK_BYTES, wantedBytes) });
  inflater.onData = (data): void => {
    const chunk = data instanceof Uint8Array ? data : new Uint8Array(data);
    chunks.push(chunk);
    collected += chunk.length;
    if (collected >= wantedBytes) throw new EnoughOutput();
  };
  // `@types/pako` omits `ended`. onEnd is the typed seam: Z_OK means the gzip
  // trailer arrived, so later RFC 1952 zero padding must not be pushed.
  let finished = false;
  const onEnd = inflater.onEnd.bind(inflater);
  inflater.onEnd = (status: number): void => {
    onEnd(status);
    if (status === 0) finished = true;
  };

  try {
    for (let offset = 0; offset < bytes.length; offset += COMPRESSED_CHUNK_BYTES) {
      const next = Math.min(offset + COMPRESSED_CHUNK_BYTES, bytes.length);
      // Never mark the stream finished: this may stop mid-member, and a
      // complete member may be followed by RFC 1952 zero padding that some
      // inflaters reject. Pako ends on the gzip trailer itself.
      if (!inflater.push(bytes.subarray(offset, next), false)) {
        throw new Error(inflater.msg || `pako inflate error ${inflater.err}`);
      }
      if (finished) break;
    }
  } catch (cause) {
    if (!(cause instanceof EnoughOutput)) {
      if (cause instanceof AlfError) throw cause;
      throw new AlfError('gzip-malformed', 'Invalid gzip stream', { cause });
    }
  }

  if (collected < wantedBytes && !finished) {
    refuse('gzip-invalid', 'Truncated gzip stream');
  }

  const size = Math.min(collected, wantedBytes);
  const output = new Uint8Array(size);
  let offset = 0;
  for (const chunk of chunks) {
    if (offset >= size) break;
    const take = Math.min(chunk.length, size - offset);
    output.set(chunk.subarray(0, take), offset);
    offset += take;
  }
  return output;
}

export async function gzip(bytes: Uint8Array): Promise<Uint8Array> {
  // Pako writes a zero mtime and no filename by default, making the envelope
  // deterministic across browsers, Node and Bun. Its result is already a fresh
  // Uint8Array; copying it here would double peak memory for incompressible data.
  return pakoGzip(bytes, { level: 6 });
}
