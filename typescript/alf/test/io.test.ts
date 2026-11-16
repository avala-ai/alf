import { describe, expect, test } from 'bun:test';
import {
  AlfError,
  readAlf,
  readAlfHeader,
  writeAlc,
  writeAlf,
  writeAli,
  writeAll,
  writeAlp,
  writeAlx,
  type AlfChannel,
} from '../src/index.ts';
import { decodeFloat16, encodeFloat16 } from '../src/float16.ts';
import {
  alignedBitCursor,
  bitCursorBitOffset,
  bitCursorByteIndex,
  gunzip,
  gunzipPrefix,
  gzipMemberBoundaryWalkCount,
  huffmanTableBuildCount,
  MAX_COMPRESSED_INPUT_BYTES,
} from '../src/gzip.ts';
import {
  assertAlpPayloadFits,
  assertAlxPayloadFits,
  assertDecodedAlxCopyFits,
  assertDecodedByteChannelCopyFits,
  assertDecodedAlpFits,
  assertDecodedV2LabelsFit,
  assertEnvelopeFits,
  assertV2LabelPayloadFits,
} from '../src/io.ts';
import { roundCanonical } from '../../conformance/canonical.ts';

class SliceCountingBytes extends Uint8Array {
  static calls = 0;

  override slice(start?: number, end?: number): Uint8Array {
    SliceCountingBytes.calls += 1;
    return super.slice(start, end);
  }
}

async function roundTrip(bytes: Promise<Uint8Array>, channel: AlfChannel) {
  return readAlf(await bytes, { channel });
}

function envelope(version: number, metadata: unknown, payload: Uint8Array): Uint8Array {
  return envelopeWithHeader(version, JSON.stringify(metadata), payload);
}

function envelopeWithHeader(version: number, headerText: string, payload: Uint8Array): Uint8Array {
  const header = new TextEncoder().encode(headerText);
  const bytes = new Uint8Array(5 + header.length + payload.length);
  bytes[0] = version;
  new DataView(bytes.buffer).setUint32(1, header.length, true);
  bytes.set(header, 5);
  bytes.set(payload, 5 + header.length);
  return bytes;
}

function emptyFixedGzip(blockCount: number): Uint8Array {
  const deflate = new Uint8Array(Math.ceil((blockCount * 10) / 8));
  let bit = 0;
  for (let block = 0; block < blockCount; block += 1) {
    if (block === blockCount - 1) deflate[bit >>> 3] = (deflate[bit >>> 3] ?? 0) | (1 << (bit & 7));
    bit += 1;
    deflate[bit >>> 3] = (deflate[bit >>> 3] ?? 0) | (1 << (bit & 7)); // BTYPE=01 (fixed).
    bit += 9; // Second BTYPE bit plus the seven-bit end-of-block symbol.
  }
  const bytes = new Uint8Array(10 + deflate.length + 8);
  bytes.set([0x1f, 0x8b, 8, 0, 0, 0, 0, 0, 0, 3]);
  bytes.set(deflate, 10);
  // Empty output has CRC32=0 and ISIZE=0, already supplied by zero fill.
  return bytes;
}

function emptyDynamicGzip(blockCount: number): Uint8Array {
  const deflate = new Uint8Array(Math.ceil((blockCount * 90) / 8));
  let bit = 0;
  const writeBits = (value: number, count: number): void => {
    for (let index = 0; index < count; index += 1) {
      if (((value >>> index) & 1) !== 0) {
        const byteIndex = Math.floor(bit / 8);
        deflate[byteIndex] = (deflate[byteIndex] ?? 0) | (1 << (bit % 8));
      }
      bit += 1;
    }
  };
  for (let block = 0; block < blockCount; block += 1) {
    writeBits(block === blockCount - 1 ? 1 : 0, 1); // BFINAL
    writeBits(2, 2); // BTYPE=10 (dynamic)
    writeBits(0, 5); // 257 literal/length codes
    writeBits(0, 5); // 1 distance code
    writeBits(14, 4); // 18 code-length codes
    for (let index = 0; index < 18; index += 1) writeBits(index === 2 || index === 17 ? 1 : 0, 3);
    writeBits(1, 1); // Code-length symbol 18.
    writeBits(127, 7); // 138 zero literal lengths.
    writeBits(1, 1); // Code-length symbol 18.
    writeBits(107, 7); // 118 more zero literal lengths.
    writeBits(0, 1); // Literal 256 has length 1.
    writeBits(0, 1); // Distance 0 has length 1.
    writeBits(0, 1); // End-of-block symbol 256.
  }
  const bytes = new Uint8Array(10 + deflate.length + 8);
  bytes.set([0x1f, 0x8b, 8, 0, 0, 0, 0, 0, 0, 3]);
  bytes.set(deflate, 10);
  return bytes;
}

function churningDynamicGzip(blockCount: number, variantCount: number): Uint8Array {
  const deflateStorage = new Uint8Array(Math.ceil((blockCount * 200) / 8));
  let bit = 0;
  const writeBits = (value: number, count: number): void => {
    for (let index = 0; index < count; index += 1) {
      if (((value >>> index) & 1) !== 0) {
        const byteIndex = Math.floor(bit / 8);
        deflateStorage[byteIndex] = (deflateStorage[byteIndex] ?? 0) | (1 << (bit % 8));
      }
      bit += 1;
    }
  };
  const pairs: readonly (readonly [number, number])[] = Array.from({ length: variantCount }, (_, variant) => {
    let remaining = variant;
    for (let first = 0; first < 28; first += 1) {
      const choices = 28 - first;
      if (remaining < choices) return [first, first + 1 + remaining] as const;
      remaining -= choices;
    }
    throw new Error('dynamic-Huffman variant count exceeds the available pairs');
  });
  for (let block = 0; block < blockCount; block += 1) {
    const pair = pairs[block % pairs.length];
    if (pair === undefined) throw new Error('dynamic-Huffman variants must not be empty');
    writeBits(block === blockCount - 1 ? 1 : 0, 1); // BFINAL
    writeBits(2, 2); // BTYPE=10 (dynamic)
    writeBits(29, 5); // 286 literal/length codes
    writeBits(0, 5); // 1 distance code
    writeBits(14, 4); // 18 code-length codes
    for (let index = 0; index < 18; index += 1) {
      // Complete code-length tree: 18=>1 bit, 0=>2 bits, 1/2=>3 bits.
      writeBits(index === 2 ? 1 : index === 3 ? 2 : index === 15 || index === 17 ? 3 : 0, 3);
    }
    writeBits(0, 1); // Code-length symbol 18.
    writeBits(127, 7); // 138 zero literal lengths.
    writeBits(0, 1); // Code-length symbol 18.
    writeBits(107, 7); // 118 more zero literal lengths.
    writeBits(3, 3); // Literal 256 has length 1.
    for (let extra = 0; extra < 29; extra += 1) {
      writeBits(extra === pair[0] || extra === pair[1] ? 7 : 1, extra === pair[0] || extra === pair[1] ? 3 : 2);
    }
    writeBits(3, 3); // Distance 0 has length 1.
    writeBits(0, 1); // End-of-block symbol 256.
  }
  const deflate = deflateStorage.subarray(0, Math.ceil(bit / 8));
  const bytes = new Uint8Array(10 + deflate.length + 8);
  bytes.set([0x1f, 0x8b, 8, 0, 0, 0, 0, 0, 0, 3]);
  bytes.set(deflate, 10);
  return bytes;
}


describe('readAlfHeader', () => {
  test('returns the envelope without inflating the payload', async () => {
    const bytes = await writeAlp([[1.37, -2.11, 0.43], [87.62, 41.19, -3.07]]);
    const header = await readAlfHeader(bytes, { channel: 'alp' });
    expect(header.channel).toBe('alp');
    expect(header.version).toBe(1);
    expect(header.totalPoints).toBe(2);
    expect(header.metadata).toBeDefined();
  });

  test('agrees with readAlf on every envelope field', async () => {
    // The header-only path must not be a second, laxer parser. Anything the two
    // can disagree about is a file one accepts and the other refuses.
    for (const bytes of [
      await writeAlp([[1, 2, 3]]),
      await writeAlp([[1, 2, 3]], { codec: 'int16', frameConvention: 'world-aligned' }),
      await writeAli(new Uint8Array([1, 2, 3])),
      await writeAll(new Uint8Array([4, 5, 6])),
    ]) {
      const channel: AlfChannel = (await (async (): Promise<AlfChannel> => {
        for (const candidate of ['alp', 'ali', 'all'] as const) {
          try {
            await readAlf(bytes, { channel: candidate });
            return candidate;
          } catch {
            continue;
          }
        }
        throw new Error('fixture matched no channel');
      })())!;
      const full = await readAlf(bytes, { channel });
      const header = await readAlfHeader(bytes, { channel });
      expect(header.version).toBe(full.version);
      expect(header.totalPoints).toBe(full.totalPoints);
      expect(header.metadata).toEqual(full.metadata);
      expect(header.channel).toBe(full.channel);
    }
  });

  test('reads a gzipped file without inflating all of it', async () => {
    // The point of the function, asserted as a MEASUREMENT rather than trusted
    // from the implementation: the probe must touch far less than the payload.
    const points: number[] = [];
    for (let i = 0; i < 120_000; i += 1) points.push(i * 0.001, 0, 0);
    const bytes = await writeAlp(points, { gzip: true, checkBudget: false });

    const walksBefore = gzipMemberBoundaryWalkCount();
    const header = await readAlfHeader(bytes, { channel: 'alp' });
    expect(header.totalPoints).toBe(120_000);
    // A header scan that still Huffman-walks the member to find the trailer
    // has paid the payload cost this path exists to avoid — including twice
    // when the header outruns the probe.
    expect(gzipMemberBoundaryWalkCount()).toBe(walksBefore);

    const whole = await gunzip(bytes);
    expect(gzipMemberBoundaryWalkCount()).toBeGreaterThan(walksBefore);
    const probeWalks = gzipMemberBoundaryWalkCount();
    const probe = await gunzipPrefix(bytes, 64 * 1024);
    expect(gzipMemberBoundaryWalkCount()).toBe(probeWalks);
    expect(probe.length).toBeLessThan(whole.length);
    // 120k points is 720_000 payload bytes; a 64 KiB probe is a small fraction.
    expect(probe.length).toBeLessThanOrEqual(64 * 1024);
  });

  test('handles a header longer than the probe with a second inflate', async () => {
    // Forces the two-step path: a header past HEADER_PROBE_BYTES must still parse.
    const filler = 'x'.repeat(80 * 1024);
    const bytes = await writeAlp([[1, 2, 3]], { gzip: true, metadata: { note: filler } });
    const walksBefore = gzipMemberBoundaryWalkCount();
    const header = await readAlfHeader(bytes, { channel: 'alp' });
    expect(header.totalPoints).toBe(1);
    expect((header.metadata as Record<string, unknown>)['note']).toBe(filler);
    expect(gzipMemberBoundaryWalkCount()).toBe(walksBefore);
  });

  test('enforces the CHANNEL-scoped header rules, not only the envelope ones', async () => {
    // These depend on which channel you opened, and they lived only inside the
    // channel decoders — which a header-only read never reaches. So a scan
    // accepted `.alx` files with no declared count and `.all` files naming a
    // label layout that does not exist, both refused by `readAlf` and both
    // decidable from the header alone. Every implementation had this gap.
    const cases: [Uint8Array, 'alx' | 'all'][] = [
      [envelope(1, { _source_total_points: 10 }, new Uint8Array(4)), 'alx'],
      [envelope(1, { _total_points: 1, _format_version: 9 }, new Uint8Array(1)), 'all'],
    ];
    for (const [bytes, channel] of cases) {
      await expect(readAlf(bytes, { channel })).rejects.toBeInstanceOf(AlfError);
      await expect(readAlfHeader(bytes, { channel })).rejects.toBeInstanceOf(AlfError);
    }
  });

  test('does not promote a lazily refused rule into the header read', async () => {
    // `.alx` without `_source_total_points` is refused where the indices are
    // decoded. A header-only read reaches neither, and adding it here would
    // make a scan STRICTER than the read it stands in for — the opposite
    // error, and just as wrong.
    const bytes = envelope(1, { _total_points: 1 }, new Uint8Array(4));
    await expect(readAlfHeader(bytes, { channel: 'alx' })).resolves.toBeDefined();
  });

  test('refuses the same malformed envelopes readAlf refuses', async () => {
    const cases: Uint8Array[] = [
      new Uint8Array([9, 0, 0, 0, 0]), // unsupported version byte
      new Uint8Array([1, 200, 0, 0, 0]), // header longer than the file
      new Uint8Array([1, 2]), // shorter than the 5-byte envelope
    ];
    for (const bytes of cases) {
      await expect(readAlfHeader(bytes, { channel: 'alp' })).rejects.toBeInstanceOf(AlfError);
      await expect(readAlf(bytes, { channel: 'alp' })).rejects.toBeInstanceOf(AlfError);
    }
  });

  test('requires a channel, like readAlf', async () => {
    const bytes = await writeAlp([[1, 2, 3]]);
    // @ts-expect-error the channel is required
    await expect(readAlfHeader(bytes, {})).rejects.toBeInstanceOf(AlfError);
    await expect(readAlfHeader(bytes, { channel: 'nope' as AlfChannel })).rejects.toBeInstanceOf(AlfError);
  });

  test('rejects a version/channel mismatch the same way readAlf does', async () => {
    const v3 = await writeAlp([[1, 2, 3]], { codec: 'int16', frameConvention: 'world-aligned' });
    await expect(readAlfHeader(v3, { channel: 'ali' })).rejects.toBeInstanceOf(AlfError);
  });
});

describe('gunzipPrefix', () => {
  test('returns only the requested bytes', async () => {
    const bytes = await writeAli(new Uint8Array(50_000).fill(7), { gzip: true });
    const prefix = await gunzipPrefix(bytes, 128);
    expect(prefix.length).toBe(128);
    const whole = await gunzip(bytes);
    expect(Array.from(prefix)).toEqual(Array.from(whole.subarray(0, 128)));
  });

  test('returns the whole member when it is shorter than the request', async () => {
    const bytes = await writeAli(new Uint8Array([1, 2, 3]), { gzip: true });
    const whole = await gunzip(bytes);
    const prefix = await gunzipPrefix(bytes, 1024 * 1024);
    expect(Array.from(prefix)).toEqual(Array.from(whole));
  });

  test('does not Huffman-walk the member to find its trailer', async () => {
    // firstGzipMemberEnd decodes every DEFLATE symbol. A prefix read that
    // still calls it has already paid the full-member cost, so comparing
    // output length alone cannot catch the regression.
    const bytes = await writeAli(new Uint8Array(80_000).fill(7), { gzip: true });
    const walksBefore = gzipMemberBoundaryWalkCount();
    const prefix = await gunzipPrefix(bytes, 64);
    expect(prefix.length).toBe(64);
    expect(gzipMemberBoundaryWalkCount()).toBe(walksBefore);
    await gunzip(bytes);
    expect(gzipMemberBoundaryWalkCount()).toBeGreaterThan(walksBefore);
  });

  test('stops at the first member when RFC-permitted zero padding follows', async () => {
    const bytes = await writeAli(new Uint8Array([1, 2, 3]), { gzip: true });
    const padded = new Uint8Array(bytes.length + 16);
    padded.set(bytes);
    const whole = await gunzip(bytes);
    const prefix = await gunzipPrefix(padded, 1024 * 1024);
    expect(Array.from(prefix)).toEqual(Array.from(whole));
  });
});

describe('channel writers', () => {
  test('writes and reads v1 fp16 .alp geometry', async () => {
    const file = await roundTrip(writeAlp([[1.37, -2.11, 0.43], [87.62, 41.19, -3.07]]), 'alp');
    expect(file.channel).toBe('alp');
    if (file.channel !== 'alp') return;
    expect(file.version).toBe(1);
    expect(file.codec).toBe('fp16');
    expect(file.points.length).toBe(6);
    expect(file.points[0]).toBeCloseTo(1.37, 2);
  });

  test('float16 rounding carries a full mantissa into the exponent', async () => {
    const file = await roundTrip(writeAlp([1.9996, 0, 0]), 'alp');
    expect(file.channel).toBe('alp');
    if (file.channel !== 'alp') return;
    expect(file.points[0]).toBe(2);
  });

  test('float16 rounds the upper half of the smallest subnormal interval up', () => {
    const minimumSubnormal = 2 ** -24;
    expect(encodeFloat16(minimumSubnormal / 2)).toBe(0);
    expect(encodeFloat16((minimumSubnormal * 3) / 4)).toBe(1);
    expect(decodeFloat16(encodeFloat16((minimumSubnormal * 3) / 4))).toBe(minimumSubnormal);
  });

  test('float16 converts directly from binary64 without float32 double rounding', () => {
    const exactTie = 1 + 2 ** -11;
    const justAboveTieButSameFloat32 = exactTie + 2 ** -25;
    expect(Math.fround(justAboveTieButSameFloat32)).toBe(Math.fround(exactTie));
    expect(encodeFloat16(exactTie)).toBe(0x3c00);
    expect(encodeFloat16(justAboveTieButSameFloat32)).toBe(0x3c01);
  });

  test('float16 accepts the finite upper rounding interval and refuses infinity', () => {
    expect(encodeFloat16(65519)).toBe(0x7bff);
    for (const value of [65520, 65536, 70_000, 131072, -70_000]) {
      expect(() => encodeFloat16(value)).toThrow();
    }
  });

  test('float16 writer refuses every non-finite exponent even without the quality budget', async () => {
    for (const value of [65536, 70_000, 131072, -70_000]) {
      await expect(writeAlp([value, 0, 0], { checkBudget: false })).rejects.toMatchObject({
        code: 'writer-input-invalid',
      });
    }
  });

  test('int16 carries a whole invalid return through the sentinel', async () => {
    const file = await roundTrip(
      writeAlp([1, 2, 3, NaN, NaN, NaN], { codec: 'int16', frameConvention: 'world-aligned', checkBudget: false }),
      'alp',
    );
    if (file.channel !== 'alp') return;
    // The index-correlation invariant is the whole point of the sentinel:
    // dropping the invalid point would slide every later channel entry by one.
    expect(file.points.length).toBe(6);
    expect(file.points[0]).toBeCloseTo(1, 3);
    expect(Number.isNaN(file.points[3] ?? 0)).toBe(true);
    expect(Number.isNaN(file.points[4] ?? 0)).toBe(true);
    expect(Number.isNaN(file.points[5] ?? 0)).toBe(true);
  });

  test('an int16 invalid return decodes to NaN, never to the sentinel coordinate', async () => {
    // -32768 * scale + offset is a perfectly ordinary coordinate, so a decoder
    // that forgets the sentinel does not crash — it silently reports a point at
    // the far edge of the grid. That silence is why this is pinned.
    const file = await roundTrip(
      writeAlp([1000, 0, 0, NaN, NaN, NaN], { codec: 'int16', frameConvention: 'world-aligned', checkBudget: false }),
      'alp',
    );
    if (file.channel !== 'alp') return;
    expect(Number.isNaN(file.points[3] ?? 0)).toBe(true);
    expect(file.points[0]).toBeCloseTo(1000, 0);
  });

  test('int16 writes a frame of nothing BUT invalid returns', async () => {
    // A fully-occluded sweep. There is no finite extent to centre the grid on,
    // so a writer deriving its offset from the Infinity/-Infinity seeds emits a
    // NaN origin and then rejects its own header.
    const file = await roundTrip(
      writeAlp([NaN, NaN, NaN, NaN, NaN, NaN], { codec: 'int16', frameConvention: 'world-aligned', checkBudget: false }),
      'alp',
    );
    if (file.channel !== 'alp') return;
    expect(file.points.length).toBe(6);
    expect([...file.points].every((value) => Number.isNaN(value))).toBe(true);
  });

  test('a PARTIAL sentinel triple is refused, not decoded as a position', async () => {
    // Hand-built: no conforming writer can emit this, since -32768 is reserved
    // and the encoder clamps real coordinates to +/-32767. Decoded
    // arithmetically it reads -327.68 — finite, plausible, never measured.
    for (const words of [
      [-32768, 0, 0],
      [0, -32768, 0],
      [0, 0, -32768],
    ]) {
      const header = new TextEncoder().encode(
        JSON.stringify({
          _total_points: 1,
          _geom_codec: 'int16', _frame_convention: 'world-aligned', frameConvention: 'world-aligned',
          _geom_scale: 0.01,
          _geom_offset: [0, 0, 0],
        }),
      );
      const payload = new Int16Array(words);
      const bytes = new Uint8Array(1 + 4 + header.length + payload.byteLength);
      bytes[0] = 3;
      new DataView(bytes.buffer).setUint32(1, header.length, true);
      bytes.set(header, 5);
      bytes.set(new Uint8Array(payload.buffer), 5 + header.length);

      await expect(readAlf(bytes, { channel: 'alp' })).rejects.toMatchObject({
        code: 'geom-partial-no-return',
      });
    }
  });

  test('int16 refuses a PARTLY invalid point, which has no interpretation', async () => {
    await expect(writeAlp([1, NaN, 3], { codec: 'int16', frameConvention: 'world-aligned', checkBudget: false })).rejects.toMatchObject({
      code: 'writer-input-invalid',
    });
  });

  test('writes and reads v3 int16 .alp geometry off-centre', async () => {
    const original = new Float64Array([1001.37, -2.11, 0.43, 1087.62, 41.19, -3.07]);
    const file = await roundTrip(writeAlp(original, { codec: 'int16', frameConvention: 'world-aligned' }), 'alp');
    expect(file.channel).toBe('alp');
    if (file.channel !== 'alp') return;
    expect(file.version).toBe(3);
    expect(file.codec).toBe('int16');
    expect(Math.abs((file.points[0] ?? 0) - (original[0] ?? 0))).toBeLessThanOrEqual(file.maxGeomErrorM + 1e-12);
  });

  test('floors an underflowing non-empty v3 half-scale bound upward', async () => {
    const file = await roundTrip(
      writeAlp([0, 0, 0], {
        codec: 'int16', frameConvention: 'world-aligned',
        geomScale: Number.MIN_VALUE,
        geomOffset: [0, 0, 0],
      }),
      'alp',
    );
    expect(file.channel).toBe('alp');
    if (file.channel !== 'alp') return;
    expect(file.metadata['_geom_max_err_m']).toBe(Number.MIN_VALUE);
    expect(file.maxGeomErrorM).toBe(Number.MIN_VALUE);
  });

  test('writer emits an exact binary64 token for a wide integer grid', async () => {
    const exact = 2 ** 64;
    const file = await roundTrip(
      writeAlp([exact, 0, 0], { codec: 'int16', frameConvention: 'world-aligned', geomScale: 0.0001, geomOffset: [exact, 0, 0] }),
      'alp',
    );
    expect(file.channel === 'alp' ? file.points[0] : 0).toBe(exact);
  });

  test('writes finite v3 grids at same-sign and opposite binary64 extremes', async () => {
    const maximum = Number.MAX_VALUE;
    const constant = await roundTrip(
      writeAlp([maximum, maximum, maximum], { codec: 'int16', frameConvention: 'world-aligned', checkBudget: false }),
      'alp',
    );
    expect(constant.channel === 'alp' ? [...constant.points] : []).toEqual([maximum, maximum, maximum]);

    const extremes = await roundTrip(
      writeAlp([-maximum, -maximum, -maximum, maximum, maximum, maximum], {
        codec: 'int16', frameConvention: 'world-aligned',
        checkBudget: false,
      }),
      'alp',
    );
    expect(extremes.channel).toBe('alp');
    if (extremes.channel !== 'alp') return;
    expect([...extremes.points].every(Number.isFinite)).toBe(true);
    expect(extremes.points[0]).toBeLessThan(0);
    expect(extremes.points[3]).toBeGreaterThan(0);
    expect(Number.isFinite(extremes.metadata['_geom_scale'])).toBe(true);
    expect((extremes.metadata['_geom_offset'] as number[]).every(Number.isFinite)).toBe(true);
  });

  test('recomputes the automatic int16 grid instead of inheriting stale codec metadata', async () => {
    const file = await roundTrip(
      writeAlp([1, 0, 0, 2, 0, 0], {
        codec: 'int16', frameConvention: 'world-aligned',
        metadata: { _geom_scale: 100, _geom_offset: [1_000, 2_000, 3_000] },
      }),
      'alp',
    );
    expect(file.channel).toBe('alp');
    if (file.channel !== 'alp') return;
    expect(file.metadata['_geom_scale']).toBe(0.0001);
    expect(file.metadata['_geom_offset']).toEqual([1.5, 0, 0]);
  });

  test('validates and normalizes inherited geometry anchors', async () => {
    const file = await roundTrip(
      writeAlp([1, 2, 3], { metadata: { device_position: [4, 5, 6] } }),
      'alp',
    );
    expect(file.metadata['device_position']).toEqual({ x: 4, y: 5, z: 6 });

    for (const metadata of [
      { device_position: { x: 1 } },
      { world_centroid_epsg3857: { x: true, y: 0, z: 0 } },
      { world_centroid_epsg3857: [1, 2, 3, 4] },
    ]) {
      await expect(writeAlp([1, 2, 3], { metadata })).rejects.toMatchObject({ code: 'writer-input-invalid' });
    }
  });

  test('preserves anchor application members without mutating metadata', async () => {
    const metadata = {
      device_position: {
        z: 3,
        note: { source: 'gps', flags: ['rtk', { fixed: true }] },
        x: 1,
        quality: 0.875,
        y: 2,
      },
    };
    const before = JSON.stringify(metadata);

    const file = await roundTrip(writeAlp([0, 0, 0], { metadata }), 'alp');
    expect(file.metadata['device_position']).toEqual({
      note: { source: 'gps', flags: ['rtk', { fixed: true }] },
      quality: 0.875,
      x: 1,
      y: 2,
      z: 3,
    });
    expect(JSON.stringify(metadata)).toBe(before);
  });

  test('writes exact wide v3 offsets with a binary64-preserving JSON token', async () => {
    const bytes = await writeAlp([2 ** 64, 0, 0], { codec: 'int16', frameConvention: 'world-aligned', checkBudget: false });
    const headerLength = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength).getUint32(1, true);
    const header = new TextDecoder().decode(bytes.subarray(5, 5 + headerLength));
    expect(header).toContain('1.8446744073709552e+19');
    const file = await readAlf(bytes, { channel: 'alp' });
    expect(file.channel === 'alp' ? file.points[0] : 0).toBe(2 ** 64);
  });

  test('writes and reads normalized .ali intensity', async () => {
    const file = await roundTrip(writeAli([0, 5, 10], { inputRange: [0, 10] }), 'ali');
    expect(file.channel).toBe('ali');
    if (file.channel !== 'ali') return;
    expect([...file.intensity]).toEqual([0, 127, 255]);
    expect(file.metadata['_intensities_every_10_percent']).toEqual([0, 0, 0, 127, 127, 127, 255, 255, 255]);
  });

  test('requires exactly two runtime intensity range bounds', async () => {
    const malformed = [null, [0], [0, 1, 2], { 0: 0, 1: 1, length: 2 }, new Float64Array([0, 1])];
    for (const inputRange of malformed) {
      await expect(writeAli([0], { inputRange: inputRange as never })).rejects.toMatchObject({
        code: 'writer-input-invalid',
      });
    }
  });

  test('derives automatic deciles from the exact payload snapshot', async () => {
    const count = 10;
    let reads = 0;
    const changing = new Proxy({ length: count } as ArrayLike<number>, {
      get(target, property, receiver) {
        if (typeof property === 'string' && /^(0|[1-9][0-9]*)$/.test(property)) {
          reads += 1;
          return reads <= count ? 0 : 255;
        }
        return Reflect.get(target, property, receiver);
      },
    });

    const file = await roundTrip(writeAli(changing), 'ali');
    expect(reads).toBe(count);
    expect([...file.intensity]).toEqual(new Array<number>(count).fill(0));
    expect(file.metadata['_intensities_every_10_percent']).toEqual(new Array<number>(9).fill(0));
  });

  test('accepts equal intensity bounds only for matching constant samples', async () => {
    const file = await roundTrip(writeAli([7, 7, 7], { inputRange: [7, 7] }), 'ali');
    expect(file.channel).toBe('ali');
    if (file.channel !== 'ali') return;
    expect([...file.intensity]).toEqual([0, 0, 0]);
    await expect(writeAli([7, 8], { inputRange: [7, 7] })).rejects.toMatchObject({ code: 'writer-input-invalid' });
  });

  test('refuses intensity samples outside their declared input range', async () => {
    await expect(writeAli([0, 100], { inputRange: [0, 1] })).rejects.toMatchObject({ code: 'writer-input-invalid' });
  });

  test('refuses integral intensity inputs at the binary64 consecutive-integer boundary', async () => {
    await expect(
      writeAli([2 ** 53, 2 ** 53 + 1], { inputRange: [2 ** 53, 2 ** 53 + 1] }),
    ).rejects.toMatchObject({ code: 'writer-input-invalid' });
    await expect(writeAli([0, 1], { inputRange: [0, 2 ** 53] })).rejects.toMatchObject({
      code: 'writer-input-invalid',
    });
  });

  test('computes intensity deciles with the same percentile positions', async () => {
    const samples = Uint8Array.from({ length: 1000 }, (_, index) => index % 256);
    const file = await roundTrip(writeAli(samples), 'ali');
    expect(file.metadata['_intensities_every_10_percent']).toEqual([25, 50, 75, 100, 125, 150, 175, 200, 225]);
  });

  test('writes and reads .alc palette indexes including the no-colour sentinel', async () => {
    const file = await roundTrip(writeAlc([0, 1, 2], ['#FF0000', '00ff00']), 'alc');
    expect(file.channel).toBe('alc');
    if (file.channel !== 'alc') return;
    expect(file.palette).toEqual(['ff0000', '00ff00']);
    expect([...file.paletteIndices]).toEqual([0, 1, 2]);
  });

  test('palette reader normalizes one optional hash and refuses two', async () => {
    const prefixed = await readAlf(envelope(1, { _total_points: 1, palette: ['#AbCdEf'] }, new Uint8Array([0])), {
      channel: 'alc',
    });
    expect(prefixed.channel === 'alc' ? prefixed.palette : []).toEqual(['abcdef']);
    await expect(writeAlc([0], ['##ff0000'])).rejects.toMatchObject({ code: 'palette-entry-malformed' });
    await expect(
      readAlf(envelope(1, { _total_points: 1, palette: ['##ff0000'] }, new Uint8Array([0])), { channel: 'alc' }),
    ).rejects.toMatchObject({ code: 'palette-entry-malformed' });
  });

  test('refuses an oversized palette before iterating or copying it', async () => {
    const oversized = new Array<string>(256);
    const palette = new Proxy(oversized, {
      get: (target, property, receiver) => {
        if (property === Symbol.iterator) throw new Error('oversized palette was iterated');
        return Reflect.get(target, property, receiver) as unknown;
      },
    });

    await expect(writeAlc([0], palette)).rejects.toMatchObject({ code: 'writer-input-invalid' });
    await expect(writeAlc([255], new Array<string>(255).fill('000000'))).resolves.toBeInstanceOf(Uint8Array);
  });

  test('writes and reads v1 semantic .all labels', async () => {
    const file = await roundTrip(writeAll([1, 2, 3]), 'all');
    expect(file.channel).toBe('all');
    if (file.channel !== 'all') return;
    expect(file.formatVersion).toBe(1);
    expect([...file.categories]).toEqual([1, 2, 3]);
    expect(file.instanceIds).toBeNull();
  });

  test('writes and reads v2 panoptic .all labels', async () => {
    const file = await roundTrip(
      writeAll([1, 2], { instanceIds: [1, 258], instanceMap: { '1': 'uuid-a', '258': 'uuid-b' } }),
      'all',
    );
    expect(file.channel).toBe('all');
    if (file.channel !== 'all') return;
    expect(file.version).toBe(2);
    expect(file.formatVersion).toBe(2);
    expect([...file.instanceIds!]).toEqual([1, 258]);
  });

  test('treats an explicit null instance map as absent', async () => {
    const file = await roundTrip(writeAll([1], { instanceIds: [0], instanceMap: null }), 'all');
    expect(file.channel === 'all' ? file.instanceMap : {}).toBeNull();
    expect(Object.prototype.hasOwnProperty.call(file.metadata, 'instance_map')).toBe(false);
    await expect(writeAll([1], { instanceIds: [7], instanceMap: null })).rejects.toMatchObject({
      code: 'instance-map-incomplete',
    });
  });

  test('requires maps for newly authored nonzero instance IDs', async () => {
    await expect(writeAll([1], { instanceIds: [7] })).rejects.toMatchObject({ code: 'instance-map-incomplete' });
    await expect(
      writeAll([1], { instanceIds: [7], metadata: { instance_map: { '7': 'uuid-seven' } } }),
    ).resolves.toBeInstanceOf(Uint8Array);
    await expect(writeAll([1], { instanceIds: [0] })).resolves.toBeInstanceOf(Uint8Array);
  });

  test('writes and reads .alx decimation indexes', async () => {
    const file = await roundTrip(writeAlx([8, 2, 5], { alpTotalPoints: 3, sourceTotalPoints: 10 }), 'alx');
    expect(file.channel).toBe('alx');
    if (file.channel !== 'alx') return;
    expect([...file.keptIndices]).toEqual([8, 2, 5]);
    expect(file.sourceTotalPoints).toBe(10);
  });

  test('snapshots an accessor-backed ALX source count once', async () => {
    let reads = 0;
    const options = {
      alpTotalPoints: 1,
      get sourceTotalPoints(): number {
        reads += 1;
        return reads === 4 ? 1 : 10;
      },
    };

    const file = await roundTrip(writeAlx([5], options), 'alx');
    expect(reads).toBe(1);
    expect(file.channel === 'alx' ? file.sourceTotalPoints : 0).toBe(10);
    expect(file.channel === 'alx' ? [...file.keptIndices] : []).toEqual([5]);
  });

  test('re-encodes an exactly representable wide .alx source count', async () => {
    const source = envelopeWithHeader(
      1,
      '{"_total_points":1,"_source_total_points":18446744073709551616}',
      new Uint8Array([0, 0, 0, 0]),
    );
    const first = await readAlf(source, { channel: 'alx' });
    const second = await readAlf(await writeAlf(first), { channel: 'alx' });
    expect(second.channel === 'alx' ? second.sourceTotalPoints : 0).toBe(2 ** 64);
  });

  test('rejects malformed runtime geometry input before coercion', async () => {
    await expect(writeAlp([true, 0, 0] as unknown as number[])).rejects.toMatchObject({ code: 'writer-input-invalid' });
    await expect(writeAlp(['1', 0, 0] as unknown as number[])).rejects.toMatchObject({ code: 'writer-input-invalid' });
    const sparse = new Array<readonly [number, number, number]>(2);
    sparse[0] = [0, 0, 0];
    await expect(writeAlp(sparse)).rejects.toMatchObject({ code: 'writer-input-invalid' });
    await expect(writeAlp([0, 0, 0], { codec: 'fp-16' as never })).rejects.toMatchObject({
      code: 'writer-input-invalid',
    });
    for (const geomOffset of [
      [0, 0],
      [0, 0, 0, 0],
    ]) {
      await expect(
        writeAlp([0, 0, 0], { codec: 'int16', frameConvention: 'world-aligned', geomOffset: geomOffset as unknown as [number, number, number] }),
      ).rejects.toMatchObject({ code: 'writer-input-invalid' });
    }
  });

  test('enforces geometry quality budgets unless explicitly waived', async () => {
    await expect(writeAlp([512, 0, 0])).rejects.toMatchObject({ code: 'writer-input-invalid' });
    await expect(writeAlp([-20_000, 0, 0, 20_000, 0, 0], { codec: 'int16', frameConvention: 'world-aligned' })).rejects.toMatchObject({
      code: 'writer-input-invalid',
    });
    expect((await writeAlp([512, 0, 0], { checkBudget: false })).length).toBeGreaterThan(0);
  });

  test('snapshots an accessor-backed explicit geometry offset once', async () => {
    const reads = [0, 0, 0];
    const expected = [100, 200, 300];
    const geomOffset = [0, 0, 0] as [number, number, number];
    for (let axis = 0; axis < 3; axis += 1) {
      Object.defineProperty(geomOffset, axis, {
        get: () => {
          reads[axis] = (reads[axis] ?? 0) + 1;
          return (reads[axis] ?? 0) <= 3 ? (expected[axis] ?? 0) : 0;
        },
      });
    }

    const file = await roundTrip(
      writeAlp(expected, { codec: 'int16', frameConvention: 'world-aligned', geomScale: 1, geomOffset, checkBudget: false }),
      'alp',
    );
    expect(reads).toEqual([1, 1, 1]);
    expect(file.channel === 'alp' ? [...file.points] : []).toEqual(expected);
    expect(file.metadata['_geom_offset']).toEqual(expected);
  });

  test('normalizes and writes the same snapshot of an accessor-backed rotation', async () => {
    for (const gzip of [false, true]) {
      const reads = { w: 0, x: 0, y: 0, z: 0 };
      const frameRotation = { w: 0, x: 0, y: 0, z: 0 };
      for (const axis of ['w', 'x', 'y', 'z'] as const) {
        Object.defineProperty(frameRotation, axis, {
          // Both poses normalize to valid rotations, but the second turns the
          // cloud in the opposite direction. Reader validation cannot catch it.
          get: () => ++reads[axis] === 1 || axis === 'w' ? 0.525 : -0.525,
        });
      }
      const file = await roundTrip(
        writeAlp([1.234567, 2.345678, 3.456789], {
          codec: 'int16', frameConvention: 'device-aligned', frameRotation, gzip,
        }),
        'alp',
      );
      expect(file.channel === 'alp' ? file.frameRotation : null).toEqual({ w: 0.5, x: 0.5, y: 0.5, z: 0.5 });
      expect(reads).toEqual({ w: 1, x: 1, y: 1, z: 1 });
    }
  });

  test('refuses a null runtime rotation with a named writer error', async () => {
    await expect(writeAlp([0, 0, 0], {
      codec: 'int16', frameConvention: 'device-aligned', frameRotation: null as never,
    })).rejects.toMatchObject({ code: 'writer-input-invalid' });
  });

  test('replaces stale geometry bounds from decoded output and removes them for empty frames', async () => {
    const metadata = { box: '999:999:999:1000:1000:1000', _box_frame: 'world' };
    const file = await roundTrip(writeAlp([1, 2, 3], { metadata }), 'alp');
    expect(file.metadata['box']).toBe('1:2:3:1:2:3');
    expect(file.metadata['_box_frame']).toBe('local');
    const empty = await roundTrip(writeAlp([], { metadata }), 'alp');
    expect(empty.metadata['box']).toBeUndefined();
    expect(empty.metadata['_box_frame']).toBeUndefined();
  });
});

describe('envelope and regressions', () => {
  test('canonical rounding preserves large finite coordinates without overflow', () => {
    expect(roundCanonical(1e308, 6)).toBe(1e308);
    expect(roundCanonical(-1e308, 6)).toBe(-1e308);
    expect(roundCanonical(1.23456789, 6)).toBe(1.234568);
    expect(roundCanonical(0.0000025, 6)).toBe(0.000003);
    expect(roundCanonical(-0.0000025, 6)).toBe(-0.000003);
  });

  test('refuses a writer envelope above the reader ceiling before allocation', () => {
    const ceiling = 512 * 1024 * 1024;
    expect(() => assertEnvelopeFits(0, ceiling - 5)).not.toThrow();
    expect(() => assertEnvelopeFits(0, ceiling - 4)).toThrow();
  });

  test('refuses an oversized v2 label payload before allocating interleaved rows', () => {
    const maximumDecodedBytes = 256 * 1024 * 1024;
    const maximumRowsByDecodedStorage = Math.floor(maximumDecodedBytes / 3);
    expect(() => assertV2LabelPayloadFits(maximumRowsByDecodedStorage)).not.toThrow();
    expect(() => assertV2LabelPayloadFits(maximumRowsByDecodedStorage + 1)).toThrow(
      expect.objectContaining({ code: 'writer-input-invalid' }),
    );
    expect(() => assertDecodedV2LabelsFit(maximumRowsByDecodedStorage)).not.toThrow();
    expect(() => assertDecodedV2LabelsFit(maximumRowsByDecodedStorage + 1)).toThrow(
      expect.objectContaining({ code: 'input-too-large' }),
    );
  });

  test('bounds one-byte result copies while retaining the input envelope', () => {
    const ceiling = 512 * 1024 * 1024;
    expect(() => assertDecodedByteChannelCopyFits(ceiling / 2, ceiling / 2)).not.toThrow();
    expect(() => assertDecodedByteChannelCopyFits(ceiling / 2 + 1, ceiling / 2)).toThrow(
      expect.objectContaining({ code: 'input-too-large' }),
    );
  });

  test('bounds ALX result copies while retaining the input envelope', () => {
    const ceiling = 512 * 1024 * 1024;
    expect(() => assertDecodedAlxCopyFits(ceiling / 2, ceiling / 2)).not.toThrow();
    expect(() => assertDecodedAlxCopyFits(ceiling / 2 + 1, ceiling / 2)).toThrow(
      expect.objectContaining({ code: 'input-too-large' }),
    );
  });

  test('writer applies the decoded v2 row ceiling before allocating', async () => {
    const maximumRowsByDecodedStorage = Math.floor((256 * 1024 * 1024) / 3);
    const tooManyRows = maximumRowsByDecodedStorage + 1;
    const values = { length: tooManyRows } as unknown as ArrayLike<number>;
    const originalUint8Array = Object.getOwnPropertyDescriptor(globalThis, 'Uint8Array');
    if (originalUint8Array === undefined) throw new Error('Uint8Array global descriptor is missing');
    Object.defineProperty(globalThis, 'Uint8Array', {
      ...originalUint8Array,
      value: new Proxy(Uint8Array, {
        construct: (): never => {
          throw new Error('oversized v2 Uint8Array allocation attempted');
        },
      }),
    });
    let write: Promise<Uint8Array>;
    try {
      write = writeAll(values, { instanceIds: values });
    } finally {
      Object.defineProperty(globalThis, 'Uint8Array', originalUint8Array);
    }
    await expect(write).rejects.toMatchObject({ code: 'writer-input-invalid' });
  });

  test('refuses oversized geometry payloads before cloning float64 coordinates', async () => {
    const maximumDecodedBytes = 512 * 1024 * 1024;
    const maximumCoordinatesByDecodedStorage = Math.floor(maximumDecodedBytes / Float64Array.BYTES_PER_ELEMENT);
    expect(() => assertAlpPayloadFits(maximumCoordinatesByDecodedStorage)).not.toThrow();
    expect(() => assertAlpPayloadFits(maximumCoordinatesByDecodedStorage + 1)).toThrow();
    expect(() => assertDecodedAlpFits(maximumCoordinatesByDecodedStorage)).not.toThrow();
    expect(() => assertDecodedAlpFits(maximumCoordinatesByDecodedStorage + 1)).toThrow(
      expect.objectContaining({ code: 'input-too-large' }),
    );
    const impossible = { length: Number.MAX_SAFE_INTEGER + 1 } as unknown as ArrayLike<number>;
    await expect(writeAlp(impossible)).rejects.toMatchObject({ code: 'writer-input-invalid' });

    const tooManyPoints = Math.floor(maximumCoordinatesByDecodedStorage / 3) + 1;
    await expect(readAlf(envelope(1, { _total_points: tooManyPoints }, new Uint8Array()), { channel: 'alp' })).rejects.toMatchObject({
      code: 'input-too-large',
    });
  });

  test('snapshots flat geometry shape before allocation preflight', async () => {
    let reads = 0;
    const changing: unknown[] = [0, [0, 0, 0], [0, 0, 0]];
    Object.defineProperty(changing, 0, {
      configurable: true,
      get: () => {
        reads += 1;
        return reads === 1 ? 0 : [0, 0, 0];
      },
    });
    await expect(writeAlp(changing as ArrayLike<number>)).rejects.toMatchObject({ code: 'writer-input-invalid' });
    expect(reads).toBe(2);
  });

  test('refuses oversized ALX payloads before allocating index buffers', async () => {
    const ceiling = 512 * 1024 * 1024;
    const maximumRowsByPayloadAlone = Math.floor((ceiling - 5) / 4);
    expect(() => assertAlxPayloadFits(maximumRowsByPayloadAlone)).not.toThrow();
    expect(() => assertAlxPayloadFits(maximumRowsByPayloadAlone + 1)).toThrow();
    const impossible = { length: Number.MAX_SAFE_INTEGER + 1 } as unknown as ArrayLike<number>;
    await expect(
      writeAlx(impossible, { alpTotalPoints: Number.MAX_SAFE_INTEGER + 1, sourceTotalPoints: 2 ** 32 }),
    ).rejects.toMatchObject({ code: 'writer-input-invalid' });
  });

  test('shares one ALX index backing buffer between wire encoding and duplicate validation', async () => {
    const originalUint8Array = Object.getOwnPropertyDescriptor(globalThis, 'Uint8Array');
    const originalUint32Array = Object.getOwnPropertyDescriptor(globalThis, 'Uint32Array');
    if (originalUint8Array === undefined || originalUint32Array === undefined) {
      throw new Error('typed-array global descriptor is missing');
    }
    const payloadBytes = 3 * Uint32Array.BYTES_PER_ELEMENT;
    const backingBuffers = new Set<ArrayBufferLike>();
    Object.defineProperty(globalThis, 'Uint8Array', {
      ...originalUint8Array,
      value: new Proxy(Uint8Array, {
        construct: (target, argumentsList, newTarget): Uint8Array => {
          const value = Reflect.construct(target, argumentsList, newTarget) as Uint8Array;
          if (value.byteLength === payloadBytes) backingBuffers.add(value.buffer);
          return value;
        },
      }),
    });
    Object.defineProperty(globalThis, 'Uint32Array', {
      ...originalUint32Array,
      value: new Proxy(Uint32Array, {
        construct: (target, argumentsList, newTarget): Uint32Array => {
          const value = Reflect.construct(target, argumentsList, newTarget) as Uint32Array;
          if (value.byteLength === payloadBytes) backingBuffers.add(value.buffer);
          return value;
        },
      }),
    });
    let encoded: Uint8Array;
    try {
      encoded = await writeAlx([9, 0, 4], { alpTotalPoints: 3, sourceTotalPoints: 10 });
    } finally {
      Object.defineProperty(globalThis, 'Uint8Array', originalUint8Array);
      Object.defineProperty(globalThis, 'Uint32Array', originalUint32Array);
    }

    expect(backingBuffers.size).toBe(1);
    const file = await readAlf(encoded, { channel: 'alx' });
    expect(file.channel === 'alx' ? [...file.keptIndices] : []).toEqual([9, 0, 4]);
    await expect(writeAlx([1, 1], { alpTotalPoints: 2, sourceTotalPoints: 3 })).rejects.toMatchObject({
      code: 'writer-input-invalid',
    });
  });

  test('refuses oversized one-byte channels before allocating payloads', async () => {
    const ceiling = 512 * 1024 * 1024;
    const payloadOnlyBoundary = { length: ceiling - 5 } as unknown as ArrayLike<number>;
    const callerMetadataBoundary = { length: ceiling - 5 - 64 } as unknown as ArrayLike<number>;
    const callerMetadata = { padding: 'x'.repeat(128) };
    const originalUint8Array = Object.getOwnPropertyDescriptor(globalThis, 'Uint8Array');
    if (originalUint8Array === undefined) throw new Error('Uint8Array global descriptor is missing');
    const allocationRefusingUint8Array = new Proxy(Uint8Array, {
      construct: (): never => {
        throw new Error('oversized Uint8Array allocation attempted');
      },
    });
    let writes: readonly Promise<Uint8Array>[];
    Object.defineProperty(globalThis, 'Uint8Array', {
      ...originalUint8Array,
      value: allocationRefusingUint8Array,
    });
    try {
      writes = [
        writeAli(payloadOnlyBoundary),
        writeAlc(payloadOnlyBoundary, ['ffffff']),
        writeAll(payloadOnlyBoundary),
        writeAli(callerMetadataBoundary, { metadata: callerMetadata }),
        writeAlc(callerMetadataBoundary, ['ffffff'], { metadata: callerMetadata }),
        writeAll(callerMetadataBoundary, { metadata: callerMetadata }),
      ];
    } finally {
      Object.defineProperty(globalThis, 'Uint8Array', originalUint8Array);
    }
    for (const write of writes) {
      await expect(write).rejects.toMatchObject({ code: 'writer-input-invalid' });
    }
  });

  test('reserves fixed-width channel headers before allocating payload buffers', async () => {
    const ceiling = 512 * 1024 * 1024;
    const callerMetadata = { padding: 'x'.repeat(128) };
    const fitTriples = (length: number): number => length - (length % 3);
    const geometryBoundary = fitTriples(Math.floor((ceiling - 5) / 2));
    const geometryMetadataBoundary = fitTriples(Math.floor((ceiling - 5 - 64) / 2));
    const v2Boundary = Math.floor((ceiling - 5) / 3);
    const v2MetadataBoundary = Math.floor((ceiling - 5 - 64) / 3);
    const alxBoundary = Math.floor((ceiling - 5) / 4);
    const alxMetadataBoundary = Math.floor((ceiling - 5 - 64) / 4);
    const arrayLike = (length: number): ArrayLike<number> => ({ length }) as ArrayLike<number>;

    const constructors = ['Uint8Array', 'Uint16Array', 'Uint32Array', 'Float64Array'] as const;
    const originals = new Map<string, PropertyDescriptor>();
    for (const name of constructors) {
      const descriptor = Object.getOwnPropertyDescriptor(globalThis, name);
      if (descriptor === undefined) throw new Error(`${name} global descriptor is missing`);
      originals.set(name, descriptor);
      Object.defineProperty(globalThis, name, {
        ...descriptor,
        value: new Proxy(descriptor.value as typeof Uint8Array, {
          construct: (): never => {
            throw new Error(`oversized ${name} allocation attempted`);
          },
        }),
      });
    }

    let writes: readonly Promise<unknown>[];
    try {
      writes = [
        writeAlp(arrayLike(geometryBoundary), { codec: 'fp16' }),
        writeAlp(arrayLike(geometryBoundary), { codec: 'int16', frameConvention: 'world-aligned' }),
        writeAlp(arrayLike(geometryMetadataBoundary), { codec: 'fp16', metadata: callerMetadata }),
        writeAlp(arrayLike(geometryMetadataBoundary), { codec: 'int16', frameConvention: 'world-aligned', metadata: callerMetadata }),
        writeAll(arrayLike(v2Boundary), { instanceIds: arrayLike(v2Boundary) }),
        writeAll(arrayLike(v2MetadataBoundary), {
          instanceIds: arrayLike(v2MetadataBoundary),
          metadata: callerMetadata,
        }),
        writeAlx(arrayLike(alxBoundary), { alpTotalPoints: alxBoundary, sourceTotalPoints: alxBoundary + 1 }),
        writeAlx(arrayLike(alxMetadataBoundary), {
          alpTotalPoints: alxMetadataBoundary,
          sourceTotalPoints: alxMetadataBoundary + 1,
          metadata: callerMetadata,
        }),
      ].map((write) => write.catch((error: unknown) => error));
    } finally {
      for (const [name, descriptor] of originals) Object.defineProperty(globalThis, name, descriptor);
    }
    for (const write of writes) {
      expect(await write).toMatchObject({ code: 'writer-input-invalid' });
    }
  });

  test('validates byte-channel ArrayLike lengths before typed-array coercion', async () => {
    const fractional = { 0: 0, 1: 10, length: 1.5 } as ArrayLike<number>;
    const notANumber = { 0: 7, length: Number.NaN } as ArrayLike<number>;
    const negative = { 0: 7, length: -1 } as ArrayLike<number>;
    await expect(writeAli(fractional, { inputRange: [0, 10] })).rejects.toMatchObject({ code: 'writer-input-invalid' });
    await expect(writeAlc(notANumber, ['ffffff'])).rejects.toMatchObject({ code: 'writer-input-invalid' });
    await expect(writeAll(negative)).rejects.toMatchObject({ code: 'writer-input-invalid' });
  });

  test('preflights v2 row counts before normalizing category storage', async () => {
    const impossible = { length: Number.MAX_SAFE_INTEGER + 1 } as unknown as ArrayLike<number>;
    await expect(writeAll(impossible, { instanceIds: impossible })).rejects.toMatchObject({ code: 'writer-input-invalid' });
  });

  test('uses UTF-8 byte length for non-ASCII metadata', async () => {
    const bytes = await writeAll([7], { metadata: { note: 'robot 🤖' } });
    const headerLength = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength).getUint32(1, true);
    expect(new TextDecoder().decode(bytes.subarray(5, 5 + headerLength))).toContain('🤖');
    const decoded = await readAlf(bytes, { channel: 'all' });
    expect(decoded.metadata['note']).toBe('robot 🤖');
  });

  test('gzip and uncompressed inputs decode identically', async () => {
    const plain = await writeAll([4, 5]);
    const compressed = await writeAll([4, 5], { gzip: true });
    const [a, b] = await Promise.all([readAlf(plain, { channel: 'all' }), readAlf(compressed, { channel: 'all' })]);
    expect(a.channel === 'all' && [...a.categories]).toEqual(b.channel === 'all' && [...b.categories]);
  });

  test('keeps one payload view and copies only the exposed byte channel', async () => {
    const source = envelope(1, { _total_points: 3 }, new Uint8Array([4, 5, 6]));
    const input = new SliceCountingBytes(source);
    SliceCountingBytes.calls = 0;
    const file = await readAlf(input, { channel: 'ali' });
    expect(file.channel === 'ali' ? [...file.intensity] : []).toEqual([4, 5, 6]);
    expect(SliceCountingBytes.calls).toBe(1);
    input[input.length - 1] = 99;
    expect(file.channel === 'ali' ? [...file.intensity] : []).toEqual([4, 5, 6]);
  });

  test('gzip output is deterministic', async () => {
    const first = await writeAll([4, 5], { metadata: { sequence: 'same' }, gzip: true });
    const second = await writeAll([4, 5], { metadata: { sequence: 'same' }, gzip: true });
    expect(second).toEqual(first);
  });

  test('bounds gzip expansion before accepting decompressed output', async () => {
    expect(MAX_COMPRESSED_INPUT_BYTES).toBe(513 * 1024 * 1024);
    const compressed = await writeAll(new Uint8Array(4096), { gzip: true });
    await expect(gunzip(compressed, 64)).rejects.toMatchObject({ code: 'gzip-invalid' });

    // A forged modulo-2^32 ISIZE cannot bypass the callback-side bound: the
    // inflater refuses as soon as emitted chunks exceed the bounded buffer.
    const forgedSmallSize = Uint8Array.from(compressed);
    new DataView(forgedSmallSize.buffer).setUint32(forgedSmallSize.length - 4, 32, true);
    await expect(gunzip(forgedSmallSize, 64)).rejects.toMatchObject({ code: 'gzip-malformed' });
  });

  test('reuses immutable fixed Huffman tables across many empty blocks', async () => {
    const constructions = huffmanTableBuildCount();
    expect(await gunzip(emptyFixedGzip(10_000))).toEqual(new Uint8Array());
    expect(huffmanTableBuildCount()).toBe(constructions);
  });

  test('caches repeated dynamic Huffman definitions during the gzip pre-scan', async () => {
    const constructions = huffmanTableBuildCount();
    expect(await gunzip(emptyDynamicGzip(10_000))).toEqual(new Uint8Array());
    expect(huffmanTableBuildCount() - constructions).toBeLessThanOrEqual(3);
  });

  test('accepts valid dynamic Huffman churn without a TypeScript-only refusal', async () => {
    expect(await gunzip(churningDynamicGzip(3000, 256))).toEqual(new Uint8Array());
  });

  test('keeps large DEFLATE cursors out of signed 32-bit arithmetic', () => {
    expect(alignedBitCursor(2 ** 31 + 1)).toBe(2 ** 31 + 8);
    expect([bitCursorByteIndex(2 ** 31 + 9), bitCursorBitOffset(2 ** 31 + 9)]).toEqual([2 ** 28 + 1, 1]);
    expect([bitCursorByteIndex(2 ** 32 + 7), bitCursorBitOffset(2 ** 32 + 7)]).toEqual([2 ** 29, 7]);
  });

  test('validates gzip CRC32 and ISIZE trailers', async () => {
    const compressed = await writeAll([4, 5], { gzip: true });
    const badCrc = Uint8Array.from(compressed);
    badCrc[badCrc.length - 8] = (badCrc[badCrc.length - 8] ?? 0) ^ 1;
    await expect(readAlf(badCrc, { channel: 'all' })).rejects.toMatchObject({ code: 'gzip-malformed' });

    const badSize = Uint8Array.from(compressed);
    badSize[badSize.length - 4] = (badSize[badSize.length - 4] ?? 0) ^ 1;
    await expect(readAlf(badSize, { channel: 'all' })).rejects.toMatchObject({ code: 'gzip-malformed' });
  });

  test('accepts RFC-permitted zero padding after a gzip member', async () => {
    const compressed = await writeAli([0, 127, 255], { gzip: true });
    const padded = new Uint8Array(compressed.length + 8);
    padded.set(compressed);
    const file = await readAlf(padded, { channel: 'ali' });
    expect(file.channel).toBe('ali');
    if (file.channel !== 'ali') return;
    expect([...file.intensity]).toEqual([0, 127, 255]);
  });

  test('still refuses nonzero data after a gzip member', async () => {
    const compressed = await writeAli([0, 127, 255], { gzip: true });
    const trailingJunk = new Uint8Array(compressed.length + 1);
    trailingJunk.set(compressed);
    trailingJunk[trailingJunk.length - 1] = 1;
    await expect(readAlf(trailingJunk, { channel: 'ali' })).rejects.toMatchObject({ code: 'multiple-gzip-members' });
  });

  test('writeAlf preserves channel layout and codec', async () => {
    const first = await readAlf(await writeAlp([1.37, 2.11, 3.07], { codec: 'int16', frameConvention: 'world-aligned' }), { channel: 'alp' });
    const second = await readAlf(await writeAlf(first), { channel: 'alp' });
    expect(second.channel).toBe('alp');
    if (second.channel !== 'alp') return;
    expect(second.codec).toBe('int16');
    expect(second.points).toEqual(first.channel === 'alp' ? first.points : new Float64Array());
  });

  test('writeAlf preserves a larger decoded v3 geometry error bound', async () => {
    const first = await readAlf(
      envelope(
        3,
        {
          _total_points: 1,
          _geom_codec: 'int16', _frame_convention: 'world-aligned', frameConvention: 'world-aligned',
          _geom_scale: 0.0001,
          _geom_offset: [0, 0, 0],
          _geom_max_err_m: 0.2,
        },
        new Uint8Array(6),
      ),
      { channel: 'alp' },
    );
    const second = await readAlf(await writeAlf(first), { channel: 'alp' });
    expect(first.channel === 'alp' ? first.maxGeomErrorM : 0).toBe(0.2);
    expect(second.channel === 'alp' ? second.maxGeomErrorM : 0).toBe(0.2);
    expect(second.metadata['_geom_max_err_m']).toBe(0.2);
  });

  test('writeAlf applies an inherited v3 error bound to the default budget', async () => {
    const first = await readAlf(
      envelope(
        3,
        {
          _total_points: 1,
          _geom_codec: 'int16', _frame_convention: 'world-aligned', frameConvention: 'world-aligned',
          _geom_scale: 0.1,
          _geom_offset: [0, 0, 0],
          _geom_max_err_m: 0.3,
        },
        new Uint8Array(6),
      ),
      { channel: 'alp' },
    );

    await expect(writeAlf(first)).rejects.toMatchObject({ code: 'writer-input-invalid' });
    await expect(writeAlf(first, { checkBudget: false })).resolves.toBeInstanceOf(Uint8Array);
  });

  test('writeAlf preserves an absent v2 instance map', async () => {
    const payload = new Uint8Array([4, 7, 0]);
    const first = await readAlf(envelope(2, { _total_points: 1, _format_version: 2 }, payload), { channel: 'all' });
    expect(first.channel === 'all' ? first.instanceMap : {}).toBeNull();
    const second = await readAlf(await writeAlf(first), { channel: 'all' });
    expect(second.channel === 'all' ? second.instanceMap : {}).toBeNull();
    expect(second.channel === 'all' ? [...second.instanceIds!] : []).toEqual([7]);
  });

  test('writeAlf normalizes decoded negative-zero metadata without relaxing direct writers', async () => {
    const first = await readAlf(
      envelopeWithHeader(1, '{"_total_points":1,"custom":{"offset":-0}}', new Uint8Array([7])),
      { channel: 'ali' },
    );
    expect(Object.is((first.metadata['custom'] as { offset: number }).offset, -0)).toBe(true);
    const second = await readAlf(await writeAlf(first), { channel: 'ali' });
    expect(Object.is((second.metadata['custom'] as { offset: number }).offset, -0)).toBe(false);
    expect((second.metadata['custom'] as { offset: number }).offset).toBe(0);

    const geometry = await readAlf(
      envelopeWithHeader(
        3,
        '{"_total_points":1,"_geom_codec":"int16","_frame_convention":"world-aligned","_geom_scale":1,"_geom_offset":[-0,0,0]}',
        new Uint8Array(6),
      ),
      { channel: 'alp' },
    );
    await expect(writeAlf(geometry, { checkBudget: false })).resolves.toBeInstanceOf(Uint8Array);

    await expect(writeAli([7], { metadata: { custom: { offset: -0 } } })).rejects.toMatchObject({
      code: 'writer-input-invalid',
    });
  });

  test('assigns stable refusal codes', async () => {
    const invalid = envelope(2, { _total_points: 1 }, new Uint8Array([0, 0, 0, 0, 0, 0]));
    await expect(readAlf(invalid, { channel: 'alp' })).rejects.toMatchObject<Partial<AlfError>>({
      code: 'label-version-on-non-label-channel',
    });
    const unsupportedCodec = envelope(
      1,
      { _total_points: 1, _geom_codec: 'int24' },
      new Uint8Array(6),
    );
    await expect(readAlf(unsupportedCodec, { channel: 'alp' })).rejects.toMatchObject<
      Partial<AlfError>
    >({ code: 'codec-unsupported' });
  });

  test('refuses unknown geometry codecs with their shared code', async () => {
    const invalid = envelope(1, { _total_points: 1, _geom_codec: 'int24' }, new Uint8Array(6));
    await expect(readAlf(invalid, { channel: 'alp' })).rejects.toMatchObject({ code: 'codec-unsupported' });
  });

  test('bounds nested JSON before recursive duplicate-key validation', async () => {
    const nested = `${'['.repeat(20_000)}0${']'.repeat(20_000)}`;
    await expect(
      readAlf(envelopeWithHeader(1, `{"_total_points":0,"custom":${nested}}`, new Uint8Array()), {
        channel: 'ali',
      }),
    ).rejects.toMatchObject({ code: 'header-nesting-too-deep' });
  });

  test('refuses .alx without its own point count', async () => {
    const missing = envelope(1, { _source_total_points: 4 }, new Uint8Array([1, 0, 0, 0]));
    await expect(readAlf(missing, { channel: 'alx' })).rejects.toMatchObject({ code: 'total-points-missing' });
  });

  test('refuses an ambiguous disagreement between .all layout signals', async () => {
    const ambiguous = envelope(1, { _format_version: 2 }, new Uint8Array([1, 0, 0]));
    await expect(readAlf(ambiguous, { channel: 'all' })).rejects.toMatchObject({ code: 'label-layout-ambiguous' });

    const empty = envelope(1, { _format_version: 2, _total_points: 0 }, new Uint8Array());
    await expect(readAlf(empty, { channel: 'all' })).rejects.toMatchObject({ code: 'label-layout-ambiguous' });
  });

  test('refuses an empty disagreement between .all layout signals', async () => {
    const ambiguous = envelope(1, { _total_points: 0, _format_version: 2 }, new Uint8Array());
    await expect(readAlf(ambiguous, { channel: 'all' })).rejects.toMatchObject({ code: 'label-layout-ambiguous' });
  });

  test('preserves integer lexemes for reserved count and layout fields', async () => {
    const cases: Array<readonly [AlfChannel, string, Uint8Array]> = [
      ['all', '{"_total_points":1.00000000000000001}', new Uint8Array([1])],
      ['all', '{"_total_points":1,"_format_version":1.00000000000000001}', new Uint8Array([1])],
      ['alx', '{"_total_points":1,"_source_total_points":2.00000000000000001}', new Uint8Array([1, 0, 0, 0])],
    ];
    for (const [channel, header, payload] of cases) {
      await expect(readAlf(envelopeWithHeader(1, header, payload), { channel })).rejects.toBeInstanceOf(AlfError);
    }
  });

  test('rejects duplicate reserved keys before numeric-token preflight', async () => {
    const bytes = envelopeWithHeader(1, '{"_total_points":"discarded","_total_points":1}', new Uint8Array([7]));
    await expect(readAlf(bytes, { channel: 'ali' })).rejects.toMatchObject({ code: 'header-key-duplicate' });
  });

  test('preserves geometry-anchor numeric tokens before JSON parsing', async () => {
    for (const anchor of ['device_position', 'world_centroid_epsg3857']) {
      const header = `{"_total_points":1,"${anchor}":{"x":9007199254740993,"y":0,"z":0}}`;
      await expect(
        readAlf(envelopeWithHeader(1, header, new Uint8Array(6)), { channel: 'alp' }),
      ).rejects.toMatchObject({ code: 'geom-number-inexact' });
    }

    const exact = await readAlf(
      envelopeWithHeader(
        1,
        '{"_total_points":1,"device_position":{"x":18446744073709551616,"y":0,"z":0}}',
        new Uint8Array(6),
      ),
      { channel: 'alp' },
    );
    expect((exact.metadata['device_position'] as { x: number }).x).toBe(2 ** 64);
  });

  test('validates geometry anchor shape during the envelope read', async () => {
    await expect(
      readAlf(
        envelopeWithHeader(
          1,
          '{"_total_points":1,"device_position":{"x":1,"y":2}}',
          new Uint8Array(6),
        ),
        { channel: 'alp' },
      ),
    ).rejects.toMatchObject({ code: 'geometry-anchor-malformed' });
  });

  test('treats geometry anchor names as ordinary metadata outside .alp', async () => {
    const file = await readAlf(
      envelopeWithHeader(
        1,
        '{"_total_points":1,"device_position":{"x":9007199254740993,"y":0,"z":0}}',
        new Uint8Array([7]),
      ),
      { channel: 'ali' },
    );

    expect(file.channel).toBe('ali');
    expect(file.channel === 'ali' ? [...file.intensity] : []).toEqual([7]);
  });

  test('accepts exactly integral count spellings and exact wide source counts', async () => {
    for (const token of ['1.0', '1e0', '1e+00000']) {
      const file = await readAlf(envelopeWithHeader(1, `{"_total_points":${token}}`, new Uint8Array([7])), {
        channel: 'ali',
      });
      expect(file.totalPoints).toBe(1);
    }
    const negativeZero = await readAlf(envelopeWithHeader(1, '{"_total_points":-0.0}', new Uint8Array()), {
      channel: 'ali',
    });
    expect(negativeZero.totalPoints).toBe(0);
    const alx = await readAlf(
      envelopeWithHeader(
        1,
        '{"_total_points":1.0,"_source_total_points":18446744073709551616}',
        new Uint8Array([0, 0, 0, 0]),
      ),
      { channel: 'alx' },
    );
    expect(alx.channel === 'alx' ? alx.sourceTotalPoints : 0).toBe(2 ** 64);
  });

  test('cancels integral count trailing zeros before bounding the exponent', async () => {
    const exactOne = `1${'0'.repeat(10_000)}e-10000`;
    const file = await readAlf(
      envelopeWithHeader(1, `{"_total_points":${exactOne}}`, new Uint8Array([7])),
      { channel: 'ali' },
    );
    expect(file.totalPoints).toBe(1);

    await expect(
      readAlf(envelopeWithHeader(1, '{"_total_points":1e-10000000}', new Uint8Array()), {
        channel: 'ali',
      }),
    ).rejects.toMatchObject({ code: 'total-points-not-an-integer' });
  });

  test('scopes format-owned numeric tokens to their owning channels', async () => {
    const file = await readAlf(
      await writeAli([7], { metadata: { _format_version: 'legacy', _source_total_points: 'legacy' } }),
      { channel: 'ali' },
    );
    expect(file.metadata['_format_version']).toBe('legacy');
    expect(file.metadata['_source_total_points']).toBe('legacy');

    const unrelatedAnchor = await readAlf(
      envelopeWithHeader(
        1,
        '{"_total_points":1,"device_position":{"x":9007199254740993,"y":0,"z":0}}',
        new Uint8Array([7]),
      ),
      { channel: 'ali' },
    );
    expect((unrelatedAnchor.metadata['device_position'] as { x: number }).x).toBe(9007199254740992);

    await expect(
      readAlf(envelopeWithHeader(1, '{"_total_points":1,"_format_version":"legacy"}', new Uint8Array([7])), {
        channel: 'all',
      }),
    ).rejects.toMatchObject({ code: 'total-points-not-an-integer' });
    await expect(
      readAlf(
        envelopeWithHeader(1, '{"_total_points":1,"_source_total_points":"legacy"}', new Uint8Array(4)),
        { channel: 'alx' },
      ),
    ).rejects.toMatchObject({ code: 'total-points-not-an-integer' });
  });

  test('lets the label layout validator classify exact unsupported versions', async () => {
    await expect(
      readAlf(envelopeWithHeader(1, '{"_total_points":1,"_format_version":-1.0}', new Uint8Array([7])), {
        channel: 'all',
      }),
    ).rejects.toMatchObject({ code: 'label-layout-unsupported' });
    await expect(
      readAlf(envelopeWithHeader(1, '{"_total_points":1,"_format_version":1.5}', new Uint8Array([7])), {
        channel: 'all',
      }),
    ).rejects.toMatchObject({ code: 'total-points-not-an-integer' });
  });

  test('v3 error bound includes reconstruction rounding and ignores stale lower headers', async () => {
    const payload = new Uint8Array(18);
    const payloadView = new DataView(payload.buffer);
    [0, 0, 0, 40, 0, 0, 80, 0, 0].forEach((value, index) => payloadView.setInt16(index * 2, value, true));
    const metadata = {
      _total_points: 3,
      _geom_codec: 'int16', _frame_convention: 'world-aligned', frameConvention: 'world-aligned',
      _geom_scale: 0.0001,
      _geom_offset: [-4.488e11, 0, 0],
      _geom_max_err_m: 1e-12,
    };
    const file = await readAlf(envelope(3, metadata, payload), { channel: 'alp' });
    expect(file.channel).toBe('alp');
    if (file.channel !== 'alp') return;
    expect(file.maxGeomErrorM).toBe(0.00007832031250000035);

    const { _geom_max_err_m: _ignored, ...withoutRecordedBound } = metadata;
    const missing = await readAlf(envelope(3, withoutRecordedBound, payload), { channel: 'alp' });
    expect(missing.channel === 'alp' ? missing.maxGeomErrorM : 0).toBe(file.maxGeomErrorM);
  });

  test('refuses lossy numeric tokens in v3 grid metadata before JSON rounding', async () => {
    const payload = new Uint8Array(6);
    const headers = [
      '{"_total_points":1,"_geom_codec":"int16","_frame_convention":"world-aligned","_geom_scale":1,"_geom_offset":[9007199254740993,0,0]}',
      '{"_total_points":1,"_geom_codec":"int16","_frame_convention":"world-aligned","_geom_scale":9007199254740993,"_geom_offset":[0,0,0]}',
      '{"_total_points":1,"_geom_codec":"int16","_frame_convention":"world-aligned","_geom_scale":9007199254740993.0,"_geom_offset":[0,0,0]}',
      '{"_total_points":1,"_geom_codec":"int16","_frame_convention":"world-aligned","_geom_scale":9007199254740993e0,"_geom_offset":[0,0,0]}',
      '{"_total_points":1,"_geom_codec":"int16","_frame_convention":"world-aligned","_geom_scale":1,"_geom_offset":[9007199254740993.5,0,0]}',
      '{"_total_points":1,"_geom_codec":"int16","_frame_convention":"world-aligned","_geom_scale":1,"_geom_offset":[18446744073709552000,0,0]}',
      '{"_total_points":1,"_geom_codec":"int16","_frame_convention":"world-aligned","_geom_scale":1.0000000000000001,"_geom_offset":[0,0,0]}',
      '{"_total_points":1,"_geom_codec":"int16","_frame_convention":"world-aligned","_geom_scale":1,"_geom_offset":[1000000000000000100,0,0]}',
      '{"_total_points":1,"_geom_codec":"int16","_frame_convention":"world-aligned","_geom_scale":1,"_geom_offset":[1000000000000000100.0,0,0]}',
      '{"_total_points":1,"_geom_codec":"int16","_frame_convention":"world-aligned","_geom_scale":1,"_geom_offset":[1000000000000000100e0,0,0]}',
      '{"_total_points":1,"_geom_codec":"int16","_frame_convention":"world-aligned","_geom_scale":1,"_geom_offset":[9.007199254740993e15,0,0]}',
    ];
    for (const header of headers) {
      await expect(readAlf(envelopeWithHeader(3, header, payload), { channel: 'alp' })).rejects.toMatchObject({
        code: 'geom-number-inexact',
      });
    }
  });

  test('accepts an exactly representable large integer v3 grid token', async () => {
    for (const exact of [
      '9007199254740992',
      '9007199254740994',
      '18446744073709551616',
      '18446744073709551616.0',
      '18446744073709551616e0',
    ]) {
      const header =
        `{"_total_points":1,"_geom_codec":"int16","_frame_convention":"world-aligned","_geom_scale":${exact},"_geom_offset":[${exact},0,0]}`;
      const file = await readAlf(envelopeWithHeader(3, header, new Uint8Array(6)), { channel: 'alp' });
      expect(file.channel === 'alp' ? file.points[0] : 0).toBe(Number(exact));
      expect(file.channel === 'alp' ? file.metadata['_geom_scale'] : 0).toBe(Number(exact));
    }
  });

  test('refuses duplicate JSON member names at any header depth', async () => {
    const header = '{"_total_points":1,"custom":{"sensor":"left","sensor":"right"}}';
    await expect(readAlf(envelopeWithHeader(1, header, new Uint8Array([7])), { channel: 'ali' })).rejects.toMatchObject({
      code: 'header-key-duplicate',
    });
  });

  test('accepts v3 decimals and exponents preserved by binary64 shortest form', async () => {
    const header =
      '{"_total_points":1,"_geom_codec":"int16","_frame_convention":"world-aligned","_geom_scale":0.1,"_geom_offset":[1e+308,0.30000000000000004,0]}';
    const file = await readAlf(envelopeWithHeader(3, header, new Uint8Array(6)), { channel: 'alp' });
    expect(file.channel).toBe('alp');
    if (file.channel !== 'alp') return;
    expect(file.points[0]).toBe(1e308);
    expect(file.points[1]).toBe(0.30000000000000004);
  });

  test('metadata preflight refuses values JSON would drop or rewrite', async () => {
    const cases: unknown[] = [
      { value: NaN },
      { value: Infinity },
      { value: undefined },
      { value: new Map([['key', 'value']]) },
      { value: new Date(0) },
      { value: -0 },
      { value: [1, undefined] },
    ];
    for (const metadata of cases) {
      await expect(writeAll([1], { metadata: metadata as Record<string, unknown> })).rejects.toMatchObject({
        code: 'writer-input-invalid',
      });
    }
    await expect(writeAll([1], { metadata: new Map() as unknown as Record<string, unknown> })).rejects.toMatchObject({
      code: 'writer-input-invalid',
    });
  });

  test('bounds reader and writer metadata nesting with stable ALF errors', async () => {
    let allowed: unknown = 0;
    for (let depth = 0; depth < 255; depth += 1) allowed = [allowed];
    const allowedFile = await readAlf(await writeAli([], { metadata: { custom: allowed } }), { channel: 'ali' });
    expect(allowedFile.metadata['custom']).toBeDefined();

    let tooDeep: unknown = allowed;
    tooDeep = [tooDeep];
    await expect(writeAli([], { metadata: { custom: tooDeep } })).rejects.toMatchObject({ code: 'writer-input-invalid' });

    const header = `{"_total_points":0,"custom":${'['.repeat(256)}0${']'.repeat(256)}}`;
    await expect(readAlf(envelopeWithHeader(1, header, new Uint8Array()), { channel: 'ali' })).rejects.toMatchObject({
      code: 'header-nesting-too-deep',
    });
  });

  test('refuses decoded non-finite numbers anywhere in reader metadata', async () => {
    const header = '{"_total_points":0,"custom":{"nested":[1e309]}}';
    await expect(readAlf(envelopeWithHeader(1, header, new Uint8Array()), { channel: 'ali' })).rejects.toMatchObject({
      code: 'header-not-json',
    });
  });

  test('rejects duplicate and out-of-range .alx indexes by their own rules', async () => {
    const duplicate = envelope(1, { _total_points: 2, _source_total_points: 4 }, new Uint8Array([1, 0, 0, 0, 1, 0, 0, 0]));
    await expect(readAlf(duplicate, { channel: 'alx' })).rejects.toMatchObject({ code: 'kept-index-duplicate' });
    const outside = envelope(1, { _total_points: 1, _source_total_points: 4 }, new Uint8Array([4, 0, 0, 0]));
    await expect(readAlf(outside, { channel: 'alx' })).rejects.toMatchObject({ code: 'kept-index-out-of-range' });
  });

  test('validates ALX duplicates without cloning the complete index array', async () => {
    const originalSlice = Object.getOwnPropertyDescriptor(Uint32Array.prototype, 'slice');
    Object.defineProperty(Uint32Array.prototype, 'slice', {
      configurable: true,
      value: (): never => {
        throw new Error('full Uint32 clone attempted');
      },
    });
    try {
      const payload = new Uint8Array(12);
      const view = new DataView(payload.buffer);
      [9, 0, 4].forEach((value, index) => view.setUint32(index * 4, value, true));
      const file = await readAlf(envelope(1, { _total_points: 3, _source_total_points: 10 }, payload), {
        channel: 'alx',
      });
      expect(file.channel === 'alx' ? [...file.keptIndices] : []).toEqual([9, 0, 4]);
      await expect(
        writeAlx([9, 0, 4], { alpTotalPoints: 3, sourceTotalPoints: 10 }),
      ).resolves.toBeInstanceOf(Uint8Array);
    } finally {
      if (originalSlice === undefined) {
        Reflect.deleteProperty(Uint32Array.prototype, 'slice');
      } else {
        Object.defineProperty(Uint32Array.prototype, 'slice', originalSlice);
      }
    }
  });
});
