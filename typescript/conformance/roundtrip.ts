#!/usr/bin/env bun
import {
  writeAlc,
  writeAli,
  writeAll,
  writeAlp,
  writeAlx,
  type AlfMetadata,
  type GeometryCodec,
} from '../alf/src/index.ts';

const [inputPath, channelValue, outputPath] = Bun.argv.slice(2);
if (inputPath === undefined || channelValue === undefined || outputPath === undefined) {
  throw new Error('usage: roundtrip.ts <request.json> <channel> <output-path>');
}
if (!['alp', 'ali', 'alc', 'all', 'alx'].includes(channelValue)) throw new Error(`Unknown ALF channel ${channelValue}`);

interface WriterRequestBase {
  metadata: AlfMetadata;
}

type WriterRequest =
  | (WriterRequestBase & { channel: 'alp'; codec: GeometryCodec; points: (number | string)[] })
  | (WriterRequestBase & { channel: 'ali'; intensity: number[] })
  | (WriterRequestBase & {
      channel: 'all';
      categories: number[];
      instance_ids: number[] | null;
      instance_map: Record<string, string> | null;
    })
  | (WriterRequestBase & { channel: 'alc'; palette_indices: number[]; palette: string[] })
  | (WriterRequestBase & {
      channel: 'alx';
      kept_indices: number[];
      alp_total_points: number;
      source_total_points: number;
    });

/** Parse one coordinate from the request's canonical spelling.
 *
 * JSON has no NaN, so a non-finite component travels as a string. A candidate
 * writer that ignores this and coerces the string to a number writes garbage
 * rather than an invalid return, which is what this gate exists to catch.
 */
function canonicalComponentValue(value: number | string): number {
  if (typeof value !== 'string') return value;
  if (value === 'NaN') return Number.NaN;
  if (value === 'Infinity') return Number.POSITIVE_INFINITY;
  if (value === '-Infinity') return Number.NEGATIVE_INFINITY;
  throw new Error(`Unknown non-finite coordinate spelling ${value}`);
}

const request = (await Bun.file(inputPath).json()) as WriterRequest;
if (request.channel !== channelValue) throw new Error('Request channel disagrees with command channel');
const gzip = outputPath.endsWith('.gz');
const encoded = await (async () => {
  switch (request.channel) {
    case 'alp':
      return writeAlp(request.points.map(canonicalComponentValue), {
        metadata: request.metadata,
        codec: request.codec,
        gzip,
        checkBudget: false,
        // A writer INPUT: the coordinates of a world-aligned and a
        // device-aligned frame are identical, so an encoder cannot derive it.
        ...(request.frame_convention === null
          ? {}
          : { frameConvention: request.frame_convention }),
        ...(request.frame_rotation === null ? {} : { frameRotation: request.frame_rotation }),
      });
    case 'ali':
      return writeAli(request.intensity, { metadata: request.metadata, gzip });
    case 'all':
      return writeAll(request.categories, {
        metadata: request.metadata,
        instanceIds: request.instance_ids,
        instanceMap: request.instance_map,
        gzip,
      });
    case 'alc':
      return writeAlc(request.palette_indices, request.palette, {
        metadata: request.metadata,
        gzip,
      });
    case 'alx':
      return writeAlx(request.kept_indices, {
        metadata: request.metadata,
        alpTotalPoints: request.alp_total_points,
        sourceTotalPoints: request.source_total_points,
        gzip,
      });
  }
})();
await Bun.write(outputPath, encoded);
