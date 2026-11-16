#!/usr/bin/env bun
import { AlfError, readAlf, type AlfChannel, type AlfFile } from '../alf/src/index.ts';
import { roundCanonical } from './canonical.ts';

function inferChannel(path: string): AlfChannel {
  const withoutGzip = path.endsWith('.gz') ? path.slice(0, -3) : path;
  const suffix = withoutGzip.slice(withoutGzip.lastIndexOf('.') + 1);
  if (suffix === 'alp' || suffix === 'ali' || suffix === 'alc' || suffix === 'all' || suffix === 'alx') return suffix;
  throw new Error(`Cannot infer ALF channel from ${path}; pass it as the second argument`);
}

/** One coordinate, in the spelling every language must produce.
 *
 * JSON has no NaN: `JSON.stringify(NaN)` silently writes `null`, while Python's
 * `json.dumps` emits the bare non-JSON token `NaN` — so three conforming
 * readers would disagree on the text while agreeing on the bytes. Non-finite
 * components are valid `.alp` content (spec/format.md §5.1), and the three
 * spellings stay distinct because a reader that turned +Infinity into NaN would
 * have changed the file's meaning.
 */
function canonicalComponent(value: number): number | string {
  if (Number.isNaN(value)) return 'NaN';
  if (!Number.isFinite(value)) return value > 0 ? 'Infinity' : '-Infinity';
  return roundCanonical(value, 6);
}

function canonical(file: AlfFile): Record<string, unknown> {
  const common: Record<string, unknown> = { version: file.version, extension: file.channel };
  switch (file.channel) {
    case 'alp': {
      const points: (number | string)[][] = [];
      for (let index = 0; index < file.points.length; index += 3) {
        points.push([
          canonicalComponent(file.points[index] ?? 0),
          canonicalComponent(file.points[index + 1] ?? 0),
          canonicalComponent(file.points[index + 2] ?? 0),
        ]);
      }
      const anchors = Object.fromEntries(
        ['device_position', 'world_centroid_epsg3857']
          .filter((key) => Object.prototype.hasOwnProperty.call(file.metadata, key))
          .map((key) => [key, file.metadata[key]]),
      );
      return {
        ...common,
        codec: file.codec,
        points,
        max_geom_error_m: roundCanonical(file.maxGeomErrorM, 12),
        frame_convention: file.frameConvention,
        ...(file.frameRotation === null
          ? {}
          : {
              frame_rotation: {
                w: canonicalComponent(file.frameRotation.w),
                x: canonicalComponent(file.frameRotation.x),
                y: canonicalComponent(file.frameRotation.y),
                z: canonicalComponent(file.frameRotation.z),
              },
            }),
        ...(Object.keys(anchors).length === 0 ? {} : { anchors }),
      };
    }
    case 'ali':
      return { ...common, intensity: [...file.intensity] };
    case 'alc':
      return { ...common, palette: file.palette, palette_indices: [...file.paletteIndices] };
    case 'all':
      return {
        ...common,
        format_version: file.formatVersion,
        categories: [...file.categories],
        instance_ids: file.instanceIds === null ? null : [...file.instanceIds],
        instance_map: file.instanceMap,
      };
    case 'alx':
      return {
        ...common,
        kept_indices: [...file.keptIndices],
        source_total_points: file.sourceTotalPoints,
      };
  }
}

function canonicalJson(value: unknown): string {
  if (typeof value === 'number') {
    return Number.isInteger(value) && !Number.isSafeInteger(value) ? value.toExponential() : JSON.stringify(value);
  }
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(',')}]`;
  if (value !== null && typeof value === 'object') {
    return `{${Object.entries(value)
      .map(([key, item]) => `${JSON.stringify(key)}:${canonicalJson(item)}`)
      .join(',')}}`;
  }
  return JSON.stringify(value);
}

const path = Bun.argv[2];
if (path === undefined) throw new Error('usage: read.ts <path> [channel]');
const channel = (Bun.argv[3] as AlfChannel | undefined) ?? inferChannel(path);

try {
  const file = await readAlf(await Bun.file(path).bytes(), { channel });
  console.log(canonicalJson(canonical(file)));
} catch (error) {
  if (error instanceof AlfError) {
    console.log(JSON.stringify({ refused: error.code }));
  } else {
    throw error;
  }
}
