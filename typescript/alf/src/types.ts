export const ALF_VERSION_V1 = 1 as const;
export const ALF_VERSION_V2 = 2 as const;
export const ALF_VERSION_V3 = 3 as const;

export type AlfVersion = 1 | 2 | 3;
export type AlfChannel = 'alp' | 'ali' | 'alc' | 'all' | 'alx';
export type AlfMetadata = Record<string, unknown>;
export type GeometryCodec = 'fp16' | 'int16';

/** How a v3 `.alp`'s local points relate to world axes. See `spec/format.md` §5.3. */
export type FrameConvention = 'world-aligned' | 'device-aligned';
export const FRAME_WORLD_ALIGNED = 'world-aligned' as const;
export const FRAME_DEVICE_ALIGNED = 'device-aligned' as const;

/** A unit quaternion, named rather than ordered so no spelling choice exists. */
export interface FrameRotation {
  readonly w: number;
  readonly x: number;
  readonly y: number;
  readonly z: number;
}

interface AlfFileBase {
  readonly channel: AlfChannel;
  readonly version: AlfVersion;
  readonly metadata: Readonly<AlfMetadata>;
  readonly totalPoints: number | null;
}

export interface AlpFile extends AlfFileBase {
  readonly channel: 'alp';
  readonly codec: GeometryCodec;
  /** Flat XYZ triples, always decoded as JavaScript float64 numbers. */
  readonly points: Readonly<Float64Array>;
  readonly maxGeomErrorM: number;
  /** `null` for v1/v2, which do not say and cannot. */
  readonly frameConvention: FrameConvention | null;
  /** Present only on a `device-aligned` frame. */
  readonly frameRotation: FrameRotation | null;
}

export interface AliFile extends AlfFileBase {
  readonly channel: 'ali';
  readonly intensity: Readonly<Uint8Array>;
}

export interface AlcFile extends AlfFileBase {
  readonly channel: 'alc';
  readonly palette: readonly string[];
  readonly paletteIndices: Readonly<Uint8Array>;
}

export interface AllFile extends AlfFileBase {
  readonly channel: 'all';
  readonly formatVersion: 1 | 2;
  readonly categories: Readonly<Uint8Array>;
  readonly instanceIds: Readonly<Uint16Array> | null;
  readonly instanceMap: Readonly<Record<string, string>> | null;
}

export interface AlxFile extends AlfFileBase {
  readonly channel: 'alx';
  readonly keptIndices: Readonly<Uint32Array>;
  readonly sourceTotalPoints: number;
}

export type AlfFile = AlpFile | AliFile | AlcFile | AllFile | AlxFile;

/**
 * What a header-only read returns: the envelope, and nothing that would have
 * required inflating the payload.
 *
 * A distinct TYPE rather than an `AlfFile` with absent fields, because the
 * difference is a contract and the compiler can enforce it. The Python
 * reference returns an `AlfFile` whose payload accessors raise, since a
 * dataclass cannot express "these fields are not present here"; TypeScript can,
 * so reaching for `.points` on a header-only read is a compile error rather
 * than a runtime one.
 */
export type AlfHeader = AlfFileBase & { readonly channel: AlfChannel };

export interface ReadAlfOptions {
  channel: AlfChannel;
}

export interface WriteOptions {
  metadata?: AlfMetadata;
  gzip?: boolean;
}

export interface WriteAlpOptions extends WriteOptions {
  codec?: GeometryCodec;
  /**
   * REQUIRED when `codec` is `int16`. Says whether the caller's points were
   * rotated into the device frame as well as translated — the half of the
   * transform the coordinates cannot reveal. See `spec/format.md` §5.3.
   */
  frameConvention?: FrameConvention;
  /** REQUIRED when `frameConvention` is `device-aligned`. Normalized on write. */
  frameRotation?: FrameRotation;
  geomScale?: number;
  geomOffset?: readonly [number, number, number];
  /** Enforce the annotation-grade geometry error budget. Defaults to true. */
  checkBudget?: boolean;
}

export interface WriteAliOptions extends WriteOptions {
  /** Normalize source values from this inclusive range into uint8. */
  inputRange?: readonly [number, number];
  /** Sequence-level 10th through 90th percentiles for viewer contrast. */
  deciles?: readonly number[];
}

export interface WriteAllOptions extends WriteOptions {
  instanceIds?: ArrayLike<number> | null;
  instanceMap?: Readonly<Record<string, string>> | null;
}

export interface WriteAlxOptions extends WriteOptions {
  alpTotalPoints: number;
  sourceTotalPoints: number;
}
