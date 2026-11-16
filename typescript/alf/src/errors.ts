export type AlfErrorCode =
  | 'version-unsupported'
  | 'label-version-on-non-label-channel'
  | 'geometry-version-on-non-geometry-channel'
  | 'codec-unsupported'
  | 'codec-version-disagreement'
  | 'codec-unsupported'
  | 'geom-scale-missing'
  | 'geom-scale-not-positive'
  | 'geom-offset-malformed'
  | 'geom-number-inexact'
  | 'geometry-anchor-malformed'
  | 'frame-convention-missing'
  | 'frame-convention-unsupported'
  | 'frame-convention-on-legacy-version'
  | 'frame-rotation-missing'
  | 'frame-rotation-malformed'
  | 'frame-rotation-not-normalized'
  | 'frame-rotation-unexpected'
  | 'payload-length-mismatch'
  | 'total-points-missing'
  | 'total-points-not-an-integer'
  | 'payload-not-whole-points'
  | 'coordinate-non-finite'
  | 'geom-partial-no-return'
  | 'label-layout-unsupported'
  | 'label-layout-ambiguous'
  | 'instance-id-reserved'
  | 'instance-map-incomplete'
  | 'instance-map-malformed'
  | 'palette-index-out-of-range'
  | 'palette-entry-malformed'
  | 'header-not-an-object'
  | 'header-truncated'
  | 'header-not-utf8'
  | 'header-not-json'
  | 'header-nesting-too-deep'
  | 'header-key-duplicate'
  | 'header-too-large'
  | 'input-too-large'
  | 'multiple-gzip-members'
  | 'gzip-malformed'
  | 'gzip-invalid'
  | 'channel-unknown'
  | 'channel-required'
  | 'source-total-points-missing'
  | 'kept-index-duplicate'
  | 'kept-index-out-of-range'
  | 'writer-input-invalid';

/** A format refusal whose code is stable across runtimes and releases. */
export class AlfError extends Error {
  readonly code: AlfErrorCode;

  constructor(code: AlfErrorCode, message: string, options?: ErrorOptions) {
    super(message, options);
    this.name = 'AlfError';
    this.code = code;
  }
}

export function refuse(code: AlfErrorCode, message: string): never {
  throw new AlfError(code, message);
}
