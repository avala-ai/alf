"""float16 coordinate-magnitude guard for ALF (.alp) writers.

Part of the ALF surface, not a general utility: it encodes the annotation-grade
budget the fp16 codec is judged against.

.alp stores each coordinate component as binary16, whose rounding error scales with the
component's magnitude (half-ULP doubles per power of two): ~3.1 cm in the [64, 128) band,
~6.3 cm in [128, 256), 25 cm in [512, 1024). Points must therefore be written LOCAL to a
nearby origin (`device_position`, a tile centroid, ...) — a converter that forgets the
localization writes coordinates whose quantization error is measured in meters.

This guard is called by writers on the max localized |component| per frame:

- >= FP16_WARN_COMPONENT_M (64 m): warn — error there is already past the ~3 cm/axis
  budget annotation-grade geometry needs.
- >= FP16_ERROR_COMPONENT_M (512 m): raise — error is >= 0.25 m per component, unusable
  for any annotation purpose, and in practice means the localization step failed or was
  skipped, so writing the file would persist garbage.

See the precision guide (website/docs/guides/precision.md) for the derivation
and the per-band error table.
"""

import math

FP16_WARN_COMPONENT_M = 64.0
FP16_ERROR_COMPONENT_M = 512.0


def check_fp16_component_budget(max_abs_component: float, file_label: str) -> None:
    """Warn/fail when localized coordinates are too large for float16 storage.

    Raises ValueError at/above FP16_ERROR_COMPONENT_M — at that magnitude the
    quantization error is >= 0.25 m per component and the frame is unusable, which
    indicates the writer skipped or botched its localization (device-position or
    centroid subtraction) rather than a genuinely usable large scene. Scenes that
    legitimately span more than ~1 km must be tiled with per-tile origins first,
    which is what the aerial tile writers already do.
    """
    if max_abs_component >= FP16_ERROR_COMPONENT_M:
        raise ValueError(
            f"{file_label}: localized coordinate component reaches {max_abs_component:.1f} m "
            f"(>= {FP16_ERROR_COMPONENT_M:.0f} m). float16 quantization error at this magnitude "
            f"is >= 0.25 m per component — the localization (device-position/centroid "
            f"subtraction) has almost certainly failed; refusing to write a garbage .alp. "
            f"Legitimately large scenes must be tiled with per-tile origins instead."
        )
    if max_abs_component >= FP16_WARN_COMPONENT_M:
        # Half-ULP for binary16 doubles per power of two: ~3.1 cm in [64, 128),
        # ~6.3 cm in [128, 256), etc.
        exponent = math.floor(math.log2(max_abs_component))
        max_err_m = 2.0 ** (exponent - 11)
        print(
            f"[WARN] {file_label}: localized coordinate component reaches "
            f"{max_abs_component:.1f} m; float16 error there is up to {max_err_m * 100:.1f} cm "
            f"per axis (annotation budget ~3 cm). Consider cropping or re-centering."
        )
