"""ALF — Annotated Local Frames.

A container for annotated point-cloud sequences in a local coordinate frame,
streamed per frame. See ``README.md`` for the format and ``spec/`` for the
normative specification.

Everything public is re-exported here, but imported LAZILY: the codecs need
NumPy, and a caller that only wants to know what channel a filename is should
not pay for it.
"""

__version__ = "0.1.0"

_LAZY = frozenset(
    {
        "AlfError",
        "AlfFile",
        "FRAME_CONVENTIONS",
        "FRAME_DEVICE_ALIGNED",
        "FRAME_WORLD_ALIGNED",
        "GEOM_FP16",
        "GEOM_INT16",
        "choose_geom_offset",
        "choose_geom_scale",
        "geom_error_report",
        "iter_frames",
        "map_intensity",
        "memory_limits",
        "read_alf",
        "read_alf_header",
        "set_memory_limits",
        "write_ali",
        "write_alc",
        "write_all",
        "write_alp",
        "write_alx",
    }
)

__all__ = sorted(_LAZY) + ["__version__"]


def __getattr__(name: str):
    if name in _LAZY:
        from importlib import import_module

        value = getattr(import_module("alf._io"), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | _LAZY)
