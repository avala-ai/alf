# Guides

ALF — **A**rrays in **L**ocal **F**rames — is a container for point-cloud sequences in a local
coordinate frame, streamed one frame at a time. Each file is one typed array plus a JSON header,
and that is the whole of the name: an `.alp` is an _array of local points_.

- [Concepts](./concepts.md) — the five channels, the envelope, and the one invariant, in about a
  page
- [Precision and the two codecs](./precision.md) — why float16 and int16 both cost 6 bytes per
  point, and which one to write
- [Invalid returns](./invalid-returns.md) — NaN geometry is content, not damage, and what that
  obliges a reader to do
- [Why it fails closed](./failing-closed.md) — the design stance, and why refusals carry stable
  identifiers

## Getting started

- [Read in Python](./getting-started/read-python.md) — the reference implementation
- [Write in Python](./getting-started/write-python.md) — every channel, both geometry codecs
- [Read in a browser](./getting-started/read-web.md) — `@avala-ai/alf`, browser and Node ESM
- [Read in Dart](./getting-started/read-dart.md) — Dart VM, Flutter web and Wasm
- [Read in Rust](./getting-started/read-rust.md) — the reader `wasm32` consumers bind to

## Where the authority lives

The [specification](/spec/format) is normative; these guides are the readable path in. What each
SDK actually implements is the [support matrix](/spec/support), and what proves it is the
[conformance suite](../reference/conformance.md). A refusal identifier you saw in an error message
is in the [refusal catalogue](../reference/refusals.md), and a header field you found in a file is
in the [header field index](../reference/header-fields.md).

## Scope

The format, its specification, its conformance suite, and reader/writer SDKs. Viewers, renderers,
annotation tooling and pipeline glue live with their consumers — see
[AGENTS.md](https://github.com/avala-ai/alf/blob/main/AGENTS.md).

ALF is **not** a geospatial archive. LAS/LAZ/COPC own that and do it better. ALF gives up global
coordinates on purpose and gets its precision back from the trade.
