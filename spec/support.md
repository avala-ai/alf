# ALF implementation support

Support is recorded per feature and per boundary. A check means the shared
suite proves the claim; code existing without that evidence is marked
unverified. SDK conformance and application adoption are separate
checks because one does not imply the other.

## Format-owned SDKs

`read` in a combined read/write row means the implementation reads that channel
and has no writer for it. The Rust crate provides file, header, and geometry-word
readers. Writer capabilities and gates are `n/a` for that crate.

| Feature                                   | Python | Dart | TypeScript | Rust |
| ----------------------------------------- | ------ | ---- | ---------- | ---- |
| `.alp` v1 float16 read                    | ✅     | ✅   | ✅         | ✅   |
| `.alp` v1 float16 write                   | ✅     | ✅   | ✅         | n/a  |
| `.alp` v3 fixed-point read                | ✅     | ✅   | ✅         | ✅   |
| `.alp` v3 fixed-point write               | ✅     | ✅   | ✅         | n/a  |
| `.ali` read/write                         | ✅     | ✅   | ✅         | read |
| `.all` v1 semantic read/write             | ✅     | ✅   | ✅         | read |
| `.all` v2 panoptic read/write             | ✅     | ✅   | ✅         | read |
| `.alc` read/write                         | ✅     | ✅   | ✅         | read |
| `.alx` read/write                         | ✅     | ✅   | ✅         | read |
| gzip detected by magic                    | ✅     | ✅   | ✅         | ✅   |
| named refusals                            | ✅     | ✅   | ✅         | ✅   |
| header-only envelope read                 | ✅     | ✅   | ✅         | ✅   |
| fp16 invalid returns (NaN/Inf) read+write | ✅     | ✅   | ✅         | read |
| v3 frame convention read                  | ✅     | ✅   | ✅         | ✅   |
| v3 frame convention write                 | ✅     | ✅   | ✅         | n/a  |

Evidence behind every check:

| Gate                                     | Python   | Dart                      | TypeScript                          | Rust                                              |
| ---------------------------------------- | -------- | ------------------------- | ----------------------------------- | ------------------------------------------------- |
| Unit/regression suite                    | CI-gated | CI-gated                  | CI-gated                            | CI-gated                                          |
| Shared reader conformance                | 117/117  | 117/117                   | 117/117                             | 117/117                                           |
| Writer variants decoded by all four SDKs | 45/45    | 45/45                     | 45/45                               | n/a — reader only                                 |
| Runtime portability                      | n/a      | VM, JS and Wasm execution | Bun 1.2.14 + 1.3.11; Node ESM smoke | Wasm compile + executed count and geometry checks |

The 117 reader cases are 45 valid variants and 72 one-rule invalid variants.
The 45 writer variants cover every channel, both geometry codecs, both label
layouts, both v3 frame conventions, and gzip zero padding. Each of the three
writers is checked by all four readers, for 540 writer/reader checks in total.
The writer gate compares canonical decoded
meaning, preserves normalized application metadata, and inspects writer-owned
canonical fields—including v3 measured-error metadata and palette
spelling—before compatibility readers can normalize them. Exact decoded fp16
geometry and the positive v3 half-scale error floor are checked before the
human-readable canonical projection can round either difference away.
The writer receives a decoded typed request rather than an already-valid ALF
wire image, so copying its input or a sibling fixture cannot earn evidence.
Exact wide integral geometry values are accepted in integer, decimal, and
exponent spellings while nearby lossy integer collisions and expanded
integral claims remain refused. Normalized scientific writer spellings remain
self-readable when they preserve the decoded binary64 value.
Every oracle and decoded source meaning is snapshotted before candidate code
runs, and the encoder receives an isolated read-only decoded request rather than
a corpus path, so candidate code cannot choose or rewrite the evidence used to
score it.
It does not otherwise require byte identity because gzip and JSON serialization
may differ without changing the format.

Header-only reading is a READER-API capability, not a wire feature, so its
row is not backed by the shared corpus and cannot be: the corpus is
bytes-on-disk evidence keyed by a refusal id, and the same bytes read whole or
read header-first yield the same header. Its evidence is each suite asserting
the resource property directly, on inputs the full reader refuses outright —
Python with an instrumented stream counting bytes consumed, Dart with a member
truncated after its header, a corrupt payload CRC32, and a bounded prefix
inflate over a 4 MiB payload.

Rust adds a stronger form of it, because Rust is the implementation that can
reach the corpus directly from its own test binary: for every valid corpus file
the header read must agree with the full read, and for every INVALID one it must
either refuse with the same identifier or decline to have an opinion — never a
different one. The second half is the drift that matters, stated as a property
over ~70 files rather than as a list of cases somebody remembered to write.

All four implementations agree on which refusals belong to a header-only read.
Building the Dart entry point against Python and TypeScript turned up a split: Python validated
`_total_points` lazily, in the accessor, so a malformed count survived a scan
and raised later from whichever frame a caller looked at closely, while
TypeScript refused the same bytes at the call. The count is decidable from the
header alone, so refusing it is header-scoped; all four now do it eagerly.

These are format-boundary interoperability tests. They are not product
end-to-end tests: they do not load an application's frame sequence, execute a
label merge, or pass coordinates through a WebAssembly point-processing
boundary.

## Application adoption

The SDK gates above establish format interoperability. Each adopter must also
verify its application's complete file path: producer, storage, loader, and
consumers, including workers and WebAssembly boundaries.

Keep that deployment inventory with the application. A consumer that delegates
parsing to one of these SDKs needs integration tests and a guard against local
parsers reappearing. A consumer that implements the format itself needs the
shared conformance gate.

The migration order and the type-erased WebAssembly hazard are in
[`adoption.md`](adoption.md). v3 remains opt-in until every reader on a written
file's path is migrated and tested. An SDK release does not establish that any
particular deployment is ready to change its writer default.
