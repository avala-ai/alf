# Security policy

## Reporting a vulnerability

Report privately via GitHub's "Report a vulnerability" flow on this repository's Security tab, or by
email to security@avala.ai. Please do not open a public issue for a vulnerability.

Expect an acknowledgement within three working days and an assessment within ten.

## Threat model

**Decoders in this repository parse untrusted bytes.** An ALF file may arrive from anywhere — an
upload, an object store, a browser fetch — and a decoder must treat every field in it as hostile
until validated. That makes the following in scope for a security report:

- Memory-safety failures, out-of-bounds reads, or crashes from a malformed file
- A panic where a named refusal is required. The Rust crate is consumed from WebAssembly, where a
  panic is not something a caller can handle, so it is a security-relevant defect rather than a
  robustness one
- Unbounded allocation driven by a header field: a decompression bomb, an implausible
  `_total_points`, a declared header length that exceeds the resource, a JSON nesting depth that
  overflows a parser stack
- Pathological runtime from a crafted number. A compact token such as `1e100000000` must be bounded
  before it is expanded, and a negative exponent must be compared against the significand's
  available trailing zeros before a divisor is constructed
- Path traversal or arbitrary writes from names embedded in a file

Out of scope: memory exhaustion from a legitimately enormous but well-formed file — that is what
`set_memory_limits` and the per-implementation ceilings are for — and issues in applications that
consume these libraries.

## A wrong answer is a security issue here

Channels are index-correlated: entry _i_ of every channel describes point _i_. A reader that
resolves an ambiguity by guessing does not fail — it returns a frame in which every label and
intensity reading after the gap is misattributed, and nothing downstream can detect it.

So a report that a reader **accepts** something the specification says it must refuse is a valid
security report, not only a conformance bug. The same is true of a reader that refuses for the
wrong reason, because the refusal identifier is what a caller routes on.

## Expectations of implementations

Any implementation claiming conformance is expected to:

- Apply declared-size and nesting bounds **before** allocating or invoking a recursive parser, not
  after
- Enforce decompression ceilings incrementally rather than on a finished buffer
- Produce a named refusal for every malformed input in the shared invalid corpus, and exit
  successfully while doing so — a refusal is a result, not a crash
- Never infer a channel from payload shape

## How that is checked

The invalid corpus in `tests/conformance/` is 72 cases that each break exactly one rule and name the
refusal a conforming reader must produce. Several exist specifically for the resource-exhaustion
routes above: `header-too-large`, `header-nesting-too-deep`, `total-points-hostile-positive-exponent`,
`total-points-hostile-negative-exponent`, `multiple-gzip-members`.

`tests/test_rust_panic_safety.py` runs the whole corpus through the Rust reader and requires a named
refusal rather than a panic on every entry.

New rules land with a corpus case. See [CONTRIBUTING.md](CONTRIBUTING.md) and
[AGENTS.md](AGENTS.md).
