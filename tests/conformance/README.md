# Conformance suite

The contract between implementations. An implementation claims a feature by
**passing this suite for it**, and the support matrix in the root README records
only what the suite proves.

## How it works

`generate.py` reconstructs the ignored binary fixture for each variant. The
explicit `generate.py --update` maintainer command also rewrites:

- `data/<variant>.<ext>.gz` — a real file, **not committed**
- `data/<variant>.json` — exactly what a correct reader must produce, committed

Each implementation ships a small runner that reads one file and prints the
canonical JSON on stdout. `run.py` runs the runner over every variant and diffs
against the committed expectation.

```bash
python3 tests/conformance/generate.py           # build fixtures, preserve oracle
python3 tests/conformance/generate.py --update  # intentionally update oracle
python3 tests/conformance/run.py                # score the reference runner
python3 tests/conformance/run.py --runner "dart run dart/conformance/bin/conformance.dart"
```

The current corpus has 117 cases: 45 valid channel/layout/transport variants and 72 one-rule invalid variants.
It covers every channel, both `.all`
layouts, both `.alp` geometry codecs, version/channel direction mismatches,
gzip zero padding, `.alp` anchor application metadata, and `.alx` mapping
integrity. Geometry-number cases distinguish exact wide integral decimal and
exponent spellings from nearby expanded integral claims that binary64 changes,
while preserving normalized scientific writer tokens.

Readers and writers make different claims. `run.py` proves a reader against the
valid and invalid corpus. `encode_roundtrip.py` invokes a language writer for
all 45 valid variants, then has the Python reference decode its output:

```bash
python3 tests/conformance/encode_roundtrip.py \
  --encoder "dart run dart/conformance/bin/encode_roundtrip.dart" \
  --reader "bun run typescript/conformance/read.ts" \
  --reader "rust/alf/target/release/alf-conformance"
```

Repeat `--reader` to verify the candidate's actual output with additional SDKs.
Each receives a separate copy and must agree with Python's decoding of that
candidate, including its geometry error bound. CI exercises all three writers
(Python, Dart, TypeScript) against all four readers, covering 540 writer/reader
cases across the 45 valid variants. Rust currently exposes a reader only.

The decoded JSON request, channel, and output paths are appended to the command.
A request contains typed values and application metadata, never an already-valid
ALF wire image or a sibling answer a candidate could copy. Exact output bytes
may differ; canonical decoded meaning may not. Writer-owned canonical fields
that compatibility readers normalize, such as hash-free palette entries, are
also checked directly in the candidate envelope. Every v3 candidate must carry
a finite, non-negative writer-measured `_geom_max_err_m` in its raw header, and
that raw value must cover both the independently measured reconstruction error
and the upward-rounded positive half-scale floor for a non-empty fixed-point
payload. Decoded fp16 geometry is compared exactly before the six-decimal
canonical projection can hide a changed subnormal component.
Application metadata is compared after portable binary64 normalization, with
only explicitly regenerated format-owned fields excluded. ALP anchor arrays
and x/y/z objects compare after their documented writer normalization; the same
member names on non-geometry channels remain ordinary metadata whose structure
must survive unchanged.
All expectations, source meaning, and decoded requests are snapshotted before
candidate code starts. The candidate receives an isolated read-only request,
and mutating a committed expectation is itself a gate failure. Each runner or
encoder starts in a private process session; the harness terminates the entire
group before inspecting output, so forked descendants cannot retain diagnostic
pipes or continue changing a candidate after their parent exits.

## No binaries in the repository

The corpus files are **generated, not committed** — `.gitignore` excludes them
and `generate.py` rebuilds them deterministically. What is committed is the JSON
expectations plus `data/CHECKSUMS.txt`.

`generate.py --verify` is the gate. It regenerates everything in scratch space without
rewriting the committed oracle, asserts every
checksum matches, and asserts that **two consecutive runs produce byte-identical
output**. That second check catches accidental nondeterminism in a writer —
iteration order, a timestamp, a hash seed — which is otherwise invisible until
someone else's CI fails on a diff they did not cause.

It earned its keep immediately: the first run flagged the concatenated-gzip
fixture, because bare `gzip.compress()` stamps the current time into the header.

## The invalid corpus: files that must be refused

A corpus of valid files can only prove a reader **accepts** what it should. It
cannot prove a reader **refuses** what it should — and most of this format's
specification is refusals. A reader that ignores every one of them passes all
45 valid variants.

`generator/invalid.py` declares the other half. Each entry breaks **exactly one**
rule and names the refusal a conforming reader must produce:

```json
{ "refused": "codec-version-disagreement" }
```

### Why an identifier and not an exception type

"Both readers raised an error" is not agreement. `AlfError` covers a hundred
distinct faults, so a negative test that only checks that _something_ was raised
passes when a reader refuses for the **wrong reason** — which is the failure a
negative test exists to catch. The identifier says _which rule_, and it is the
same string in every language.

Identifiers are declared once, in `invalid.py`. A language does not invent one.

### The runner contract

A refusal is a **result, not a crash**: the runner prints
`{"refused": "<id>"}` on stdout and **exits 0**, and the harness diffs it like
any other answer. Exiting non-zero would collapse "refused correctly" and "fell
over" into one outcome, and those are exactly what this corpus exists to tell
apart.

A refusal a runner cannot name prints an empty identifier, matches no
expectation, and fails with a readable diff. **A refusal a library cannot name is
one the suite cannot check, and it should look like a gap rather than a pass.**

### Why these rules

Every rule here already existed in the reference implementation — this corpus
does not invent policy, it makes the policy portable. Each entry carries a `why`
recording what goes wrong when a reader lets it through, because "reject
malformed input" is not a specification and the failure mode is what makes the
rule worth having.
