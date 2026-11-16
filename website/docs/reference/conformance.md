# Conformance suite

The contract between implementations. An implementation claims a feature by **passing this suite for
it**, and the [support matrix](/spec/support) records only what the suite proves.

The corpus lives in [`tests/conformance/`](https://github.com/avala-ai/alf/tree/main/tests/conformance).

## What is in it

**117 reader cases**: 45 valid channel/layout/transport variants and 72 invalid variants that each
break exactly one rule.

The valid corpus covers every channel, both `.all` label layouts, both `.alp` geometry codecs, gzip
zero padding, `.alp` anchor application metadata, and `.alx` mapping integrity. The geometry-number
cases distinguish exact wide integral decimal and exponent spellings from nearby expanded integral
claims that binary64 changes, while preserving normalized scientific writer tokens.

**45 writer variants**, the same valid set, used by a separate gate described below.

## How it runs

`generate.py` reconstructs the binary fixture for each variant. What is committed is the JSON
expectations plus `data/CHECKSUMS.txt`; the fixtures themselves are generated, never committed, and
`.gitignore` excludes them.

Each implementation ships a small runner that reads one file and prints canonical JSON on stdout.
`run.py` runs the runner over every variant and diffs against the committed expectation.

```bash
python3 tests/conformance/generate.py           # build fixtures, preserve the oracle
python3 tests/conformance/generate.py --update  # intentionally update the oracle
python3 tests/conformance/run.py                # score the reference runner
python3 tests/conformance/run.py --runner "dart run dart/conformance/bin/conformance.dart"
python3 tests/conformance/run.py --runner "bun run typescript/conformance/read.ts"
python3 tests/conformance/run.py --runner "rust/alf/target/release/alf-conformance"
```

Regeneration is an explicit maintainer command so that **a codec change cannot bless its own bytes**.

## The determinism gate

`generate.py --verify` is what CI runs. It regenerates everything in scratch space without rewriting
the committed oracle, asserts every checksum matches, and asserts that **two consecutive runs
produce byte-identical output**.

That second check catches accidental nondeterminism in a writer — iteration order, a timestamp, a
hash seed — which is otherwise invisible until someone else's CI fails on a diff they did not cause.
It earned its keep immediately: the first run flagged the concatenated-gzip fixture, because bare
`gzip.compress()` stamps the current time into the header.

## The invalid corpus

A corpus of valid files can only prove a reader **accepts** what it should. It cannot prove a reader
**refuses** what it should — and most of this format's specification is refusals. A reader that
ignores every one of them passes all 45 valid variants.

So `generator/invalid.py` declares the other half. Each entry breaks **exactly one** rule and names
the refusal a conforming reader must produce:

```json
{ "refused": "codec-version-disagreement" }
```

Each entry also carries a `why`, recording what goes wrong when a reader lets it through, because
"reject malformed input" is not a specification and the failure mode is what makes the rule worth
having.

### Why an identifier and not an exception type

"Both readers raised an error" is not agreement. `AlfError` covers a hundred distinct faults, so a
negative test that only checks that _something_ was raised passes when a reader refuses for the
**wrong reason** — which is the failure a negative test exists to catch. The identifier says _which
rule_, and it is the same string in every language.

Identifiers are declared once, in `invalid.py`. A language does not invent one. See the
[refusal catalogue](./refusals.md).

### The runner contract

A refusal is a **result, not a crash**. The runner prints `{"refused": "<id>"}` on stdout and
**exits 0**, and the harness diffs it like any other answer. Exiting non-zero would collapse
"refused correctly" and "fell over" into one outcome, and those are exactly what this corpus exists
to tell apart.

A refusal a runner cannot name prints an empty identifier, matches no expectation, and fails with a
readable diff. **A refusal a library cannot name is one the suite cannot check, and it should look
like a gap rather than a pass.**

A crash, an unnamed error, extra stdout, or a different refusal id all fail the case.

### Canonical output

Geometry output rounds the exact decoded binary64 rational to six decimal places with ties-to-even
semantics. It must **not** first multiply in binary64, because an intermediate rounding can move a
value onto the wrong side of a decimal midpoint.

Non-finite components are spelled `"NaN"`, `"Infinity"` and `"-Infinity"` — JSON has no NaN, and the
languages disagreed on the bare form.

## The writer gate is separate

Readers and writers make different claims, so passing `run.py` says nothing about a writer.
`encode_roundtrip.py` invokes a language writer for all 45 valid variants and has the Python
reference decode its output:

```bash
python3 tests/conformance/encode_roundtrip.py \
  --encoder "dart run dart/conformance/bin/encode_roundtrip.dart"
```

**A self-round-trip does not prove interoperability.** The candidate receives a decoded typed
request — values and application metadata, with the channel and output paths appended to the command
— never an already-valid ALF wire image or a sibling fixture it could copy as its output.

Exact output bytes may differ; canonical decoded meaning may not. Gzip and JSON serialization can
vary without changing the format, so byte identity is not required. What _is_ checked:

- **Writer-owned canonical fields are inspected in the raw envelope**, before a compatibility reader
  can normalize them away — hash-free palette entries, for instance.
- **Every v3 candidate must carry a finite, non-negative writer-measured `_geom_max_err_m`** in its
  raw header, and that raw value must cover both the independently measured reconstruction error and
  the upward-rounded positive half-scale floor for a non-empty fixed-point payload.
- **Decoded fp16 geometry is compared exactly**, before the six-decimal canonical projection can hide
  a changed subnormal component.
- **Application metadata is compared after portable binary64 normalization**, excluding only the
  fields the writer explicitly regenerates from its typed payload or API arguments. `.alp` anchor
  arrays and x/y/z objects are the same metadata after their documented shape normalization; the same
  member names on a non-geometry channel remain ordinary metadata whose structure must survive
  unchanged.

An interoperability gate may allow only an ULP-scaled arithmetic allowance when comparing the error
bound. A fixed absolute tolerance would erase the complete bound for sufficiently small valid grids.

## The harness does not trust the candidate

Every oracle, source meaning and decoded request is snapshotted **before** candidate code runs. The
encoder receives an isolated read-only decoded request rather than a corpus path, so candidate code
cannot choose or rewrite the evidence used to score it, and mutating a committed expectation is
itself a gate failure.

Each runner or encoder starts in a **private process session**, and the harness terminates the
entire group before inspecting output — so forked descendants cannot retain diagnostic pipes or keep
changing a candidate after their parent has exited.

## Adding a rule

A rule with no conformance case is a rule only one language enforces.

1. Implement it in `python/alf/_io.py`.
2. Add a case to `tests/conformance/generator/invalid.py` with a **new refusal identifier** and a
   `why` that says what goes wrong if a reader lets it through.
3. Map the identifier in `tests/conformance/runners/python_runner.py`.
4. Regenerate: `python3 tests/conformance/generate.py --update`.

Then port it, and prove the port with `run.py --runner "<their runner>"`.

## What this suite is not

These are **format-boundary interoperability tests**. They are not product end-to-end tests: they do
not load an application's frame sequence, execute a label merge, or pass coordinates through an
application's WebAssembly boundary.

There are also properties the corpus **cannot** express, and forcing a case for one is a mistake.
Header-only reading is the example: the corpus is bytes-on-disk evidence keyed by a refusal id, and
"did the reader avoid inflating the payload?" is a property of an API call, not of a byte sequence —
the same file read either way yields the same header. Its evidence is the Python suite asserting the
resource property directly, with a source the full reader cannot read at all plus an instrumented
stream counting bytes consumed.
