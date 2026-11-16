# Working in this repository

## What this is

ALF — Arrays in Local Frames. A container format: its specification, its
conformance suite, and the reference implementation.

Viewers, renderers, annotation tooling and pipeline glue are **out of scope**.
If a change would only make sense to one consumer, it belongs in that consumer.

## Public repository content

Use synthetic examples and fixtures. Keep customer identities, production
measurements, private repository paths, and deployment inventories in the
consuming repository. This applies to source comments, commit messages, issues,
pull requests, and review comments as well as documentation. Describe a bug's
reproducible behavior without copying internal incident reports.

## Dependencies

`import alf` must not load NumPy. The codecs need it; answering "what channel is
this file?" or "what does the format say?" does not, and a caller that only
needs the latter should not pay for the former. Public names are re-exported
lazily in `python/alf/__init__.py`; a test enforces it.

`_guard.py` uses `math` alone for the same reason.

## The rule that generates most of the code

**Channels are index-correlated.** Entry _i_ of every channel describes point
_i_. A length mismatch does not corrupt one file — it silently misattributes
every label and intensity reading after the gap, and the file stays perfectly
readable while it does.

That is why this module fails closed. Where guessing would return
plausible-but-wrong data, it raises.

## Adding a validation rule

1. Implement it in `python/alf/_io.py`.
2. Add a case to `tests/conformance/generator/invalid.py` with a **new refusal
   identifier** and a `why` that says what goes wrong if a reader lets it
   through.
3. Map the identifier in `tests/conformance/runners/python_runner.py`.
4. Regenerate: `python3 tests/conformance/generate.py --update`.

The bare generator command materializes ignored binary fixtures without
rewriting the committed expectations or checksums. Use `--update` whenever a
contract case changes the oracle.

A rule with no conformance case is a rule only one language enforces.

## Tests

```bash
python3 -m pytest tests/                        # the implementation
ruff check .                                    # the lint gate, same as CI
ruff format --check .                           # the formatting gate
python3 tests/conformance/generate.py --verify  # corpus determinism gate
python3 tests/conformance/run.py                # cross-language contract
```

Python 3.11 is the floor. Ruff replaced black, isort and flake8, which were
declared as dev dependencies and never actually run; its rules and line length
live in the root `ruff.toml` so the commands above are the ones CI runs.
`ruff format` excludes `.md`, deliberately: it reformats Python blocks inside
Markdown, and the guides align trailing comments into a column a reader scans
down. Prettier owns the prose.

### Verify a regression test actually fails

A test that has never been seen to fail is a comment. Before committing one,
revert the fix and confirm the test goes red. Several tests in this suite's
history asserted things that were already true by construction — including one
that compared a header field against itself for twenty review rounds.

Fixtures need the same scrutiny: coordinates like `1.5` or `100.0` are exactly
representable in float16, so a codec comparison built from them passes
trivially and proves nothing.

## Reader entry points that are not wire features

`read_alf_header` reads the envelope and stops — it never fetches or inflates
the payload. TypeScript's `readAlfHeader` is the same contract, backed by
`gunzipPrefix`. Two consequences worth knowing before touching it or adding
something like it:

**Its refusals are a strict subset, and the split is the contract.** Every
HEADER-scoped rule still applies, because it is decidable from the envelope
alone; both readers get them from the one shared `_parse_envelope`. The
PAYLOAD-scoped rules cannot apply, because their evidence is the bytes it
declined to read: `_check_payload_length`'s count-vs-payload cross-check, the
concatenated-gzip-member refusal, the whole-frame decompression cap, and the
`fstat`-snapshot length equality that catches a file growing or shrinking
mid-read. That is why it is a separate function rather than a
`read_alf(..., header_only=True)` flag — a boolean lets the weaker contract be
selected by a variable at a call site that reads like the strong one.

**A missing payload must never be modelled as an empty one.** The substitute is
`_WithheldPayload`, which raises from `__len__`/`__bool__`/`__getitem__`. `b""`
would make `.intensity` return an empty array and `format_version`'s stride
disambiguation "measure" zero bytes and pick a layout — wrong ANSWERS, which is
the one failure mode this repo does not accept. The explicit `_require_payload`
guards sit on top only for wording and because `np.frombuffer` reaches the C
buffer protocol, which Python cannot intercept before 3.12.

**No conformance case is expressible for it.** The corpus is bytes-on-disk
evidence keyed by a refusal id (`spec/format.md` §10). "Did the reader avoid
inflating the payload?" is a property of an API call, not of a byte sequence —
the same file read both ways yields the same header. Its evidence is therefore
the Python suite asserting the resource property directly (a source the full
reader cannot read at all, plus an instrumented stream counting bytes). Do not
force a corpus case for a feature the corpus cannot observe.

**TypeScript `gunzipPrefix` must not call `firstGzipMemberEnd`.** That helper
Huffman-decodes every DEFLATE symbol to find the trailer — the payload cost
this path exists to avoid, paid twice when a header outruns the probe.
`gunzip` still needs the walk (CRC/ISIZE, concatenated-member refusal). Prefix
inflate feeds pako until it has enough output or the member ends. The
TypeScript suite asserts this with `gzipMemberBoundaryWalkCount()`.

## Wire format changes

The extensions (`.alp`, `.ali`, `.alc`, `.all`, `.alx`) are load-bearing in
stored data — readers infer the channel from the suffix and every object in S3
is named by it. They do not change.

The format has **no skip-unknown mechanism**: an unrecognised version byte is
fatal by design, so adding a version means every reader must be updated before
anything writes it. That is why v3 is opt-in and why `spec/adoption.md` exists.
