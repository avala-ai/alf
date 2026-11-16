# Contributing

Thanks for helping build ALF. This guide covers the development setup, the checks CI runs, and the
processes that keep the format portable. Repository conventions for automated agents — which apply
to humans too — live in [AGENTS.md](AGENTS.md).

## Development setup

Python 3.11 or newer is the only hard requirement; the reference implementation and the shared
conformance tooling are Python.

```bash
pip install -e ".[dev]"    # alf, pytest, and the pinned ruff
```

Node runs the built ESM and WebAssembly runtime checks and the prose gates (fetched on demand by `npx`). Bun, Dart and Rust are needed
only when you work on those SDKs — [each language directory's README](README.md#repository-layout)
lists its own commands.

## Before a pull request

```bash
python3 -m pytest tests/                        # the implementation
ruff check .                                    # the lint gate, same as CI
ruff format --check .                           # the formatting gate
python3 tests/conformance/generate.py --verify  # corpus determinism gate
python3 tests/conformance/run.py                # cross-language contract
```

If you touched any Markdown — the specification and the site both are — run the two prose gates with
the same pinned versions CI uses (exact versions, not `@3` and `@8`: those are ranges, and a tool
that floats turns CI red without anyone changing a line):

```bash
npx --yes prettier@3.9.6 --check "**/*.md" ".github/**/*.md"
npx --yes cspell@8.19.4 --no-progress --config cspell.config.yaml "**/*.md" ".github/**/*.md"
```

Add `--write` to the prettier command to fix formatting in place. `.github/**/*.md` is listed
separately because a leading-dot directory is not matched by `**`, which is how the issue and
pull-request templates once went unchecked.

## A new validation rule needs a conformance case

Implementing a refusal in Python only makes it a Python rule. The shared suite is what makes it
portable — see [AGENTS.md](AGENTS.md) for the four steps: implement, add an invalid-corpus case with
a new refusal identifier, map the identifier in the Python runner, regenerate with `--update`.

## Adding a language SDK

1. Put the publishable, transport-neutral package under `<language>/`; keep its file/CLI runner in a
   separate private conformance package.
2. Implement readers and writers against [`spec/format.md`](spec/format.md), with stable refusal
   identifiers from the shared invalid corpus.
3. Pass the shared reader corpus and have the Python reference decode that language's writer output.
   A self-round-trip is necessary but insufficient.
4. Update the support matrix from what those gates prove. Use “untested” or a warning state for code
   that exists without cross-language evidence.

Keep one language per pull request. When language packages depend on shared corpus changes, stack
them above the corpus/spec layer and merge bottom-up.

## Regression tests must be seen to fail

Revert the fix, watch the test go red, restore it. A test that has never failed is a comment, and
this repository has a documented history of tests that asserted things already true by construction.

## Scope

Use synthetic examples and fixtures in contributions. Keep customer identities,
production measurements, private repository paths, and deployment inventories
out of source, documentation, commit messages, and public discussions. A bug
report should describe reproducible behavior without copying an internal
incident report.

**In scope:** the format — `spec/`, the conformance suite, and reader/writer SDKs under `python/`,
`dart/`, `typescript/` and `rust/` — and the documentation of those, which is `website/docs/` and
the specification it serves.

**Out of scope:** viewers, renderers, annotation tooling and pipeline glue live with their
consumers. So does anything that would only make sense to one of them: if a change reads as obvious
to exactly one caller and arbitrary to the rest, it belongs in that caller. The line is not about
size — a decoder is in scope and a 200-line one is welcome; a three-line convenience that encodes
one application's assumption about what a frame means is not.

## What CI exercises

A pull request is checked on every one of these, and the list is worth knowing before you add a
dependency:

| Leg                                                      | What it pins                                                                                       |
| -------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| Python 3.11, 3.12, 3.13, 3.14                            | The declared floor, and the newest release                                                         |
| Bun 1.2.14 and 1.3.11, Node 22                           | TypeScript source tests and runtime checks through the built ESM package                           |
| Dart 3.9.0 and 3.13.3 on VM, JavaScript, and WebAssembly | The same synthetic v3 scenarios execute on all three runtimes                                      |
| Rust 1.98.0, plus `--target wasm32-unknown-unknown`      | A dependency that cannot cross-compile is a break found here rather than in a consumer             |
| `ruff`, `prettier`, `cspell`                             | Code and prose                                                                                     |
| The corpus determinism gate                              | Two generator runs must be byte-identical                                                          |
| Writer/reader interoperability                           | Every Python, Dart, and TypeScript writer variant is decoded by Python, Dart, TypeScript, and Rust |

`import alf` is separately checked in a clean environment to prove it does not pull NumPy: answering
"what channel is this file?" should not cost a caller an array library. Each registry artifact is
also assembled and inspected on every pull request, so a release tag tests credentials and
publication rather than discovering that a package cannot be built.

## Code of conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). Report unacceptable behaviour
to conduct@avala.ai.
