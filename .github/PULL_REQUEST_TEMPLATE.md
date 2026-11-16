## What this changes

<!-- One or two sentences. -->

## Scope

- [ ] This change is about the format: the specification, SDK read/write, the conformance suite, or
      documentation of those.

Viewers, renderers, annotation tooling and pipeline glue live with their consumers — see
[CONTRIBUTING.md](../CONTRIBUTING.md).

## Checks

- [ ] `python3 -m pytest tests/` passes
- [ ] `python3 tests/conformance/generate.py --verify` passes
- [ ] `python3 tests/conformance/run.py` passes, plus the runner for any language this touches
- [ ] If this touches a writer, `encode_roundtrip.py` passes for it — a self-round-trip is not
      evidence of interoperability
- [ ] If this adds a validation rule, it has a conformance case with a **new refusal identifier**
      and a `why`, and the identifier is mapped in every language's runner
- [ ] If this changes the corpus, it was regenerated with `--update` and the checksums are committed
- [ ] `spec/support.md` still reflects what runs

## If this is a regression test

- [ ] I reverted the fix and watched the test go red

A test that has never been seen to fail is a comment, and this repository has a documented history
of tests that asserted things already true by construction.
