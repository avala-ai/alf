---
name: New language SDK
about: Propose or track an ALF implementation in a language this repository does not cover
labels: sdk
---

## Language and runtime

<!-- And what consumes it: an SDK exists to be used by something. -->

## Reader, writer, or both

The two make different claims and earn different gates. The Rust crate is a reader, so it earns the
reader conformance gate and is `n/a` for the writer one rather than ❌ on a gate it never entered.

- [ ] Reader
- [ ] Writer

## The four steps

From [CONTRIBUTING.md](../../CONTRIBUTING.md):

- [ ] The publishable, transport-neutral package lives under `<language>/`, with its file/CLI runner
      in a **separate private conformance package**
- [ ] Readers and writers implemented against [`spec/format.md`](../../spec/format.md), with the
      stable refusal identifiers from the shared invalid corpus — a language does not invent one
- [ ] `python3 tests/conformance/run.py --runner "<their runner>"` passes 117/117
- [ ] For a writer: `python3 tests/conformance/encode_roundtrip.py --encoder "<theirs>"` passes 45/45.
      A self-round-trip is necessary but insufficient
- [ ] `spec/support.md` updated from what those gates prove, using "untested" for code that exists
      without cross-language evidence

## Runner contract

- [ ] A refusal prints `{"refused": "<id>"}` on stdout and **exits 0**
- [ ] A refusal the library cannot name prints an empty identifier rather than being mapped to a
      nearby one — it should look like a gap, not a pass

## Portability

<!-- What CI has to install, and which targets are load-bearing. The Rust crate exists partly
     because it must build for wasm32-unknown-unknown; the Dart package must compile for web and
     Wasm, not only the VM. If your language has an equivalent constraint, say so now — it decides
     which dependencies are admissible. -->

## Scope check

- [ ] This is a reader/writer for the format, not a viewer, renderer, or pipeline integration
