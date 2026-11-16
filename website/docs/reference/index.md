# Reference

The specification says what a conforming file is. This section is the material around it: what
proves the claims, what every refusal identifier means, and what every reserved header field is.

- **[Conformance suite](./conformance.md)** — how the shared corpus is generated, run and scored,
  and why a refusal is a result rather than a crash
- **[Refusals](./refusals.md)** — the catalogue of stable refusal identifiers, what each one means,
  and what goes wrong if a reader lets it through
- **[Header fields](./header-fields.md)** — every reserved `_`-prefixed field and every recognised
  anchor, with the numeric rule that applies to it

## What is proven, and where that is recorded

Support is recorded **per feature and per boundary**, not per language. A check means the shared
suite proves the claim; code existing without that evidence is marked unverified.

That matrix lives in the specification, at [`spec/support.md`](/spec/support), and there is exactly
one copy of it on purpose — a support table that exists in two places is a support table that is
wrong in one of them.

Two distinctions it makes that are worth understanding before reading it:

**An implementation parses the format**, so it owes the shared conformance suite a passing run.
All four format-owned SDKs — Python, Dart, TypeScript and Rust — are through that gate at 117/117.

**A consumer reads ALF files but delegates the parsing**, so running the suite against it would only
re-test the implementation underneath. What a consumer owes is proof that it _delegates_: the check
that fits is the absence of a local envelope parser, not a conformance run. A consumer that can pass
the suite still has the parser you were trying to remove.

**Readers and writers make different claims.** The Rust crate is a reader, so it earns the reader
gate and is `n/a` for the writer one rather than failing a gate it never entered.

## Adoption

[`spec/adoption.md`](/spec/adoption) tracks what would have to be true before v3 fixed-point
geometry becomes the default. The short version: the format has no skip-unknown mechanism, so v3
cannot be written until every consumer of a file can read it. Application readiness must be
verified separately from SDK conformance.

It also documents the trap that makes this hard to verify — a compile-fail gate proves nothing
across a **type-erased boundary**. A WebAssembly consumer needs a source audit and runtime
integration evidence in addition to a successful build.
