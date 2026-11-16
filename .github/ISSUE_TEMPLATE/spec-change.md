---
name: Specification change
about: Propose a change to what a conforming reader or writer must do
labels: spec
---

## What the specification says today

<!-- Quote the rule, with its section from spec/format.md. -->

## What it should say

## What goes wrong under the current rule

The failure mode is the argument. "Reject malformed input" is not a specification — what makes a
rule worth having is what a reader that lets it through returns, and whether anything downstream can
tell.

State it concretely: which channel, what a consumer receives, and why no later check catches it.

## Does this change the interpretation of any existing file?

- [ ] No — no conforming writer has ever emitted this wire pattern
- [ ] Yes, and here is the migration

The format has **no skip-unknown mechanism**. An unrecognised version byte is fatal by design, so a
change that needs a new version cannot be written until every reader on a file's path can read it.
If that applies, say which consumers are affected — see [`spec/adoption.md`](../../spec/adoption.md).

## Conformance

- [ ] A new refusal identifier, if this adds a refusal
- [ ] A case that breaks **exactly one** rule, with a `why`
- [ ] The case fails against `main` and passes with the change

## Affected implementations

<!-- Python, Dart, TypeScript, Rust. A rule with no conformance case is a rule only one language
     enforces, so a change landing in one language alone is not this proposal being done. -->
