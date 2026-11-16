# Governance

This is an early-stage project and its governance is deliberately minimal. This document will grow
as the project does; it is a stub, not a constitution.

## Decisions

- **Specification changes** — anything under `spec/` that changes what a conforming reader or writer
  must do — require an issue, a pull request, two approvals from maintainers, and a conformance case
  demonstrating the change. A rule with no conformance case is a rule only one language enforces.
- **SDK changes** follow ordinary review: one approval, conformance green.
- **Version registry additions** (a new wire version byte, a new `_geom_codec`) follow the
  specification path, and additionally require the adoption question to be answered before anything
  writes the new value. The format has no skip-unknown mechanism, so an unrecognised version byte is
  fatal by design — see [`spec/adoption.md`](spec/adoption.md).
- **Changes to the project's charter** — the scope sentence in the README and CONTRIBUTING — require
  the same process as specification changes.

## What cannot change

The channel suffixes (`.alp`, `.ali`, `.alc`, `.all`, `.alx`) are load-bearing in stored data:
readers infer the channel from the suffix and stored objects are named by it. They are not subject
to the process above, because they are not subject to change.

## The support matrix is not a decision

`spec/support.md` records what the shared suite proves. Moving a cell is a consequence of evidence
landing, never a thing to argue for. A pull request that changes a cell without changing what runs
is the one shape of change this project rejects on sight.

## Maintainers

Maintained by Avala AI. A formal maintainer list and a path to maintainership will be added when
there are contributors outside the founding organization.
