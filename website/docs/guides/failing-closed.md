# Why it fails closed

Where guessing would return plausible-but-wrong data, ALF raises.

That is the whole design stance, and it is why the format exists as a small number of shared
implementations rather than the two dozen hand-rolled parsers it replaced. Three languages
disagreeing about which malformed files to accept is worse than any one of their answers.

## The failure mode being defended against

Channels are index-correlated. A length mismatch does not corrupt one file — it silently
misattributes every label and intensity reading after the gap, **and the file stays perfectly
readable while it does**. Nothing downstream throws. A segmentation review looks normal. The
annotation is wrong by one row for the rest of the frame.

Almost every refusal in the specification traces back to a failure with that shape: it produces an
answer, the answer is wrong, and no later stage can tell. Compare:

| Guessing would give you                                                 | Refusing gives you       |
| ----------------------------------------------------------------------- | ------------------------ |
| A `.all` whose stride was chosen from a length that fits both layouts   | `label-layout-ambiguous` |
| A geometry grid shifted because `_geom_offset` lost precision on decode | `geom-number-inexact`    |
| A plausible coordinate reconstructed from a reserved sentinel word      | `geom-partial-no-return` |
| A header whose duplicate keys resolved differently in each language     | `header-key-duplicate`   |
| A second gzip member's envelope silently appended as payload            | `multiple-gzip-members`  |

None of those are exotic. Each has a case in the invalid corpus because each was reachable.

## Refusals are identified, not just raised

Every refusal carries a **stable identifier**, and it is the same string in every language:

```json
{ "refused": "codec-version-disagreement" }
```

"Both readers raised an error" is not agreement. `AlfError` covers a hundred distinct faults, so a
negative test that only checks that _something_ was raised passes when a reader refuses for the
**wrong reason** — which is exactly the failure a negative test exists to catch. The identifier says
_which rule_.

Identifiers are declared once, in the corpus generator. A language does not invent one. The full
list is the [refusal catalogue](../reference/refusals.md).

In code they surface as:

```python
try:
    alf = read_alf(path)
except AlfError as exc:
    ...            # Python: the message names the rule
```

```ts
try {
  await readAlf(bytes, { channel: "alp" });
} catch (error) {
  if (error instanceof AlfError) console.error(error.code); // e.g. 'geom-scale-not-positive'
}
```

## A refusal is a result, not a crash

This is the contract the conformance runners are held to, and it is a real distinction:

- The runner prints `{"refused": "<id>"}` on stdout and **exits 0**.
- The harness diffs it like any other answer.

Exiting non-zero would collapse "refused correctly" and "fell over" into one outcome, and telling
those apart is the entire point of the invalid corpus. A refusal a runner cannot name prints an
empty identifier, matches no expectation, and fails with a readable diff — because **a refusal a
library cannot name is one the suite cannot check, and it should look like a gap rather than a
pass.**

The same standard applies to malformed input inside a library. The Rust crate has a panic-safety
suite for precisely this: a malformed file must produce a named refusal, never a panic, because a
panic across the WASM boundary is not a refusal a caller can handle.

## Where refusals do not apply

Two things are explicitly **not** refusals, and conflating them with damage is itself a
conformance failure:

- **Non-finite float16 geometry.** NaN means invalid return, and it is the normal shape of
  production data. See [Invalid returns](./invalid-returns.md).
- **A legacy `device_position: {}`.** The empty form is valid and means the zero origin. It is the
  only empty-or-partial anchor object the format accepts, kept because deployed files contain it.

## Bounds are applied before allocation, not after

Failing closed is also a resource property. A reader that parses first and checks size afterwards
has already paid the cost the check was meant to avoid, so:

- the 10,000,000-byte header bound is applied **before** UTF-8 decoding or JSON parsing allocates
  another copy;
- the 256-level nesting depth is enforced **before** a recursive parser is invoked;
- exponent work is bounded **before** an integral token is expanded, so a compact `1e100000000`
  never allocates a hundred-million-digit intermediate, and a zero significand is short-circuited
  for any syntactically valid exponent;
- gzip decompression ceilings are enforced incrementally rather than on the finished buffer.

Writers enforce the same bounds they read against, so a successful writer never emits a header its
paired reader must refuse.

## And the rules are portable, or they are not rules

A refusal implemented in one language is that language's opinion. What makes it part of the format
is a case in the shared invalid corpus that breaks **exactly one** rule and names the refusal every
conforming reader must produce.

That is the workflow: implement it, add the corpus case with a new identifier and a `why` recording
what goes wrong if a reader lets it through, map the identifier in each runner, regenerate. A rule
with no conformance case is a rule only one language enforces — see the
[conformance suite](../reference/conformance.md).
