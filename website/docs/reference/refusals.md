# Refusals

Every ALF refusal carries a **stable identifier**. It is the same string in Python, Dart,
TypeScript and Rust, and it names _which rule_ was broken rather than merely reporting that
something went wrong.

Identifiers are declared once, in
[`tests/conformance/generator/invalid.py`](https://github.com/avala-ai/alf/blob/main/tests/conformance/generator/invalid.py).
A language does not invent one. Each identifier below is backed by one or more corpus cases that
break **exactly one** rule, and the "what goes wrong" column is the reason the rule exists — not a
restatement of it.

Where to find the identifier at runtime:

| Language   | Where                                 |
| ---------- | ------------------------------------- |
| Python     | in the `AlfError` message             |
| TypeScript | `error.code`, typed as `AlfErrorCode` |
| Dart       | `error.refusal` on `AlfException`     |
| Rust       | `err.id`, carried in the error itself |

The invalid corpus is **70 cases across 41 identifiers**. Several identifiers have more than one
case because the same rule is reachable by more than one route, and a reader that catches one route
and misses another is not conforming.

## Envelope and header

| Identifier                | What goes wrong if a reader accepts it                                                                                                                                                                                                                               |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `header-truncated`        | A declared header longer than the file is how these objects go bad mid-transfer.                                                                                                                                                                                     |
| `header-too-large`        | A complete valid-JSON header above the 10,000,000-byte ceiling must be refused before parsers allocate another hostile copy.                                                                                                                                         |
| `header-not-utf8`         | A reader that lets a `UnicodeDecodeError` escape aborts a pipeline run that would otherwise quarantine the frame.                                                                                                                                                    |
| `header-not-json`         | Malformed JSON syntax has no portable metadata interpretation. Also covers any metadata number outside finite binary64 — an arbitrary-precision integer or a non-finite value becomes infinity in common decoders and silently poisons unrelated metadata consumers. |
| `header-not-an-object`    | Every accessor does a keyed lookup, so a list or scalar root fails later and further from the cause.                                                                                                                                                                 |
| `header-key-duplicate`    | JSON parsers disagree on which duplicate object member wins; accepting one makes the same ALF bytes declare different metadata across languages.                                                                                                                     |
| `header-nesting-too-deep` | Unbounded recursive JSON decoding can overflow a reader stack before malformed input reaches its ALF quarantine path.                                                                                                                                                |

`header-not-json` carries the most cases of any identifier after `geom-number-inexact`, because
it is the general rule that specific ones are carved out of. Order matters: `1e309` under
`device_position` is `geom-number-inexact`, the same token under `_total_points` is
`total-points-not-an-integer`, and the same token anywhere else is `header-not-json`. The specific
checks run first so the general one cannot claim a case that belongs to it.

Two subtle members of that family are worth naming:

- **Numbers beyond an anchor object's x/y/z axes** remain ordinary application metadata and must
  still fit the shared finite-binary64 domain.
- **Version-1 geometry keys are stale format metadata**, not active v3 codec parameters, so their
  numbers belong to the generic domain rather than the exact-spelling geometry rule.

## Version and channel

| Identifier                                 | What goes wrong if a reader accepts it                                                                                                          |
| ------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `version-unsupported`                      | A version byte nobody has defined cannot be guessed at; the payload layout is exactly what it selects.                                          |
| `label-version-on-non-label-channel`       | Wire version 2 selects the interleaved `.all` label layout; accepting it on another channel reinterprets label bytes as unrelated measurements. |
| `geometry-version-on-non-geometry-channel` | v3 is the `.alp` geometry codec. On `.all` it would fall through to the v1 accessors and hand back coordinates as category bytes.               |

## Counts and payload length

| Identifier                    | What goes wrong if a reader accepts it                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `total-points-not-an-integer` | `int()` truncates `1.9` to `1`, and a one-point payload then satisfies the very cross-check that guards against length mismatch. Also covers a present `null`, a value outside finite binary64, and hostile exponent spellings — a compact exponent must be bounded before a reader expands it into a hundred-million-digit integer, and a negative exponent larger than the significand's available trailing zeros must be refused before a reader constructs a hostile power-of-ten divisor. |
| `payload-length-mismatch`     | Channels are correlated by index, so a count that disagrees with the payload misattributes every value after the gap.                                                                                                                                                                                                                                                                                                                                                                          |
| `payload-not-whole-points`    | A payload that is not a whole number of XYZ triples is truncated; decoding the prefix silently drops the rest of the frame.                                                                                                                                                                                                                                                                                                                                                                    |
| `total-points-missing`        | `.alx` only. Without the kept-point count, the sidecar cannot be proven to contain one source mapping per decimated geometry point.                                                                                                                                                                                                                                                                                                                                                            |
| `source-total-points-missing` | `.alx` only. Without the original cloud size, a sidecar index cannot be proven to address a real source point.                                                                                                                                                                                                                                                                                                                                                                                 |

## Geometry — `.alp`

| Identifier                   | What goes wrong if a reader accepts it                                                                                                                                                                                                                                        |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `codec-unsupported`          | An unknown geometry codec has no defined payload interpretation; treating it as fp16 produces plausible but invented coordinates.                                                                                                                                             |
| `codec-version-disagreement` | A writer that set the header key and forgot the wire byte — or the mirror image. Resolving the disagreement by picking a side decodes to plausible garbage either way.                                                                                                        |
| `geom-scale-missing`         | Without the step, every coordinate decodes to the wrong magnitude, and nothing about the payload reveals it.                                                                                                                                                                  |
| `geom-scale-not-positive`    | A zero or negative step is not a grid. Zero collapses the frame to its offset; negative mirrors it.                                                                                                                                                                           |
| `geom-offset-malformed`      | A scalar or 2-vector offset broadcasts across axes and yields a uniformly shifted cloud rather than an error.                                                                                                                                                                 |
| `geometry-anchor-malformed`  | A geometry anchor without all three axes cannot place the frame, and must be refused during the envelope read rather than only when an accessor is called.                                                                                                                    |
| `geom-number-inexact`        | A geometry number whose stated decimal value changes during binary64 decoding shifts **the whole coordinate grid**, not one sample.                                                                                                                                           |
| `geom-partial-no-return`     | A point with some but not all words equal to the reserved marker `-32768` cannot come from a conforming writer, which clamps real coordinates to ±32767. Reconstructing it arithmetically yields a plausible position that no later check can distinguish from measured data. |

`geom-number-inexact` has eleven corpus cases, more than any other identifier, because a decimal
token can lie about its value in a surprising number of ways. Each case is a distinct route:

- an integer offset outside binary64's exact range;
- a decimal offset whose stated precision is lost in binary64;
- an integer token that equals binary64's shortest decimal spelling while naming a different exact
  integer;
- normalized-_looking_ scientific notation that still states a different integer;
- an expanded integral decimal, which states an exact mathematical integer;
- the same three failures on `_geom_scale`, where the consequence is a different step rather than a
  different origin;
- a `device_position` component outside binary64's exact integer range, which shifts every local
  point in the frame;
- a non-finite decoded anchor axis, which cannot place a frame at all;
- a `world_centroid_epsg3857` whose stated precision disappears, which misplaces the complete aerial
  frame.

What stays **valid** is just as load-bearing: `0.1`, large values whose shortest representation is
`1e+308`, exactly representable wide integers such as 2⁶⁴ in any spelling, and a writer's normalized
scientific token when it is the decoded value's shortest round-trip decimal. See
[Precision and the two codecs](../guides/precision.md#the-precision-rule-on-the-numbers-themselves).

## The v3 frame convention — `.alp`

A `.alp`'s points are the world points with an origin subtracted, which is **half** of a rigid
transform. Producers disagreed about the other half — some translated and stopped, some also
rotated into the device frame — and the two agree exactly whenever the device heading is identity,
so the mismatch stayed invisible until a vehicle turned. Version 3 refuses to be ambiguous; see
[the frame convention](../guides/precision.md#the-frame-convention).

| Identifier                           | What goes wrong if a reader accepts it                                                                                                                                                                        |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `frame-convention-missing`           | A v3 frame that does not say whether its points were rotated into the device frame cannot be placed in the world; reading it as either convention is right for one producer and silently wrong for the other. |
| `frame-convention-unsupported`       | A convention nobody has defined cannot be guessed at, and it is the field that says where these points are.                                                                                                   |
| `frame-convention-on-legacy-version` | Readers apply no rotation to v1/v2 geometry, so a v1 file declaring `device-aligned` states a promise the format does not keep — the producer believes a rotation will be undone and it will not be.          |
| `frame-rotation-missing`             | A device-aligned frame without its rotation cannot be returned to the world at all, and falling back to the world-aligned reading is the exact error the declaration exists to prevent.                       |
| `frame-rotation-malformed`           | A quaternion missing a component is not a rotation; defaulting the absent member to zero would produce a confident rotation the producer never described.                                                     |
| `frame-rotation-not-normalized`      | Conjugation by a non-unit quaternion scales by \|q\|² as well as rotating, so every coordinate in the frame moves by a factor nothing else in the file disagrees with.                                        |
| `frame-rotation-unexpected`          | `world-aligned` declares that no rotation was applied, so a rotation beside it means one of the two statements is wrong — and the reading that drops it returns plausible, wrongly-oriented geometry.         |

Two design choices here are themselves refusal-shaped. `_frame_rotation` is an **object** with named
members, because every array spelling must choose between `[x, y, z, w]` and `[w, x, y, z]` and the
two are indistinguishable in the bytes — a reader and writer that disagree produce a rotation no
length check catches. And exactly four members are allowed, because a stray `qw` beside `w` is a
producer mixing two spellings rather than application data, and only one of them would be read.

`frame-rotation-not-normalized` is refused rather than repaired on purpose: **normalization belongs
in the writer**, which knows what the caller meant. A reader that normalized would hide a producer's
bug instead of fixing it.

## Labels — `.all`

| Identifier                 | What goes wrong if a reader accepts it                                                                                                                                                                                                         |
| -------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `label-layout-unsupported` | Treating an unknown layout as v1 returns the interleaved stream as categories and silently invents instance ids. A negative integral token is syntactically an integer but names no supported layout.                                          |
| `label-layout-ambiguous`   | Without a declared count, a three-byte payload fits both the semantic and panoptic layouts; choosing either silently reinterprets labels. An empty counted payload fits both strides, so it cannot resolve disagreeing version signals either. |
| `instance-id-reserved`     | 65535 is reserved; the editor's registry caps valid local ids at 65534, so this file's instance identity cannot be resolved.                                                                                                                   |
| `instance-map-incomplete`  | A map that _claims_ to record identity and misses emitted ids exports objects with a null uuid and reports success.                                                                                                                            |
| `instance-map-malformed`   | Resolvers look up the canonical decimal spelling of each uint16 id, so a leading-zero key names no emitted instance.                                                                                                                           |

`label-layout-ambiguous` is the one place the format lets a reader resolve a disagreement between
two signals — and only when objective payload evidence leaves exactly one interpretation. A declared
`_total_points` can establish the stride. A payload length not divisible by three rules out layout 2. A length divisible by three fits both, and is refused.

## Colour — `.alc`

| Identifier                   | What goes wrong if a reader accepts it                                                                                  |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `palette-entry-malformed`    | `"zzzzzz"` passes a length check and then range-checks indexes against a palette no viewer can parse.                   |
| `palette-index-out-of-range` | An index past the palette end renders as an arbitrary colour rather than failing, so the mis-colouring looks like data. |

Index `palette.length` is the **no-colour sentinel** and is valid — it is what a writer emits for
points that matched no camera or were out of view. Only indexes beyond it are refused.

## Decimation — `.alx`

| Identifier                | What goes wrong if a reader accepts it                                                                          |
| ------------------------- | --------------------------------------------------------------------------------------------------------------- |
| `kept-index-duplicate`    | A repeated source index maps two labelled points to the same raw point while another kept point has no mapping. |
| `kept-index-out-of-range` | An index at or beyond the source count makes export address data outside the authoritative cloud.               |

## Transport

| Identifier              | What goes wrong if a reader accepts it                                                                                                                                                                                                                            |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `gzip-malformed`        | A trailer CRC mismatch proves the decompressed ALF bytes are corrupt even when the deflate stream itself decodes. A member carrying `FHCRC` explicitly authenticates its gzip header, so accepting a mismatched checksum treats marked corruption as valid input. |
| `multiple-gzip-members` | Concatenated members decode into one byte run, so the second file's envelope becomes the first one's payload.                                                                                                                                                     |

## Identifiers outside the corpus

Some refusals are real but are not corpus cases, because the corpus is bytes-on-disk evidence and
these are not properties of a byte sequence. They are still stable within their implementation:

| Identifier                             | Meaning                                                                                                                                                                         |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `coordinate-non-finite`                | A non-finite v3 decode **outside** a sentinel row — which can only mean `_geom_scale`/`_geom_offset` overflowed binary64.                                                       |
| `input-too-large`                      | The input, or a materialized typed result, exceeded an implementation memory ceiling. Bounds memory, not correctness.                                                           |
| `channel-required` / `channel-unknown` | The channel was not supplied, or is not one of the five. The envelope does not carry it, and a reader must not infer it from payload shape.                                     |
| `gzip-invalid`                         | A gzip framing fault distinct from a corrupt-but-well-framed member.                                                                                                            |
| `writer-input-invalid`                 | A writer was handed values it cannot encode conformantly. Writers enforce the rules readers enforce, so a successful writer never emits a header its paired reader must refuse. |

An identifier a library cannot name prints an empty string in the conformance runner, matches no
expectation, and fails with a readable diff — because a refusal the suite cannot check should look
like a gap rather than a pass.

## What is _not_ a refusal

Two things are explicitly valid, and refusing them is itself a conformance failure:

- **Non-finite float16 `.alp` geometry.** A NaN component means invalid return. It is the normal
  shape of production data. See [Invalid returns](../guides/invalid-returns.md).
- **A legacy `device_position: {}`.** The empty form means the zero origin `[0, 0, 0]`. It is the
  only empty-or-partial anchor object the format accepts, and writers preserve it when it is
  inherited from metadata.
