# Changelog

## 0.1.0 — 2026-11-16

First standalone release of ALF as its own repository.

### Added

- **`decodeSingleGzipMember` is public in Dart**, together with
  `gzipInputOverheadAllowance`. Byte-oriented consumers can reuse the strict,
  bounded single-member decoder without importing a private `src/` path or
  reimplementing gzip validation. An export only: the checks, the decoded
  budget and the stored bytes are unchanged.

- **`read_alf_header` in Rust**, which completes the header-only row of the
  support matrix. `read_alf_header_bytes` parses an envelope without inflating
  the payload; `read_alf_header` reads a bounded PREFIX of a file rather than
  the file, so scanning a directory of gigabyte frames costs kilobytes each. It
  returns an `AlfHeader`, so a payload accessor on a header-only read does not
  compile. `read_alf_bytes` and `read_alf_header_bytes` now share one
  `parse_header`.

- **`cargo test` runs in CI.** It never had: `clippy --all-targets` compiles the
  test targets without executing them, so `geometry.rs`'s unit tests had been
  green by assumption since they were written. Added to the tag-time gate too.

- **`readAlfHeader` in Dart**, matching TypeScript's and Python's entry points.
  It inflates about 64 KiB of a gzipped frame instead of the whole member, and
  returns an `AlfHeader` rather than an `AlfFile`, so a payload accessor on a
  header-only read does not compile. Backed by a new `decodeGzipPrefix`, which
  inflates a bounded prefix and stops — a different contract from
  `decodeSingleGzipMember`, which must reach the trailer before it can return
  anything. `readAlf` and `readAlfHeader` now share one `_parseEnvelope`, so
  they agree on every header-scoped rule by construction.

### Changed

- **The Dart package requires `archive` below 4.6.0.** From 4.6.0 that package
  buffers inflate output before a non-memory sink sees it, so the bounded
  prefix inflater would materialize the whole member before its stop signal
  ran. The range is `>=4.0.7 <4.6.0` until the inflater no longer depends on
  that behavior.

- **The Rust crate can be bootstrapped on crates.io from a reviewed commit.**
  Removing Cargo's `publish = false` kill switch does not publish anything; it
  makes the documented manual 0.1.0 bootstrap possible before trusted
  publishing is configured. A policy test keeps the manifest and that release
  procedure from drifting apart.

- **The Python distribution and Rust crate are named `avala-alf`.** The
  unscoped `alf` names on PyPI and crates.io belong to unrelated projects.
  Python still imports as `alf`, and Rust's library target is still `alf`, so
  this changes registry coordinates without renaming either source API.

- **The Dart lane tests the SDK floor as well as the newest release.** `pubspec`
  declares `sdk: ^3.9.0` and pub.dev enforces it, so a consumer on 3.9 can
  install this package — but CI ran only 3.13.1, leaving that promise
  unverified: a newer-only API would have shipped green while every consumer at
  the floor failed at `pub get`. Python and TypeScript already tested floor and
  newest; Dart was the one language that did not.

  `dart format` runs in BOTH legs, which is the opposite of the usual advice and
  deliberate. These sources are vendored into a monorepo that formats them with
  the FLOOR sdk while this repository formats them with the newest, and the two
  are known to disagree — 3.9.0 and 3.12.2 emit different trailing-comma styles,
  which left one file there impossible to format in any way that satisfied both gates.
  Requiring these files to be a fixed point of both formatters keeps that from
  arriving as somebody else's unsatisfiable CI. They agree on this tree today,
  measured under both.

### Fixed

- **Dart and Rust preserve counts across native and WebAssembly runtimes.**
  Dart count accessors return `num`, and Rust's header count is `Option<f64>`;
  neither clamps a valid wide count to a native integer. Payload length checks
  avoid multiplication overflow and 32-bit truncation. The shared corpus now
  covers counts beyond uint64, and CI executes the Rust count checks in Wasm.

- **Every registry artifact carries ALF's Apache notice.** Python already
  packaged the root `NOTICE`, but the Dart, Rust, and npm package roots are
  nested and their dry-run payloads omitted it. Each now carries the same text,
  npm's explicit files allowlist includes it, and a policy test prevents any
  package copy from drifting from the repository notice.

- **Every registry manifest identifies the public project it belongs to.** The
  npm manifest had no `repository.url`, so npm trusted publishing would have
  rejected the first TypeScript release after every build and conformance gate
  had passed. It now names the repository and monorepo package directory, and a
  regression test pins the exact OIDC identity. Python, Dart, and Rust also link
  their registry pages to alf.dev, the source, and their relevant guide or
  specification; Python's license metadata uses the current SPDX form instead
  of the deprecated table and classifier.

- **The first-release instructions distinguish registry bootstrap rules.** PyPI
  can create a project from a pending OIDC publisher; npm, pub.dev, and crates.io
  all require a manual first version before trusted publishing can be configured.
  `RELEASING.md` now gives the exact split and routes those unavoidable manual
  uploads through the release guard's human-certification path, so their GitHub
  Releases remain attributable without claiming OIDC published them.

- **The Dart web smoke test wrote to a fixed `/tmp` path on a persistent shared
  runner.** One concurrent job away from two writers, and with a matrix it is
  two legs of the same job racing on one file. It uses `$RUNNER_TEMP`, scoped by
  SDK, exactly as the isolated-environment step beside it already did.

- **The release workflow reads a package's version through one tested script.**
  It was four `grep -m1 '^version = '` invocations inside a YAML string, feeding
  the check that guards the one release mistake nothing can undo: a package
  published under a version its own manifest does not claim. A first-match grep
  does not know which table it is in, and `pyproject.toml` opens with
  `[build-system]` — so a `version` added to any earlier table silently wins,
  and the tag is then compared against the wrong number.

  `.github/scripts/package_version.sh` is section-aware for TOML, root-anchored
  for YAML, and parses JSON rather than grepping it. It also cross-checks
  `pyproject.toml` against `python/alf/__init__.py`, which used to be checked
  only at tag time — a bump that updated one and forgot the other now fails in
  the pull request instead.

- **`ruff format` ran over the Python tree, and is now a CI gate.** The lint
  rules landed first and the tree was left unformatted, so the formatter could
  not become a gate until it had been run once. Almost all of the diff is
  removing hand-wrapping the configured 120-column limit never required — 395
  lines deleted against 163 added, across ten files.

  Markdown is excluded, deliberately: ruff reformats Python blocks inside `.md`,
  and the guides align trailing comments into a column a reader scans down.
  `force-exclude` makes the exclusion hold when a path is named explicitly,
  which is exactly how someone would trip it. Prettier owns the prose.

- **The header-only readers enforce the CHANNEL-scoped header rules too, in all
  four implementations.** The envelope rules apply to every channel; these
  depend on which one you opened, and they lived only inside the channel
  decoders — which a header-only read never reaches. So every implementation
  accepted a `.alx` with no declared count and a `.all` naming a label layout
  that does not exist, both of which the full readers refuse and both of which
  are decidable from the header alone.

  Only rules the full reader enforces EAGERLY were added. `.alx` without
  `_source_total_points` is refused where its indices are decoded, so promoting
  it would make a scan stricter than the read it stands in for — the opposite
  error and just as wrong, and there is a test for that too.

- **The prefix inflate is bounded on its INPUT, in Dart and Rust.** Bounding a
  prefix read's output does not bound its work: DEFLATE may emit nothing at all
  for arbitrarily much input, so a member of empty non-final stored blocks — five
  bytes each, zero bytes out — was walked in full. Measured at ~10 ms per
  compressed MiB in Dart and ~1.3 ms in Rust, so a 100 MiB member took a second
  and 136 ms respectively to answer a question its first 35 bytes settled. Both
  now apply the same compressed-input ceiling Python and TypeScript already had,
  and Rust applies it to its FULL reader too, which had no input bound at all.

  A tighter, per-request budget was tried first and reverted: it made the header
  read refuse members the full reader accepts, which inverts the one guarantee
  the entry point makes. Both readers are held to one ceiling, and a test in each
  language pins that a low-yield member either reader accepts is accepted by
  both.

- **`read_alf_header` validates `_total_points` eagerly.** It previously left
  the check to the `total_points` property, so a malformed count survived a scan
  and raised later, from an accessor, on whichever frame a caller looked at
  closely — while TypeScript's reader refused the same bytes at the call. The
  count is decidable from the header alone, which makes refusing it a
  header-scoped rule that this function documents itself as enforcing. Found
  while building the Dart entry point against both. No valid file changes
  meaning; a file this now refuses is one `read_alf` already refused.

### Added

- **`readAlfHeader` in TypeScript**, matching Python's `read_alf_header`. A
  scan-many-frames caller can now read `_total_points`, `device_position`,
  `_geom_scale` or a `palette` without inflating a payload that may be hundreds
  of megabytes. Backed by a new `gunzipPrefix`, which inflates a bounded prefix
  and stops — a different contract from `gunzip`, which refuses a member that
  exceeds its limit.

  It returns an `AlfHeader`, not an `AlfFile`. Python returns an `AlfFile` whose
  payload accessors raise, because a dataclass cannot express "these fields are
  absent here"; TypeScript can, so reaching for `.points` on a header-only read
  fails to compile rather than at runtime.

  The two paths share one `parseEnvelopeHeader`, so a header-scoped refusal is
  identical between `readAlf` and `readAlfHeader` by construction rather than by
  review. The payload-scoped checks cannot run, and for a gzipped input the
  member's CRC32 and ISIZE go unverified — verifying them means inflating to the
  end, which is the cost being avoided. That is documented at both call sites.

  This unblocks a session-service consumer whose storage reader reads a
  version and metadata and then stops; a full `readAlf` there would have turned a
  header scan into a whole-file decompression inside a Lambda.

- **A release workflow.** `RELEASING.md` specified four tag namespaces and
  nothing implemented them, so the documented tags did nothing and the list had
  gone stale — it omitted Rust, which has existed since the crate landed.
  `.github/workflows/release.yml` now has a job per namespace, each guarded on
  its own tag prefix, each publishing through OIDC trusted publishing with no
  stored token, and each leaving a GitHub Release carrying that version's
  changelog section.

  Two checks run BEFORE the build, so a mistake stops the release while it can
  still be abandoned: the package version must match the tag, and the changelog
  must have a non-empty section for it that is not still marked unreleased. Each
  job also re-runs its language's conformance gate against the exact tagged
  commit — CI proved the branch, this proves the tag, and the one time those
  differ is the time it matters.

- **A Python lint gate, which did not exist.** `black`, `isort` and `flake8` were
  declared as dev dependencies and never run by anything, while Dart, TypeScript
  and Rust each had a lint gate in CI. Ruff replaces all three — one tool, one
  pinned version, rules in the root `ruff.toml` — and `ci.yml` now runs it.

  Bringing it up found 315 findings. Most were mechanical and are fixed here.
  Two were real: five `pytest.raises(match=...)` patterns wrote `.alp` where they
  meant a literal dot, so they matched more loosely than intended, and three
  lambdas in a memory-ratio test closed over loop variables. Two rule families
  are ignored with the reason recorded, because they fight this codebase on
  purpose rather than by accident: `B018` in the test suite, where
  `with pytest.raises(...): read_alf(path).xyz` is the assertion rather than a
  useless expression, and `E501` in the invalid-corpus generator, where each
  case's one-line `why` is prose that should not be wrapped to a column.

### Fixed

- **`gunzipPrefix` no longer Huffman-walks the whole gzip member before
  inflating.** It used `firstGzipMemberEnd` to find the trailer, and that helper
  decodes every DEFLATE symbol. `readAlfHeader` therefore still paid a
  whole-member decode (twice when the header outran the probe), which undercut
  the header-only path. Prefix inflate now feeds pako until it has enough output
  or the member ends, and never calls the boundary walk.

### Changed

- **Version 3 `.alp` states its frame convention, and is refused without one.**

  A `.alp`'s points are the world points with an origin subtracted. That is what
  makes them worth 6 bytes, and it is HALF of a rigid transform. Producers
  disagreed about the other half: some subtracted the origin and stopped,
  leaving points on world axes, while others also rotated into the device frame.
  Both called the result "the frame" and nothing in the bytes told them apart.

  The two agree exactly whenever the device heading is identity, which for a
  vehicle travelling straight is most frames of most sequences. So a consumer
  that omitted the rotation matched a producer that omitted it, the pair looked
  correct, and geometry drifted away from annotations only once the vehicle
  turned — by the accumulated heading, with nothing raising.

  A v3 `.alp` now carries `_frame_convention`, either `world-aligned` or
  `device-aligned`, and `device-aligned` carries a `_frame_rotation` unit
  quaternion beside it. `world_xyz()` applies the rotation when the file says
  one was applied, so placement is decided by the file rather than guessed by
  the caller.

  Seven refusals make it portable: `frame-convention-missing`,
  `frame-convention-unsupported`, `frame-convention-on-legacy-version`,
  `frame-rotation-missing`, `frame-rotation-malformed`,
  `frame-rotation-not-normalized` and `frame-rotation-unexpected`. The rotation
  is an OBJECT with named members rather than an array, because every array
  spelling must choose between `[x, y, z, w]` and `[w, x, y, z]` and the two are
  indistinguishable in the bytes.

  **Required rather than defaulted**, because a default is the thing that
  failed: absence is exactly what both producers had, and reading it as either
  convention re-creates the bug for the other one. This is a breaking change to
  v3 writes — `write_alp(..., codec="int16")` now needs `frame_convention=` —
  and it is made now precisely because v3 is opt-in and no consumer has adopted
  it. Versions 1 and 2 are grandfathered and keep the old behaviour; what they
  may not do is carry the field, since a reader applies no rotation to them and
  the declaration would be a promise the format does not keep.

  Normalization lives in the writer, which knows what the caller meant. A pose
  quaternion that has been through a few compositions is unit to about 1e-7, not
  to the 1e-9 the reader requires; readers refuse a non-unit quaternion rather
  than normalizing it, because conjugation by one scales by |q|² as well as
  rotating and normalizing on read would hide the producer's fault instead of
  fixing it.

- **Python 3.11 is the floor**, raised from 3.9. numpy's own floor passed it —
  `numpy>=2.2.6` requires 3.10 and 2.3 requires 3.11 — so holding 3.9 meant
  holding an increasingly old numpy under a library whose whole job is decoding
  into it. Both dropped versions are end-of-life. The CI matrix is now
  3.11/3.12/3.13, and the package declares its supported versions as
  classifiers, which it previously did not.

- **Non-finite float16 `.alp` geometry is VALID CONTENT, in every
  implementation.** A NaN component is a lidar's invalid return, not damage.
  Earlier readers refused these valid frames and reported them as corrupt.
  `.xyz` now returns invalid returns in place, and the
  writers emit them. Magnitude-derived values (the reported error bound, the
  header `box`, the fp16 localization budget) are computed over the finite
  components only — a plain maximum returns NaN, and NaN compares false against
  every threshold, so those guards had silently stopped guarding. A frame with
  no finite extent publishes no `box`.
- **Version 3 fixed-point carries an invalid return, in all four
  implementations.** `-32768` is reserved — `_geom_scale` is derived as
  `max|component| / 32767`, so a real coordinate never reaches it — and a point
  whose three words are all the marker decodes to NaN, so the same frame written
  in either codec decodes to the same array. Writers must emit the marker rather
  than let `round((NaN - offset) / scale)` narrow to a platform-defined int16
  that looks exactly like data. A point with SOME but not all words equal to the
  marker is refused (`geom-partial-no-return`) in both directions: reconstructing
  one arithmetically yields a plausible position that no later check can
  distinguish from measurement.

  This supersedes an earlier entry in this same release which said v3 could not
  represent an invalid return and that writers refused `codec="int16"` for a
  frame carrying one. Both were true when written and neither is now. No file's
  interpretation changed, because no conforming writer had ever emitted the wire
  pattern the marker now defines.

  A non-finite v3 _decode_ outside a marker row remains a refusal
  (`coordinate-non-finite`): it can only mean the codec parameters overflowed.

- Conformance: `fp16-coordinate-non-finite` moved from the invalid corpus to two
  valid cases (`alp-fp16-invalid-returns`, `alp-fp16-infinities`). Canonical
  output spells non-finite components `"NaN"`, `"Infinity"`, `"-Infinity"` —
  JSON has no NaN, and the three languages disagreed on the bare form.
- `generate.py` now prunes corpus files whose case no longer exists. A removed
  case used to leave its committed expectation on disk, and `run.py` discovers
  cases by globbing, so the deleted rule kept being enforced.

### Renamed

**ALF now stands for "Arrays in Local Frames"**, not "Avala LiDAR Files".

The extensions are unchanged and unchangeable — readers infer the channel from
the suffix and every stored object is named by it — so this is a rename of the
project, not of the format. Each letter still spells its channel:

|        | was                       | now                          |
| ------ | ------------------------- | ---------------------------- |
| `.alp` | Avala Lidar **P**oints    | Array of Local **P**oints    |
| `.ali` | Avala Lidar **I**ntensity | Array of Local **I**ntensity |
| `.alc` | Avala Lidar **C**olor     | Array of Local **C**olor     |
| `.all` | Avala Lidar **L**abels    | Array of Local **L**abels    |
| `.alx` | —                         | Array of Local inde**X**es   |

Two reasons. The clouds are no longer only lidar — photogrammetry, RGBD and
aerial tiles use the same channels, and the codec analysis is about the codec,
not the sensor. And "Local" is the property that actually distinguishes this
format from LAS/LAZ/COPC, which own the global/geospatial case and do it better.

### Added

- **`read_alf_header`** (Python) — parses an ALF envelope without reading or
  inflating the payload, for callers that scan many frames for header fields.
  Accepts paths, buffers and forward-only streams; never seeks. Shares one
  envelope parser with `read_alf`, so every header-scoped refusal is identical;
  the payload-scoped ones (count cross-check, concatenated-member detection,
  whole-frame size cap, read-length equality) cannot run and the returned
  `AlfFile` reports `header_only` and refuses every payload accessor rather than
  answering from an empty stand-in. Closes a parity gap the other direction from
  usual: the Dart compatibility package has had `headerOnly` for a while and the
  format's own SDKs did not.
- **Conformance suite** (`tests/conformance/`) — the cross-language contract.
  Deterministic generated corpus, committed expectations, `--verify` gate that
  asserts two consecutive generator runs are byte-identical.
- **Invalid corpus** — one generated file per declared refusal, each breaking
  exactly one rule and naming the reason. A valid corpus cannot prove a reader
  refuses what it should.
- v3 int16 fixed-point geometry: ~20× lower error than float16 at identical
  bytes per point, with the bound measured and written into the header.
- Pure Dart reader/writer SDK with VM, JavaScript, and Wasm compile coverage.
- Pure TypeScript reader/writer SDK over `Uint8Array`, with browser-standard
  gzip streams and no Node-specific runtime imports.
- Shared conformance coverage for all five channels, including `.alx`, and for
  version-2 rejection outside `.all`.
- Normative wire-format specification and independent language release model.

### Provenance

Extracted from a single module inside the deploying organization's ingest
pipelines, which had itself consolidated roughly 24 independent readers and 17
writers that disagreed with each other. The full
implementation and its test suite came across intact; current evidence is kept
in the synchronized [support matrix](spec/support.md).
