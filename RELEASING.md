# Releasing ALF SDKs

The specification and conformance corpus are shared; language packages are
versioned independently. A package release must not imply that application
consumers have adopted an opt-in wire version.

## Tag namespaces

```text
releases/python/vX.Y.Z
releases/dart/vX.Y.Z
releases/typescript/vX.Y.Z
releases/rust/vX.Y.Z
releases/corpus/vX.Y.Z
```

Each namespace is a job in
[`.github/workflows/release.yml`](.github/workflows/release.yml), guarded on its
own tag prefix. Nothing fires on an ordinary push, so a release is always
something someone did on purpose.

There is no repository-wide version and there should never be a bare `vX.Y.Z`
tag: the SDKs are versioned independently precisely so a package release cannot
imply that application consumers have adopted an opt-in wire version.

The corpus has its own namespace because adding a conformance case changes the
evidence every SDK is graded against without necessarily changing a package. It
publishes to no registry — its release carries the generated fixtures as an
asset, so an implementation outside this repository can be graded without
running the generator.

## Package checklist

1. Regenerate the corpus and pass `generate.py --verify`.
2. Pass that language's unit/regression suite and every conformance case it
   claims in the support matrix.
3. Pass the independent encoder round-trip gate for every writer the package
   claims.
4. Confirm the package version and released changelog section match the tag.
5. Confirm the support matrix records SDK evidence and `spec/adoption.md`
   describes the independent checks an application needs before adopting v3.
6. Publish through the registry's trusted-publishing mechanism and create a
   GitHub Release from the same changelog text.

Steps 1-4 are enforced by the workflow rather than trusted to the checklist. Each
job re-runs its language's conformance gate against the exact tagged commit — CI
proved the branch, this proves the tag, and the one time those differ is the time
it matters — and both the version check and the changelog check run BEFORE the
build, so a mismatch stops the release while it can still be abandoned.

## When a publish fails halfway

A registry version is immutable, so the interesting failures are the ones that
happen _after_ the registry has accepted the package. The workflow drafts the
GitHub Release before publishing and finalizes it after, and
[`release_guard.sh`](.github/scripts/release_guard.sh) decides what a second
attempt is allowed to do. **Re-running the failed job is the normal recovery**,
and it is safe: the guard reads evidence, not attempt counts.

| What you see                                                                 | What to do                                                                                                                      |
| ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| The publish step failed and the registry does not have the version           | Re-run the job. Nothing was published; it publishes.                                                                            |
| The publish succeeded and a later step failed                                | Re-run the job. It skips the publish, keeps the release notes, and finalizes.                                                   |
| The runner died between publishing and recording it                          | Certify it by hand — see below. The upload is retried, so this is rare, and it is the one case nothing here can decide for you. |
| It refuses: the registry has the version and nothing in the run published it | Read below. This one is deliberately not automatic.                                                                             |
| It refuses: it could not establish whether the registry has the version      | Look at the registry's status page and re-run. An unanswered question is not an answer.                                         |

The refusal about an unaccounted-for version means it cannot be attributed to this
tag — a manual publish, an accidental version reuse, or a tag that has been moved
since it was released. Publishing this tag's notes against that artifact would
tell readers the tagged commit produced code it may not contain. The fix is
normally to **bump the version**.

If you have established by hand that the published artifact really was built
from the commit the job is running on, the error message prints the two commands
that certify it, after which a re-run finalizes. That step is manual on purpose,
and it is worth being precise about why. When a job announces a publish and then
dies, two different things may have happened: the publish succeeded and the job
died before writing that down, or the publish _failed_ because somebody else
uploaded that version first. **Those two leave identical evidence.** The
workflow will not guess between them, so it hands you the question — whether the
artifact on the registry was built from this commit — and that is a claim only a
person can make.

Two runs for one tag never overlap: the workflow serializes them and does not
cancel, because cancelling is only safe before a run has published and this is
the workflow that cannot tell you when that was. A second push of the same tag
queues behind the first.

The evidence lives as two small assets on the tag's own GitHub Release —
`publishing-<target>-<run-id>.txt` before the attempt and
`published-<target>.txt` after it — one pair per registry, since the four
packages fail and recover independently. The announcement is removed once the
proof exists, so a shipped release carries only the proof.

## Before the first release of a package

The four registries do **not** have the same bootstrap rule. Only PyPI can bind
OIDC before a project exists. The other three require one interactive release to
create the package first; claiming that their trusted publisher can be configured
before version 0.1.0 makes the documented first release impossible.

| Registry  | First-release rule                                                                                                                                                                                                                                                                                                   |
| --------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| PyPI      | Create a [pending publisher](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/) for `avala-alf`: owner `avala-ai`, repository `alf`, workflow `release.yml`, environment `release`. The namespaced tag can then create the project and publish 0.1.0 through OIDC.                           |
| npm       | [`npm trust` requires the package to exist](https://docs.npmjs.com/cli/v12/commands/npm-trust/#prerequisites). Publish `@avala-ai/alf` 0.1.0 interactively, then configure GitHub trusted publishing for `avala-ai/alf`, `release.yml`, environment `release`, with `npm publish` allowed.                           |
| pub.dev   | [Automated publishing works only for an existing package](https://dart.dev/tools/pub/automated-publishing). Publish `alf` 0.1.0 interactively, transfer it to the verified `avala.ai` publisher, then enable GitHub publishing from `avala-ai/alf` with tag pattern `releases/dart/v{{version}}`.                    |
| crates.io | [The first crate release must be manual](https://blog.rust-lang.org/2025/07/11/crates-io-development-update-2025-07/#trusted-publishing). Remove `publish = false` from `rust/alf/Cargo.toml` in a reviewed change, publish `avala-alf` 0.1.0 interactively, then configure its trusted publisher for this workflow. |

The repository must be public before any package release: the packages link to
its source, and npm provenance is available only when both the repository and
package are public. The Rust job also checks `publish = false` explicitly and
fails early rather than letting `cargo publish` discover it three steps later.

### Bootstrap npm, pub.dev, and crates.io

For each of the three registries that requires an existing package:

1. Merge the version, changelog, and package metadata, then record the exact
   commit SHA. For Rust this includes the reviewed removal of `publish = false`.
2. In a clean checkout detached at that SHA, complete pre-publish steps 1-5 of
   the package checklist above. Then run the matching preparation block below
   from the repository root. Each block changes into the package directory in a
   subshell, so the registry reads the intended manifest rather than looking for
   one at the repository root.

   npm:

   ```bash
   (
     cd typescript/alf
     bun install --frozen-lockfile
     bun run build
     npm pack
     sha256sum avala-ai-alf-0.1.0.tgz
   )
   ```

   `npm pack` deliberately creates the tarball. Inspect and retain that file:
   the publish step below uploads those exact bytes instead of rebuilding from
   the directory. Do not substitute `npm publish` without the tarball argument,
   because that runs the packaging lifecycle again and leaves the manual
   artifact with nothing stable to compare later.

   pub.dev:

   ```bash
   (
     cd dart/alf
     dart pub get
     dart pub publish --dry-run
   )
   ```

   crates.io:

   ```bash
   (
     cd rust/alf
     cargo publish --dry-run
   )
   ```

3. Inspect the retained archive or dry-run file list, then publish 0.1.0
   interactively from that same checkout and package directory with the
   corresponding command below. Do not add the temporary credential to GitHub
   or this repository; revoke a one-use npm or crates.io token after the upload.
   The npm bootstrap version is necessarily the one release without OIDC
   provenance.

   ```bash
   (cd typescript/alf && npm publish avala-ai-alf-0.1.0.tgz --access public) # npm
   (cd dart/alf && dart pub publish)                                     # pub.dev
   (cd rust/alf && cargo publish)                                        # crates.io
   ```

4. Now that the package exists, configure the trusted publisher from the table
   above before preparing another version.
5. Push the corresponding namespaced tag, still pointing at the exact SHA that
   produced the registry artifact. The workflow drafts the GitHub Release and
   reruns every gate, then the release guard intentionally refuses because the
   registry already has the version but no workflow proof exists yet.
6. Compare the registry artifact's unpacked contents with the retained npm/crate
   archive or the pub.dev dry-run file set from the detached checkout. For npm,
   also compare the SHA-256 checksum with the tarball retained in step 2. If the
   contents match, run the two certification commands printed by the refusal
   and re-run the failed job. The proof marker makes the second run skip the
   upload and finalize the GitHub Release without pretending OIDC performed the
   manual bootstrap.

This is the same human certification path used when a runner dies after an
ordinary publish. It is explicit because the guard cannot distinguish a manual
bootstrap built from the tag from somebody else's artifact with the same
version.

Until a package exists at its registry, badges must say so plainly: code existing
in this repository is not evidence that a package is installable, which is why
the README carries CI and conformance badges and no registry badges yet.

## Registry names

ALF is the format name and `alf` remains the import/library name. Registry
distribution names are a separate constraint:

| Registry  | Distribution name | Consumer API                                          |
| --------- | ----------------- | ----------------------------------------------------- |
| PyPI      | `avala-alf`       | `pip install avala-alf`, then `import alf`            |
| npm       | `@avala-ai/alf`   | `import … from "@avala-ai/alf"`                       |
| pub.dev   | `alf`             | `import 'package:alf/alf.dart'`                       |
| crates.io | `avala-alf`       | `cargo add avala-alf --rename alf`, then `use alf::…` |

The unscoped `alf` names cannot be claimed: PyPI's is an unrelated OAuth client
and crates.io's belongs to an unrelated Rust project. The scoped names make
ownership explicit without renaming the ALF API in source code.
