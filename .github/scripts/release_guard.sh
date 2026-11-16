#!/usr/bin/env bash
#
# Decide whether a release job may publish, skip, or must stop — and leave
# behind the evidence that decision will be made from next time.
#
# THE PROBLEM THIS SOLVES. A registry version is immutable. If a job publishes
# it and then fails before finalizing the GitHub Release, the natural recovery
# is to re-run the job — and the re-run hits the registry's duplicate-version
# error and never reaches the finalizer. So the job has to be able to skip a
# publish it already did. But "the version exists" is NOT evidence that this tag
# produced it: a manual publish, an accidental version reuse, or a re-pushed tag
# all look identical from here, and skipping on that alone would attach this
# tag's release notes to somebody else's artifact. Neither is `run_attempt > 1`
# evidence: re-running the failed job is exactly what a maintainer does when the
# previous attempt printed an error, so that guard was defeated by the most
# natural response to its own message.
#
# THE EVIDENCE. Two markers, attached to the tag's own GitHub Release:
#
#   publishing-<target>-<run_id>.txt   "<sha>"   written BEFORE the attempt
#   published-<target>.txt             "<sha>"   written AFTER the registry accepts
#
# `published` is the ONLY thing that authorizes a skip. `publishing` authorizes
# nothing: it records that a publish was about to be attempted, which is intent
# and not outcome. An earlier revision of this script let a matching
# announcement stand in for proof, and that was wrong for a reason worth keeping
# written down — if anyone else uploads the version between this run's probe and
# its publish command, this run's publish fails as a duplicate while its
# announcement sits there looking exactly like the announcement of a publish
# that SUCCEEDED and could not be recorded. The two states are identical on
# disk. Inferring success from intent is the same "close enough" the rest of
# this file exists to refuse.
#
# So what covers the window between a registry accepting the package and this
# script writing that down? Two things, neither of them a guess:
#
#   * `write_marker` RETRIES. The common failure there is a transient GitHub API
#     error, and the honest answer to a transient error is to try again rather
#     than to reason about what it implies.
#   * Where the job dies anyway — a cancelled runner, a lost host — a person
#     certifies it. `decide` prints the exact two commands. Whether a published
#     artifact was built from a given commit is the one claim nothing here can
#     check, so a human makes it and the audit trail records that they did.
#
# The announcement earns its place by making that refusal specific: it lets the
# message say "this run announced this publish and never recorded one", which is
# the difference between a maintainer knowing where to look and guessing.
#
# The run id is in the announcement's NAME, not its contents. Two runs racing on
# one tag therefore cannot overwrite each other's evidence — `--clobber` on a
# shared name would let the later run inherit the earlier one's identity. The
# workflow serializes runs per tag as well; this holds even if that is ever
# misconfigured.
#
# WHETHER THE REGISTRY HAS IT is asked here too, rather than by each caller,
# because "the request failed" and "the version is not there" are different
# answers and a shell that checks an exit code cannot tell them apart. A rate
# limit read as absence walks this job into publishing a duplicate. Only a
# definite 404 is absence; anything else stops the job and says so.
#
#   release_guard.sh decide   <target> <version> <existence-probe-url>
#   release_guard.sh announce <target>
#   release_guard.sh record   <target>

set -euo pipefail

usage() {
  sed -n '/^#   release_guard.sh/,/^#$/p' "$0" >&2
  exit 2
}

target=${2-}
[ -n "$target" ] || usage
release=${GITHUB_REF_NAME:?GITHUB_REF_NAME is required}
sha=${GITHUB_SHA:?GITHUB_SHA is required}
run_id=${GITHUB_RUN_ID:?GITHUB_RUN_ID is required}
work=${RUNNER_TEMP:-/tmp}

attempt_marker="publishing-$target-$run_id.txt"
proof_marker="published-$target.txt"

# A marker that is absent is not an error: the first attempt has neither.
read_marker() {
  local name=$1
  if gh release download "$release" --pattern "$name" --dir "$work" --clobber >/dev/null 2>&1; then
    cat "$work/$name"
  fi
}

# The decision, in the one form a later step can branch on. Echoed as well as
# written, so the log says what happened without the reader opening the job's
# output object.
emit() {
  local decision
  for decision in "$@"; do
    echo "$decision"
    if [ -n "${GITHUB_OUTPUT-}" ]; then
      echo "$decision" >> "$GITHUB_OUTPUT"
    fi
  done
}

# RETRIED, because this is the step whose failure used to strand a release. A
# transient GitHub API error here is answered by asking again, which is both
# cheaper and more honest than any inference a later run could make from the
# wreckage.
write_marker() {
  local name=$1 body=$2 attempt
  printf '%s\n' "$body" > "$work/$name"
  for attempt in 1 2 3 4 5; do
    if gh release upload "$release" "$work/$name" --clobber; then
      return 0
    fi
    echo "upload of $name failed (attempt $attempt); retrying" >&2
    sleep $((attempt * 3))
  done
  echo "::error::Could not attach $name to $release after five attempts." >&2
  return 1
}

# 200 is presence, 404 is absence, and there is no third answer this job may
# guess at. `--retry` absorbs the transient ones first so a single blip does not
# stop a release; what reaches the case statement has already been retried.
#
# `-o` a REAL FILE, not `/dev/null`. Before retrying, curl truncates the output
# of the failed transfer; `ftruncate` on `/dev/null` fails, so curl gives up
# with exit 23 and writes no status at all. Measured on curl 8.5.0, the version
# ubuntu-latest ships: a first response of 500 produced an EMPTY code and zero
# retries, so the retry policy this function advertises did not exist and one
# transient blip stopped the release. With a regular file the same sequence
# rides out two 500s and returns the 404.
registry_has() {
  local url=$1 code body="$work/probe-$target.body"
  code=$(curl -sS -o "$body" -w '%{http_code}' \
    --retry 5 --retry-delay 2 --retry-all-errors --max-time 30 \
    -H 'User-Agent: alf-release-guard (+https://github.com/avala-ai/alf)' \
    "$url" || true)
  case $code in
    200) echo yes ;;
    404) echo no ;;
    *)
      # To stderr, not stdout: this function's stdout is the yes/no the caller
      # captures, so a diagnostic written there would BECOME the answer. Actions
      # annotates `::error::` on either stream.
      echo "::error::Could not establish whether the registry already has this version." >&2
      echo "::error::$url answered ${code:-nothing} after retries, which is neither 200 nor 404." >&2
      echo "::error::Refusing to treat an unanswered question as \"not published\": that is how" >&2
      echo "::error::a rate limit becomes a duplicate publish and then a misattributed release." >&2
      return 1
      ;;
  esac
}

case ${1-} in
  decide)
    version=${3-}
    probe_url=${4-}
    if [ -z "$version" ] || [ -z "$probe_url" ]; then
      usage
    fi

    has=$(registry_has "$probe_url")
    if [ "$has" != "yes" ]; then
      echo "$version is not on the registry yet; publishing."
      emit publish=true
      exit 0
    fi

    proof=$(read_marker "$proof_marker" | tr -d '[:space:]')
    if [ "$proof" = "$sha" ]; then
      echo "$version reached the registry from this commit already; finalizing the release."
      emit publish=false
      exit 0
    fi

    # Only THIS run's announcement is even looked for — the run id is in the
    # file name, so another run's marker is not a thing this can read by
    # accident or be handed by a race. It changes what the refusal SAYS. It does
    # not change that this refuses: an announcement is a record of intent, and a
    # publish that lost a race to another publisher leaves exactly the same one
    # as a publish that succeeded and could not be written down.
    attempt=$(read_marker "$attempt_marker" | tr -d '[:space:]')

    echo "::error::The registry already has $version, and nothing here proves this run put it there."
    if [ -n "$proof" ]; then
      echo "::error::The proof marker names commit $proof, not $sha — a moved tag, most likely."
      echo "::error::Bump the version, or release the code that artifact actually contains."
    elif [ "$attempt" = "$sha" ]; then
      echo "::error::This run announced a publish of $version and never recorded one."
      echo "::error::That is CONSISTENT with the publish having succeeded and this job dying"
      echo "::error::before it could say so — and also with the publish having failed because"
      echo "::error::someone else uploaded $version first. The two leave identical evidence, so"
      echo "::error::this will not guess. Check the artifact on the registry against $sha."
    elif [ -n "$attempt" ]; then
      echo "::error::This run announced commit $attempt, and it is now running $sha."
    else
      echo "::error::Nothing on $release records run $run_id publishing it. Bump the version,"
      echo "::error::or release the code that artifact actually contains."
    fi
    echo "::error::"
    echo "::error::If you have verified by hand that $version on the registry WAS built from"
    echo "::error::$sha, certify that and re-run. It is deliberately a human step:"
    echo "::error::it is the one claim nothing here can check for you."
    echo "::error::"
    # printf, not echo: the operator has to see a literal backslash-n, and
    # whether echo eats it depends on the shell this ends up running under.
    printf '::error::  printf %s %s > %s\n' "'%s\\n'" "$sha" "$proof_marker"
    echo "::error::  gh release upload $release $proof_marker --clobber"
    exit 1
    ;;

  announce)
    # BEFORE the publish, so the window between "the registry accepted it" and
    # "we wrote that down" is covered by something already on disk.
    write_marker "$attempt_marker" "$sha"
    echo "Announced: commit $sha, run $run_id."
    ;;

  record)
    write_marker "$proof_marker" "$sha"
    echo "Recorded: published from $sha."
    # The announcement has done its job once the proof exists, and a released
    # version should not carry scaffolding on its release page. Best-effort by
    # design: failing to tidy up is not a reason to fail a release that has
    # already published.
    gh release delete-asset "$release" "$attempt_marker" --yes >/dev/null 2>&1 || true
    ;;

  *)
    usage
    ;;
esac
