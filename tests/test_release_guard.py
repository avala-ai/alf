"""The release guard's decision table, executed rather than reasoned about.

`release.yml` has never run — there has been no tagged release — so every
defect found in it so far was found by reading. Three rounds of review went
into this one guard and each round found that the previous fix was necessary
and not sufficient:

1. publishing before drafting the release meant a rerun could never finalize;
2. skipping on "the registry has this version" attributed somebody else's
   artifact to this tag;
3. skipping on `run_attempt > 1` was defeated by re-running the failed job,
   which is the documented response to the message the guard itself printed;
4. recording the proof only AFTER a successful publish left the window between
   the two permanently unrecoverable;
5. the registry probe read every failure as "not published", so a rate limit or
   a 5xx could walk a release into publishing a duplicate;
6. and the marker added for (4) proved intent rather than outcome — a publish
   that lost a race to another publisher leaves the same announcement as one
   that succeeded — so it is a diagnostic now, and the recovery it used to
   authorize is a retry plus, where that is not enough, a person.

Each of those is a state machine question that a shell can answer in
milliseconds, which is what this does: `gh` is stubbed by a script backed by a
directory of release assets, and the real guard runs against it.
"""

import os
import re
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GUARD = ROOT / ".github" / "scripts" / "release_guard.sh"

SHA = "a" * 40
OTHER_SHA = "b" * 40
RUN_ID = "1234567890"

# `gh release download` writes the asset into --dir and exits non-zero when the
# pattern matches nothing; `upload` copies it back; `delete-asset` removes it.
# That is the whole surface the guard uses, and it is enough to drive every
# branch.
FAKE_GH = """\
#!/usr/bin/env bash
set -euo pipefail
store="$GH_FAKE_STORE"
mkdir -p "$store"
if [ "${1-}" = "release" ] && [ "${2-}" = "download" ]; then
  pattern=""; dir="."
  shift 3
  while [ $# -gt 0 ]; do
    case $1 in
      --pattern) pattern=$2; shift 2 ;;
      --dir) dir=$2; shift 2 ;;
      *) shift ;;
    esac
  done
  [ -f "$store/$pattern" ] || exit 1
  cp "$store/$pattern" "$dir/$pattern"
  exit 0
fi
if [ "${1-}" = "release" ] && [ "${2-}" = "upload" ]; then
  # Fail the first N uploads, so the guard's retry can be observed rather than
  # assumed. The counter is a file because each call is a fresh process.
  left=$(cat "$store/.upload-failures" 2>/dev/null || echo "${GH_FAKE_UPLOAD_FAILURES:-0}")
  if [ "$left" -gt 0 ]; then
    echo "$((left - 1))" > "$store/.upload-failures"
    echo "simulated upload failure" >&2
    exit 1
  fi
  cp "$4" "$store/$(basename "$4")"
  exit 0
fi
if [ "${1-}" = "release" ] && [ "${2-}" = "delete-asset" ]; then
  rm -f "$store/$4"
  exit 0
fi
exit 64
"""

# The guard asks the registry with `-o /dev/null -w '%{http_code}'`, so the only
# thing on stdout is the status. `000` is what real curl writes when it never
# got an answer at all.
# The guard backs off between retries. A no-op `sleep` on the stub PATH keeps
# that out of the test's wall clock WITHOUT putting a test-only branch in the
# script under test.
FAKE_SLEEP = """\
#!/usr/bin/env bash
exit 0
"""

FAKE_CURL = """\
#!/usr/bin/env bash
printf '%s' "${CURL_FAKE_CODE:-200}"
[ "${CURL_FAKE_CODE:-200}" != "000" ] || exit 7
exit 0
"""

PROBE = "https://registry.example/alf/0.1.0"


@pytest.fixture
def guard(tmp_path):
    """Run the real guard with `gh` stubbed and the markers on disk."""
    store = tmp_path / "assets"
    store.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    # Two PATHs: one where curl is stubbed, and one where only `gh` is, so the
    # probe can be exercised against the real curl binary.
    live_bin = tmp_path / "live-bin"
    live_bin.mkdir()
    for directory, stubs in (
        (bin_dir, (("gh", FAKE_GH), ("curl", FAKE_CURL), ("sleep", FAKE_SLEEP))),
        (live_bin, (("gh", FAKE_GH), ("sleep", FAKE_SLEEP))),
    ):
        for name, body in stubs:
            fake = directory / name
            fake.write_text(body)
            fake.chmod(0o755)
    output = tmp_path / "github_output"
    output.touch()

    def run(*args, sha=SHA, run_id=RUN_ID, code="200", upload_failures=0, real_curl=False):
        env = {
            **os.environ,
            "CURL_FAKE_CODE": code,
            "GH_FAKE_UPLOAD_FAILURES": str(upload_failures),
            "PATH": (f"{live_bin if real_curl else bin_dir}:{os.environ['PATH']}"),
            "GH_FAKE_STORE": str(store),
            "GITHUB_REF_NAME": "releases/python/v0.1.0",
            "GITHUB_SHA": sha,
            "GITHUB_RUN_ID": run_id,
            "RUNNER_TEMP": str(tmp_path / "temp"),
            "GITHUB_OUTPUT": str(output),
        }
        (tmp_path / "temp").mkdir(exist_ok=True)
        done = subprocess.run(["bash", str(GUARD), *args], env=env, capture_output=True, text=True, check=False)
        # The probe's diagnostics go to stderr — its stdout is the yes/no the
        # script itself reads back — so what a maintainer sees in the job log is
        # the two streams together, and that is what these assert against.
        done.said = done.stdout + done.stderr
        return done

    run.store = store
    run.outputs = lambda: dict(line.split("=", 1) for line in output.read_text().split() if "=" in line)
    return run


def test_a_version_the_registry_does_not_have_is_published(guard):
    done = guard("decide", "pypi", "0.1.0", PROBE, code="404")
    assert done.returncode == 0
    assert guard.outputs() == {"publish": "true"}


@pytest.mark.parametrize("code", ["500", "429", "403", "000"])
def test_an_unanswered_probe_stops_the_job(guard, code):
    """THE FIFTH ROUND. `curl -sf` exits non-zero for a rate limit, a 5xx and a
    dropped connection exactly as it does for a 404, so every one of them read
    as "not published yet". The job would announce, publish into a duplicate and
    fail — and the re-run would then find the registry holding the version and
    its OWN announcement beside it, take the recovery path, and record this
    commit as the publisher of an artifact it did not create. A guard built to
    stop misattribution reaching it through a transient network error."""
    done = guard("decide", "pypi", "0.1.0", PROBE, code=code)
    assert done.returncode == 1
    assert "neither 200 nor 404" in done.said
    assert guard.outputs() == {}


def test_an_unexplained_registry_version_is_refused(guard):
    """The first attempt on a version somebody published by hand. Nothing on the
    release accounts for it, so this tag must not claim it."""
    done = guard("decide", "pypi", "0.1.0", PROBE)
    assert done.returncode == 1
    assert f"records run {RUN_ID}" in done.said
    # And the operator is told the one way out, rather than left to invent one.
    assert "gh release upload" in done.said


def test_proof_from_this_commit_skips_and_finalizes(guard):
    guard("record", "pypi")
    done = guard("decide", "pypi", "0.1.0", PROBE)
    assert done.returncode == 0
    assert guard.outputs() == {"publish": "false"}


def test_proof_from_another_commit_is_refused(guard):
    """A re-pushed tag. The marker names the COMMIT, not the tag, because a tag
    can move and a commit cannot."""
    guard("record", "pypi", sha=OTHER_SHA)
    done = guard("decide", "pypi", "0.1.0", PROBE)
    assert done.returncode == 1
    assert f"names commit {OTHER_SHA}" in done.said


def test_an_announcement_is_intent_and_never_authorizes_a_skip(guard):
    """An earlier revision let this stand in for proof, and it cannot. If anyone
    else uploads the version between this run's probe and its publish command,
    this run's publish fails as a duplicate and leaves an announcement that is
    byte-identical to the announcement of a publish that SUCCEEDED and could not
    be recorded. Inferring success from intent is the "close enough" the rest of
    this guard exists to refuse."""
    guard("announce", "pypi")
    done = guard("decide", "pypi", "0.1.0", PROBE)
    assert done.returncode == 1
    assert guard.outputs() == {}
    # It does change what the refusal SAYS: this is the one case where the
    # maintainer has a specific thing to go and look at.
    assert "announced a publish" in done.said
    assert "identical evidence" in done.said


def test_certifying_by_hand_is_what_unblocks_it(guard):
    """The documented escape from a job that published and then died. The
    operator writes the proof marker themselves — the two commands the refusal
    prints — and the next attempt takes the ordinary proven path."""
    guard("announce", "pypi")
    assert guard("decide", "pypi", "0.1.0", PROBE).returncode == 1
    (guard.store / "published-pypi.txt").write_text(f"{SHA}\n")
    done = guard("decide", "pypi", "0.1.0", PROBE)
    assert done.returncode == 0
    assert guard.outputs() == {"publish": "false"}


def test_recording_clears_the_announcement_from_the_release(guard):
    """A shipped release should not carry the scaffolding that got it there."""
    guard("announce", "pypi")
    assert (guard.store / f"publishing-pypi-{RUN_ID}.txt").exists()
    guard("record", "pypi")
    assert not (guard.store / f"publishing-pypi-{RUN_ID}.txt").exists()
    assert (guard.store / "published-pypi.txt").exists()


def test_another_runs_announcement_is_not_even_visible(guard):
    """The run id is in the file NAME, so a marker from another run is not
    something this can read by accident or be handed by a race."""
    guard("announce", "pypi")
    done = guard("decide", "pypi", "0.1.0", PROBE, run_id="9999999999")
    assert done.returncode == 1
    assert "records run 9999999999" in done.said


def test_an_announcement_from_another_commit_is_named_in_the_refusal(guard):
    guard("announce", "pypi", sha=OTHER_SHA)
    done = guard("decide", "pypi", "0.1.0", PROBE)
    assert done.returncode == 1
    assert OTHER_SHA in done.said


def test_two_runs_on_one_tag_cannot_overwrite_each_others_announcement(guard):
    """The run id is in the marker's NAME. A shared name plus `--clobber` would
    let a second run inherit the first's identity: run A publishes commit X and
    dies before recording, run B for a moved tag overwrites the announcement and
    loses the duplicate race, and B's re-run then sees its own announcement
    beside A's artifact. Separate names make that unrepresentable — and the
    workflow serializes runs per tag as well, so this is the second lock."""
    guard("announce", "pypi", sha=OTHER_SHA, run_id="1111111111")
    guard("announce", "pypi", sha=SHA, run_id="2222222222")
    assert (guard.store / "publishing-pypi-1111111111.txt").read_text().strip() == OTHER_SHA
    assert (guard.store / "publishing-pypi-2222222222.txt").read_text().strip() == SHA


def test_the_markers_hold_what_the_guard_reads_back(guard):
    guard("announce", "pypi")
    guard("record", "pypi")
    assert (guard.store / "published-pypi.txt").read_text().split() == [SHA]


def test_targets_do_not_share_markers(guard):
    """Four registries publish independently. npm's proof must not let the
    crates.io job skip — they fail separately and recover separately."""
    guard("record", "npm")
    done = guard("decide", "crates", "0.1.0", PROBE)
    assert done.returncode == 1


@pytest.mark.parametrize("argv", [[], ["decide"], ["decide", "pypi"], ["nonsense", "pypi"]])
def test_a_malformed_invocation_stops_rather_than_guessing(guard, argv):
    assert guard(*argv).returncode == 2


def test_a_flaky_asset_upload_is_retried_rather_than_reasoned_about(guard):
    """The step whose failure used to strand a release. The common cause is a
    transient GitHub API error, and the honest answer to a transient error is to
    ask again — not to let a later run infer what its absence implied."""
    done = guard("record", "pypi", upload_failures=3)
    assert done.returncode == 0
    assert (guard.store / "published-pypi.txt").read_text().split() == [SHA]
    assert done.said.count("retrying") == 3


def test_an_upload_that_never_succeeds_fails_the_step(guard):
    done = guard("record", "pypi", upload_failures=99)
    assert done.returncode == 1
    assert "after five attempts" in done.said


class TestTheProbeAgainstRealCurl:
    """`curl` is NOT stubbed here, because the defect was in curl's behaviour.

    `-o /dev/null` with `--retry`: before retrying, curl truncates the failed
    transfer's output, `ftruncate` on `/dev/null` fails, and curl gives up with
    exit 23 having written no status at all. So `--retry 5` retried nothing and
    a single 429 or 5xx stopped the release — the opposite of what the flag was
    there for, and invisible to any test that stubs curl out.
    """

    @staticmethod
    def _serve(responses):
        """A local server answering with `responses` in order, then repeating
        the last one. Returns its base URL and a shutdown callable."""
        import http.server
        import threading

        state = {"n": 0}

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                code = responses[min(state["n"], len(responses) - 1)]
                state["n"] += 1
                body = b"{}"
                self.send_response(code)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        # Port 0: the OS picks a free one, so concurrent test runs cannot collide.
        server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return f"http://127.0.0.1:{server.server_port}/alf", server.shutdown, state

    @pytest.mark.parametrize(
        ("responses", "expected"),
        [
            ([404], "publish"),
            ([200], "refuse"),
            ([500, 500, 404], "publish"),
            ([429, 200], "refuse"),
            ([500], "stop"),
        ],
    )
    def test_transient_failures_are_ridden_out_and_real_ones_are_not(self, tmp_path, guard, responses, expected):
        url, shutdown, state = self._serve(responses)
        try:
            done = guard("decide", "pypi", "0.1.0", url, real_curl=True)
        finally:
            shutdown()
        if expected == "publish":
            assert done.returncode == 0, done.said
            assert guard.outputs() == {"publish": "true"}
        elif expected == "refuse":
            assert done.returncode == 1
            assert "already has 0.1.0" in done.said
        else:
            assert done.returncode == 1
            assert "neither 200 nor 404" in done.said
        # And the retries genuinely happened rather than the first answer
        # standing in for all of them.
        assert state["n"] >= len(responses), state["n"]


def test_every_publishing_job_routes_through_the_guard():
    """The four jobs had four copies of this logic, and each review round found
    a fix that had been applied to some of them. One implementation, called four
    times, is the property that keeps that from recurring."""
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text()
    for subcommand in ("decide", "announce", "record"):
        assert workflow.count(f"release_guard.sh {subcommand}") == 4, subcommand
    # And no job carries a hand-rolled copy any more — of the decision, or of
    # the registry probe whose exit code could not answer the question.
    assert "published_sha=" not in workflow
    assert "curl -sf" not in workflow


def test_one_release_run_per_tag_and_never_cancelled():
    """Two runs for one tag can interleave a publish with another run's
    announcement. Cancelling is not the alternative: cancelling is only safe
    before a run has published, and this is the workflow that cannot say when
    that was."""
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text()
    # Read as text because this assertion is about the exact GitHub expression
    # and cancellation literal, not the rest of the workflow structure.
    block = re.search(r"^concurrency:\n((?:[ \t]+.*\n)+)", workflow, re.MULTILINE)
    assert block is not None, "release.yml declares no top-level concurrency group"
    assert "group: release-${{ github.ref }}" in block.group(1)
    assert "cancel-in-progress: false" in block.group(1)


@pytest.mark.skipif(shutil.which("shellcheck") is None, reason="shellcheck not installed")
def test_the_guard_passes_shellcheck():
    done = subprocess.run(["shellcheck", str(GUARD)], capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stdout + done.stderr


def test_ci_runs_shellcheck_so_the_test_above_is_not_a_silent_skip():
    """The test above skips where shellcheck is absent, which on its own is the
    same "gate that never runs" this whole file exists because of. CI installs a
    pinned, checksummed shellcheck and runs it over every script in the
    directory — not just this one — so a second script cannot arrive unchecked."""
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert "shellcheck .github/scripts/*.sh" in workflow
    assert "sha256sum -c -" in workflow


def test_the_guard_is_executable():
    assert os.access(GUARD, os.X_OK), textwrap.dedent(
        """\
        release_guard.sh must be executable in the index: the workflow calls it
        directly rather than through `bash`, and a mode that is right in the
        working tree and wrong in git fails only on the runner."""
    )
