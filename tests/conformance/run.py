#!/usr/bin/env python3
"""Score an implementation's runner against the corpus.

Usage:
    python3 tests/conformance/run.py                       # score the Python runner
    python3 tests/conformance/run.py --runner "dart run ..."

An implementation claims a feature by PASSING this suite for it. The support
matrix in README.md records only what the suite proves — never what a reader is
believed to do.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shlex
import shutil
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE / "data"
DEFAULT_RUNNER = f"{sys.executable} {HERE / 'runners' / 'python_runner.py'}"
ADVERTISED_VALID_CASES = 45
ADVERTISED_TOTAL_CASES = 117

sys.path.insert(0, str(HERE))
from bounded_process import MAX_CAPTURE_BYTES, run_bounded


def _corpus_paths() -> list[pathlib.Path]:
    """Return every oracle or fixture a runner must be unable to rewrite."""
    paths = [DATA / "CHECKSUMS.txt"]
    for pattern in ("*.json", "*.gz", "invalid/*.json", "invalid/*.bin"):
        paths.extend(DATA.glob(pattern))
    return sorted(set(paths))


def _run_isolated(runner: str, name: str, payload: bytes, extension: str = "") -> dict:
    """Run one case with no future corpus case reachable through its input path."""
    root = pathlib.Path(tempfile.mkdtemp(prefix="alf-read-"))
    path = root / name
    try:
        path.write_bytes(payload)
        path.chmod(0o444)
        root.chmod(0o555)
        return _run(runner, path, extension)
    finally:
        # The runner owns the isolated copy and can revoke its read-only mode.
        # Restore permissions best-effort so cleanup never masks its result.
        try:
            root.chmod(0o755)
            for child in root.rglob("*"):
                try:
                    child.chmod(0o755 if child.is_dir() else 0o644)
                except OSError:
                    pass
        except OSError:
            pass
        shutil.rmtree(root, ignore_errors=True)


def _run(runner: str, path: pathlib.Path, extension: str = "") -> dict:
    argv = shlex.split(runner) + [str(path)] + ([extension] if extension else [])
    returncode, stdout, stderr, overflow, timed_out = run_bounded(argv, 120)
    if overflow:
        return {"_runner_output_too_large": (f"runner diagnostics exceeded the {MAX_CAPTURE_BYTES}-byte capture limit")}
    if timed_out:
        return {"_runner_crashed": "runner exceeded the 120-second timeout"}
    if returncode != 0:
        # A runner that exits non-zero has FALLEN OVER. Refusing correctly is
        # `{"refused": id}` with exit 0 — collapsing the two would hide a crash
        # behind an expected refusal.
        return {"_runner_crashed": stderr.strip()[-400:] or f"exit {returncode}"}
    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        return {"_runner_output_not_json": stdout.strip()[:400]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", default=DEFAULT_RUNNER)
    parser.add_argument("--valid-only", action="store_true")
    args = parser.parse_args()

    if not (DATA / "CHECKSUMS.txt").exists():
        print("corpus missing — run tests/conformance/generate.py first", file=sys.stderr)
        return 1

    # Freeze every expectation and fixture before the first runner. Per-case
    # copies live in distinct temporary directories, so chmod/overwrite access
    # to one input cannot change what a later case executes or expects.
    corpus_paths = _corpus_paths()
    corpus = {path: path.read_bytes() for path in corpus_paths}
    cases: list[tuple[str, object, str | None, bytes | None, str]] = []
    for expectation_path in sorted(DATA.glob("*.json")):
        name = expectation_path.stem
        expected = json.loads(corpus[expectation_path])
        candidates = sorted(DATA.glob(f"{name}.*.gz"))
        source = candidates[0] if candidates else None
        cases.append(
            (
                name,
                expected,
                None if source is None else source.name,
                None if source is None else corpus[source],
                "",
            )
        )

    if not args.valid_only:
        for expectation_path in sorted((DATA / "invalid").glob("*.json")):
            name = expectation_path.stem
            declared = json.loads(corpus[expectation_path])
            expected = {"refused": declared["refused"]}
            candidates = sorted((DATA / "invalid").glob(f"{name}.*.bin"))
            source = candidates[0] if candidates else None
            extension = "" if source is None else source.name.split(".")[-2]
            cases.append(
                (
                    f"invalid/{name}",
                    expected,
                    None if source is None else source.name,
                    None if source is None else corpus[source],
                    extension,
                )
            )

    failures: list = []
    checked = len(cases)
    for name, expected, source_name, source_bytes, extension in cases:
        if source_name is None or source_bytes is None:
            failures.append((name, expected, {"_fixture_missing": True}))
            continue
        actual = _run_isolated(args.runner, source_name, source_bytes, extension)
        if actual != expected:
            failures.append((name, expected, actual))

    advertised = ADVERTISED_VALID_CASES if args.valid_only else ADVERTISED_TOTAL_CASES
    case_failures = len(failures)
    if checked != advertised:
        failures.append(("corpus-total", {"advertised": advertised}, {"discovered": checked}))

    current_paths = _corpus_paths()
    mutated = [
        path.relative_to(DATA).as_posix()
        for path, original in corpus.items()
        if not path.is_file() or path.read_bytes() != original
    ]
    mutated.extend(path.relative_to(DATA).as_posix() for path in current_paths if path not in corpus)
    if mutated:
        failures.append(
            (
                "corpus",
                {"integrity": "unchanged"},
                {"_runner_integrity_failure": "runner mutated committed corpus: " + ", ".join(sorted(mutated))},
            )
        )

    for name, expected, actual in failures:
        print(f"FAIL {name}")
        print(f"  expected: {json.dumps(expected, sort_keys=True)[:200]}")
        print(f"  actual:   {json.dumps(actual, sort_keys=True)[:200]}")

    # Two independent invariants, both learned from this line getting it wrong:
    #
    # The DENOMINATOR is what the corpus is supposed to contain, so it must not
    # shrink to match a corpus that lost fixtures — otherwise deleting a case
    # improves the ratio. It takes the larger of advertised and discovered so a
    # GROWN corpus is not silently reported against a stale smaller target.
    #
    # The NUMERATOR stays an honest count of cases that passed, so it is NOT
    # clamped when some non-case check fails: a corpus-integrity failure does
    # not retroactively unpass the cases, and faking that would just move the
    # dishonesty. Instead the verdict rides on this same line. `min(checked,
    # advertised)` once printed "93/93 passed" three lines under its own FAILED
    # corpus-total check, because the corpus had grown to 94: the exit code was
    # right, so CI could not be fooled, but a human reading the tail saw a pass
    # — this repo's `ci-green-means-tested` failure mode inside the tool that
    # enforces conformance. A ratio alone cannot express "all cases passed AND
    # the run failed", so the ratio must not be the whole summary.
    expected_total = max(advertised, checked)
    passed = max(checked - case_failures, 0)
    verdict = f"  <-- RUN FAILED, {len(failures)} check(s) did not pass" if failures else ""
    print(f"\n{passed}/{expected_total} passed{verdict}")
    if failures:
        print(f"FAILED: {len(failures)} check(s) above did not pass")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
