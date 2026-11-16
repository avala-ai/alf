"""CI policy regressions for portable ALF evidence gates."""

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"


class _WorkflowLoader(yaml.SafeLoader):
    """GitHub-compatible safe loader that also rejects duplicate YAML keys."""


# PyYAML implements YAML 1.1, where unquoted ``on`` is a boolean. GitHub uses
# YAML 1.2-like booleans, so copy the resolver table and remove only that rule.
# Copying is important: mutating SafeLoader here would change unrelated callers.
_WorkflowLoader.yaml_implicit_resolvers = {
    key: [resolver for resolver in resolvers if resolver[0] != "tag:yaml.org,2002:bool"]
    for key, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
_WorkflowLoader.add_implicit_resolver(
    "tag:yaml.org,2002:bool",
    re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"),
    list("tTfF"),
)


def _construct_unique_mapping(loader: _WorkflowLoader, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
    """Construct a mapping after rejecting duplicate explicit keys.

    Merge keys are deliberately excluded from the duplicate scan: an explicit
    key is allowed to override an inherited value. ``flatten_mapping`` then
    resolves aliases and merges before the policy examines the workflow.
    """
    seen = set()
    for key_node, _ in node.value:
        if key_node.tag == "tag:yaml.org,2002:merge":
            continue
        key = loader.construct_object(key_node, deep=False)
        try:
            duplicate = key in seen
        except TypeError as error:
            raise ValueError(f"unhashable YAML mapping key: {key!r}") from error
        if duplicate:
            raise ValueError(f"duplicate YAML key: {key!r}")
        seen.add(key)

    loader.flatten_mapping(node)
    return dict(loader.construct_pairs(node, deep=deep))


_WorkflowLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping)


def _load_workflow(path: Path) -> dict[str, Any]:
    workflow = yaml.load(path.read_text(), Loader=_WorkflowLoader)
    assert isinstance(workflow, dict), f"{path} is not a workflow mapping"
    return workflow


def _event_names(path: Path, workflow: dict[str, Any]) -> set[str]:
    """Return only resolved trigger names, never tokens from trigger filters."""
    events = workflow.get("on")
    if events is None:
        return set()
    if isinstance(events, str):
        return {events}
    if isinstance(events, list):
        assert all(isinstance(event, str) for event in events), f"{path} has a non-string workflow event"
        return set(events)
    assert isinstance(events, dict), f"{path} has an unrecognized workflow event declaration"
    assert all(isinstance(event, str) for event in events), f"{path} has a non-string workflow event key"
    return set(events)


def _pull_request_jobs(workflows: Path = WORKFLOWS) -> list[tuple[Path, str, Any]]:
    """Enumerate each resolved job runner in PR-triggered workflows."""
    found = []
    paths = sorted(path for path in workflows.iterdir() if path.is_file() and path.suffix.lower() in {".yml", ".yaml"})
    for path in paths:
        workflow = _load_workflow(path)
        if not _event_names(path, workflow).intersection({"pull_request", "pull_request_target"}):
            continue

        jobs = workflow.get("jobs")
        assert isinstance(jobs, dict), f"{path} accepts pull requests but has no jobs mapping"
        assert jobs, f"{path} accepts pull requests but exposes no jobs to runner policy"
        for job, definition in jobs.items():
            assert isinstance(job, str), f"{path} has a non-string job name"
            assert isinstance(definition, dict), f"{path}:{job} is not a job mapping"
            found.append((path, job, definition.get("runs-on")))
    return found


def test_pull_request_ci_uses_ephemeral_github_hosted_runners() -> None:
    """Public contributions execute repository code. Every job in every
    pull-request workflow must explicitly select the ephemeral GitHub-hosted
    runner, never Avala's persistent private fleet.

    Expressions, matrices, YAML lists and reusable-workflow jobs are rejected
    rather than guessed through. A future exception needs a policy reader that
    can prove its resolved runner, not syntax this test silently overlooks.
    """
    jobs = _pull_request_jobs()
    assert jobs, "no pull-request jobs found"

    unsafe = [(path.relative_to(ROOT), job, runner) for path, job, runner in jobs if runner != "ubuntu-latest"]
    assert not unsafe, f"pull-request jobs without a literal GitHub-hosted runner: {unsafe}"


def test_runner_policy_enumerates_new_jobs_and_workflows(tmp_path) -> None:
    """The old check counted six strings in ci.yml. A seventh private job —
    or a second workflow containing one — left those six strings untouched and
    passed. Enumeration makes both additions visible."""
    workflows = tmp_path / "workflows"
    workflows.mkdir()
    (workflows / "first.yml").write_text(
        """\
on: [push, pull_request]
jobs:
  hosted:
    runs-on: ubuntu-latest
  private:
    runs-on: self-hosted
"""
    )
    (workflows / "second.YAML").write_text(
        """\
on: {pull_request_target: null}
jobs:
  another_private_job:
    runs-on: [self-hosted, linux]
  inline_private: {runs-on: self-hosted}
"""
    )
    (workflows / "indented.yml").write_text(
        """\
on:
    pull_request:
jobs:
  private_after_reindent:
    runs-on: self-hosted
"""
    )

    jobs = _pull_request_jobs(workflows)
    assert [(path.name, job, runner) for path, job, runner in jobs] == [
        ("first.yml", "hosted", "ubuntu-latest"),
        ("first.yml", "private", "self-hosted"),
        ("indented.yml", "private_after_reindent", "self-hosted"),
        ("second.YAML", "another_private_job", ["self-hosted", "linux"]),
        ("second.YAML", "inline_private", "self-hosted"),
    ]


def test_runner_policy_does_not_guess_through_an_expression(tmp_path) -> None:
    """A matrix expression can resolve to self-hosted. Requiring the literal
    hosted value keeps that ambiguity from becoming access to a private host."""
    workflows = tmp_path / "workflows"
    workflows.mkdir()
    (workflows / "matrix.yml").write_text(
        """\
"on":
  pull_request:
jobs:
  test:
    runs-on: ${{ matrix.runner }}
"""
    )

    assert [(job, runner) for _, job, runner in _pull_request_jobs(workflows)] == [("test", "${{ matrix.runner }}")]


def test_runner_policy_resolves_multiline_events_aliases_and_job_merges(tmp_path) -> None:
    """Policy follows the workflow GitHub executes, not its physical lines."""
    workflows = tmp_path / "workflows"
    workflows.mkdir()
    (workflows / "alias.yml").write_text(
        """\
events: &events [
  push,
  pull_request
]
runner: &runner ubuntu-latest
private_runners: &private_runners [self-hosted, linux]
private_job: &private_job
  runs-on: self-hosted
on: *events
jobs:
  direct:
    runs-on: *runner
  list:
    runs-on: *private_runners
  merged:
    <<: *private_job
"""
    )

    assert [(job, runner) for _, job, runner in _pull_request_jobs(workflows)] == [
        ("direct", "ubuntu-latest"),
        ("list", ["self-hosted", "linux"]),
        ("merged", "self-hosted"),
    ]


def test_runner_policy_resolves_multiline_runner_scalars(tmp_path) -> None:
    workflows = tmp_path / "workflows"
    workflows.mkdir()
    (workflows / "multiline.yml").write_text(
        """\
on: pull_request
jobs:
  private:
    runs-on: >-
      ubuntu-latest
      self-hosted
"""
    )

    assert [(job, runner) for _, job, runner in _pull_request_jobs(workflows)] == [
        ("private", "ubuntu-latest self-hosted")
    ]


@pytest.mark.parametrize("branch", ["'*pull_request*'", "pull_request/**"])
def test_runner_policy_does_not_treat_filter_values_as_events(tmp_path, branch) -> None:
    workflows = tmp_path / "workflows"
    workflows.mkdir()
    (workflows / "push.yml").write_text(
        f"""\
on:
  push:
    branches:
      - {branch}
jobs:
  private:
    runs-on: self-hosted
"""
    )

    assert _pull_request_jobs(workflows) == []


def test_runner_policy_rejects_duplicate_yaml_keys(tmp_path) -> None:
    workflows = tmp_path / "workflows"
    workflows.mkdir()
    (workflows / "duplicate.yml").write_text(
        """\
on: pull_request
jobs:
  test:
    runs-on: ubuntu-latest
    runs-on: self-hosted
"""
    )

    with pytest.raises(ValueError, match="duplicate YAML key: 'runs-on'"):
        _pull_request_jobs(workflows)


def test_typescript_writer_is_checked_by_the_python_reference_reader() -> None:
    """A same-language round trip cannot establish wire interoperability."""
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert "tests/conformance/encode_roundtrip.py" in workflow
    assert '--encoder "bun run typescript/conformance/roundtrip.ts"' in workflow


def _readme_implementation_rows() -> set[str]:
    """The directory each row of README's IMPLEMENTATION table claims for itself.

    Scoped to the first table after the `| Implementation |` header, so the
    consumer table below it — whose rows are other repositories and cannot be
    checked from here — is not swept in.
    """
    readme = (ROOT / "README.md").read_text()
    # Located by a whitespace-INSENSITIVE match on the header row. Prettier pads
    # table cells to align them, so an exact-string split on
    # "| Implementation | v1 |" found nothing the moment the formatter ran and
    # this returned an IndexError rather than a readable failure. What the test
    # is about is which directories the table claims, and that does not depend
    # on how wide the columns are.
    header = re.search(r"^\|\s*Implementation\s*\|\s*v1\s*\|.*$", readme, re.MULTILINE)
    assert header is not None, "README has no implementation table header"
    table = readme[header.end() :].split("\n\n", 1)[0]
    return set(re.findall(r"`(python|dart|typescript|rust)/", table))


def _languages_with_a_conformance_runner() -> set[str]:
    """Languages whose conformance runner `ci.yml` actually executes.

    The language is found ANYWHERE inside the runner string, not as its prefix:
    the invocations are `dart run dart/conformance/...` and
    `bun run typescript/conformance/...`, so a prefix match sees only Rust and
    would report Dart and TypeScript as unproven.
    """
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    languages = set()
    for runner in re.findall(r'--runner\s+"([^"]+)"', workflow):
        languages.update(re.findall(r"\b(python|dart|typescript|rust)/", runner))
    # `run.py` with no `--runner` IS the Python reference runner.
    languages.add("python")
    return languages


def test_every_conformance_runner_in_ci_is_listed_as_an_implementation() -> None:
    """A language whose runner CI executes must appear in README's table.

    The Rust implementation passed 95/95 and had its runner wired into `ci.yml`
    while README's status table did not mention Rust at all — so the table
    understated what was proven, and a reader deciding whether v3 was safe got
    the wrong answer from the document written to answer exactly that.

    Checked in this direction because it is the one that goes stale silently:
    adding a language means touching CI (or nothing runs), while the table is
    prose nothing forces you to open.
    """
    wired = _languages_with_a_conformance_runner()
    documented = _readme_implementation_rows()

    missing = sorted(wired - documented)
    assert not missing, (
        f"{missing} pass the conformance suite in CI but are absent from README's "
        f"implementation table. The table is what a consumer reads to decide whether "
        f"v3 is safe to write; an unlisted implementation understates that."
    )


def test_the_readme_table_does_not_claim_an_implementation_ci_never_runs() -> None:
    """The other direction: a listed language must actually be gated.

    Weaker than the check above — a row could be added with no runner and this
    would catch it — but it is the half that stops the table from drifting into
    aspiration.
    """
    unproven = sorted(_readme_implementation_rows() - _languages_with_a_conformance_runner())
    assert not unproven, (
        f"README lists {unproven} as conformance-verified implementations, but no "
        f"conformance runner for them is executed in ci.yml. Either wire the runner "
        f"or move the row to the consumer table."
    )


def test_ci_runs_the_formatter_and_not_only_the_linter() -> None:
    """`ruff check` and `ruff format` are different gates.

    The lint rules were added first and the tree was left unformatted, so the
    formatter could not become a gate until it had been run once. Now that it
    has, the check has to be in CI: formatting drift is invisible in review and
    silent in every other way.
    """
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert "ruff format --check ." in workflow


def test_the_formatter_leaves_markdown_to_prettier() -> None:
    """Ruff reformats Python code blocks inside `.md`, and the guides use
    aligned trailing comments as a column a reader scans down. Collapsing those
    to one space is right for source and wrong for a table, so `.md` is excluded
    — and `force-exclude` makes that hold when a path is named explicitly, which
    is exactly how someone would trip it."""
    config = (ROOT / "ruff.toml").read_text()
    assert "force-exclude = true" in config
    assert "[format]" in config
    assert 'exclude = ["*.md"]' in config


def _pubspec_sdk_floor() -> str:
    """The lowest SDK `dart/alf/pubspec.yaml` promises to work on.

    This intentionally checks the source text: the compact SDK constraint is
    the contract under test, and no other pubspec structure is relevant here.
    """
    pubspec = (ROOT / "dart" / "alf" / "pubspec.yaml").read_text()
    found = re.search(r"^\s*sdk:\s*\^?([0-9]+\.[0-9]+\.[0-9]+)", pubspec, re.MULTILINE)
    assert found is not None, "pubspec declares no SDK constraint"
    return found.group(1)


def _dart_matrix() -> list[str]:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    block = re.search(r"^\s*sdk:\s*\[([^\]]*)\]", workflow, re.MULTILINE)
    assert block is not None, "the dart job declares no SDK matrix"
    return re.findall(r'"([^"]+)"', block.group(1))


def test_ci_exercises_the_dart_sdk_floor_the_package_promises() -> None:
    """`sdk: ^3.9.0` is not documentation — pub.dev enforces it, so a consumer
    on that version can install this package. Testing only the newest SDK
    leaves the promise unverified: a newer-only API would ship green here and
    fail at `pub get` for every consumer at the floor.

    This is the same rule the Python matrix follows (floor plus newest) and the
    one the Dart lane was the only language not following.
    """
    assert _pubspec_sdk_floor() in _dart_matrix()


def test_the_dart_matrix_also_covers_something_newer_than_the_floor() -> None:
    """The other half. A floor-only lane misses breakage on current Dart, which
    is what consumers actually run — the monorepo that vendors this builds it
    on a Dart newer than the floor."""

    def as_version(text: str) -> tuple[int, ...]:
        # Compared as NUMBERS. Lexically "3.9.0" sorts above "3.13.1", so a
        # string max would call the floor the newest and pass on a floor-only
        # matrix — which is the exact thing this test exists to reject.
        return tuple(int(part) for part in text.split("."))

    matrix = _dart_matrix()
    assert len(matrix) >= 2
    assert max(matrix, key=as_version) != _pubspec_sdk_floor()


def test_the_dart_smoke_output_is_not_a_fixed_shared_path() -> None:
    """The runners are persistent and shared, so `/tmp/alf-smoke.js` is one
    concurrent job away from two writers — and with a matrix it is two legs of
    the same job racing on one file."""
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert "/tmp/alf-smoke" not in workflow
    assert "$RUNNER_TEMP/alf-smoke" in workflow
