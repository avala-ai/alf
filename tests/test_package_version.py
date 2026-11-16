"""The release workflow's version resolution, executed rather than trusted.

`release.yml` has never run — there has been no tagged release — and the check
this feeds guards the one release mistake that cannot be undone: a package
published under a version its own manifest does not claim. A registry version is
immutable, so there is no fixing it after the fact.

It used to be four `grep -m1 '^version = '` invocations inside a YAML string.
That reads the FIRST line-start match in the file and does not know which table
it is in, which matters because `pyproject.toml` opens with `[build-system]` and
a `version` added to any earlier table would silently win. These tests are the
decoys that catch it, and they are only possible because the reading moved out
of the workflow and into a file.
"""

import json
import os
import subprocess
import textwrap
import tomllib
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / ".github" / "scripts" / "package_version.sh"


def resolve(package: str, root: Path | None = None) -> subprocess.CompletedProcess[str]:
    env = {**os.environ}
    if root is not None:
        env["ALF_ROOT"] = str(root)
    return subprocess.run(
        ["bash", str(SCRIPT), package],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


@pytest.fixture
def tree(tmp_path):
    """A manifest tree whose files a test overwrites one at a time."""

    def write(relative: str, text: str) -> None:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(text).lstrip(), encoding="utf-8")

    write("pyproject.toml", '[project]\nname = "alf"\nversion = "1.2.3"\n')
    write("python/alf/__init__.py", '__version__ = "1.2.3"\n')
    write("typescript/alf/package.json", '{"name": "@avala-ai/alf", "version": "1.2.3"}\n')
    write("dart/alf/pubspec.yaml", "name: alf\nversion: 1.2.3\n")
    write("rust/alf/Cargo.toml", '[package]\nname = "alf"\nversion = "1.2.3"\n')
    return types.SimpleNamespace(root=tmp_path, write=write)


@pytest.mark.parametrize("package", ["python", "typescript", "dart", "rust"])
def test_it_reads_the_real_manifests(package):
    """Against the repository's own files, not only fixtures. A resolver that
    works on invented input and not on the tree it ships with is worse than no
    resolver, because the failure arrives at a tag."""
    done = resolve(package)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip(), f"{package} resolved to nothing"


def test_every_package_the_release_workflow_can_tag_is_resolvable():
    """The tag scheme names five namespaces and four of them are packages. A
    package the workflow can tag but this cannot read is a release that stops
    at its first step."""
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text()
    for package in ["python", "typescript", "dart", "rust"]:
        assert f"releases/{package}/" in workflow
        assert resolve(package).returncode == 0


def test_registry_coordinates_do_not_claim_unrelated_alf_projects():
    """The bare names are occupied on PyPI and crates.io. Keep the published
    distribution coordinates distinct while preserving both source APIs."""
    python = tomllib.loads((ROOT / "pyproject.toml").read_text())
    rust = tomllib.loads((ROOT / "rust/alf/Cargo.toml").read_text())
    workflow = (ROOT / ".github/workflows/release.yml").read_text()

    assert python["project"]["name"] == "avala-alf"
    assert rust["package"]["name"] == "avala-alf"
    assert rust["lib"]["name"] == "alf"
    assert "pypi/avala-alf/$version/json" in workflow
    assert "crates/avala-alf/$version" in workflow


def test_rust_crate_is_publishable_for_the_documented_bootstrap():
    """The first crates.io release is manual, but it still comes from this
    manifest. Leaving Cargo's publish kill switch in place makes the documented
    bootstrap command impossible and is otherwise discovered only after every
    release check has passed."""
    rust = tomllib.loads((ROOT / "rust/alf/Cargo.toml").read_text())

    assert rust["package"].get("publish") is not False


def test_nested_registry_packages_ship_the_repository_notice():
    """Python packages from the repository root and already includes NOTICE.
    The other registry roots are nested, so each needs the same notice inside
    its own artifact; npm's files allowlist must opt it in explicitly."""
    notice = (ROOT / "NOTICE").read_text()

    for package_root in ["dart/alf", "rust/alf", "typescript/alf"]:
        assert (ROOT / package_root / "NOTICE").read_text() == notice

    npm = json.loads((ROOT / "typescript/alf/package.json").read_text())
    assert "NOTICE" in npm["files"]


def test_npm_manifest_identifies_the_trusted_publishers_repository():
    """npm's GitHub trusted publisher requires `repository.url` to match the
    repository exactly. Without it every earlier release gate passes and the
    irreversible publish step is the first place OIDC authentication fails."""
    manifest = json.loads((ROOT / "typescript/alf/package.json").read_text())

    assert manifest["repository"] == {
        "type": "git",
        "url": "git+https://github.com/avala-ai/alf.git",
        "directory": "typescript/alf",
    }


def test_npm_bootstrap_builds_the_archive_and_cannot_default_to_private():
    """The first scoped npm release is interactive. A clean checkout has no
    tracked ``dist`` directory, and a scoped package can otherwise default to
    restricted access. Make both safety properties part of the manifest so
    they hold for every pack/publish invocation, not only one documentation
    example."""
    manifest = json.loads((ROOT / "typescript/alf/package.json").read_text())

    assert manifest["publishConfig"]["access"] == "public"
    assert manifest["scripts"]["prepack"] == "bun run build"


def test_manual_bootstrap_commands_use_the_package_roots():
    """The repository root has none of the nested registry manifests. The
    bootstrap instructions must enter each package and must build npm's
    untracked distribution before inspecting or publishing it."""
    instructions = (ROOT / "RELEASING.md").read_text()

    assert "pre-publish steps 1-5" in instructions
    assert "cd typescript/alf" in instructions
    assert "bun install --frozen-lockfile" in instructions
    assert "bun run build" in instructions
    assert "     npm pack\n     sha256sum avala-ai-alf-0.1.0.tgz" in instructions
    assert "npm publish --dry-run --access public" not in instructions
    assert "npm publish avala-ai-alf-0.1.0.tgz --access public" in instructions
    assert "cd dart/alf" in instructions
    assert "dart pub publish --dry-run" in instructions
    assert "cd rust/alf" in instructions
    assert "cargo publish --dry-run" in instructions


def test_a_version_in_an_earlier_table_does_not_win(tree):
    """THE DEFECT THIS PREVENTS. `pyproject.toml` opens with `[build-system]`,
    so a first-match grep is one added line away from reading a version that is
    not the project's — and the tag would be compared against that."""
    tree.write(
        "pyproject.toml",
        """
        [build-system]
        requires = ["setuptools>=84"]
        version = "9.9.9"

        [project]
        name = "alf"
        version = "1.2.3"
        """,
    )
    assert resolve("python", tree.root).stdout.strip() == "1.2.3"


def test_a_workspace_version_does_not_win_over_the_package_one(tree):
    """The same shape in Cargo.toml, which is where it is most likely: a
    workspace stanza above `[package]` is ordinary Rust layout."""
    tree.write(
        "rust/alf/Cargo.toml",
        """
        [workspace.package]
        version = "9.9.9"

        [package]
        name = "alf"
        version = "1.2.3"
        """,
    )
    assert resolve("rust", tree.root).stdout.strip() == "1.2.3"


def test_a_dependency_version_does_not_win(tree):
    """A `version` nested under a dependency is not the package's. YAML says so
    by indentation and JSON by nesting; both were reachable by a grep."""
    tree.write(
        "dart/alf/pubspec.yaml",
        """
        name: alf
        dependencies:
          archive:
            version: 9.9.9
        version: 1.2.3
        """,
    )
    assert resolve("dart", tree.root).stdout.strip() == "1.2.3"
    tree.write(
        "typescript/alf/package.json",
        '{"name": "@avala-ai/alf", "dependencies": {"x": {"version": "9.9.9"}}, "version": "1.2.3"}',
    )
    assert resolve("typescript", tree.root).stdout.strip() == "1.2.3"


def test_pythons_two_declarations_must_agree(tree):
    """`alf.__version__` is what a user quotes in a bug report and what an
    installed package answers with; the manifest is what the registry records.
    A release where they disagree ships a package that misreports itself, and
    nothing downstream notices."""
    tree.write("python/alf/__init__.py", '__version__ = "9.9.9"\n')
    done = resolve("python", tree.root)
    assert done.returncode != 0
    assert "1.2.3" in done.stderr and "9.9.9" in done.stderr


def test_the_repositorys_own_two_python_declarations_agree():
    """The same check, against the tree as it stands — so a bump that updates
    one and forgets the other fails in the pull request rather than at the tag,
    where the only remedy is a version nobody wanted to skip."""
    assert resolve("python").returncode == 0


@pytest.mark.parametrize(
    ("package", "relative"),
    [
        ("python", "pyproject.toml"),
        ("typescript", "typescript/alf/package.json"),
        ("dart", "dart/alf/pubspec.yaml"),
        ("rust", "rust/alf/Cargo.toml"),
    ],
)
def test_a_missing_manifest_stops_with_a_named_file(tree, package, relative):
    (tree.root / relative).unlink()
    done = resolve(package, tree.root)
    assert done.returncode != 0
    assert relative.rsplit("/", 1)[-1] in done.stderr


def test_a_manifest_without_a_version_stops(tree):
    tree.write("rust/alf/Cargo.toml", '[package]\nname = "alf"\n')
    done = resolve("rust", tree.root)
    assert done.returncode != 0
    assert "no version" in done.stderr


@pytest.mark.parametrize("package", ["", "nonsense", "corpus"])
def test_an_unknown_package_stops_rather_than_guessing(package):
    """`corpus` is in the tag scheme and is deliberately NOT a package: it
    publishes to no registry, so there is no version for this to read."""
    assert resolve(package).returncode == 2


def test_every_release_job_routes_through_the_script():
    """Four copies of this logic is how it came to be four different parsers.
    One implementation, called from each job, is the property that keeps the
    decoy tests above meaningful."""
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text()
    assert workflow.count("package_version.sh") >= 4
    # And no job carries a hand-rolled copy any more.
    assert "grep -m1 '^version" not in workflow
    assert "require('./package.json').version" not in workflow


def test_the_script_is_executable():
    assert os.access(SCRIPT, os.X_OK)


def test_pull_request_ci_builds_every_registry_artifact():
    """The release workflow has never run. Pull requests must build each
    registry payload so a first tag is testing credentials and publication,
    not discovering that the package itself cannot be assembled."""
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()
    typescript_job = workflow.split("\n  typescript:\n", 1)[1].split("\n  rust:\n", 1)[0]

    assert "python -m build" in workflow
    assert 'required_wheel = {"alf/__init__.py", "alf/_guard.py", "alf/_io.py"}' in workflow
    assert "dart pub publish --dry-run" in workflow
    assert "actions/setup-node@v7" in typescript_job
    assert 'node-version: "22"' in typescript_job
    assert "npm install -g npm@12.0.2" in typescript_job
    assert "npm pack --dry-run --json" in typescript_job
    assert "cargo package --list" in workflow
    for expected in ["dist/index.js", "dist/index.d.ts", "LICENSE", "NOTICE", "README.md"]:
        assert expected in workflow
