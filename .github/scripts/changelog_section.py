#!/usr/bin/env python3
"""Print one version's section of a changelog, or fail.

Used by `release.yml` before anything is built. A tag that publishes a package
and leaves behind a release nobody can read the reason for is how a changelog
and a registry drift apart, so the section is required rather than optional:
this exits non-zero when it cannot find one, and it runs early enough that the
release can still be abandoned.

Sections are `## <version>` optionally followed by anything else on the line, so
`## 0.1.0 — 2026-08-20` and a bare `## 0.1.0` both match. The section runs to the
next `## ` heading or the end of the file.

    changelog_section.py 0.1.0                       # CHANGELOG.md
    changelog_section.py 0.1.0 dart/alf/CHANGELOG.md
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


def section(text: str, version: str) -> str | None:
    """The body under `## <version>`, or None when there is no such heading."""
    # The version is matched literally — `re.escape`, because `0.1.0`'s dots
    # would otherwise be wildcards and `0.1.0` would match a `0X1X0` heading.
    #
    # `\b` is NOT enough to end it. Both `-` and `.` are non-word characters, so
    # a word boundary sits between `0.1.0` and the `-` of `0.1.0-rc.1` — and a
    # changelog lists newest first, which puts the prerelease heading ABOVE the
    # stable one. A stable tag would have found the prerelease section, passed
    # the not-unreleased check against that heading, and published its notes.
    # The version has to be followed by whitespace or the end of the line.
    pattern = re.compile(
        rf"^##\s+{re.escape(version)}(?=\s|$).*?$(.*?)(?=^##\s|\Z)",
        re.DOTALL | re.MULTILINE,
    )
    found = pattern.search(text)
    if found is None:
        return None
    body = found.group(1).strip()
    return body or None


def main(argv: list[str]) -> int:
    if not 2 <= len(argv) <= 3:
        print(__doc__, file=sys.stderr)
        return 2

    version = argv[1]
    path = Path(argv[2] if len(argv) == 3 else "CHANGELOG.md")

    if not path.is_file():
        print(f"error: {path} does not exist", file=sys.stderr)
        return 1

    text = path.read_text(encoding="utf-8")
    body = section(text, version)
    if body is None:
        print(
            f"error: {path} has no non-empty '## {version}' section. "
            f"A release with nothing written about it should not reach a registry.",
            file=sys.stderr,
        )
        return 1

    # A heading still marked unreleased is a section nobody has finished. It
    # would otherwise publish cleanly and leave the changelog claiming the
    # version is not out, which is worse than no section at all — the file that
    # is wrong is the one a reader checks to find out what shipped.
    heading = re.search(rf"^##\s+{re.escape(version)}(?=\s|$).*$", text, re.MULTILINE)
    if heading is not None and "unreleased" in heading.group(0).lower():
        print(
            f"error: {path}'s '{heading.group(0).strip()}' is still marked unreleased. "
            f"Date the heading before tagging.",
            file=sys.stderr,
        )
        return 1

    print(body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
