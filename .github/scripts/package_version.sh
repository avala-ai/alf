#!/usr/bin/env bash
#
# The version a package DECLARES, read one way for all of them.
#
# WHY THIS IS NOT FOUR GREPS IN A WORKFLOW. It used to be, and the check it
# feeds guards the one release mistake that cannot be undone: a package
# published under a version its own manifest does not claim. A registry version
# is immutable, so there is no fixing it afterwards — and the reading was done
# by `grep -m1 '^version = '`, which takes the FIRST line-start match in the
# file and does not know what section it is in.
#
# That is one line away from wrong. `pyproject.toml` opens with `[build-system]`
# and `Cargo.toml` with `[package]`; a `version = ` added to an earlier table —
# a `[tool.*]` block, a workspace stanza — silently wins, and the release
# publishes under whatever it said. Nothing in the workflow would notice,
# because the tag would be compared against the wrong number and could easily
# match it.
#
# So the TOML readers are section-aware, and this file is tested against both
# the real manifests and against decoys, which four copies inside a YAML string
# could never be.
#
#   package_version.sh python      # pyproject.toml [project], cross-checked
#                                  # against python/alf/__init__.py
#   package_version.sh typescript  # typescript/alf/package.json
#   package_version.sh dart        # dart/alf/pubspec.yaml
#   package_version.sh rust        # rust/alf/Cargo.toml [package]

set -euo pipefail

root=${ALF_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}

die() {
  echo "::error::$*" >&2
  exit 1
}

# `version = "..."` from ONE named table, ignoring every other table's.
toml_version() {
  local file=$1 section=$2 found
  [ -f "$file" ] || die "$file does not exist"
  found=$(awk -v want="$section" '
    # A table header ends the previous table, whatever it was.
    /^[[:space:]]*\[/ {
      inside = ($0 ~ "^[[:space:]]*\\[" want "\\][[:space:]]*$")
      next
    }
    inside && /^[[:space:]]*version[[:space:]]*=/ {
      if (match($0, /"[^"]*"/)) {
        print substr($0, RSTART + 1, RLENGTH - 2)
        exit
      }
    }
  ' "$file")
  [ -n "$found" ] || die "$file has no version in [$section]"
  printf '%s\n' "$found"
}

# A YAML root key: column zero, so an indented `version:` under a dependency
# cannot be mistaken for the package's own.
yaml_root_version() {
  local file=$1 found
  [ -f "$file" ] || die "$file does not exist"
  found=$(awk '/^version:[[:space:]]/ { print $2; exit }' "$file")
  [ -n "$found" ] || die "$file has no root-level version"
  printf '%s\n' "$found"
}

json_version() {
  local file=$1 found
  [ -f "$file" ] || die "$file does not exist"
  # A parser, not a grep: `"version"` also appears in dependency specifiers and
  # in any nested object, and only the top-level one is the package's.
  found=$(python3 -c '
import json, sys
with open(sys.argv[1], encoding="utf-8") as handle:
    print(json.load(handle).get("version", ""))
' "$file")
  [ -n "$found" ] || die "$file has no top-level version"
  printf '%s\n' "$found"
}

# The same anchored, quoted read the TOML side uses, for a Python source file.
python_dunder_version() {
  local file=$1 found
  [ -f "$file" ] || die "$file does not exist"
  found=$(awk '/^__version__[[:space:]]*=/ {
    if (match($0, /"[^"]*"/)) {
      print substr($0, RSTART + 1, RLENGTH - 2)
      exit
    }
  }' "$file")
  [ -n "$found" ] || die "$file has no __version__"
  printf '%s\n' "$found"
}

case ${1-} in
  python)
    manifest=$(toml_version "$root/pyproject.toml" "project")
    # TWO declarations, and they must agree. `alf.__version__` is what a user
    # reports in a bug and what a pinned install resolves to; the manifest is
    # what the registry records. A release where they disagree ships a package
    # that misreports itself, and the disagreement is invisible until someone
    # asks the installed package what it is.
    source=$(python_dunder_version "$root/python/alf/__init__.py")
    [ "$manifest" = "$source" ] ||
      die "pyproject.toml says $manifest and python/alf/__init__.py says $source"
    printf '%s\n' "$manifest"
    ;;
  typescript) json_version "$root/typescript/alf/package.json" ;;
  dart) yaml_root_version "$root/dart/alf/pubspec.yaml" ;;
  rust) toml_version "$root/rust/alf/Cargo.toml" "package" ;;
  *)
    sed -n '/^#   package_version.sh/,/^$/p' "$0" >&2
    exit 2
    ;;
esac
