"""
version_cli.py - Generate version strings

Run inside any Git working copy. By default it inspects HEAD and prints a
friendly version that scheme derived from the last git tag:

    2507               -> exactly at tag v2507 or 2507
    2507.post3         -> 3 commits after that tag
    0.0.0              -> no matching tag found

If you pass the ``--local`` switch the script also appends the commit hash,
forming a *local* version segment:

    2507.post3+g58a226e

CLI examples
------------

    $ python version_cli.py            # auto-detect
    $ python version_cli.py --local    # include +g<hash>
    $ python version_cli.py --tag 28.8 # force base tag
    $ python version_cli.py --help     # see all options

Exit status is 2 if the directory is not a Git repo.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class GitInfo:
    tag: Optional[str]
    distance: int
    node: str

# Accept plain digits or dotted calendar tags, with optional leading "v".
# Examples that match:
#   2507
#   v2507
#   28.8
#   v2024.11.03
_TAG_RE = re.compile(r"^v?(\d[\d.]*)$")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def _run_git(*args: str) -> str:
    """Run *git* and return *stdout* (stripped). Empty string on error."""
    try:
        return (
            subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def probe_git() -> GitInfo | None:
    """Return GitInfo or *None* when not inside a repo."""
    if _run_git("rev-parse", "--is-inside-work-tree") != "true":
        return None

    # "git describe --tags --long" yields: <tag>-<distance>-g<hash>
    desc = _run_git("describe", "--tags", "--long", "--match", "*[0-9]*")

    if desc:
        pieces = desc.rsplit("-", 2)
        if len(pieces) == 3:
            tag, distance_s, ghash = pieces
            distance = int(distance_s)
            node = ghash.lstrip("g")[:7]
        else:
            # Only hash, no tags reachable
            tag, distance, node = None, 0, desc[:7]
    else:
        # Fallback when *describe* produced nothing (e.g. fresh repo)
        tag, distance, node = None, 0, _run_git("rev-parse", "--short=7", "HEAD")

    # Validate tag against our calendar pattern
    if tag and not _TAG_RE.match(tag):
        tag = None
    if tag and tag.startswith("v"):
        tag = tag[1:]

    return GitInfo(tag=tag, distance=distance, node=node or "unknown")


def make_version(info: GitInfo, include_local: bool = False) -> str:
    """Turn GitInfo into the final version string.

    If *include_local* is True and the build is not exactly on the tag, append
    "+g<hash>" as a local version identifier. Otherwise omit the local part.
    """
    base = info.tag or "0.0.0"
    if info.distance == 0:
        return base

    version = f"{base}.post{info.distance}"
    if include_local:
        version += f"+g{info.node}"
    return version

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate version string from Git metadata",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--local", action="store_true", help="Append '+g<hash>' local identifier segment")
    parser.add_argument("--tag", help="Override base tag (e.g. 25.8)")
    parser.add_argument("--distance", type=int, help="Override commit distance from tag")
    parser.add_argument("--node", help="Override commit hash (7 chars)")
    args = parser.parse_args(argv)

    info = probe_git()
    if info is None:
        print("Error: not a Git repository", file=sys.stderr)
        sys.exit(2)

    # Apply overrides
    if args.tag is not None:
        info.tag = args.tag.lstrip("v")
    if args.distance is not None:
        info.distance = args.distance
    if args.node is not None:
        info.node = args.node[:7]

    print(make_version(info, include_local=args.local))


if __name__ == "__main__":
    main()
