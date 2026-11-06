"""Say whether the pinned pyspi is behind upstream main, in one or two lines.

    python scripts/check_pyspi_pin.py

Run at the start of each Claude Code session (.claude/settings.json). Needs
network; on failure it says so and exits 0. Bumping the pin: HPC.md, "pyspi".
Uses `gh api` when available: it trusts the system keychain, so it works behind
TLS-inspecting proxies (Zscaler) where Python's certificate bundle does not.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import subprocess
import tomllib
import urllib.request

REPO = "DynamicsAndNeuralSystems/pyspi"
TIMEOUT = 5


def _get(path: str, accept: str = "application/vnd.github+json") -> bytes:
    """GET a GitHub API path, through `gh` if it works, else directly."""
    if shutil.which("gh"):
        try:
            return subprocess.run(["gh", "api", "-H", f"Accept: {accept}", path],
                                  capture_output=True, check=True, timeout=TIMEOUT).stdout
        except subprocess.SubprocessError:
            pass
    request = urllib.request.Request(f"https://api.github.com/{path}",
                                     headers={"User-Agent": "mtsspi-pin-check", "Accept": accept})
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        return response.read()


def _computation(ref: str) -> str:
    source = _get(f"repos/{REPO}/contents/pyspi/_parallel.py?ref={ref}", "application/vnd.github.raw").decode()
    match = re.search(r'^COMPUTATION_VERSION = "([^"]+)"', source, re.M)
    return match.group(1).rsplit(".", 1)[-1] if match else "?"


def main() -> None:
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    pin = tomllib.loads(pyproject.read_text())["tool"]["uv"]["sources"]["pyspi"]["rev"]
    try:
        compare = json.loads(_get(f"repos/{REPO}/compare/{pin}...main"))
        head = compare["commits"][-1]["sha"] if compare["commits"] else pin
        ahead, behind = compare["ahead_by"], compare["behind_by"]
        pinned, latest = _computation(pin), _computation(head)
    except Exception as exc:  # noqa: BLE001 - a failed check must never block a session
        print(f"pyspi pin {pin[:7]}: upstream check failed ({type(exc).__name__}).")
        return
    if behind:
        print(f"pyspi pin {pin[:7]} ({pinned}) is not on upstream main; pin a merged commit.")
    if not ahead:
        print(f"pyspi pin {pin[:7]} ({pinned}) is upstream main.")
        return
    moved = f"computation {pinned} -> {latest}" if latest != pinned else f"computation still {pinned}"
    print(f"pyspi pin {pin[:7]} is {ahead} commit(s) behind upstream main {head[:7]} ({moved}). "
          f"New: {'; '.join(_merged(compare['commits']))}. Consider bumping (HPC.md, \"pyspi\").")


def _merged(commits: list[dict]) -> list[str]:
    """The pull requests merged in these commits, newest first, else the last commits."""
    titles = []
    for commit in reversed(commits):
        lines = [line for line in commit["commit"]["message"].splitlines() if line.strip()]
        merge = re.match(r"Merge pull request (#\d+)", lines[0])
        if merge:
            titles.append(f"{merge.group(1)} {lines[1] if len(lines) > 1 else ''}".strip())
        elif re.search(r"\(#\d+\)$", lines[0]):
            titles.append(lines[0])
    return titles[:8] or [c["commit"]["message"].splitlines()[0] for c in commits[-3:]]


if __name__ == "__main__":
    main()
