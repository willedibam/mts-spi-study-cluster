"""Say in one line what work is not yet safe in main; print nothing when there is none.

    python scripts/check_loose_ends.py

Run at the start of each Claude Code session (.claude/settings.json), on either
Claude account: git is the only state the accounts share. Reports branches with
changes not in origin/main (marked * if not pushed), worktrees with uncommitted
files, unpushed commits on main and stale worktree records. A branch named in
docs/context/*.md is recorded there (set aside or open on purpose) and is only
counted. Local refs only, no fetch.
"""
from __future__ import annotations

from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent.parent


def git(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True).stdout.strip()


def main() -> None:
    base = "origin/main" if git("rev-parse", "-q", "--verify", "origin/main") else "main"
    noted_text = "\n".join(p.read_text() for p in (ROOT / "docs/context").glob("*.md"))
    local = set(git("for-each-ref", "--format=%(refname:short)", "refs/heads").split())
    remote = {r.removeprefix("origin/") for r in
              git("for-each-ref", "--format=%(refname:short)", "refs/remotes/origin").split()}
    unmerged, noted = [], 0
    for branch in sorted((local | remote) - {"main", "HEAD", "origin"}):
        ref = branch if branch in local else f"origin/{branch}"
        # `git cherry` counts changes, so squashed or cherry-picked work counts as merged.
        n = sum(line.startswith("+") for line in git("cherry", base, ref).splitlines())
        if not n:
            continue
        if branch in noted_text:
            noted += 1
            continue
        pushed = branch in remote and not git("rev-list", f"origin/{branch}..{branch}") if branch in local else True
        unmerged.append(f"{branch.removeprefix('claude/')} {n}{'' if pushed else '*'}")

    dirty, stale = [], 0
    for block in git("worktree", "list", "--porcelain").split("\n\n"):
        fields = dict(line.split(" ", 1) if " " in line else (line, "") for line in block.splitlines())
        if "prunable" in fields:
            stale += 1
        elif "worktree" in fields and Path(fields["worktree"]).is_dir():
            n = len(git("status", "--porcelain", cwd=Path(fields["worktree"])).splitlines())
            if n:
                path = Path(fields["worktree"])
                dirty.append(f"{'main' if path == ROOT else path.name} {n}")
    ahead = git("rev-list", "--count", f"{base}..main") if base != "main" else "0"

    parts = []
    if unmerged:
        parts.append("unmerged " + ", ".join(unmerged) + (f" (+{noted} noted)" if noted else ""))
    if dirty:
        parts.append("uncommitted " + ", ".join(dirty))
    if ahead not in ("", "0"):
        parts.append(f"main {ahead} unpushed")
    if stale:
        parts.append(f"{stale} stale worktree record(s) (git worktree prune)")
    if parts:
        print("Loose ends: " + "; ".join(parts) + ".")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 - a failed check must never block a session
        print(f"Loose-ends check failed ({type(exc).__name__}).")
