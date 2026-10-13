"""The farm submitters only run from committed, pushed, current source."""
import os
import subprocess
from pathlib import Path

GUARD = Path(__file__).resolve().parents[1] / "jobs/gadi/require_synced_source.sh"


def _git(cwd, *args):
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
                   cwd=cwd, check=True, capture_output=True)


def _guard(cwd, **env):
    return subprocess.run(["bash", "-c", f"source {GUARD}; echo submitted"], cwd=cwd,
                          capture_output=True, text=True, env={**os.environ, **env})


def test_guard_refuses_dirty_unpushed_and_stale_checkouts(tmp_path):
    remote, work, other = tmp_path / "remote.git", tmp_path / "work", tmp_path / "other"
    _git(tmp_path, "init", "--bare", "-b", "main", str(remote))
    _git(tmp_path, "clone", str(remote), str(work))
    (work / "runner.py").write_text("a\n")
    _git(work, "add", "runner.py")
    _git(work, "commit", "-m", "one")

    unpushed = _guard(work)
    assert unpushed.returncode == 2 and "no remote branch" in unpushed.stderr
    _git(work, "push", "-u", "origin", "HEAD:main")
    assert _guard(work).stdout.strip() == "submitted"

    (work / "notes.txt").write_text("untracked files do not matter\n")
    assert _guard(work).returncode == 0
    (work / "runner.py").write_text("b\n")
    dirty = _guard(work)
    assert dirty.returncode == 2 and "uncommitted" in dirty.stderr
    assert _guard(work, ALLOW_UNSYNCED_SOURCE="1").stdout.strip() == "submitted"
    _git(work, "checkout", "runner.py")

    _git(tmp_path, "clone", str(remote), str(other))
    (other / "runner.py").write_text("c\n")
    _git(other, "commit", "-am", "two")
    _git(other, "push", "origin", "HEAD:main")
    stale = _guard(work)
    assert stale.returncode == 2 and "1 commits behind" in stale.stderr


def test_guard_passes_a_source_snapshot(tmp_path):
    # GIT_CEILING_DIRECTORIES: the temporary directory may sit inside a checkout.
    result = _guard(tmp_path, GIT_CEILING_DIRECTORIES=str(tmp_path.parent))
    assert result.stdout.strip() == "submitted"
