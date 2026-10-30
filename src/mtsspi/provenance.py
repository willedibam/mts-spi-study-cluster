"""Identity, atomic writes and resource checks shared by every runner.

One definition each: what code and pyspi produced a result, how a result file
is written so a killed job cannot leave a partial one, and whether the process
has the cores and thread settings its worker count assumes.
"""
from __future__ import annotations

from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np

from .utils import project_root


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.view(np.uint8))
    return digest.hexdigest()


def repository_provenance() -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(project_root()), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(project_root()), "status", "--porcelain"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "unknown", None
    return {"git_commit": commit, "git_dirty": dirty}


def pyspi_commit() -> str:
    """The pyspi commit that was imported: a checkout's HEAD, else the locked git pin."""
    import pyspi

    source = Path(pyspi.__file__).resolve().parent.parent
    try:
        # An install inside this repo's .venv would otherwise report this repo.
        if not (source / ".git").exists():
            raise OSError("not a pyspi checkout")
        commit = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(source), "status", "--porcelain", "--", "pyspi"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
        return commit + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        pass
    try:
        from importlib.metadata import distribution

        dist = distribution("pyspi")
        if Path(dist.locate_file("pyspi")).resolve() == source / "pyspi":
            info = json.loads(dist.read_text("direct_url.json") or "{}")
            return info.get("vcs_info", {}).get("commit_id", "unknown")
    except (PackageNotFoundError, OSError, ValueError):
        pass
    # A snapshot directory under operations/sources/ is named by its commit.
    return "unknown"


def pyspi_version() -> dict[str, str]:
    try:
        dist = version("pyspi")
    except PackageNotFoundError:
        dist = "unknown"
    import pyspi

    computation = pyspi.COMPUTATION_VERSION
    source = str(Path(pyspi.__file__).resolve().parent)
    # `source` separates a run made from a pinned snapshot (PYSPI_SOURCE) from
    # one made with the environment's install at the same computation revision.
    return {"dist": dist, "computation": str(computation), "source": source}


def atomic_savez(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


THREAD_VARIABLES = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
)


def bound_cores() -> int:
    """CPUs this process may run on: its affinity mask where the OS has one."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # macOS
        return os.cpu_count() or 1


def worker_resources(n_jobs: int) -> dict[str, Any]:
    """Refuse a worker count the launcher did not provide for; record what it did.

    The launcher owns placement: it binds this process to ``n_jobs`` cores and
    sets the numerical libraries to one thread each. This checks that it did,
    because more workers than bound cores, or threaded libraries under several
    workers, silently time-share instead of failing.
    """
    cores = bound_cores()
    threads = {name: os.environ.get(name) for name in THREAD_VARIABLES}
    if n_jobs > cores:
        raise RuntimeError(
            f"n_jobs={n_jobs} exceeds the {cores} CPU(s) this process is bound to")
    if n_jobs > 1:
        threaded = {name: value for name, value in threads.items() if value != "1"}
        if threaded:
            raise RuntimeError(
                f"n_jobs={n_jobs} needs single-threaded numerical libraries; "
                f"set to 1: {sorted(threaded)}")
    return {"bound_cores": cores, "thread_variables": threads}


def job_peak_memory_mib(cgroups: Path = Path("/proc/self/cgroup"),
                        mount: Path = Path("/sys/fs/cgroup")) -> float | None:
    """Peak memory of the PBS job this process runs in, from the job's memory cgroup.

    The figure PBS reports as memory used. It covers every process of the job,
    so for a job running one recording it is that recording's peak. ``None``
    outside a PBS job or where the cgroup cannot be read.
    """
    job = os.environ.get("PBS_JOBID")
    try:
        lines = cgroups.read_text().splitlines() if job else []
    except OSError:
        return None
    for line in lines:
        _, controllers, path = line.split(":", 2)
        if job not in path:
            continue
        if "memory" in controllers.split(","):      # cgroup v1 (Gadi)
            file = mount / "memory" / path.lstrip("/") / "memory.max_usage_in_bytes"
        elif not controllers:                         # cgroup v2
            file = mount / path.lstrip("/") / "memory.peak"
        else:
            continue
        try:
            return round(int(file.read_text()) / 2**20, 1)
        except (OSError, ValueError):
            return None
    return None


def peak_memory_mib() -> dict[str, float]:
    """Peak resident size of this process and of its largest finished worker,
    and under PBS the measured peak of the whole job (``job``).

    Size a request from ``job`` of a one-recording job. Without it,
    ``parent + n_jobs * largest_worker`` bounds the recording from above, by
    two to four times: only the worker on an all-channel SPI is that large.
    """
    unit = 1 if sys.platform == "darwin" else 1024  # ru_maxrss: bytes or KiB
    def peak(who: int) -> float:
        return round(resource.getrusage(who).ru_maxrss * unit / 2**20, 1)
    peaks = {"parent": peak(resource.RUSAGE_SELF),
             "largest_worker": peak(resource.RUSAGE_CHILDREN)}
    job = job_peak_memory_mib()
    return peaks if job is None else {**peaks, "job": job}
