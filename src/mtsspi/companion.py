"""Join a companion corpus's SPIs onto a primary corpus, recording by recording.

A companion corpus runs an extra SPI set over the recordings of an existing
corpus, from its own corpus YAML into its own output directory, so the primary
corpus is never extended or re-run. ``configs/pyspi/subsets/phasealigned.yaml``
is the first such set. The join checks that

- the two corpora hold the same recordings (the same relative directories), and
  each pair is the same data: the archive member hash, or the bytes of
  ``timeseries.npy`` for a generated corpus, with equal ``M``, ``T`` and
  ``normalise``;
- the SPI names of the two are disjoint;
- within each corpus one pyspi config and one computation version made every
  recording, and across the two the computation version is the same unless
  ``allow_mixed_computation=True``. Disjoint SPI sets keep each SPI's values
  from one version, so the opt-in mixes versions between SPIs, never within one;
  the versions of both sides are in ``provenance`` for the analysis to record.

    join = join_companion(p90_root, phasealigned_root)
    for primary_dir, companion_dir in join.pairs:
        mpis = load_joined_mpis(primary_dir, companion_dir)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .provenance import file_sha256
from .utils import load_json


@dataclass(frozen=True)
class CompanionJoin:
    pairs: list[tuple[Path, Path]]
    provenance: dict[str, Any]


def _spi_names(meta: dict[str, Any]) -> list[str]:
    return [spi["name"] for spi in meta["pyspi"]["spis"]]


def _recordings(root: Path) -> dict[str, tuple[Path, dict[str, Any]]]:
    found = {}
    for meta_path in sorted(root.rglob("meta.json")):
        meta = load_json(meta_path)
        if meta.get("status") != "complete":
            raise ValueError(f"{meta_path.parent}: recording is not complete")
        found[meta_path.parent.relative_to(root).as_posix()] = (meta_path.parent, meta)
    if not found:
        raise FileNotFoundError(f"no recordings under {root}")
    return found


def _data_identity(directory: Path, meta: dict[str, Any]) -> tuple[str, str]:
    member = (meta.get("source") or {}).get("member_sha256")
    if member:
        return "archive member", member
    series = directory / "timeseries.npy"
    if series.is_file():
        return "timeseries.npy", file_sha256(series)
    raise ValueError(f"{directory}: no archive member hash or timeseries.npy identifies its data")


def join_companion(
    primary: str | Path, companion: str | Path, *, allow_mixed_computation: bool = False
) -> CompanionJoin:
    """Pair the recordings of two corpus output roots, refusing anything but the same data."""
    roots = {"primary": Path(primary), "companion": Path(companion)}
    sides = {side: _recordings(root) for side, root in roots.items()}
    unmatched = sorted(set(sides["primary"]) ^ set(sides["companion"]))
    if unmatched:
        raise ValueError(f"{len(unmatched)} recording(s) in only one corpus, e.g. {unmatched[:3]}")

    provenance: dict[str, Any] = {}
    for side, recordings in sides.items():
        catalogues = {
            (meta["pyspi"].get("config_sha256"), meta["pyspi"]["version"]["computation"])
            for _, meta in recordings.values()
        }
        if len(catalogues) != 1:
            raise ValueError(f"{side} corpus mixes pyspi configs or computation versions: "
                             f"{sorted(catalogues, key=str)}")
        ((config_sha256, computation),) = catalogues
        provenance[side] = {"root": str(roots[side]), "config_sha256": config_sha256,
                            "computation": computation, "recordings": len(recordings)}
    mixed = provenance["primary"]["computation"] != provenance["companion"]["computation"]
    if mixed and not allow_mixed_computation:
        raise ValueError(
            f"computation versions differ: primary {provenance['primary']['computation']}, "
            f"companion {provenance['companion']['computation']}; pass "
            f"allow_mixed_computation=True only if the analysis records both")
    provenance["mixed_computation"] = mixed

    pairs = []
    for key in sorted(sides["primary"]):
        (primary_dir, primary_meta), (companion_dir, companion_meta) = (
            sides["primary"][key], sides["companion"][key])
        for field in ("M", "T", "normalise"):
            if primary_meta.get(field) != companion_meta.get(field):
                raise ValueError(f"{key}: {field} differs between the corpora")
        if _data_identity(primary_dir, primary_meta) != _data_identity(companion_dir, companion_meta):
            raise ValueError(f"{key}: the corpora hold different data")
        shared = set(_spi_names(primary_meta)) & set(_spi_names(companion_meta))
        if shared:
            raise ValueError(f"{key}: SPI(s) in both corpora: {sorted(shared)[:5]}")
        pairs.append((primary_dir, companion_dir))
    return CompanionJoin(pairs, provenance)


def load_joined_mpis(primary_dir: str | Path, companion_dir: str | Path) -> dict[str, np.ndarray]:
    """One recording's matrices: the primary's SPIs in stored order, then the companion's."""
    joined: dict[str, np.ndarray] = {}
    for directory in map(Path, (primary_dir, companion_dir)):
        names = _spi_names(load_json(directory / "meta.json"))
        with np.load(directory / "spi_mpis.npz", allow_pickle=False) as archive:
            if archive.files != names:
                raise ValueError(f"{directory}: SPI archive does not match its catalogue")
            for name in names:
                if name in joined:
                    raise ValueError(f"SPI {name!r} in both corpora")
                joined[name] = archive[name]
    return joined
