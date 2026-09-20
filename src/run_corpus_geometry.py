"""Build reproducible legacy and fitted corpus geometries from YAML."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Sequence

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .corpus_geometry import (
    condensed_distances,
    fit_geometry_transform,
    fit_pca_projection,
    legacy_geometry,
    load_feature_rows,
    neighbour_overlap,
    retrieval_summary,
)
from .utils import load_yaml


def _atomic_npz(path: Path, **payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _paths(root: Path, values: Sequence[str]) -> list[Path]:
    return [path if path.is_absolute() else root / path for path in map(Path, values)]


def _row_ids(metadata: dict[str, np.ndarray]) -> np.ndarray:
    labels = np.asarray(metadata["y"]).astype(str)
    if len(np.unique(labels)) == len(labels):
        return labels
    instances = np.asarray(metadata["instance"], dtype=object)
    if np.any(instances != None):  # noqa: E711 - object sentinel comparison
        return np.asarray(
            [
                f"{label}|M{m}|T{t}|I{instance}"
                for label, m, t, instance in zip(
                    labels, metadata["M"], metadata["T"], instances
                )
            ]
        )
    return labels


def _run_study(
    name: str,
    config: dict[str, Any],
    *,
    root: Path,
    output_root: Path,
    random_state: int,
) -> list[dict[str, Any]]:
    reference, reference_meta = load_feature_rows(
        _paths(root, config["reference_artifacts"]),
        matrix_key=str(config.get("matrix_key", "auto")),
    )
    query_paths = config.get("query_artifacts", [])
    if query_paths:
        query, query_meta = load_feature_rows(
            _paths(root, query_paths),
            matrix_key=str(config.get("matrix_key", "auto")),
        )
        independent_query = True
    else:
        query, query_meta = reference, reference_meta
        independent_query = False

    dimensions = int(config.get("pca_dimensions", 50))
    validity = float(config.get("minimum_valid_fraction", 0.95))
    variance = float(config.get("variance_threshold", 1e-8))
    recipes = [str(value) for value in config.get("recipes", ["center"])]
    output_dir = output_root / name
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics: list[dict[str, Any]] = []
    query_representations: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    reference_ids = _row_ids(reference_meta)
    query_ids = _row_ids(query_meta)

    for recipe in recipes:
        if recipe == "legacy":
            combined = (
                np.concatenate((reference, query), axis=0)
                if independent_query
                else reference
            )
            fitted = legacy_geometry(
                combined,
                minimum_feature_valid_fraction=float(
                    config.get("legacy_minimum_feature_valid_fraction", 0.90)
                ),
                minimum_row_valid_fraction=float(
                    config.get("legacy_minimum_row_valid_fraction", 0.80)
                ),
            )
            split = len(reference)
            reference_mask = fitted.row_mask[:split]
            query_mask = (
                fitted.row_mask[split:] if independent_query else fitted.row_mask
            )
            reference_count = int(reference_mask.sum())
            reference_values = fitted.values[:reference_count]
            query_values = (
                fitted.values[reference_count:]
                if independent_query
                else fitted.values
            )
            keep_indices = np.flatnonzero(fitted.feature_mask)
            location = fitted.feature_mean
            scale = fitted.feature_scale
            impute = np.zeros_like(location)
            valid_fraction = np.mean(np.isfinite(combined[:, fitted.feature_mask]), axis=0)
        else:
            transform = fit_geometry_transform(
                reference,
                scaling=recipe,  # type: ignore[arg-type]
                minimum_valid_fraction=validity,
                variance_threshold=variance,
            )
            reference_values = transform.transform(reference)
            query_values = transform.transform(query)
            keep_indices = transform.keep_indices
            location = transform.location
            scale = transform.scale
            impute = transform.impute_values
            valid_fraction = transform.valid_fraction
            reference_mask = np.ones(len(reference), dtype=bool)
            query_mask = np.ones(len(query), dtype=bool)

        pca, reference_scores, query_scores = fit_pca_projection(
            reference_values,
            query_values,
            n_components=dimensions,
            random_state=random_state,
        )
        query_representations[recipe] = (query_ids[query_mask], query_scores)
        distances = condensed_distances(query_scores)
        row: dict[str, Any] = {
            "study": name,
            "recipe": recipe,
            "reference_rows": len(reference),
            "query_rows": len(query),
            "retained_reference_rows": int(reference_mask.sum()),
            "retained_query_rows": int(query_mask.sum()),
            "raw_features": reference.shape[1],
            "retained_features": len(keep_indices),
            "pca_dimensions": query_scores.shape[1],
            "pca_cumulative_variance": float(np.sum(pca.explained_variance_ratio_)),
            "transductive": recipe == "legacy",
        }
        if independent_query:
            row.update(
                retrieval_summary(
                    query_scores,
                    query_meta["y"][query_mask],
                    reference_scores,
                    reference_meta["y"][reference_mask],
                )
            )
        diagnostics.append(row)
        _atomic_npz(
            output_dir / f"{recipe}.npz",
            recipe=np.asarray(recipe),
            reference_names=reference_ids[reference_mask],
            query_names=query_ids[query_mask],
            reference_class=reference_meta["y"][reference_mask],
            query_class=query_meta["y"][query_mask],
            reference_labels=reference_meta["labels"][reference_mask],
            query_labels=query_meta["labels"][query_mask],
            reference_M=reference_meta["M"][reference_mask],
            reference_T=reference_meta["T"][reference_mask],
            query_M=query_meta["M"][query_mask],
            query_T=query_meta["T"][query_mask],
            keep_indices=keep_indices,
            valid_fraction=valid_fraction.astype(np.float32),
            impute_values=impute.astype(np.float32),
            location=location.astype(np.float32),
            scale=scale.astype(np.float32),
            reference_pca=reference_scores.astype(np.float32),
            query_pca=query_scores.astype(np.float32),
            pca_components=pca.components_.astype(np.float32),
            pca_explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32),
            query_distances=distances.astype(np.float32),
            transductive=np.asarray(recipe == "legacy"),
        )

    if len(query_representations) > 1:
        for first_index, first in enumerate(recipes):
            for second in recipes[first_index + 1 :]:
                first_ids, first_values = query_representations[first]
                second_ids, second_values = query_representations[second]
                common, first_index, second_index = np.intersect1d(
                    first_ids, second_ids, return_indices=True
                )
                diagnostics.append(
                    {
                        "study": name,
                        "recipe": f"{first} vs {second}",
                        "neighbour_overlap_15": neighbour_overlap(
                            first_values[first_index],
                            second_values[second_index],
                            k=min(15, len(common) - 1),
                        ),
                        "comparison_rows": len(common),
                    }
                )
    return diagnostics


def run(config_path: str | Path) -> pd.DataFrame:
    path = Path(config_path).resolve()
    config = load_yaml(path)
    root = path.parents[2]
    output_root = Path(config["output_dir"])
    if not output_root.is_absolute():
        output_root = root / output_root
    rows: list[dict[str, Any]] = []
    with threadpool_limits(limits=int(config.get("threads", 1))):
        for name, study in config["studies"].items():
            rows.extend(
                _run_study(
                    str(name),
                    study,
                    root=root,
                    output_root=output_root,
                    random_state=int(config.get("random_state", 1729)),
                )
            )
    table = pd.DataFrame(rows)
    output_root.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_root / "geometry-comparison.csv", index=False)
    (output_root / "geometry-comparison.json").write_text(
        json.dumps(rows, indent=2, allow_nan=False, default=str) + "\n",
        encoding="utf-8",
    )
    return table


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    print(run(args.config).to_string(index=False))


if __name__ == "__main__":
    main()
