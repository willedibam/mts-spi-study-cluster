# Historical compatibility layer

These modules adapt the useful `old/` operations to versioned NPZ artifacts.
Run them from the repository root with `python -m scripts.legacy_compat.<name>`.
They are reproducibility controls, not the default analysis API.

| Old file | Compatibility module | Current equivalent | Material difference |
|---|---|---|---|
| `compute_distance_matrix.py` | same name | `corpus-geometry` | Historical transductive z-score/zero-fill is preserved; output is compact NPZ rather than square CSV. |
| `plot_dataspace.py` | same name | `src.corpus_visualization.plot_embedding` and atlas coordinates | Compatibility module preserves 2-D t-SNE clustering; current clustering is fitted in PCA space. |
| `proof_of_principle.py` | same name | `proof_p90_geometry_comparison.ipynb` | Old UMAP settings remain available; the current proof fits transforms on development and evaluates confirmation. |
| `plot_clusters.py` | same name | `zenodo_1053_geometry.ipynb` | Current plot keeps points neutral and projects low-alpha cluster envelopes. |
| `plot_clustermap.py` | same name | feature-matrix panel in the global notebook | Display-only feature subsampling prevents an unreadable 41,616-column figure. |
| `plot_data.py` | same name | `plot_mts_heatmap` | The old cross-process scaling is preserved as `legacy`; per-process scaling is primary. |
| `plot_nearest_neighbours.py` | same name | `nearest_neighbour_table` | Current focal rows may be algorithmic medoids and retrieval uses the fitted PCA geometry. |
| `utils.py` | reduced helper module | `src/corpus_visualization.py` | The stateful multipurpose `plotter` class was not copied. |

Not ported here:

- `find_discriminating_features.py` and `find_non-overlapping_distributions.py`:
  supervised analysis is intentionally deferred.
- `plot_dendrogram.py`: excluded by scope.
- `get_reduced_feature_set.py`: its K-medoids indices select dataset rows but are
  applied as feature indices. Feature gating and PCA already provide valid
  reduction; a corrected discrete feature-selection study should be separate.
