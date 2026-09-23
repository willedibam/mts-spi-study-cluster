# Gadi storage layout

Use `/scratch/ql44/we2614/mts-spi-study` for working storage and `/g/data/ql44/we2614/mts-spi-study` for retained outputs. These roots share a naming scheme, not identical contents. A link to Scratch does not make its target durable. The project README and `documentation/storage-index.csv` identify physical locations; `documentation/path-map.json` resolves historical paths without keeping old root aliases.

## Placement

| Path under the project | Contents |
| --- | --- |
| `proof/` | Shared multi-system proof input banks and proof-specific features |
| `order-parameter-inference/<system>/` | CML2D, quadratic CML, Miller–Huse, Stuart–Landau, Kuramoto, Desai–Zwanzig, Vicsek and finite-regime studies |
| `zenodo/7118947/` | Zenodo inputs, authoritative seed1729 MPI bank and features |
| `representation/` | Synthetic and empirical representation studies, cross-M/T, model comparisons |
| `pyspi-optimisation/large-m-260917/` | Large-M estimator/runtime and pair-baseline work |
| `archives/<workstream>/` | Verified inactive scientific archives, physically on gdata |
| `operations/{sources,logs,maintenance}/` | Frozen source worktrees, execution records and storage-maintenance evidence |
| `environments/` | One shared uv-created MTS environment, physically on Scratch; checkout `.venv` links reference it |
| `dev/<experiment>/` | Disposable named experiments on Scratch; each needs a purpose and retirement condition |
| `legacy-non-p90/` | Existing nonconforming outputs flagged for review, not a destination for new runs |
| `documentation/` | Storage index, historical path map and audit evidence |

EEML, Baseten and TUSZ are sibling projects. Each owns its own archives and environment if needed. Shared machine caches live in `.cache/`; active editor/runtime temporary files may remain in `tmp/`. Neither is a scientific workstream.

## Rules for people and agents

1. Read this document and the relevant `docs/context/INDEX.md` entry before cluster work. Consult the storage index before scanning an entire filesystem or creating a new folder.
2. Put each run under its workstream, then its role: `data/` or existing `runs/`, `features/`, `analysis/`, `models/`, `configs/`. Preserve pilot, development, confirmation and control distinctions. Create a new workstream or subfolder when its purpose differs; these categories are examples, not a closed schema. Prefer a simple move over additional navigation links or custom migration machinery.
3. Give a run a stable descriptive ID with a date. Record purpose, generator/config and code revision, seed, input dataset IDs, SPI catalogue hash, output location and whether it extends or supersedes another run. Existing metadata remain authoritative; do not rewrite them merely to change storage paths.
4. Keep one physical copy of each artifact. Use explicit shared-input paths or the index for cross-workstream access. Avoid blanket mirrors and links back and forth between the two stores. Each intentional link must resolve directly to its canonical target.
5. Launch with an explicit output path inside the project. Production SPIs use `configs/pyspi/benchmarked_p90.yaml`; alternate catalogues require an explicit experiment decision and separate labelling. Physics-only validation is not an SPI catalogue violation.
6. Use the shared MTS uv environment. Record installed versions and editable-source revisions before changing dependencies; do not create a new environment for each source snapshot. Frozen code and a shared evolving environment do not by themselves guarantee historical reproducibility.
7. Retain scientific failure evidence and useful execution provenance. For inactive runs, archive only after a retirement decision, compare archive contents before removing loose files, record a checksum and update the index. Keep supersession separate from byte duplication.
8. Before submitting a large run, check inode and byte quota and run `python scripts/check_gadi_storage_layout.py`. After a run or move, update its index entry and check again. The check is read-only; it never deletes or moves files.
9. Keep operational scripts in Git. Historical snapshots may contain old paths: resolve them with `python scripts/gadi_storage_path.py OLD_PATH`, then use a current launcher or a separately recorded replay configuration. Never edit frozen metadata or snapshot source in place to disguise a historical path change.

A directory layout cannot prevent clutter by itself. The enforceable habits are explicit output paths, stable run identities, a small maintained index, one physical owner, and a completion/retirement decision for every experiment. Do not create periodic cleanup automation that deletes scientific data without an explicit retention policy.

Keep source checkouts only in `operations/sources/`. Do not add per-system source links or checkout `data` links back to the entire project: these create recursive, misleading folder views. New scientifically appropriate workstream names are allowed without changing the validator.
