# USYD Physics headnode context

Verified live on 2026-08-25. This is the USYD Physics PBS cluster, not NCI Gadi; use
`AGENTS_CLUSTER_CONTEXT.md` for Gadi.

## Layout and storage

- Login: `wedi0306@headnode.physics.usyd.edu.au`; home: `/suphys/wedi0306`.
- Main repo: `/suphys/wedi0306/mts-spi-study-cluster`, a symlink to
  `/import/taiji1/wedi0306/mts-spi-study-cluster`.
- pyspi repo: `/suphys/wedi0306/pyspi-fork` (regular home-directory checkout).
- The main repo's `.venv` and `data` directories therefore live physically on
  `/import/taiji1`. At verification they used 5.4 GiB and 11 GiB respectively.
- Home quota: 42.8/51.2 GiB soft (52.2 GiB hard), 206k/500k files soft.
  `/import/taiji1` had 7.6 TiB globally free (93% used); no per-user quota was visible.
- Keep repositories and small durable files in their current locations; keep large
  environments, data and outputs on `/import/taiji1`.

## Repositories and Python

- Main: branch `refactor-lagged-warping`, commit `7784a9e`; pyspi: branch `v3`,
  commit `65317c9` (`pyspi 3.0.0`). Both track `origin`.
- Python 3.12.12 in `.venv`; `uv 0.10.4` at `~/.local/bin/uv`.
- Both repositories are editable-installed. Code pulls are immediately live; rerun
  `uv pip install -e . -e ../pyspi-fork` only when dependency/build metadata changes.
- Use `source scripts/activate.sh` for pyspi work; it activates the environment and
  pins BLAS/OpenMP threads to one.
- Old PBS logs remain in the main tree but are now ignored. Its only visible untracked
  files are `proof-p90-light.yaml` and `proof-p90-m32.yaml`; pyspi has 37 untracked
  benchmark outputs. Preserve them unless cleanup is explicitly authorised.

## PBS

- PBS Pro server `headnode.physics.usyd.edu.au`; default route queue `defaultQ`.
- Primary CPU execution queue: `physics`; GPU/special queues also exist.
- Server defaults: 1 CPU, 1 GiB, 1 hour; `physics` allows at most 48 CPUs, no GPU,
  and defaults to 2 hours. Maximum array size is 950.
- Pin numerical-library threads when running one pyspi process per allocated core.

## Verified update procedure

1. Confirm no active jobs and no tracked changes.
2. `git fetch --prune`, then fast-forward only; do not overwrite untracked artifacts.
3. Refresh editable installs only if metadata changed.
4. Verify `import src, pyspi`, run relevant tests, then a one-dataset smoke test before
   any production submission.

On 2026-08-25 imports succeeded after the v3 update. Main tests passed 64/65; the
only failure required a Kuramoto contract artifact absent from this cluster, rather
than exposing a code/environment failure. pyspi passed 609 tests, with 5 skipped and
1 expected failure.

## Known unknowns

- `/import/taiji1` per-user quota and scheduler fair-share/accounting were not exposed.
