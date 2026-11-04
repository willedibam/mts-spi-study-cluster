# Gadi storage audit, 5 November 2026 (read-only)

Read-only survey from a login node (`lquota`, `du`, `find`, `nci-file-expiry`); nothing was moved or deleted. Sizes are allocated MiB; "newest" is the newest file mtime in the directory. Placement rules: [storage layout](gadi-storage-layout.md).

## Quotas (project ql44, all users)

| fs | bytes | inodes |
|---|---|---|
| gdata | 83.17 / 100 GiB | 16,647 / 70,000 |
| scratch | 397.81 GiB / 1 TiB | 136,829 / 202,000 |

gdata bytes are the binding limit. Inodes are not pressing on either filesystem.

## Age criterion

Nothing under `mts-spi-study` on either filesystem is older than 3.5 months (cut-off 20 July 2026). The 2 November packing already sealed the older units. The oldest live content is scratch `representation/empirical/hcp-working-memory/raw` (78 GiB, newest 15 September) and `operations/sources/registered-worktrees` (1.7 GiB, 7,162 inodes, newest 16 September). An age rule reclaims almost nothing; role and retirement do.

## Reclaimable by role (decision needed; none actioned)

| Item | Where | Size | Why it can go | Recovery |
|---|---|---|---|---|
| Siena raw EDF (`siena-confirmation-261102/raw`, 45 files) | gdata | 16.2 GiB | public PhysioNet `siena-scalp-eeg/1.0.0` input to a completed, negative confirmation; derived bank/features stay | URL recorded in `manifest.json`, `source-lock.json` and `input-audit.json` |
| CCEP pilot raw (`ccep-pilot-261014/raw`) | gdata | 13.3 GiB | public OpenNeuro ds004080 input; pilot complete; `prepared/`, `corpora/` and `analysis/` stay | `acquisition.json` holds the git tree and per-file SHA-256 |
| CCEP cohort raw (`ccep-cohort-261014/raw`) | scratch | 68.9 GiB | same public source; cohort computation complete (15 October) | `acquisition.json` as above |
| HCP working-memory raw | scratch | 78 GiB | study stalled on family structure (see `docs/context/spi-spi-representation-evaluation.md`); access-controlled but re-downloadable from ConnectomeDB | needs the user's HCP account |
| `registered-worktrees` (6 source worktrees) | scratch | 1.7 GiB, 7,162 inodes | September source snapshots; pack to one `.tar.zst` with a `README.md` restore pointer | the archive itself |

## Durability risk

Scratch files unread for 100 days are quarantined, then deleted after 14 days ([CMS wiki summary of NCI policy](https://coecms.github.io/cms-wiki/starting/starting-gadi-scratch.html)). The `packed-261102/` archives on scratch are exposed to this from about mid-February 2027:
- `order-parameter-inference` 19.7 GiB;
- `proof` 5.5 GiB;
- `operations/sources` 3.2 GiB.

Clearing the two public raw inputs from gdata (≈ 29.5 GiB) would free enough room to move those ≈ 28.4 GiB of archives onto gdata.

Also flagged by `nci-file-expiry`, outside this project: the sibling EEML scratch checkout (`/scratch/ql44/we2614/eeml-2026-application`) began expiring on 4–6 November. Its code is on `origin`, and its `archives/` entries are links to gdata, so no unique data is known to be at risk.
