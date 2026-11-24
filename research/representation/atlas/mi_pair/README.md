# MI pair and related tests: scripts, preregistrations and outputs

This folder holds the pyspi-free code behind the Gaussian-against-kNN MI result and its follow-up tests, ported on 17 November 2026 from the review sessions' notes folder and transcript. The claims and numbers are in [`../findings.md`](../findings.md), section "Gaussian against model-free MI". Small outputs are in `results/representation/atlas-mi-pair-261117/`, force-added because `results/` is gitignored. These scripts were run as written, from flat working directories; they were not refactored into the package, so import paths and input paths are those of the original runs.

## Contents

| Folder | Test | Status | Outputs (`results/representation/atlas-mi-pair-261117/`) |
|---|---|---|---|
| `review/` | Macaque NeuroTycho discovery and robustness. 2-s IAAFT pairs (`surr_pairs.py`, `surr_eval.py`, `focal.py`, `split.py`); 10-s normal-scored rerun (`zgk_ns.py`, `sumall.py`); sleep and delta matching (`sleepmatch.py`, `sleepmatch_ns.py`); hub test (`hubs.py`, `hubtest.py`); form checks (`mech.py`, `mech_ns.py`, `edgemi.py`, `calib*.py`, `decide*.py`) | Recovered from the transcript (see below) | `zns/`: the lost `zns_[AB]_*.npz`, recomputed with `edges.py recompute` (job 184230163, same seeds and invocations; matches every reported value) and now with per-edge MI. Arrays on gdata `.../atlas-mi-pair-261117/zns/`; checksums and `sumall.txt` here. |
| `claar/` | Mouse isoflurane EEG (Claar et al. 2023, DANDI 000458). Epoch primary, induction (Addendum A) and late-awake drift (Addendum B) | Frozen: [`preregs/claar-prereg.md`](preregs/claar-prereg.md) | `claar/claar_z_*.npz`, `claar_sel_qc.json` |
| `kauvar/` | Human sub-anaesthetic ketamine sEEG (Kauvar et al. 2025, DANDI 001347) | Frozen: [`preregs/kauvar-prereg.md`](preregs/kauvar-prereg.md) | `kauvar/kv_z.npz`, `qc.json` |
| `edges.py`, `edges_score.py`, `edges_posthoc.py` | Recompute of the lost arrays and the all-channel edge-concentration test | Frozen: [`edges-prereg.md`](edges-prereg.md); result [`edges-result.md`](edges-result.md) | `zns/` (checksums, `sumall.txt`); `full/` (scores; per-window arrays on gdata) |
| `montage/` | Macaque bipolar montage, stages A and B | Robustness check, not frozen | `montage/zbip_[AB].npz` |
| `amplitude/` | Amplitude-gate form test, macaque and mouse, with synthetic calibrations | Frozen: [`preregs/amplitude-prereg.md`](preregs/amplitude-prereg.md); inconclusive | `amplitude/gate_*.npz` |
| `nonstat/` | Non-stationary (segment) surrogates against the stationary IAAFT null, 10-s macaque, with synthetic calibration | Frozen: [`preregs/nonstat-prereg.md`](preregs/nonstat-prereg.md); calibration shows it cannot separate gain from a coupling nonlinearity, so the real-data run is descriptive | `nonstat/calib1.*`, `nonstat/` scores |
| `transient/` | Cut of samples beyond 5 robust SDs (±0.2 s) in any channel, against a shifted-mask control, 10-s macaque | Frozen: [`preregs/transient-prereg.md`](preregs/transient-prereg.md); result [`preregs/transient-result.md`](preregs/transient-result.md): independent of transients | `transient/score.txt`, `transient-score.json`, `SHA256SUMS` |
| `cumulants/` | Cross-cumulant (J3 odd / J4 even) and envelope decomposition of the MI excess, 10-s macaque, with synthetic calibration | Frozen: [`preregs/cumulants-prereg.md`](preregs/cumulants-prereg.md) (J3 share .07: symmetric); post hoc `split.py` (J4 mostly along the pair difference direction) | `cumulants/calib1.*`, `split-calib.*`, `score.txt`, `split-score.*` |

Each folder's `*_result.txt`, `*_qc.txt` or `*_sum.py` output reproduces the numbers quoted in `findings.md`.

## Inputs

- **Macaque 10-s windows.** `zgk_ns.py raw/nt10 zns_A_<agent>.npz <agent>` and `raw/nt10b zns_B_<agent>.npz <agent>`. `raw/nt10` and `raw/nt10b` were local copies of Gadi `/scratch/ql44/we2614/mts-spi-study/atlas/banks/neurotycho-long-stage-a/` and the stage-B windows of `neurotycho-long/` (`manifest.json` and `windows.npz`). The windows come from the atlas `prepare.py`.
- **Mouse.** `extract_claar.py` reads DANDI 000458 NWB files by asset ID (`claar_jobs.txt`; `curlfile.py` fetches them). `select_claar.py`, `select_induction.py` and `select_late.py` write selected windows. The selected windows (about 90 MB) are not committed; they regenerate from DANDI with these scripts.
- **Human ketamine.** `extract_kv.py` and `select_kv.py` read DANDI 001347 by asset ID (`kv_jobs*.txt`).
- Paths to the venv (`/Users/wedi0306/Code/mts-spi-study-cluster/.venv`) and scratch directories are hard-coded as in the original runs.

## Figure inputs (`figwin/`)

`plot_character_draft.py` (fig:mi, also `--c-top delta` for `mi-figure-cdelta`) and `plot_delta_supp.py` read the arrays above plus `figwin/`, `human/prios-sfull/z.npz` (from gdata `.../atlas-mi-pair-261117/human/prios-sfull/`) and `zns/` (gdata, checked against `zns/SHA256SUMS`). `figwin/` is local-only and was lost once with a deleted worktree; rebuilt on 24 November 2026 with the commands below, the committed script reproduced the committed figure byte for byte. `R` is `results/representation/atlas-mi-pair-261117`.

- `windows.npz`: on Gadi, `fig_windows.py /scratch/ql44/we2614/mts-spi-study/atlas/banks windows.npz <ids>`, the ids being those of `CALLOUT`, `PANEL_B`, `TRACE_MAC` and `GLYPH_PAIR[0]` in `plot_character_draft.py`.
- `start_s.csv`: on Gadi, columns `stage,id,unit,state,session,start_s`, one row per entry of `rows` in `neurotycho-long-stage-{a,b}/manifest.json` under the banks directory.
- `brain_{Chibi,George}.npz`: the first `.mat` in `results/representation/neurotycho/neurotycho_audit_260910/montage/{Chibi,George}.zip`, read with `scipy.io.loadmat(simplify_cells=True)` and saved as `I`, `X`, `Y` and `source` (the `.mat` name).
- `prios_electrodes_mni305.npy`: x, y, z of the rows with `group` grid or strip and `silicon` "no" (NaN coordinates dropped), concatenated over `sub-PRIOS0{1-6,9}_ses-1_electrodes.tsv` of OpenNeuro ds004370 in sorted order (381 x 3). Local copies in `$R/figwin-src/prios-electrodes/`.
- `assets.npz`: `fig_assets.py <download dir> $R/figwin`; the downloads and their URLs are listed in its docstring.
- `traces_d.npz`: `fig_traces.py $R <claar_sel> <claar_ind_sel> $R/figwin-src/prios`. The Claar selections are in the notes folder `local_89f26329-.../claar/claar_sel` and `.../critic-2/data/claar_ind_sel`; the PRIOS windows are gdata `.../human/prios-sfull/windows/*.npz`.

## Provenance of `review/`

- **Source.** Recovered on 17 November from the review session's transcript, `~/.claude/projects/-Users-wedi0306-Code-mts-spi-study-cluster--claude-worktrees-real-world-apps-9b53a1/e32304ac-64a6-4158-8b6e-6f8d3e9556fc.jsonl`.
- **Version.** Each file is the last version written there by heredoc or Write.
- **Later edits are not replayed.** That session also edited some files in place with `sed`. The 19 commands are in [`review/EDITS-LOG.txt`](review/EDITS-LOG.txt), with heredoc bodies omitted.
  - Files named in those commands: `edgemi.py`, `zgk_ns.py`, `between.py`, `split.py`, `calib.py`, `calib10.py`, `decide.py`, `beyond.py`, `sumall.py`, `surr_pairs.py`, plus the Claar extraction scripts.
  - Treat these as close to, but not certainly identical with, the code that produced the numbers.
- **One rebuilt file.** `sleepmatch_ns.py` was made in that session by a `sed` transform of `sleepmatch.py`. It is rebuilt here with the same command (logged), and was checked only for that substitution.
- **Duplicates.** Where a script also exists in `claar/`, `kauvar/`, `montage/` or `amplitude/`, those copies came from the notes folder and are the later, authoritative versions. The mouse versions of `claar_z.py`, `zgk_ns.py`, `surr_pairs.py` and `extract_claar.py` are in `claar/`.
- **Exploratory.** `review/` also holds exploratory scripts from the same session: embeddings, figures, NWB inspection and the lag null (`lagnull*.py`). They are kept for completeness.

## Not ported

- critic-2's scans and frozen tests (`mcd10`, `lag10`, `lagx`, `scan10` and others): 143 MB including data.
- The robust-axis channel check (`robev/`).
- Both remain in the notes folder `side-session-notes/local_89f26329-f8af-40ac-adc4-5cbb414827de/`. Port them separately if they enter the paper.
