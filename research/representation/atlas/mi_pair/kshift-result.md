# kshift side by side: how much of each completed MI-pair drug effect survives the shift-corrected KSG

Post-hoc secondary, declared on 26 November 2026 in [`preregs/claar-prereg.md`](preregs/claar-prereg.md), [`preregs/human-prereg.md`](preregs/human-prereg.md) and [`preregs/kauvar-prereg.md`](preregs/kauvar-prereg.md) (commit d8caa4f8), before any kshift value was computed. Code: [`kshift/kz.py`](kshift/kz.py) (k4, kshift and k2te on the same windows and IAAFT draws) and [`kshift/kscore.py`](kshift/kscore.py) (gate, the frozen scorers rerun per estimator, R table). No frozen primary or verdict changes.

## Submission (26 November)

Snapshot `operations/sources/mi-kshift-d8caa4f8` (src and `research/representation/atlas` only, 413 files), job `jobs/kshift.pbs`, 48 cores and 64 GB each. Claar and Kauvar windows were copied from the session-notes archive to `atlas/kshift-inputs/` on scratch (62 files, sha256 matched). PRIOS reads its windows on gdata. Outputs go to `atlas-mi-pair-261117/kshift/<study>/` on gdata.

| Study | Job | Windows | Expected kSU | Worst kSU |
|---|---|---|---|---|
| Claar (primary, induction, late) | 185283890 | 983 + 406 + 235 | 0.1 | 0.38 (4 h) |
| PRIOS post-deviation primary | 185283891 | 1,064 | 0.03 | 0.38 (4 h) |
| Kauvar | 185283892 | 600 | 0.05 | 0.38 (4 h) |
| PRIOS S-full | 185283893 | 3,911 (all channels, 10 draws) | 0.35 | 0.96 (10 h) |

Before submission:
- `kz.py check`: the per-edge k4, kshift and k2te equal `ksgbias.edge_all` exactly.
- k4 gate, k4 only, so no kshift value was printed. It was exact on the machine that made each stored z: macOS for 4 Claar primary, 2 induction, 2 late and 3 Kauvar windows; a Gadi login node for 2 PRIOS windows.
- The same PRIOS windows on macOS gave a real-window gap of 1.8e-4, with the surrogates exact. Normal-scored windows share one quantile set, so their distance ties are exact and different hardware breaks them differently (inferred from that pattern). Claar and Kauvar run on Gadi may therefore show a gap of this size on real windows. R is computed against the same run's k4.

Results: pending.
