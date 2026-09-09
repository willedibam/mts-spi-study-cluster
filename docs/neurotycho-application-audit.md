# NeuroTycho: access, labels and transfer-design audit

Audited 2026-09-10. This is an application feasibility assessment, not a new
classification result. No ECoG waveform data or application models were used.

## Live catalogue and access

The site's live [JSON catalogue](https://neurotycho.org/data/detail.json) contains
119 entries, including **30** with Task="Anesthesia and Sleep". The previous
indexed-page count of31 is superseded by this live snapshot. The relevant current
task ID is78; task40 selects the older Sleep category. Three additional entries
have Task="Anesthesia", animal aliases C2/G, and older Event3 formatting. They are
kept separate; animal identities and possible duplicate experiments are unverified.

| Animal | KTMD | Propofol | Ketamine | Medetomidine | Sleep | Total archives |
|---|---:|---:|---:|---:|---:|---:|
| Chibi | 2 | 2 | 2 | 2 | 4 | 12 |
| George | 3 | 2 | 2 | 2 | 3 | 12 |
| Kin2 | 3 | 0 | 0 | 0 | 0 | 3 |
| Su | 3 | 0 | 0 | 0 | 0 | 3 |
| Total | 11 | 4 | 4 | 4 | 7 | 30 |

KTMD is ketamine–medetomidine. These are archive/experimental-date counts, not
independent animals or numbers of training examples. An archive may contain
multiple acquisition sessions and labelled state intervals.

All30 MAT archive URLs return HTTP200 after redirecting to HTTPS on the RIKEN
data host. Bounded byte-range requests successfully read ZIP directories and
Condition.mat members from six representative animal/agent cells: KTMD in all
four animals, and propofol in Chibi and George. Every extracted member passes
its ZIP size and CRC checks. The actual large archives therefore exist and small
annotation reads work; no bulk waveform transfer is required for this audit.

## What the annotations support

All six sampled archives have explicit AwakeEyesClosed-Start/End and
Anesthetized-Start/End labels. Their awake/anesthetized intervals span approximately
605–1321/551–1937seconds respectively. ConditionIndex and ConditionTime obey
`time=(index−1)/1000`, consistent with one-based indices at1kHz.

The annotations support a sustained awake-eyes-closed versus anaesthetized
contrast. They do not contain a trial-by-trial responsiveness trace in the sampled
Condition.mat files, so they are not by themselves validated onset-time or
subjective-consciousness labels. The original experiment used behavioural responses
to define anaesthetic unresponsiveness, with slow-wave confirmation. Its sleep
labels were defined through spatial slow-wave synchrony; exclude sleep from an
independent validation of an interaction marker unless separate label evidence is
established. [Original methods](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0080845).

One Chibi propofol recovery annotation starts at index−100/time−.101seconds,
consistent with a boundary preceding that acquisition. Do not use negative indices
as Python array positions. This concerns recovery, outside the proposed sustained
awake/anaesthetized contrast, and is not evidence that the recordings are corrupt.

## Transfer potential and limits

The catalogue permits a bounded pilot that trains on KTMD recordings and evaluates
propofol in a held-out animal, with Chibi and George as two possible target animals.
Training must exclude that animal's KTMD data when claiming animal transfer.
Kin2 and Su provide additional source animals but no propofol target replication.
This is only two target animals; many windows cannot establish broad population
generalization or create a substantial independent-subject learning curve.

KTMD dates are January–June2011; propofol dates July–August2012. Agent and recording
period are therefore confounded for this proposed transfer. Success would support
transfer across their combined change, not isolated pharmacological invariance.
All four drug conditions in the later period occur in only Chibi and George.

Before waveform extraction, specify referencing, bad-channel rules, sampling
rate, scientifically relevant frequency coverage, stable-state windows and grouped
splits. Those choices must reflect ECoG physiology, not inherit the synthetic
model's100Hz sampling or3–20Hz Hilbert band by default. Audit montage/reference
metadata before selecting32channels. Use eyes-closed awake intervals to reduce
an avoidable eye-condition difference; do not mix open/closed states casually.

Judgment: **accessible and suitable for a small, carefully qualified external
pilot; insufficient by itself for broad cross-subject/intervention claims**.
A raw-data quality and strong spectral/domain-baseline pilot is the next gate.
Do not launch the full pyspi/neural comparison until that contract is specified.
If the intended claim requires a large independent-subject evaluation, choose
another cohort rather than inflating the number of windows here.

## Reproduction and provenance

Artifacts: `results/neurotycho_audit_260910/{detail.json,catalogue.csv,audit.json}`
and the six annotation/member-list directories. `scripts/audit_neurotycho_catalogue.py`
reproduces access checks and bounded annotation reads using the site's advertised
URLs. It uses curl's normal certificate verification; this environment's Python
requests certificate bundle failed, while curl validated the connection.

An initial audit mistakenly applied a2MB body-size limit to HEAD requests, which
curl interpreted against the archives' full Content-Length. It was corrected by
limiting GET bodies only; the initial diagnostic is preserved in
`audit-head-limit-diagnostic.json`. The final audit finds30/30 accessible archives.
This was an audit-client issue, not a dataset-access failure.
