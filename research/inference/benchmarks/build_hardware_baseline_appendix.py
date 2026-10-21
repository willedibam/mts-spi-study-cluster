"""Append the hardware mean-|r| / mean-MPI baselines to the baselines notebook; execute only the new cells.

Existing cells and outputs are copied verbatim and verified unchanged. Rerunning replaces only the
previously appended section (cells tagged TAG), never other edits.
"""
from copy import deepcopy
from pathlib import Path
import hashlib
import json

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[3]
TARGET = ROOT/'research/inference/benchmarks/notebooks/order-parameter-benchmarks-baselines.ipynb'
TAG = 'hardware-baselines-261021'


def appendix():
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    cells = [
        md(r'''## 7. Electronic oscillator hardware — carrier frequency locking (appended 2026-10-21)

**System.** 28 coupled electronic oscillators from the public hybrid-intermittency recordings ([Vera-Ávila et al. 2026](https://doi.org/10.1016/j.dib.2026.112785)): an ER wiring sweep (29 selected coupling settings) and a small-world (SW) sweep (16), each $M=N=28$, $T=5000$ at 500 Hz (the full 10 s). The SPI–SPI coordinate $q$ is the frozen r9 centred PC1 of $z$ fitted on all 29 ER recordings and projected onto SW ([hardware pilot protocol](../../../empirical/hardware_pilot/notes/hardware-pilot-261012.md)).

**Physical quantity.** No universal order parameter is known for this apparatus. $Q$ is the across-channel range of mean instantaneous carrier frequency (90–140 Hz band, central 9 s); its collapse marks the cessation of phase slips (ER 23→24, SW 31→34). Mean carrier phase coherence $R_1$ is reported as an alternative.

**Baselines,** with the same definitions as for the locking banks. Mean-$|r|$: mean absolute zero-lag Pearson correlation over all $M(M-1)$ ordered off-diagonal entries of the exact input given to pyspi. Mean-MPI PC1: each SPI's mean over the same entries, then a standardized PC1 fitted on the ER recordings only and projected onto SW; SPIs not finite on every ER recording are dropped; display sign oriented by voltage on ER rows. Inputs are the local frozen r9 observations and MPIs (hash-checked); no pyspi computation or new extraction. The display range is settings ≤ 87 (0–0.435 V), as in the three-systems notebook; the table also reports the full sweep. This is a retrospective comparison: $q$'s behaviour on this sweep was already known.'''),
        code('''import sys
from pathlib import Path
ROOT = next(p for p in (Path.cwd().resolve(), *Path.cwd().resolve().parents) if (p/'src/mtsspi/spi_spi_contract.py').is_file())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import json
import matplotlib.pyplot as plt
import pandas as pd
from research.inference.benchmarks import hardware_baselines as hb
from research.inference.benchmarks.composite_figures import style
frame, provenance = hb.compute()
table = hb.save(frame, provenance)
print({k: provenance[k] for k in ('records', 'fit', 'mean_PC1_evr', 'mean_PC1_spis', 'max_missing')})
table.round(3)'''),
        code('''style()  # imports three_systems, which selects Agg; restore inline display afterwards
%matplotlib inline
fig = hb.figure(frame)
plt.show()'''),
        md(r'''**Interpretation.** Rank association does not favour $q$ on this sweep. On ER settings ≤ 87, $|\rho_s|$ with the frequency range is .89 for $q$, .95 for mean-$|r|$ and .98 for mean-MPI PC1; on SW it is .84, .88 and .93. The transition is strength-correlated: mean-$|r|$ is a monotone function of $R_1$ ($|\rho_s|=1.00$ on ER).

The difference is localisation, and it is descriptive. In ER, $q$ crosses 43% of its displayed range over the single slip-cessation interval 23→24 (log $Q$: 42%). Mean-$|r|$ crosses 4.5% and mean-MPI 8.9% there; their steepest slopes lie before locking (18→19, 16→18). In SW all three readouts change most over 31→34 (52%, 29% and 37%), on a coarser grid. Shares and slopes depend on the adaptive control grid and are not tests. For this system the defensible statement is therefore that $q$ tracks the measured quantity and steps where phase slips stop; it is not that $q$ outperforms simple statistics.'''),
    ]
    for cell in cells:
        cell.metadata['tags'] = [TAG]
    return cells


def build(target=TARGET):
    before = target.read_bytes()
    original = nbf.read(target, as_version=4)
    kept = [c for c in original.cells if TAG not in c.metadata.get('tags', [])]
    check = deepcopy(kept)
    executable = nbf.v4.new_notebook(cells=appendix(), metadata=deepcopy(original.metadata))
    NotebookClient(executable, timeout=600, kernel_name='python3', resources={'metadata': {'path': str(target.parent)}}).execute()
    errors = [o for c in executable.cells if c.cell_type == 'code' for o in c.outputs if o.output_type == 'error']
    assert not errors, errors
    result = deepcopy(original)
    result.cells = kept+executable.cells
    nbf.validate(result)
    assert result.cells[:len(kept)] == check
    assert target.read_bytes() == before, 'Notebook changed during execution; preserve concurrent edits.'
    nbf.write(result, target)
    receipt = dict(target=str(target.relative_to(ROOT)), before_sha256=hashlib.sha256(before).hexdigest(),
                   after_sha256=hashlib.sha256(target.read_bytes()).hexdigest(), preserved_cells=len(kept),
                   appended_cells=len(executable.cells), errors=0)
    out = ROOT/'results/order-parameter-inference'/TAG
    out.mkdir(parents=True, exist_ok=True)
    (out/'notebook-audit.json').write_text(json.dumps(receipt, indent=2)+'\n')
    return receipt


if __name__ == '__main__':
    print(json.dumps(build(), indent=2))
