"""Simple baselines for the hardware oscillator recordings: mean |r| and mean-MPI PC1.

Inputs are the frozen r9 bank (45 records: 29 ER, 16 small-world; M=28, T=5000 at 500 Hz),
read locally; no pyspi computation. Definitions match the locking banks:

- mean-|r|: mean absolute zero-lag Pearson correlation over all M(M-1) ordered off-diagonal
  entries of the exact M x T input given to pyspi (the full 10 s record).
- mean-MPI: each SPI's mean over all M(M-1) ordered off-diagonal entries (`summarize`), then a
  standardized PC1 (`coordinate`) fitted on the ER records only, as the SPI-SPI z-PC1 was, and
  projected onto SW. SPIs not finite on every ER record are dropped by `coordinate`; no target,
  voltage or physical diagnostic enters the fit. The display sign is oriented by voltage on ER rows.

Physical diagnostics come from the 90-140 Hz carrier band (central 9 s): the across-channel range
of mean instantaneous frequency (phase-slip cessation) and mean phase coherence R1.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT/'results/hardware-pilot-261012/full-p90-r9-261014'
OUT = ROOT/'results/order-parameter-inference/hardware-baselines-261021'
LOCKING = {'ER': (23, 24), 'SW': (31, 34)}
READOUTS = {'er_pc1': 'SPI–SPI PC1', 'mean_abs_r': 'mean-|r|', 'mean_PC1': 'mean-MPI PC1'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mean_abs_r(x):
    """Mean |Pearson r| over ordered off-diagonal entries (identical to the upper triangle)."""
    r = np.corrcoef(x)
    return float(abs(r[~np.eye(len(r), dtype=bool)]).mean())


def mean_mpi_pc1(means, fit):
    """Standardized PC1 of per-SPI means, fitted on `fit` rows; sign arbitrary until oriented."""
    from research.inference.candidate_search.report_dependence_transition import coordinate
    return coordinate(means, fit, True)


def compute(run=RUN):
    from research.proof.p90.spi_baseline_exploration import summarize
    from research.empirical.hardware_pilot.calibrate_hardware_pilot import phase_properties
    rows = json.loads((run/'manifest.json').read_text())['rows']
    summary = json.loads((run/'analysis/summary.json').read_text())
    hashes = {s['name']: s['mpi_sha256'] for s in summary['sources']}
    scores = pd.read_csv(run/'analysis/scores.csv').set_index('name')
    records, means, order = [], [], None
    with np.load(run/'observations.npz', allow_pickle=False) as observations:
        for row in rows:
            x = observations[row['name']]
            if x.shape != (28, 5000):
                raise ValueError(f"Unexpected input shape for {row['name']}")
            path, = (run/'mpis').glob(f"*/????-{row['name']}/spi_mpis.npz")
            if sha(path) != hashes[row['name']]:
                raise ValueError(f"MPI archive changed: {row['name']}")
            with np.load(path, allow_pickle=False) as archive:
                order = order or list(archive.files)
                if list(archive.files) != order or len(order) != 289:
                    raise ValueError('SPI catalogue differs between records')
                means.append(summarize({k: archive[k] for k in order}, order)[:, 0])
            carrier = phase_properties(x, (90, 140))
            records.append(dict(name=row['name'], topology=row['topology'], step=row['step'],
                                voltage=row['voltage'], frequency_range=float(np.ptp(carrier['frequency'])),
                                R1=carrier['R1'], mean_abs_r=mean_abs_r(x),
                                er_pc1=float(scores.loc[row['name'], 'er_pc1'])))
    frame = pd.DataFrame(records)
    means = np.array(means)
    fit = frame.topology.eq('ER').to_numpy()
    q, evr, missing = mean_mpi_pc1(means, fit)
    sign = 1 if spearmanr(q[fit], frame.voltage[fit]).statistic >= 0 else -1
    frame['mean_PC1'] = sign*q
    provenance = dict(
        run=str(run.relative_to(ROOT)), records=len(frame), fit='ER records only (29); SW projected',
        mean_PC1_evr=float(evr), mean_PC1_spis=int(np.isfinite(means[fit]).all(0).sum()),
        max_missing=float(missing.max()), display_sign=sign,
        definitions=__doc__.split('\n\n', 1)[1].strip(),
        observations_sha256=sha(run/'observations.npz'), scores_sha256=sha(run/'analysis/scores.csv'),
        source_sha256=sha(__file__))
    return frame, provenance


def steepest(part, column):
    """Largest |change|/volt between adjacent settings, as a share of the displayed range."""
    part = part.sort_values('voltage')
    values = np.log10(part[column]) if column == 'frequency_range' else part[column]
    values, volts, steps = values.to_numpy(), part.voltage.to_numpy(), part.step.to_numpy()
    change = np.diff(values)
    slope = abs(change)/np.diff(volts)
    span = np.ptp(values)
    i, j = int(np.argmax(abs(change))), int(np.argmax(slope))
    return dict(largest_step=f'{steps[i]}→{steps[i+1]}', largest_step_share=float(abs(change[i])/span),
                steepest_slope=f'{steps[j]}→{steps[j+1]}')


def metrics(frame, max_step=87):
    """Rank association per topology, display range (settings <= max_step) and full sweep."""
    rows = []
    for topology, (before, after) in LOCKING.items():
        for scope, part in [(f'settings ≤{max_step}', frame.query('topology == @topology and step <= @max_step')),
                            ('all settings', frame.query('topology == @topology'))]:
            part = part.sort_values('step')
            locked = part.set_index('step')
            for column, label in [('frequency_range', 'Q: log frequency range'), *READOUTS.items()]:
                values = np.log10(locked[column]) if column == 'frequency_range' else locked[column]
                share = abs(values.loc[after]-values.loc[before])/np.ptp(values)
                rows.append(dict(topology=topology, scope=scope, n=len(part), readout=label,
                                 abs_rho_frequency_range=float(abs(spearmanr(part[column], part.frequency_range).statistic)),
                                 abs_rho_R1=float(abs(spearmanr(part[column], part.R1).statistic)),
                                 locking_share=float(share), **steepest(part, column)))
    return pd.DataFrame(rows)


def save(frame, provenance, out=OUT):
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out/'scores.csv', index=False)
    table = metrics(frame)
    table.to_csv(out/'metrics.csv', index=False)
    provenance['scores_csv_sha256'] = sha(out/'scores.csv')
    (out/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    return table


def load(out=OUT):
    frame = pd.read_csv(out/'scores.csv')
    provenance = json.loads((out/'provenance.json').read_text())
    if sha(out/'scores.csv') != provenance['scores_csv_sha256']:
        raise ValueError('Hardware baseline scores changed after saving')
    return frame


def figure(frame, max_step=87):
    """Q (carrier frequency range, log) with each readout, ER fit and SW projection."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(9.2, 5.0), layout='constrained', sharex=True)
    colors = {'er_pc1': '#26828e', 'mean_abs_r': '#7570b3', 'mean_PC1': '#d95f02'}
    for row, topology in enumerate(LOCKING):
        part = frame.query('topology == @topology and step <= @max_step').sort_values('voltage')
        before, after = LOCKING[topology]
        for ax, (column, label) in zip(axes[row], READOUTS.items()):
            ax.plot(part.voltage, part.frequency_range, 'o-', color='#222222', ms=2.7, lw=1.2)
            ax.set_yscale('log')
            ax.axvspan(before*.005, after*.005, color='.5', alpha=.15, lw=0)
            right = ax.twinx()
            right.spines['right'].set_visible(True)
            right.plot(part.voltage, part[column], 's-', color=colors[column], ms=2.7, lw=1.2)
            right.tick_params(axis='y', labelcolor=colors[column])
            rho = abs(spearmanr(part[column], part.frequency_range).statistic)
            ax.set_title(f'{topology} · {label}\n' + rf'$|\rho_s|={rho:.2f}$', fontsize=9)
            if ax is axes[row, 0]:
                ax.set_ylabel('Carrier frequency range (Hz)')
            if row == 1:
                ax.set_xlabel('Coupling-control voltage (V)')
    return fig
