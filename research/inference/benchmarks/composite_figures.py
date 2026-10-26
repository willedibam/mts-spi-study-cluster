"""Composite order-parameter figure: q = PC1 of SPI–SPI z tracking a physical Q without labels.

Presentation and verification only. Every q, baseline and z value is an archived, target-blind
readout; nothing here refits a coordinate or uses Q to choose what is shown. Snapshot records
follow fixed, pre-existing rules (never Q or q); displayed edge profiles are replayed from archived
MPIs over all M(M-1) ordered off-diagonal entries (`unified_ordered_v3`), and physical glyphs are
regenerated from the archived generators and checked against the archived inputs.

Layouts. `composite_stacked(main, stack)`: the main system as a full column (headline Q/q with
in-axes MTS insets and glyphs, mean-|r| and mean-MPI glyphs, z_ab feature curves, edge planes) beside
compact panels for further systems (MTS heatmaps with glyphs above Q/q curves; baselines in text).
`composite(systems)` puts full columns side by side. Spread across instances: faint per-instance
paths ('trajectories'), 10-90% bands ('band'), 'both' or 'none'; feature curves always use bands.

Registry: `SYSTEMS[name]` is a `Spec` (presentation choices) whose `loader` returns a `SystemData`
(per-record frame with control/Q/q/mean-|r|/mean-MPI/cohort[/trajectory, z0..zk], three `Snapshot`s,
metrics, provenance). Registered: locking, hardware, stuart-landau, kuramoto, cml2d.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
import hashlib
import json
import textwrap
from typing import Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'results/order-parameter-inference/composites-261021'
Q_COLOR, Z_COLOR = '#222222', '#26828e'
BASELINES = {'mean-|r|': '#7570b3', 'mean-MPI': '#d95f02'}
BASELINE_TITLES = {'mean-|r|': r'mean-$|r|$', 'mean-MPI': 'mean-MPI'}
GLYPH_COLOR = '#3b3b3b'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(condition, message):
    if not condition:
        raise ValueError(message)


def style(tex=None):
    """Computer Modern typography: LaTeX (`text.usetex`) when latex and dvipng exist, else mathtext.

    Pass tex=False to force the native renderer (no external processes). All figure strings are written
    to be valid in both modes. With usetex, SVG/PDF text is embedded as glyph outlines.
    """
    import logging
    import shutil
    from research.inference.benchmarks.three_systems import style as base
    base()
    logging.getLogger('fontTools').setLevel(logging.ERROR)  # Type 42 embedding warns about font timestamps
    tex = bool(shutil.which('latex') and shutil.which('dvipng')) if tex is None else tex
    plt.rcParams.update({'text.usetex': tex, 'font.family': 'serif',
                         'font.serif': ['Computer Modern Roman'] if tex else ['DejaVu Serif'],
                         'text.latex.preamble': r'\usepackage{amsmath}',
                         'svg.fonttype': 'none', 'pdf.fonttype': 42, 'axes.linewidth': .6,
                         'xtick.major.width': .6, 'ytick.major.width': .6, 'xtick.major.size': 2.5,
                         'ytick.major.size': 2.5, 'legend.fontsize': 7, 'axes.titlesize': 8,
                         'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                         'figure.dpi': 180, 'savefig.dpi': 600})
    return tex


# --------------------------------------------------------------------------- data structures

@dataclass(frozen=True)
class Feature:
    spi_a: str
    spi_b: str
    label: str      # short pair name
    reading: str    # what calibration supports; never a mechanism the evidence does not establish


@dataclass
class Snapshot:
    control: float
    label: str
    row_id: str
    x: np.ndarray                                   # observed M x T input, as given to pyspi
    glyph: dict                                     # arrays for the physical glyphs
    edges: dict = field(default_factory=dict)       # feature index -> (a_profile, b_profile, z)
    groups: np.ndarray | None = None                # per ordered edge group index (edge planes)


@dataclass
class SystemData:
    spec: 'Spec'
    frame: pd.DataFrame          # one row per displayed record
    snapshots: list
    metrics: dict
    provenance: dict
    primary: str                 # cohort the headline curves and q fit belong to


@dataclass(frozen=True)
class Spec:
    name: str
    title: str
    loader: Callable[['Spec'], SystemData]
    control_label: str
    Q_label: str
    q_label: str = r'$q$ (PC1 of $z$)'
    Q_log: bool = False
    boundary: float | tuple | None = None           # dotted line or shaded interval (control units)
    features: tuple = ()
    edge_features: tuple = (0,)
    glyphs: tuple = ()                               # names in GLYPHS; first is the default
    glyph_titles: dict = field(default_factory=dict)
    heatmap_window: tuple | None = None              # (start, count) samples; None = full record
    dt: float = 1.
    communities: int = 0                             # equal blocks marked in heatmaps
    edge_groups: tuple = ()                          # (label, colour) per group index
    secondary_style: dict = field(default_factory=dict)  # cohort -> linestyle for non-primary cohorts
    Q_source: str = ''                               # loader-specific choice of physical quantity
    subtitle: str = ''
    control_short: str = ''
    legend_loc: str | tuple = 'lower right'           # or (loc, bbox_to_anchor)
    edge_scale: str = 'shared'                       # 'shared' within a row, or 'panel' (z is scale-free)
    primary_label: str = 'held records'
    community_labels: tuple = ()                     # bottom-to-top block letters on the first heatmap
    snapshot_controls: tuple = ()                    # loader-specific fixed snapshot controls
    stack_legend_loc: str | tuple = 'lower left'     # legend position when baselines are overlaid
    inset_boxes: tuple = ((.05, .62), (.55, .12), (.78, .12))   # heatmap lower-left, headline axes fraction
    Q_limits: tuple | None = None
    q_limits: tuple | None = None
    control_ticks: tuple | None = None
    note: str = ''


# --------------------------------------------------------------------------- shared helpers

def pair_index(order, a, b):
    """Column of unordered SPI pair (a, b) in a unified z row (triu order of `spi_order`)."""
    order = list(order)
    i, j = sorted((order.index(a), order.index(b)))
    n = len(order)
    return i*n - i*(i+1)//2 + (j-i-1)


def ordered_edges(matrix):
    """All M(M-1) ordered off-diagonal entries, row-major (the unified_ordered_v3 vector)."""
    return np.asarray(matrix)[~np.eye(len(matrix), dtype=bool)]


def edge_z(mpi_a, mpi_b):
    x, y = ordered_edges(mpi_a), ordered_edges(mpi_b)
    return x, y, float(np.corrcoef(x, y)[0, 1])


def upper_symmetrized_z(mpi_a, mpi_b):
    """Legacy contract value, used only to show the ordered-contract check can discriminate."""
    upper = np.triu_indices(len(mpi_a), 1)
    sa, sb = ((m+m.T)/2 for m in (np.asarray(mpi_a), np.asarray(mpi_b)))
    return float(np.corrcoef(sa[upper], sb[upper])[0, 1])


def verify_ordered_row(mpis, order, stored, atol=2e-6):
    """Rebuild a whole z row with the library's unified (ordered) builder and compare to the bank."""
    from mtsspi.spi_spi_contract import build_unified_feature_values
    z, _, _ = build_unified_feature_values(mpis, list(order))
    stored = np.asarray(stored, float)
    same_missing = bool(np.array_equal(np.isfinite(z), np.isfinite(stored)))
    finite = np.isfinite(z) & np.isfinite(stored)
    error = float(np.max(abs(z[finite]-stored[finite]))) if finite.any() else np.nan
    return dict(max_error=error, same_missing=same_missing, finite=int(finite.sum()),
                passes=bool(same_missing and error < atol))


def curve(frame, column, cohort=None):
    part = frame if cohort is None else frame[frame.cohort.eq(cohort)]
    group = part.groupby('control')[column]
    return group.mean(), group.quantile(.1), group.quantile(.9)


def value_at(frame, column, control, cohort):
    mean, _, _ = curve(frame, column, cohort)
    return float(np.interp(control, mean.index.to_numpy(float), mean.to_numpy(float)))


def wrap_cycles(phase):
    return (np.asarray(phase)/(2*np.pi)+.5) % 1-.5


def robust_scaled(xs):
    from mtsspi.corpus_visualization import scale_timeseries
    scaled = [scale_timeseries(x, 'robust')[0] for x in xs]
    return scaled, max(float(np.quantile(abs(s), .99)) for s in scaled)


def rho(frame, column, cohort):
    part = frame[frame.cohort.eq(cohort)]
    return float(abs(spearmanr(part[column], part.Q).statistic))


# --------------------------------------------------------------------------- locking

def locking_trajectory(gamma, seed, *, n=None, delta=None, dt=.02, burn=500., reference=1000., samples=None,
                       eps=None, every_fine=5):
    """Mirror of `cross_frequency_locking.simulate` that also records community mean-field phases.

    The random stream and update order are identical, so `raw` and `truth` equal `simulate`'s output
    (tested); the extra record is arg Z_A1, arg Z_A2 and arg Z_B every `every_fine` integration steps
    across the observed window, after each update, as for the observed channels.
    """
    from scipy.special import erfinv
    from research.inference.locking import cross_frequency_locking as L
    n = L.N_COMMUNITY if n is None else n
    delta = L.DELTA if delta is None else delta
    eps = L.EPS if eps is None else eps
    samples = L.T if samples is None else samples
    index = int(np.argmin(np.abs(L.GAMMAS-gamma))) if np.min(np.abs(L.GAMMAS-gamma)) < 1e-9 else int(round(gamma*1e6))+1000
    rng = np.random.default_rng(np.random.SeedSequence([L.SEED, n, index, seed]))
    q = np.sqrt(2)*L.WIDTH*erfinv(2*(np.arange(n)+.5)/n-1)
    wa = 1+rng.permutation(q); wb = 2+delta+rng.permutation(q); wc = L.FREQ_C+rng.permutation(q)
    a, b, c = rng.uniform(0, 2*np.pi, (3, n))
    every = round(L.STEP/dt); start = round(burn/dt)
    x, fields, fine, times = [], [], [], []
    for k in range(round((burn+samples*L.STEP+reference)/dt)):
        da, db, dc, z = L.velocity(a, b, c, gamma, wa, wb, wc, eps)
        e = L.SIGMA*np.sqrt(dt)*rng.normal(size=(3, n))
        a = a+dt*da+e[0]; b = b+dt*db+e[1]; c = c+dt*dc+e[2]
        if k >= start and len(x) < samples and (k+1-start) % every_fine == 0:
            fine.append((np.exp(1j*a).mean(), np.exp(2j*a).mean(), np.exp(1j*b).mean()))
            times.append((k+1-start)*dt)
        if k >= start and (k+1) % every == 0:
            if len(x) < samples:
                x.append(np.r_[np.sin(a), np.sin(b), np.sin(c)])
            else:
                fields.append(z)
    fields = np.array(fields); phase = np.unwrap(np.angle(fields[:, 2]*np.conj(fields[:, 1])))
    halves = [abs(np.exp(1j*h).mean()) for h in np.array_split(phase, 2)]
    truth = dict(Q_lock=float(abs(np.exp(1j*phase).mean())), Q_lock_first=float(halves[0]), Q_lock_second=float(halves[1]),
                 slip=float(abs(phase[-1]-phase[0])/(len(phase)*L.STEP)), X1=float(abs(fields[:, 0]).mean()),
                 X2=float(abs(fields[:, 1]).mean()), Y=float(abs(fields[:, 2]).mean()), RC=float(abs(fields[:, 3]).mean()))
    z1, z2, zb = np.array(fine).T
    # Fine samples stop at the last observed sample; keep only those inside the observed window.
    keep = np.array(times) <= samples*L.STEP+1e-9
    mean_field = dict(t=np.array(times)[keep], theta=np.unwrap(np.angle(z1[keep])), psi=np.unwrap(np.angle(zb[keep])),
                      phi=np.unwrap(np.angle(zb[keep]*np.conj(z2[keep]))))
    return np.array(x).T, truth, mean_field


def locking_observation(row, raw, truth):
    """Re-create the random-noise arm's observation of `raw` (same noise stream as the archive)."""
    from research.inference.locking import cross_frequency_locking_snr as S
    arm = row['system']
    rng = np.random.default_rng(np.random.SeedSequence([S.NOISE_SEED, S.ARMS.index(arm), int(round(row['control']*1000)), int(row['seed'])]))
    eta = .5 if arm == S.ARMS[0] else float(rng.uniform(.3, .9))
    x, truth = S.observe(raw, dict(truth), rng, eta)
    return x, truth, eta


LOCKING_RUN = 'cross-frequency-locking-snr-261006'
LOCKING_GROUPS = (('A--B', '#CC79A7'), ('A--A', '#0072B2'), ('B--B', '#56B4E9'), ('pairs with C', '#b5b5b5'))


def locking_groups(n=8):
    """Group per ordered edge: 0 = A–B (the locking pair), 1 = A–A, 2 = B–B, 3 = involving C (C is the reference)."""
    i, j = np.where(~np.eye(3*n, dtype=bool))
    a, b = i//n, j//n
    return np.select([((a == 0) & (b == 1)) | ((a == 1) & (b == 0)), (a == 0) & (b == 0), (a == 1) & (b == 1)], [0, 1, 2], 3)


def choose_locking_records(scores, controls):
    """Held random-noise records with sensor noise nearest eta = .6 at each control (ties by row ID).

    The rule of `plot_locking_m24_edge_profiles.choose_records`, for any controls; never uses Q or q.
    """
    held = scores[scores.system.eq('random-noise') & scores.role.eq('evaluation') & scores.comparison_eligible]
    chosen = []
    for gamma in controls:
        part = held[np.isclose(held.control, gamma)].sort_values('row_id')
        chosen.append(part.iloc[np.argmin(abs(part.eta.to_numpy()-.6))])
    return chosen


def load_locking(spec):
    result = ROOT/'results/order-parameter-inference'/LOCKING_RUN
    scores = pd.read_csv(result/'scores.csv')
    analysis = json.loads((result/'analysis.json').read_text())
    check(sha(result/'features.npz') == analysis['features_sha256'], 'Locking feature bank changed')
    sources = {r['row_id']: r for r in json.loads((result/'sources.json').read_text())['sources']}
    arm = scores.system.eq('random-noise')
    held = arm & scores.role.eq('evaluation') & scores.comparison_eligible
    check(held.sum() == 168, 'Locking held scope changed')
    with np.load(result/'features.npz') as bank:
        np.testing.assert_array_equal(bank['row_id'], scores.row_id)
        order = list(bank['spi_order'])
        columns = [pair_index(order, f.spi_a, f.spi_b) for f in spec.features]
        z_rows = bank['z'][held.to_numpy()][:, columns].astype(float)
        chosen = choose_locking_records(scores, spec.snapshot_controls)
        stored_rows = {r.row_id: bank['z'][int(np.flatnonzero(scores.row_id.eq(r.row_id))[0])].astype(float) for r in chosen}
    part = scores[held]
    frame = pd.DataFrame(dict(row_id=part.row_id.to_numpy(), control=part.control.to_numpy(), Q=part.Q_lock.to_numpy(),
                              q=part.z_PC1.to_numpy(), seed=part.seed.to_numpy(), eta=part.eta.to_numpy(), cohort='held',
                              trajectory=part.seed.to_numpy()))  # same seed index; realisations independent per control
    frame['mean-|r|'] = part.mean_abs_r.to_numpy()
    frame['mean-MPI'] = part.mean_PC1.to_numpy()
    for k in range(len(columns)):
        frame[f'z{k}'] = z_rows[:, k]
    snapshots, receipts, checks = [], [], []
    groups = locking_groups()
    with np.load(ROOT/'data/order-parameter-inference'/LOCKING_RUN/'observations.npz') as observations:
        for row in chosen:
            archived = observations[row.row_id]
            raw, truth, mean_field = locking_trajectory(row.control, int(row.seed))
            x, truth, eta = locking_observation(row, raw, truth)
            replay_error = float(np.max(abs(x-archived)))
            check(replay_error < 1e-10 and abs(truth['Q_lock']-row.Q_lock) < 1e-12 and abs(eta-row.eta) < 1e-12,
                  f'Locking replay failed for {row.row_id}')
            path = ROOT/'results/order-parameter-inference/locking-features-m24-261009/mpi-examples'/f"{row.row_id.removeprefix('xfreq21-random-noise-')}.npz"
            check(sha(path) == sources[row.row_id]['mpi_sha256'], f'Locking MPI changed: {row.row_id}')
            with np.load(path) as m:
                mpis = {k: m[k] for k in order}
            contract = verify_ordered_row(mpis, order, stored_rows[row.row_id])
            check(contract['passes'], f'Locking z row is not the ordered contract: {row.row_id}')
            edges = {}
            for k, f in enumerate(spec.features):
                a, b, z = edge_z(mpis[f.spi_a], mpis[f.spi_b])
                stored = float(stored_rows[row.row_id][columns[k]])
                check(abs(z-stored) < 2e-6, f'Edge replay failed: {f.label} {row.row_id}')
                edges[k] = (a, b, z)
                receipts.append(dict(row_id=row.row_id, feature=f.label, z=z, stored=stored,
                                     legacy_upper_symmetrized=upper_symmetrized_z(mpis[f.spi_a], mpis[f.spi_b])))
            fine = mean_field
            glyph = dict(t=fine['t'], phase=fine['phi'][None], lissajous=(np.cos(fine['theta']), np.cos(fine['psi'])),
                         window=(0., float(fine['t'][-1])))
            snapshots.append(Snapshot(control=float(row.control), label=rf'$\gamma={row.control:.2f}$', row_id=row.row_id,
                                      x=x, glyph=glyph, edges=edges, groups=groups))
            checks.append(dict(row_id=row.row_id, observation_replay_max_error=replay_error, row_contract=contract))
    metrics = {key: rho(frame, key, 'held') for key in ['q', 'mean-|r|', 'mean-MPI']}
    for k, f in enumerate(spec.features):
        metrics[f'z{k}'] = rho(frame, f'z{k}', 'held')
    provenance = dict(run=LOCKING_RUN, scores_sha256=sha(result/'scores.csv'), features_sha256=analysis['features_sha256'],
                      held_records=int(held.sum()), snapshots=[s.row_id for s in snapshots], snapshot_checks=checks,
                      edge_receipts=receipts,
                      selection=f'held, eta nearest .6 (ties by row ID) at gamma {spec.snapshot_controls}; the gamma=.20 MPI '
                                'was fetched read-only from Gadi on 2026-10-21 and matches sources.json.')
    return SystemData(spec, frame, snapshots, metrics, provenance, primary='held')


# --------------------------------------------------------------------------- hardware

HARDWARE_RUN = ROOT/'results/hardware-pilot-261012/full-p90-r9-261014'
HARDWARE_GROUPS = (('pairs with a late-locking channel', '#d95f02'), ('coherent-core pairs', '#26828e'))


def late_channels(x, threshold=.9):
    """Channels whose mean incident carrier PLV is below `threshold` (fixed at ER setting 23)."""
    from research.empirical.hardware_pilot.calibrate_hardware_pilot import phase_properties
    plv = phase_properties(x, (90, 140))['plv']
    incident = (plv.sum(1)-np.diag(plv))/(len(plv)-1)
    return np.flatnonzero(incident < threshold), int(np.argmax(incident))


def hardware_glyph(x, late, reference):
    from scipy.ndimage import uniform_filter1d
    from research.empirical.hardware_pilot.calibrate_hardware_pilot import phase_properties, FS, TRIM
    p = phase_properties(x, (90, 140))
    u = p['unit']
    relative = np.unwrap(np.angle(u[late]*u[reference].conj()), axis=1)
    synchrony = uniform_filter1d(abs(u.mean(0)), round(5*FS/116), mode='nearest')
    t = (TRIM+np.arange(u.shape[1]))/FS
    # Every channel's wrapped carrier phase relative to the reference, late-locking channels on top.
    rows = np.r_[np.setdiff1d(np.arange(len(u)), late), late]
    raster = np.angle(u[rows]*u[reference].conj()).astype(np.float32)
    return dict(t=t, phase=relative, synchrony=synchrony, raster=raster, raster_marked=len(late),
                window=(float(t[0]), float(t[-1])))


def load_hardware(spec, steps=(0, 23, 87), display_max=87):
    from research.inference.benchmarks import hardware_baselines
    run = HARDWARE_RUN
    rows = json.loads((run/'manifest.json').read_text())['rows']
    names = [r['name'] for r in rows]
    summary = json.loads((run/'analysis/summary.json').read_text())
    hashes = {s['name']: s['mpi_sha256'] for s in summary['sources']}
    baseline = hardware_baselines.load()
    check(baseline.name.tolist() == names, 'Hardware baseline rows differ from the manifest')
    with np.load(run/'analysis/features.npz') as bank:
        z = bank['z']; order = list(bank['spi_order'])
        replay = (z[:, bank['er_features']].astype(float)-bank['er_center'])@bank['er_components'][0]
        np.testing.assert_allclose(replay, baseline.er_pc1, atol=1e-8)
        columns = [pair_index(order, f.spi_a, f.spi_b) for f in spec.features]
        z_rows = z[:, columns].astype(float)
    frame = pd.DataFrame(dict(row_id=names, control=baseline.voltage, step=baseline.step, Q=baseline[spec.Q_source or 'frequency_range'],
                              R1=baseline.R1, q=baseline.er_pc1, cohort=baseline.topology))
    frame['mean-|r|'] = baseline.mean_abs_r
    frame['mean-MPI'] = baseline.mean_PC1
    for k in range(len(columns)):
        frame[f'z{k}'] = z_rows[:, k]
    shown = frame[frame.step.le(display_max)].reset_index(drop=True)
    lookup = {(r['topology'], r['step']): i for i, r in enumerate(rows)}
    snapshots, receipts, checks = [], [], []
    with np.load(run/'observations.npz', allow_pickle=False) as observations:
        late, reference = late_channels(observations[rows[lookup['ER', 23]]['name']])
        check(late.tolist() == [11, 17, 20, 23], 'Late-locking channels differ from the calibrated 12/18/21/24')
        i, j = np.where(~np.eye(28, dtype=bool))
        groups = np.where(np.isin(i, late) | np.isin(j, late), 0, 1)
        for step in steps:
            index = lookup['ER', step]; name = rows[index]['name']
            x = observations[name]
            path, = (run/'mpis').glob(f'*/????-{name}/spi_mpis.npz')
            check(sha(path) == hashes[name], f'Hardware MPI changed: {name}')
            with np.load(path) as m:
                mpis = {k: m[k] for k in order}
            contract = verify_ordered_row(mpis, order, z[index])
            check(contract['passes'], f'Hardware z row is not the ordered contract: {name}')
            edges = {}
            for k, f in enumerate(spec.features):
                a, b, value = edge_z(mpis[f.spi_a], mpis[f.spi_b])
                check(abs(value-float(z[index, columns[k]])) < 2e-6, f'Edge replay failed: {f.label} {name}')
                edges[k] = (a, b, value)
                receipts.append(dict(row_id=name, feature=f.label, z=value, stored=float(z[index, columns[k]]),
                                     legacy_upper_symmetrized=upper_symmetrized_z(mpis[f.spi_a], mpis[f.spi_b])))
            checks.append(dict(row_id=name, row_contract=contract))
            snapshots.append(Snapshot(control=float(rows[index]['voltage']), label=f'{rows[index]["voltage"]:.3f} V',
                                      row_id=name, x=x, glyph=hardware_glyph(x, late, reference), edges=edges, groups=groups))
    metrics = {key: rho(shown, key, 'ER') for key in ['q', 'mean-|r|', 'mean-MPI']}
    metrics.update({f'SW {key}': rho(shown, key, 'SW') for key in ['q', 'mean-|r|', 'mean-MPI']})
    provenance = dict(run=str(run.relative_to(ROOT)), scores_sha256=sha(run/'analysis/scores.csv'),
                      baseline_provenance='results/order-parameter-inference/hardware-baselines-261021/provenance.json',
                      displayed=f'settings <= {display_max} (0-{display_max*.005:.3f} V); q fitted on all 29 ER records',
                      late_channels_1based=(late+1).tolist(), reference_channel_1based=int(reference+1),
                      snapshots=[s.row_id for s in snapshots], snapshot_checks=checks, edge_receipts=receipts)
    return SystemData(spec, shown, snapshots, metrics, provenance, primary='ER')


# --------------------------------------------------------------------------- stacked systems (no SPI–SPI inspection)

BASELINE_SCORES = ROOT/'results/baseline-comparison_260921'


def load_cml2d(spec):
    """Collective period-two CML: confirmation bank curves (544 rows, 32 seeds, M=32 of N=65,536).

    Snapshot MTS, global mean and final lattice come from the fixed pilot seed used in three_systems
    (26091115, a held seed of the pilot), at r = 3.84, 3.86212 and 3.89. The lattice is the archived
    state after the future reference window, not simultaneous with the observed window."""
    from research.inference.benchmarks.three_systems import load_cml
    data = load_cml()       # checks frozen hashes, replays every q from the frozen model
    base = data['frame']
    sign = json.loads((ROOT/'data/order-parameter-inference/cml2d/cml2d_confirmation_260911/primary-analysis/summary.json')
                      .read_text())['display_sign']
    frame = pd.DataFrame(dict(row_id=base.row_id, control=base.r, Q=base.Q_reference, q=sign*base.q,
                              trajectory=base.seed, cohort='held'))
    frame['mean-|r|'] = base.mean_abs_correlation.to_numpy()
    frame['mean-MPI'] = base.mean_PC1.to_numpy()
    snapshots = [Snapshot(control=e['control'], label=rf"$r={e['control']:g}$", row_id=e['row_id'], x=e['x'],
                          glyph=dict(t=np.arange(100), trace=e['mean'], lattice=e['field'], sensors=e['sensors'],
                                     window=(0., 99.)))
                 for e in data['examples']]
    metrics = {key: rho(frame, key, 'held') for key in ['q', 'mean-|r|', 'mean-MPI']}
    provenance = dict(scores_sha256=data['score_sha256'], q_replay_max_error=data['replay_error'],
                      snapshots=[s.row_id for s in snapshots], note=load_cml2d.__doc__.split('\n\n', 1)[1].strip())
    return SystemData(spec, frame, snapshots, metrics, provenance, primary='held')


SL_RUN = ROOT/'data/order-parameter-inference/stuart-landau/stuart_landau_locking_boundary_confirmation/stuart-landau-locking-boundary-confirmation'
SL_ANALYSIS = ROOT/'results/order-parameter-inference/stuart-landau/stuart_landau_locking_boundary_confirmation_analysis'


def load_stuart_landau(spec, controls=(.68, .725, .77), instance=0):
    """Fine Stuart–Landau locking-to-unsteady confirmation (152 rows: 19 gamma x 8 instances, M=N=32).

    q is the frozen coordinate applied without refitting. Snapshots: instance 0 at the range ends and the
    steepest interval; |Z(t)| is regenerated from the stored seed and checked against the archived input."""
    from mtsspi.generators.order_parameter import generate_stuart_landau
    scores = pd.read_csv(SL_ANALYSIS/'scores.csv')
    check(len(scores) == 152 and scores.instance.nunique() == 8 and scores.gamma.nunique() == 19, 'SL fine scope changed')
    base = pd.read_csv(BASELINE_SCORES/'inference-StuartLandauFine.csv').query('evaluation')
    merged = scores.merge(base[['instance', 'gamma', 'q', 'mean_PC1']], on=['instance', 'gamma'], validate='one_to_one',
                          suffixes=('', '_baseline'))
    np.testing.assert_allclose(merged.q, merged.q_baseline, atol=1e-8)
    frame = pd.DataFrame(dict(row_id=[f'I{i}-gamma{g:.3f}' for i, g in zip(merged.instance, merged.gamma)],
                              control=merged.gamma, Q=merged.Q_R_mean, q=merged.q, trajectory=merged.instance, cohort='held'))
    frame['mean-|r|'] = merged.mean_abs_correlation.to_numpy()
    frame['mean-MPI'] = merged.mean_PC1.to_numpy()
    snapshots, checks = [], []
    for gamma in controls:
        folder = SL_RUN/f"M32_T1000_I{instance}_gamma{gamma:.3f}".replace('gamma0.', 'gamma0p')
        meta = json.loads((folder/'meta.json').read_text())
        x = np.load(folder/'timeseries.npy').T.astype(float)
        params = dict(meta['generator']['resolved_params']); params.pop('N_full')
        regenerated, internals = generate_stuart_landau(M=32, T=1000, N_full=32, **params, return_internals=True,
                                                        rng=np.random.default_rng(meta['generator']['seed']))
        error = float(np.max(abs(regenerated.T-x)))
        check(error < 1e-6, f'Stuart–Landau regeneration differs from the archive: {folder.name}')
        R = np.abs(internals.order_parameter)
        snapshots.append(Snapshot(control=gamma, label=rf'$\gamma={gamma:g}$', row_id=folder.name, x=x,
                                  glyph=dict(t=np.arange(len(R))*params['sample_dt'], synchrony=R,
                                             window=(0., (len(R)-1)*params['sample_dt']))))
        checks.append(dict(row_id=folder.name, regeneration_max_error=error))
    metrics = {key: rho(frame, key, 'held') for key in ['q', 'mean-|r|', 'mean-MPI']}
    provenance = dict(scores_sha256=sha(SL_ANALYSIS/'scores.csv'), snapshot_checks=checks,
                      note=load_stuart_landau.__doc__.split('\n\n', 1)[1].strip())
    return SystemData(spec, frame, snapshots, metrics, provenance, primary='held')


KURAMOTO_RUN = ROOT/'data/order-parameter-inference/kuramoto/kuramoto_full_observation_260916/primary'


def load_kuramoto(spec, controls=(.625, 1.0075, 1.65), seed=260916009):
    """Full-observation Kuramoto (M=N=32): 128 held rows, 16 kappa x 8 seeds; exploratory (dominance screen
    1.478 < 1.5 unmet). Snapshots: the first held seed; R(t) regenerated with the pilot's generator call."""
    from research.inference.kuramoto import kuramoto_full_observation_260916 as K
    from mtsspi.generators.order_parameter import generate_kuramoto_order_parameter
    scores = pd.read_csv(KURAMOTO_RUN/'analysis/scores.csv').query("role == 'evaluation' and eligible")
    sign = json.loads((KURAMOTO_RUN/'analysis/summary.json').read_text())['display_sign']
    base = pd.read_csv(BASELINE_SCORES/'inference-Kuramoto.csv').set_index('row_id').loc[scores.row_id]
    np.testing.assert_allclose(base.q, scores.q, atol=1e-8)
    frame = pd.DataFrame(dict(row_id=scores.row_id, control=scores.control, Q=scores.Q_reference, q=sign*scores.q,
                              trajectory=scores.seed, cohort='held'))
    frame['mean-|r|'] = scores.mean_abs_correlation.to_numpy()
    frame['mean-MPI'] = base.mean_PC1.to_numpy()
    rows = json.loads((KURAMOTO_RUN/'manifest.json').read_text())['rows']
    snapshots, checks = [], []
    with np.load(KURAMOTO_RUN/'observations.npz') as observations:
        for kappa in controls:
            row, = [r for r in rows if r['seed'] == seed and np.isclose(r['control'], kappa)]
            x = observations[row['row_id']]
            y, truth = generate_kuramoto_order_parameter(
                M=32, T=1000, N_full=32, K=kappa*np.sqrt(8/np.pi), dt=.02, sample_dt=.1, burn_time=200,
                future_truth_T=10000, omega_mean=1, omega_std=1, frequency_distribution='gaussian',
                frequency_sampling='random', output='cos', rng=np.random.default_rng(seed),
                return_internals=True, store_full_phases=False)
            error = float(np.max(abs(y.T-x)))
            check(error < 1e-6, f"Kuramoto regeneration differs from the archive: {row['row_id']}")
            snapshots.append(Snapshot(control=kappa, label=rf'$\kappa={kappa:g}$', row_id=row['row_id'], x=x,
                                      glyph=dict(t=np.arange(len(truth.r_full))*.1, synchrony=truth.r_full,
                                                 window=(0., (len(truth.r_full)-1)*.1))))
            checks.append(dict(row_id=row['row_id'], regeneration_max_error=error))
    assert K.CONTROLS[0] == controls[0] and K.CONTROLS[-1] == controls[-1]
    metrics = {key: rho(frame, key, 'held') for key in ['q', 'mean-|r|', 'mean-MPI']}
    provenance = dict(display_sign=sign, snapshot_checks=checks, note=load_kuramoto.__doc__.split('\n\n', 1)[0])
    return SystemData(spec, frame, snapshots, metrics, provenance, primary='held')


# --------------------------------------------------------------------------- glyphs

def _bare(ax, spines=('bottom',)):
    ax.set_xticks([]); ax.set_yticks([])
    for name, spine in ax.spines.items():
        spine.set_visible(name in spines)
        spine.set_linewidth(.5)


def glyph_wrapped(ax, g, color=GLYPH_COLOR):
    """Wrapped relative phase: a sawtooth while phases slip, a flat band once locked."""
    stride = max(1, g['phase'].shape[1]//2500)
    for row in g['phase']:
        ax.scatter(g['t'][::stride], wrap_cycles(row)[::stride], s=.3, lw=0, color=color,
                   alpha=.55 if len(g['phase']) > 1 else .8, rasterized=True)
    ax.set(xlim=g['window'], ylim=(-.56, .56))
    ax.axhline(0, color='.82', lw=.4, zorder=0)
    _bare(ax, ('bottom', 'left'))


def glyph_unwrapped(ax, g, color=GLYPH_COLOR, limits=None):
    for row in g['phase']:
        ax.plot(g['t'], (row-row[0])/(2*np.pi), color=color, lw=.6, alpha=.75)
    ax.set_xlim(g['window'])
    if limits is not None:
        ax.set_ylim(limits)
    _bare(ax, ('bottom', 'left'))


def glyph_spiral(ax, g, color=GLYPH_COLOR):
    """Relative phase as angle, elapsed time as radius: slipping winds, locking draws a ray."""
    t = g['t']; radius = .18+.82*(t-t[0])/(t[-1]-t[0])
    for row in g['phase']:
        ax.plot(radius*np.cos(row-row[0]), radius*np.sin(row-row[0]), color=color, lw=.35, alpha=.8)
    ax.add_patch(plt.Circle((0, 0), 1., fill=False, lw=.4, color='.75'))
    ax.set(xlim=(-1.06, 1.06), ylim=(-1.06, 1.06)); ax.set_aspect('equal')
    _bare(ax, ())


def glyph_synchrony(ax, g, color=GLYPH_COLOR):
    """Order-parameter magnitude over the record; dotted line at 1."""
    ax.plot(g['t'], g['synchrony'], color=color, lw=.6)
    ax.axhline(1, color='.75', lw=.4, ls=':')
    ax.set(xlim=g['window'], ylim=(0, 1.06))
    _bare(ax, ('bottom', 'left'))


def glyph_lissajous(ax, g, color=GLYPH_COLOR, span=60.):
    """Slow-A against fast-B mean-field waveform over a fixed span: a closed 1:2 figure when locked."""
    t = g['t']; keep = t <= t[0]+span
    a, b = (v[keep] for v in g['lissajous'])
    ax.plot(a, b, color=color, lw=.35, alpha=.85)
    ax.set(xlim=(-1.08, 1.08), ylim=(-1.08, 1.08)); ax.set_aspect('equal')
    _bare(ax, ())


def glyph_raster(ax, g, color=None):
    """Channels x time, wrapped relative phase on a cyclic map: stripes are slips, flat rows are locked."""
    image = g['raster']
    ax.imshow(image, origin='lower', aspect='auto', cmap='twilight', vmin=-np.pi, vmax=np.pi,
              extent=(*g['window'], .5, len(image)+.5), interpolation='antialiased', rasterized=True)
    ax.axhline(len(image)-g['raster_marked']+.5, color='white', lw=.4)
    _bare(ax, ())


def glyph_trace(ax, g, color=GLYPH_COLOR, steps=40):
    """Global mean over the first `steps` iterations: a large zigzag is collective period-two order."""
    ax.plot(g['t'][:steps], g['trace'][:steps], color=color, lw=.6)
    ax.set(xlim=(0, steps-1), ylim=(.3, .9))
    _bare(ax, ('bottom', 'left'))


def glyph_lattice(ax, g, color=None):
    """Whole 256 x 256 lattice (state in [0, 1]); white circles mark the 32 observed sites."""
    field = g['lattice']
    side = field.shape[0]
    ax.imshow(field, origin='lower', cmap='viridis', vmin=0, vmax=1, interpolation='nearest', rasterized=True,
              extent=(-.5, side-.5, -.5, side-.5))
    sensors = np.asarray(g['sensors'])
    ax.scatter(sensors % side, sensors//side, s=1.1, facecolors='none', edgecolors='white', lw=.3)
    _bare(ax, ())


GLYPHS = {'wrapped': glyph_wrapped, 'unwrapped': glyph_unwrapped, 'spiral': glyph_spiral,
          'synchrony': glyph_synchrony, 'lissajous': glyph_lissajous, 'raster': glyph_raster, 'trace': glyph_trace,
          'lattice': glyph_lattice}
SQUARE_GLYPHS = ('spiral', 'lissajous', 'lattice')


def draw_glyphs(axes, data, kind):
    limits = None
    if kind == 'unwrapped':
        values = np.concatenate([((g := s.glyph)['phase']-g['phase'][:, :1]).ravel()/(2*np.pi) for s in data.snapshots])
        pad = .05*max(np.ptp(values), 1.)
        limits = (values.min()-pad, values.max()+pad)
    for ax, snapshot in zip(axes, data.snapshots, strict=True):
        if kind == 'unwrapped':
            glyph_unwrapped(ax, snapshot.glyph, limits=limits)
        else:
            GLYPHS[kind](ax, snapshot.glyph)
    title = data.spec.glyph_titles.get(kind)
    if title:
        axes[0].set_ylabel(title, fontsize=6.5, labelpad=1.5, rotation=0, ha='right', va='center')


# --------------------------------------------------------------------------- panels

def _box(ax, lw=.6):
    for spine in ax.spines.values():
        spine.set_visible(True); spine.set_linewidth(lw)


def _no_ticks(*axes):
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([]); ax.set_xticks([], minor=True); ax.set_yticks([], minor=True)


def italic(lines):
    """Italic multi-line text for either renderer (usetex ignores fontstyle)."""
    if plt.rcParams['text.usetex']:
        return dict(s='\n'.join(rf'\textit{{{line}}}' for line in lines))
    return dict(s='\n'.join(lines), style='italic')


def draw_heatmaps(axes, data):
    spec = data.spec
    start, count = spec.heatmap_window or (0, data.snapshots[0].x.shape[1])
    scaled, limit = robust_scaled([s.x for s in data.snapshots])
    from seaborn import color_palette
    cmap = color_palette('icefire', as_cmap=True)
    for ax, x in zip(axes, scaled, strict=True):
        m = x.shape[0]
        ax.imshow(x[:, start:start+count], origin='lower', aspect='auto', cmap=cmap, vmin=-limit, vmax=limit,
                  extent=(start, start+count, .5, m+.5), interpolation='nearest', rasterized=True)
        if spec.communities:
            for k in range(1, spec.communities):
                ax.axhline(k*m/spec.communities+.5, color='white', lw=.4)
            if ax is axes[0] and spec.community_labels:
                for k, letter in enumerate(spec.community_labels):
                    ax.text(-.025, (k+.5)/spec.communities, letter, transform=ax.transAxes, ha='right', va='center',
                            fontsize=4.5, color='.25')
        _no_ticks(ax)
        _box(ax, .5)


def mark_boundary(ax, spec):
    if isinstance(spec.boundary, tuple):
        ax.axvspan(*spec.boundary, color='.5', alpha=.16, lw=0)
    elif spec.boundary is not None:
        ax.axvline(spec.boundary, color='.5', lw=.6, ls=':')


SPREADS = ('trajectories', 'band', 'both', 'none')


def _series(axis, data, column, color, marker, *, ms=1.6, lw=1.3, alpha=1., spread='band', secondary=False, label=None):
    """Mean across instances (thick), with faint per-instance paths and/or a 10-90% band behind it."""
    if spread not in SPREADS:
        raise ValueError(f'spread must be one of {SPREADS}')
    part = data.frame[data.frame.cohort.eq(data.primary)]
    mean, lo, hi = curve(data.frame, column, data.primary)
    replicated = part.groupby('control').size().max() > 1
    if replicated and spread in ('trajectories', 'both') and 'trajectory' in part:
        paths = part.trajectory.nunique()
        alpha = float(np.clip(2.4/paths, .1, .32))
        for _, path in part.sort_values('control').groupby('trajectory'):
            axis.plot(path.control, path[column], color=color, lw=.45, alpha=alpha, zorder=1, solid_capstyle='round')
    if replicated and spread in ('band', 'both'):
        axis.fill_between(mean.index, lo, hi, color=color, alpha=.15, lw=0, zorder=1)
    handle, = axis.plot(mean.index, mean, marker+'-', color=color, ms=ms, lw=lw, alpha=alpha, label=label, zorder=3)
    if secondary:
        for cohort, ls in data.spec.secondary_style.items():
            mean, _, _ = curve(data.frame, column, cohort)
            axis.plot(mean.index, mean, ls=ls, marker=marker, mfc='white', color=color, ms=ms*.85, lw=lw*.7, alpha=.8)
    return handle


def display_readouts(data, columns=('q', 'mean-|r|', 'mean-MPI')):
    """Copy of `data` with readouts affinely rescaled so their control-mean curves span [0, 1] (display only).

    q keeps its displayed sign and mean-|r| its natural sign; the mean-MPI PC1 sign is arbitrary, so it is
    shown with the sign that correlates positively with q. No value enters any statistic.
    """
    frame = data.frame.copy()
    for column in columns:
        values = frame[column].astype(float)
        if column == 'mean-MPI' and spearmanr(values, frame.q).statistic < 0:
            values = -values
        mean = values[frame.cohort.eq(data.primary)].groupby(frame.control).mean()
        frame[column+' (display)'] = (values-mean.min())/(mean.max()-mean.min())
    return replace(data, frame=frame)


def draw_curves(ax, data, *, legend=True, spread='band', fontsize=7, baselines=False):
    """Control against Q (left, black, physical units) and readouts (right, arbitrary units, no ticks).

    With `baselines`, q, mean-|r| and mean-MPI PC1 are each rescaled to their own range and overlaid (baselines as
    mean lines only, to keep four curves legible); legend entries carry |rho_s| with Q over the displayed records.
    """
    spec = data.spec
    right = ax.twinx()
    handles = [_series(ax, data, 'Q', Q_COLOR, 'o', spread=spread, label='$Q$')]
    if baselines:
        shown = display_readouts(data)
        for column, color, marker, label in [('q', Z_COLOR, 's', '$q$'), ('mean-|r|', BASELINES['mean-|r|'], '', BASELINE_TITLES['mean-|r|']),
                                             ('mean-MPI', BASELINES['mean-MPI'], '', BASELINE_TITLES['mean-MPI'])]:
            handles.append(_series(right, shown, column+' (display)', color, marker, spread=spread if column == 'q' else 'none',
                                   lw=1.3 if column == 'q' else .8, alpha=1. if column == 'q' else .8,  # baselines: context, not focus
                                   label=f"{label} {data.metrics[column]:.2f}"))
        right.set_ylim(-.12, 1.12)
        right.set_ylabel('readouts, rescaled', labelpad=2, fontsize=fontsize)
    else:
        handles.append(_series(right, data, 'q', Z_COLOR, 's', spread=spread, label=f"$q$ {data.metrics['q']:.2f}"))
        if spec.q_limits:
            right.set_ylim(spec.q_limits)
        right.set_ylabel(spec.q_label, color=Z_COLOR, labelpad=2, fontsize=fontsize)
    if spec.Q_log:
        ax.set_yscale('log')
        ax.yaxis.set_major_locator(plt.LogLocator(base=10, numticks=4))
        ax.yaxis.set_minor_locator(plt.NullLocator())
    if spec.Q_limits:
        ax.set_ylim(spec.Q_limits)
    mark_boundary(ax, spec)
    Q_label = spec.Q_label.replace('freq. ', 'freq.\\ ') if plt.rcParams['text.usetex'] else spec.Q_label  # inter-word space
    ax.set_ylabel(Q_label, color=Q_COLOR, labelpad=1.5, fontsize=fontsize)
    right.set_yticks([]); right.set_yticks([], minor=True)       # q and rescaled readouts have no physical units
    ax.tick_params(labelsize=fontsize-.5, pad=1.5)
    ax.set_xlabel(spec.control_label, labelpad=1.5, fontsize=fontsize)
    if spec.control_ticks:
        ax.set_xticks(spec.control_ticks)
    _box(ax); _box(right)
    if legend:
        loc = spec.stack_legend_loc if baselines else spec.legend_loc
        loc, anchor = loc if isinstance(loc, tuple) else (loc, None)
        right.legend(handles=handles, loc=loc, bbox_to_anchor=anchor, fontsize=fontsize-1, handlelength=1.3,
                     borderaxespad=.3, labelspacing=.25, title=r'$|\rho_s|$ with $Q$', title_fontsize=fontsize-1.5,
                     alignment='left')
    return right


def _leader(ax, source, anchor, control, value):
    ax.plot(control, value, 'o', ms=3.6, mfc='none', mec='.25', mew=.55, zorder=6, clip_on=False)
    ax.figure.add_artist(ConnectionPatch(xyA=anchor, coordsA=source.transAxes, xyB=(control, value),
                                         coordsB=ax.transData, color='.45', lw=.4, zorder=7))


def draw_insets(ax, right, data, glyph):
    """MTS heatmap (boxed) with its physical glyph attached above, placed in clear space of the headline.

    Positions are axes fractions from `spec.inset_boxes` (heatmap lower-left corner); sizes are fixed in
    inches so text and marks stay legible. Leader lines join each inset to its control's point on Q.
    """
    g, spec = GEOMETRY, data.spec
    fig = ax.figure
    bbox = ax.get_position()
    fw, fh = fig.get_size_inches()
    aw, ah = bbox.width*fw, bbox.height*fh
    square = glyph in SQUARE_GLYPHS
    w, hh = g['inset_width']/aw, g['inset_heat']/ah
    gh = (g['inset_glyph_square'] if square else g['inset_glyph'])/ah
    gw = (g['inset_glyph_square']/aw) if square else w
    glyph_axes, heat_axes = [], []
    for (x, y) in spec.inset_boxes:
        heat = right.inset_axes([x, y, w, hh])
        heat.set_facecolor('white')
        glyph_axis = right.inset_axes([x+(w-gw)/2, y+hh+.012, gw, gh])
        glyph_axis.set_facecolor('none')
        heat_axes.append(heat); glyph_axes.append(glyph_axis)
    draw_heatmaps(heat_axes, data)
    draw_glyphs(glyph_axes, data, glyph)
    for gax, hax, snapshot, (x, y) in zip(glyph_axes, heat_axes, data.snapshots, spec.inset_boxes):
        hax.text(.5, -.06, snapshot.label, transform=hax.transAxes, ha='center', va='top', fontsize=6)
        value = value_at(data.frame, 'Q', snapshot.control, data.primary)
        px, py = ax.transAxes.inverted().transform(ax.transData.transform((snapshot.control, value)))
        top = y+hh+.012+gh
        if py < y:
            anchor, source = (.5, -.32), hax          # below the label under the heatmap
        elif py > top:
            anchor, source = (.5, 1.02), gax
        else:
            anchor, source = ((1., .5) if px > x+w else (0., .5)), hax
        _leader(ax, source, anchor, snapshot.control, value)
    return glyph_axes, heat_axes


def draw_baseline(ax, data, key, *, spread='band'):
    """Q (black) and one simple baseline against control: boxed, no ticks."""
    spec = data.spec
    right = ax.twinx()
    _series(ax, data, 'Q', Q_COLOR, 'o', ms=1.2, lw=.9, spread=spread)
    _series(right, data, key, BASELINES[key], 's', ms=1.2, lw=.9, spread=spread)
    if spec.Q_log:
        ax.set_yscale('log')
    if spec.Q_limits:
        ax.set_ylim(spec.Q_limits)
    mark_boundary(ax, spec)
    _no_ticks(ax, right)
    _box(ax, .5); _box(right, .5)
    ax.set_title(f"{BASELINE_TITLES[key]}   " + rf"$|\rho_s|={data.metrics[key]:.2f}$", fontsize=6.5, pad=2,
                 color=BASELINES[key])
    ax.set_xlabel(spec.control_short, fontsize=6, labelpad=1)
    return right


def draw_feature(ax, data, k, *, ylabel=True):
    """z_ab against control on a fixed [-1, 1] range (10-90% band): boxed, no ticks, snapshots marked."""
    spec = data.spec
    f = spec.features[k]
    ax.axhline(0, color='.85', lw=.5, zorder=0)
    _series(ax, data, f'z{k}', Z_COLOR, 'o', ms=1.2, lw=.9, spread='band', secondary=True)
    mark_boundary(ax, spec)
    for snapshot in data.snapshots:
        ax.plot(snapshot.control, 1.0, 'v', color='.45', ms=2.2, zorder=5)
    ax.set(ylim=(-1.1, 1.1))
    _no_ticks(ax)
    _box(ax, .5)
    reading = textwrap.wrap(f.reading, 26)
    ax.set_title(f.label, fontsize=6.5, pad=3+6.3*len(reading), linespacing=1.0)
    ax.text(.5, 1.03, transform=ax.transAxes, ha='center', va='bottom', fontsize=5.6, color='.35', linespacing=1.0,
            **italic(reading))
    ax.set_xlabel(spec.control_short, fontsize=6, labelpad=1)
    if ylabel:
        ax.set_ylabel(r'$z_{ab}\in[-1,1]$', fontsize=6, labelpad=1)


def edge_limits(data, k):
    limits = []
    for dim in (0, 1):
        values = np.concatenate([s.edges[k][dim] for s in data.snapshots])
        values = values[np.isfinite(values)]
        pad = max(float(np.ptp(values))*.05, 1e-9)
        limits.append((values.min()-pad, values.max()+pad))
    return limits


CORNERS = {'upper left': (.05, .95, 'left', 'top'), 'upper right': (.95, .95, 'right', 'top'),
           'lower left': (.05, .05, 'left', 'bottom'), 'lower right': (.95, .05, 'right', 'bottom')}


def emptiest_corner(axes, points, box=(.42, .2)):
    """Corner whose (box-sized) region holds the fewest points across all `axes` (axes-fraction units)."""
    counts = dict.fromkeys(CORNERS, 0)
    for ax, (a, b) in zip(axes, points):
        (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
        u, v = (np.asarray(a)-x0)/(x1-x0), (np.asarray(b)-y0)/(y1-y0)
        for name, (x, y, ha, va) in CORNERS.items():
            inside_u = u < box[0] if ha == 'left' else u > 1-box[0]
            inside_v = v > 1-box[1] if va == 'top' else v < box[1]
            counts[name] += int(np.sum(inside_u & inside_v))
    return min(counts, key=counts.get)


def annotate_z(ax, z, corner):
    x, y, ha, va = CORNERS[corner]
    ax.text(x, y, rf'$z={z:+.2f}$', transform=ax.transAxes, ha=ha, va=va, fontsize=6,
            bbox=dict(boxstyle='square,pad=.15', facecolor='white', edgecolor='none', alpha=.8))


def draw_edge_plane(ax, data, k, i, *, limits=None, axis_labels=True, z_corner=None):
    """SPI a against SPI b across all ordered channel pairs at one snapshot: boxed, no ticks.

    z is written in `z_corner` (one corner per row, chosen as the emptiest), or under the title if None.
    """
    spec, snapshot = data.spec, data.snapshots[i]
    a, b, z = snapshot.edges[k]
    order = np.argsort(snapshot.groups, kind='stable')[::-1]   # first-listed group drawn on top
    colors = np.array([c for _, c in spec.edge_groups])
    ax.scatter(a[order], b[order], s=1.8, lw=0, c=colors[snapshot.groups[order]], alpha=.6, rasterized=True)
    if limits is not None:
        ax.set(xlim=limits[0], ylim=limits[1])
    else:
        ax.margins(.06)
    _no_ticks(ax)
    _box(ax, .5)
    if z_corner is None:
        ax.set_title(snapshot.label+'\n'+rf'$z={z:+.2f}$', fontsize=6, pad=2, linespacing=1.1)
    else:
        ax.set_title(snapshot.label, fontsize=6, pad=2)
        annotate_z(ax, z, z_corner)
    f = spec.features[k]
    if axis_labels:
        ax.set_xlabel(short_spi(f.spi_a), fontsize=6, labelpad=1)
        ax.set_ylabel(short_spi(f.spi_b), fontsize=6, labelpad=1)


SPI_NAMES = {'lcss': 'LCSS', 'te_kraskov_NN-4_DCE-AUTO_k-1_kt-1_l-1_lt-1': 'TE (KSG)',
             'coint_johansen_max_eig_stat_order-0_ardiff-10': 'Johansen max-eig.',
             'coint_johansen_trace_stat_order-1_ardiff-1': 'Johansen trace',
             'corr_pearson_tau-2': 'Pearson lag 2', 'corr_pearson_tau-20-sq': r'Pearson lag 20$^2$',
             'si_kernel_W-0.5_k-1': 'SI (kernel)', 'ids': 'IDS',
             'corr_pearson_tau-1': 'Pearson 2 ms', 'corr_pearson_tau-10': 'Pearson 20 ms'}


def short_spi(name):
    return SPI_NAMES.get(name, name)


def edge_legend(ax, spec, ncol=None):
    handles = [Line2D([], [], marker='o', ls='', color=c, ms=2.6, label=g) for g, c in spec.edge_groups]
    ax.legend(handles=handles, loc='center', fontsize=6, handletextpad=.15, borderaxespad=0, frameon=False,
              ncol=ncol or len(handles), columnspacing=1.0)
    ax.axis('off')


def cohort_legend(ax, spec):
    handles = [Line2D([], [], color=Z_COLOR, marker='o', ms=2, lw=.85, label=spec.primary_label)]
    handles += [Line2D([], [], color=Z_COLOR, ls=ls, marker='o', mfc='white', ms=2, lw=.6, label=f'{c}: frozen projection')
                for c, ls in spec.secondary_style.items()]
    handles += [Line2D([], [], color='.45', marker='v', ls='', ms=2.6, label='snapshot controls')]
    ax.legend(handles=handles, loc='center left', fontsize=5.8, frameon=False, borderaxespad=0, handlelength=1.5,
              labelspacing=.35)
    ax.axis('off')


# --------------------------------------------------------------------------- inch-based layout

GEOMETRY = dict(width=3.55, left=.42, right=.30, title=.20, curves=2.55, curves_bottom=.30,
                inset_width=.62, inset_heat=.24, inset_glyph=.22, inset_glyph_square=.40,
                row_gap=.10, baseline_title=.14, baseline=.50, baseline_bottom=.16,
                feature_title=.36, feature=.80, feature_bottom=.20, edge_title=.14, edge=.80, edge_legend=.32)
STACK = dict(title=.20, glyph=.22, glyph_square=.44, glyph_label=.10, gap=.03, heat=.33, leader=.05, curves=1.05,
             curves_bottom=.30)


def axes_at(fig, x, y, w, h, **kw):
    """Axes from inch coordinates measured from the figure's top-left corner."""
    fw, fh = fig.get_size_inches()
    return fig.add_axes([x/fw, 1-(y+h)/fh, w/fw, h/fh], **kw)


def headline_height(curves=None):
    g = GEOMETRY
    return g['title']+(curves or g['curves'])+g['curves_bottom']


def column_height(spec, edge_rows, curves=None):
    g = GEOMETRY
    return (headline_height(curves)+g['row_gap']+g['baseline_title']+g['baseline']+g['baseline_bottom']+g['row_gap']
            +g['feature_title']+g['feature']+g['feature_bottom']+len(edge_rows)*(g['edge_title']+g['edge'])
            +(g['edge_legend'] if edge_rows else 0))


def stack_height(glyph, curves=None):
    s = STACK
    return (s['title']+s['glyph_label']+(s['glyph_square'] if glyph in SQUARE_GLYPHS else s['glyph'])+s['gap']+s['heat']
            +s['leader']+(s['curves'] if curves is None else curves)+s['curves_bottom'])


def bold(text):
    """Bold text for either renderer (usetex ignores fontweight)."""
    if plt.rcParams['text.usetex']:
        return dict(s=rf'\textbf{{{text}}}')
    return dict(s=text, fontweight='bold')


def panel_label(fig, x, y, text):
    if text:
        fw, fh = fig.get_size_inches()
        fig.text(x/fw, 1-y/fh, fontsize=9, va='top', ha='left', **bold(text))


def _titles(fig, x, y, title):
    fw, fh = fig.get_size_inches()
    fig.text(x/fw, 1-(y+.02)/fh, fontsize=7.5, va='top', **bold(title))


def draw_headline(fig, x0, y0, data, glyph=None, *, label=None, spread='band', curves=None):
    """Title, boxed Q/q curves, and MTS+glyph insets inside the axes; returns the bottom (in)."""
    g, spec = GEOMETRY, data.spec
    glyph = glyph or spec.glyphs[0]
    left, width = x0+g['left'], g['width']-g['left']-g['right']
    _titles(fig, left, y0, spec.title)
    panel_label(fig, x0, y0, label)
    y = y0+g['title']
    height = curves or g['curves']
    ax = axes_at(fig, left, y, width, height)
    right = draw_curves(ax, data, spread=spread)
    glyphs, heats = draw_insets(ax, right, data, glyph)
    return y+height+g['curves_bottom'], dict(curves=ax, right=right, glyphs=glyphs, heatmaps=heats)


def draw_stack_panel(fig, x0, y0, data, glyph=None, *, label=None, spread='band', curves=None):
    """Compact system panel: MTS heatmaps with glyphs above the Q/q curves; baselines summarised in text."""
    g, s, spec = GEOMETRY, STACK, data.spec
    glyph = glyph or spec.glyphs[0]
    left, width = x0+g['left'], g['width']-g['left']-g['right']
    _titles(fig, left, y0, spec.title)
    panel_label(fig, x0, y0, label)
    y = y0+s['title']+s['glyph_label']
    gap = .14
    inset = (width-2*gap)/3
    square = glyph in SQUARE_GLYPHS
    gh = s['glyph_square'] if square else s['glyph']
    glyph_axes, heat_axes = [], []
    for k in range(3):
        xk = left+k*(inset+gap)
        glyph_axes.append(axes_at(fig, xk+(inset-gh)/2, y, gh, gh) if square else axes_at(fig, xk, y, inset, gh))
        heat_axes.append(axes_at(fig, xk, y+gh+s['gap'], inset, s['heat']))
    draw_glyphs(glyph_axes, data, glyph)
    for gax, snapshot in zip(glyph_axes, data.snapshots):
        gax.set_title(snapshot.label, fontsize=6, pad=1.5)
    draw_heatmaps(heat_axes, data)
    y += gh+s['gap']+s['heat']+s['leader']
    height = curves or s['curves']
    ax = axes_at(fig, left, y, width, height)
    right = draw_curves(ax, data, spread=spread, fontsize=6.5, baselines=True)
    for hax, snapshot in zip(heat_axes, data.snapshots):
        _leader(ax, hax, (.5, 0), snapshot.control, value_at(data.frame, 'Q', snapshot.control, data.primary))
    return y+height+s['curves_bottom']


def draw_column(fig, x0, y0, data, glyph=None, edge_rows=None, labels='abcd', spread='band', curves=None):
    g, spec = GEOMETRY, data.spec
    edge_rows = spec.edge_features if edge_rows is None else edge_rows
    left, width = x0+g['left'], g['width']-g['left']-g['right']
    y, _ = draw_headline(fig, x0, y0, data, glyph, label=labels[0], spread=spread, curves=curves)
    y += g['row_gap']
    panel_label(fig, x0, y, labels[1])
    y += g['baseline_title']
    gap = .22
    for j, key in enumerate(BASELINES):
        draw_baseline(axes_at(fig, left+j*((width-gap)/2+gap), y, (width-gap)/2, g['baseline']), data, key, spread=spread)
    y += g['baseline']+g['baseline_bottom']+g['row_gap']
    # Features and edge planes share columns: three equal boxes spanning the curve width.
    side = g['feature']
    step = (width-side)/2
    panel_label(fig, x0, y, labels[2])
    y += g['feature_title']
    for k in range(3):
        if k < len(spec.features):
            draw_feature(axes_at(fig, left+k*step, y, side, side), data, k, ylabel=k == 0)
        else:
            cohort_legend(axes_at(fig, left+k*step-.05, y, side+.4, side), spec)
    y += side+g['feature_bottom']
    for r, k in enumerate(edge_rows):
        panel_label(fig, x0, y, labels[3] if r == 0 else None)
        y += g['edge_title']
        limits = edge_limits(data, k) if spec.edge_scale == 'shared' else None
        axes = [axes_at(fig, left+i*step, y, side, side) for i in range(3)]
        for i, ax in enumerate(axes):
            draw_edge_plane(ax, data, k, i, limits=limits, axis_labels=i == 0)
        corner = emptiest_corner(axes, [s.edges[k][:2] for s in data.snapshots])
        for ax, snapshot in zip(axes, data.snapshots):
            ax.set_title(snapshot.label, fontsize=6, pad=2)
            annotate_z(ax, snapshot.edges[k][2], corner)
        y += side
    if edge_rows:
        edge_legend(axes_at(fig, left, y+.17, width, g['edge_legend']-.17), spec)
        y += g['edge_legend']
    return y


def composite(systems, glyphs=None, edge_rows=None, gap=.08, spread='band'):
    """System columns side by side (full columns); `glyphs` maps name to glyph variant."""
    style()
    glyphs, edge_rows = glyphs or {}, edge_rows or {}
    heights = [column_height(d.spec, edge_rows.get(d.spec.name, d.spec.edge_features)) for d in systems]
    width = len(systems)*GEOMETRY['width']+(len(systems)-1)*gap
    fig = plt.figure(figsize=(width, max(heights)))
    letters = 'abcdefghijklmnop'
    for c, data in enumerate(systems):
        labels = [letters[c+len(systems)*r] for r in range(4)]
        draw_column(fig, c*(GEOMETRY['width']+gap), 0, data, glyphs.get(data.spec.name),
                    edge_rows.get(data.spec.name), labels, spread)
    return fig


def composite_stacked(main, stack, glyphs=None, edge_rows=None, gap=.08, spread='band'):
    """Main system as a full column (left); `stack` systems as compact panels (right), heights balanced."""
    style()
    glyphs = glyphs or {}
    rows = main.spec.edge_features if edge_rows is None else edge_rows
    left = column_height(main.spec, rows)
    kinds = [glyphs.get(d.spec.name, d.spec.glyphs[0]) for d in stack]
    fixed = sum(stack_height(k, 0) for k in kinds)
    curves = max(STACK['curves'], (left-fixed)/len(stack))
    height = max(left, fixed+curves*len(stack))
    main_curves = GEOMETRY['curves']+height-left      # the main headline absorbs any height difference
    fig = plt.figure(figsize=(2*GEOMETRY['width']+gap, height))
    draw_column(fig, 0, 0, main, glyphs.get(main.spec.name), rows, 'abcd', spread, main_curves)
    y = 0.
    for data, kind, label in zip(stack, kinds, 'efgh'):
        y = draw_stack_panel(fig, GEOMETRY['width']+gap, y, data, kind, label=label, spread=spread, curves=curves)
    return fig


def save(fig, name, out=OUT, formats=('svg', 'pdf', 'png'), dpi=600):
    out.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        fig.savefig(out/f'{name}.{ext}', dpi=dpi, bbox_inches='tight', pad_inches=.03)
    plt.close(fig)
    return out/f'{name}.png'


def export_panels(data, out=OUT, *, stacked=False, spread='band'):
    """Every panel as its own SVG/PDF/PNG (600 dpi), at the size it occupies in the composite."""
    style()
    g, spec, folder, written = GEOMETRY, data.spec, out/'panels'/data.spec.name, []
    width = g['width']-g['left']-g['right']

    def emit(fig, name):
        written.append(save(fig, name, folder))

    def single(w, h):
        fig = plt.figure(figsize=(w+.6, h+.5))
        return fig, axes_at(fig, .3, .2, w, h)

    for glyph in spec.glyphs:
        if stacked:
            fig = plt.figure(figsize=(g['width'], stack_height(glyph)))
            draw_stack_panel(fig, 0, 0, data, glyph, spread=spread)
            emit(fig, f'panel-{glyph}')
        else:
            fig = plt.figure(figsize=(g['width'], headline_height()))
            draw_headline(fig, 0, 0, data, glyph, spread=spread)
            emit(fig, f'headline-{glyph}')
        square = glyph in SQUARE_GLYPHS
        gw, gh = (g['inset_glyph_square'],)*2 if square else (g['inset_width'], g['inset_glyph'])
        pairs = [single(gw, gh) for _ in data.snapshots]
        draw_glyphs([ax for _, ax in pairs], data, glyph)
        for i, (fig, _) in enumerate(pairs):
            emit(fig, f'glyph-{glyph}-{i+1}')
    for mode in ('trajectories', 'band'):
        fig = plt.figure(figsize=(g['width'], (STACK['curves'] if stacked else g['curves'])+g['curves_bottom']+.2))
        draw_curves(axes_at(fig, g['left'], .1, width, STACK['curves'] if stacked else g['curves']), data, spread=mode)
        emit(fig, f'curves-{mode}')
    pairs = [single(g['inset_width'], g['inset_heat']) for _ in data.snapshots]
    draw_heatmaps([ax for _, ax in pairs], data)
    for i, (fig, _) in enumerate(pairs):
        emit(fig, f'heatmap-{i+1}')
    if stacked:
        return written
    for key, slug in zip(BASELINES, ('mean-abs-r', 'mean-mpi')):
        fig, ax = single((width-.22)/2, g['baseline'])
        draw_baseline(ax, data, key, spread=spread)
        emit(fig, f'baseline-{slug}')
    for k in range(len(spec.features)):
        fig, ax = single(g['feature'], g['feature'])
        draw_feature(ax, data, k)
        emit(fig, f'feature-{k+1}')
        limits = edge_limits(data, k) if spec.edge_scale == 'shared' else None
        pairs = [single(g['edge'], g['edge']) for _ in data.snapshots]
        for i, (_, ax) in enumerate(pairs):
            draw_edge_plane(ax, data, k, i, limits=limits)
        corner = emptiest_corner([ax for _, ax in pairs], [s.edges[k][:2] for s in data.snapshots])
        for i, ((fig, ax), snapshot) in enumerate(zip(pairs, data.snapshots)):
            ax.set_title(snapshot.label, fontsize=6, pad=2)
            annotate_z(ax, snapshot.edges[k][2], corner)
            emit(fig, f'edges-{k+1}-{i+1}')
    fig, ax = single(2.4, .2)
    edge_legend(ax, spec)
    emit(fig, 'edges-legend')
    fig, ax = single(1.0, .5)
    cohort_legend(ax, spec)
    emit(fig, 'feature-legend')
    return written


def glyph_gallery(datasets, size=(.95, .34)):
    """Every glyph variant (rows) at the three snapshots (columns), one block per system, for choosing."""
    style()
    rows = [(d, kind) for d in datasets for kind in d.spec.glyphs]
    fig = plt.figure(figsize=(3*(size[0]+.15)+1.1, sum((.62 if k in SQUARE_GLYPHS else size[1])+.28 for _, k in rows)+.1))
    y = .1
    for data, kind in rows:
        h = .62 if kind in SQUARE_GLYPHS else size[1]
        axes = [axes_at(fig, 1.0+i*(size[0]+.15)+((size[0]-h)/2 if kind in SQUARE_GLYPHS else 0), y+.14,
                        h if kind in SQUARE_GLYPHS else size[0], h) for i in range(3)]
        draw_glyphs(axes, data, kind)
        for ax, snapshot in zip(axes, data.snapshots):
            ax.set_title(snapshot.label, fontsize=6, pad=1)
        fw, fh = fig.get_size_inches()
        fig.text(.05/fw, 1-(y+.14+h/2)/fh, f'{data.spec.name}\n{kind}', fontsize=7, va='center')
        y += h+.28
    return fig


def compare_Q(datasets, titles, spread='band'):
    """Headline curves only, side by side, e.g. for alternative physical quantities or systems."""
    style()
    g = GEOMETRY
    fig = plt.figure(figsize=(len(datasets)*g['width'], g['curves']*.6+.75))
    for i, (data, title) in enumerate(zip(datasets, titles)):
        ax = axes_at(fig, i*g['width']+g['left'], .3, g['width']-g['left']-g['right'], g['curves']*.6)
        draw_curves(ax, data, spread=spread)
        m = data.metrics
        ax.set_title(title+rf"; $|\rho_s|$: $q$ {m['q']:.2f}, mean-$|r|$ {m['mean-|r|']:.2f}, mean-MPI {m['mean-MPI']:.2f}",
                     fontsize=6.5)
    return fig


def receipt(datasets, out=OUT):
    out.mkdir(parents=True, exist_ok=True)
    record = {d.spec.name: dict(metrics=d.metrics, provenance=d.provenance) for d in datasets}
    record['source_sha256'] = sha(__file__)
    record['claims'] = ('q tracks Q across controls; no calibration, mechanism or superiority is claimed. '
                        'Faint lines are individual held instances (paired across controls except locking, whose '
                        'realisations are independent per control); bands are 10-90% across instances, not confidence '
                        'intervals; hardware has one record per setting.')
    (out/'provenance.json').write_text(json.dumps(record, indent=2, default=float)+'\n')
    return out/'provenance.json'


# --------------------------------------------------------------------------- registry

LOCKING_FEATURES = (
    Feature('lcss', 'te_kraskov_NN-4_DCE-AUTO_k-1_kt-1_l-1_lt-1', 'LCSS\n$\\times$ TE',
            'predictive dependence vs waveform similarity (candidate)'),
    Feature('coint_johansen_max_eig_stat_order-0_ardiff-10', 'coint_johansen_trace_stat_order-1_ardiff-1',
            'Johansen $\\times$\nJohansen', 'short- vs long-history model sensitivity'),
    Feature('corr_pearson_tau-2', 'corr_pearson_tau-20-sq', 'Pearson lag 2\n$\\times$ lag 20$^2$',
            'lag profile (frequency entrainment)'),
)
HARDWARE_FEATURES = (
    Feature('si_kernel_W-0.5_k-1', 'ids', 'kernel SI\n$\\times$ IDS', 'recruitment-associated profile reorganisation'),
    Feature('corr_pearson_tau-1', 'corr_pearson_tau-10', 'Pearson 2 ms\n$\\times$ 20 ms', 'lag-profile / spectral organisation'),
)

SYSTEMS = {
    'locking': Spec(
        name='locking', title='2:1 cross-frequency locking', loader=load_locking,
        subtitle='$M=N=24$, $T=1000$, varying sensor noise; 168 held records',
        control_label=r'Cross-coupling $\gamma$', control_short=r'$\gamma$', Q_label=r'Locking index $Q$',
        boundary=.1017, features=LOCKING_FEATURES, edge_features=(0,), glyphs=('wrapped', 'spiral', 'lissajous'),
        glyph_titles={'wrapped': r'$\Phi$'}, heatmap_window=(0, 80), dt=.5, communities=3, edge_groups=LOCKING_GROUPS,
        control_ticks=(0, .1, .2), legend_loc='upper left', inset_boxes=((.06, .386), (.535, .10), (.76, .535)),
        Q_limits=(-.04, 1.06), community_labels=('A', 'B', 'C'), snapshot_controls=(0., .10, .20),
        note='M=N=24, T=1000, recording-specific sensor noise; 168 held records (8 seeds x 21 controls).'),
    'hardware': Spec(
        name='hardware', title='Electronic oscillators', loader=load_hardware,
        subtitle='$M=N=28$, $T=5000$ at 500 Hz; ER sweep, one recording per setting',
        control_label='Coupling-control voltage (V)', control_short='V', Q_label='Carrier freq. range (Hz)',
        Q_log=True, boundary=(.115, .12), features=HARDWARE_FEATURES, edge_features=(0,),
        glyphs=('wrapped', 'synchrony', 'raster'),
        glyph_titles={'wrapped': r'$\Delta\phi$', 'synchrony': '$R$', 'raster': r'$\Delta\phi$'},
        heatmap_window=(500, 40), dt=.002, edge_groups=HARDWARE_GROUPS, secondary_style={'SW': '--'},
        control_ticks=(0, .2, .4), legend_loc=('center left', (.0, .68)), edge_scale='panel', primary_label='ER (fit)',
        stack_legend_loc='center right',
        inset_boxes=((.04, .30), (.40, .34), (.70, .34)),
        note='M=N=28, T=5000 at 500 Hz; one record per setting; q fitted on all 29 ER records, SW projected.'),
    'stuart-landau': Spec(
        name='stuart-landau', title='Stuart--Landau: locked to unsteady', loader=load_stuart_landau,
        subtitle='$M=N=32$, $T=1000$; 8 held instances', control_label=r'Frequency half-width $\gamma$',
        control_short=r'$\gamma$', Q_label=r'Mean $R$', glyphs=('synchrony',), glyph_titles={'synchrony': '$R$'},
        heatmap_window=(0, 150), dt=.1, control_ticks=(.68, .72, .76), legend_loc='lower left', boundary=(.72, .725),
        stack_legend_loc='lower left',
        note='Fine confirmation, frozen coordinate; Matthews-Strogatz K=.8 path.'),
    'kuramoto': Spec(
        name='kuramoto', title='Kuramoto: phase synchronisation', loader=load_kuramoto,
        subtitle='$M=N=32$, $T=1000$; 8 held seeds; exploratory', control_label=r'Reduced coupling $\kappa$',
        control_short=r'$\kappa$', Q_label=r'Mean $R$', glyphs=('synchrony',), glyph_titles={'synchrony': '$R$'},
        heatmap_window=(0, 200), dt=.1, control_ticks=(.75, 1., 1.25, 1.5), legend_loc='lower right', boundary=1.,
        stack_legend_loc='upper left',
        note='Full observation; PC1 dominance screen unmet (1.478 < 1.5).'),
    'cml2d': Spec(
        name='cml2d', title='2D coupled maps: collective period two', loader=load_cml2d,
        subtitle='$M=32$ of $N=65{,}536$ sites, $T=1000$; 32 confirmation seeds',
        control_label='Local nonlinearity $r$', control_short='$r$', Q_label='Period-two $Q$', boundary=3.86212,
        glyphs=('lattice', 'trace'), glyph_titles={'trace': r'$\bar x$'}, heatmap_window=(0, 40),
        control_ticks=(3.84, 3.865, 3.89), legend_loc='lower left',
        note='Confirmation curves; snapshots from the fixed pilot seed (lattice after the reference window).'),
}


def load(name, **changes):
    """Load a registered system; keyword changes override its Spec (e.g. Q, labels, glyph order)."""
    if name not in SYSTEMS:
        raise KeyError(f'Unknown system {name!r}; registered: {sorted(SYSTEMS)}')
    spec = replace(SYSTEMS[name], **changes)
    return spec.loader(spec)
