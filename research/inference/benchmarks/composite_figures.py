"""Composite order-parameter figure: q = PC1 of SPI–SPI z tracking a physical Q without labels.

Presentation and verification only. Every q, baseline and z value is an archived, target-blind
readout; nothing here refits a coordinate or uses Q to choose what is shown. Snapshot records
follow fixed, pre-existing rules (never Q or q), and every displayed edge profile is replayed from
its archived MPI over all M(M-1) ordered off-diagonal entries (`unified_ordered_v3`).

Layout per system (one column each): a headline (control against Q and q) with three MTS insets,
leader lines to the Q curve and a physical glyph above each inset; two baseline glyphs (mean-|r|,
mean-MPI PC1); z_ab against control for named features; and edge planes (SPI a against SPI b across
ordered channel pairs) at the three snapshot controls.

Registry: `SYSTEMS[name]` is a `Spec` (presentation choices) whose `loader` returns a `SystemData`.
Adding a benchmarked system (Kuramoto, Stuart–Landau, ...) means writing one loader that returns a
per-record frame with columns control/Q/q/mean-|r|/mean-MPI/cohort plus `z0..zk` for the listed
features, three `Snapshot`s (MTS, glyph arrays, ordered edge profiles) and provenance; the drawing
code is shared. `cml2d` is registered as a working example outside the two-system composite.
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


def style():
    """Portable typography (no external TeX); editable text in SVG/PDF for external assembly."""
    import logging
    from research.inference.benchmarks.three_systems import style as base
    base()
    logging.getLogger('fontTools').setLevel(logging.ERROR)  # Type 42 embedding warns about font timestamps
    plt.rcParams.update({'svg.fonttype': 'none', 'pdf.fonttype': 42, 'axes.linewidth': .6,
                         'xtick.major.width': .6, 'ytick.major.width': .6, 'xtick.major.size': 2.5,
                         'ytick.major.size': 2.5, 'legend.fontsize': 7, 'axes.titlesize': 8,
                         'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7})


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
LOCKING_GROUPS = (('A–B pairs', '#CC79A7'), ('within a community', '#0072B2'), ('pairs with C', '#b5b5b5'))


def locking_groups(n=8):
    """Group per ordered edge: 0 = A–B (the locking pair), 1 = within a community, 2 = involving C."""
    i, j = np.where(~np.eye(3*n, dtype=bool))
    a, b = i//n, j//n
    return np.where(((a == 0) & (b == 1)) | ((a == 1) & (b == 0)), 0, np.where(a == b, 1, 2))


def load_locking(spec):
    from research.inference.locking.plot_locking_m24_edge_profiles import choose_records
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
        chosen = choose_records(scores)
        stored_rows = {r.row_id: bank['z'][int(np.flatnonzero(scores.row_id.eq(r.row_id))[0])].astype(float) for r in chosen}
    part = scores[held]
    frame = pd.DataFrame(dict(row_id=part.row_id.to_numpy(), control=part.control.to_numpy(), Q=part.Q_lock.to_numpy(),
                              q=part.z_PC1.to_numpy(), seed=part.seed.to_numpy(), eta=part.eta.to_numpy(), cohort='held'))
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
                      selection='plot_locking_m24_edge_profiles.choose_records: held, eta nearest .6 at gamma 0/.10/.14; '
                                'the only held records whose MPIs are local.')
    return SystemData(spec, frame, snapshots, metrics, provenance, primary='held')


# --------------------------------------------------------------------------- hardware

HARDWARE_RUN = ROOT/'results/hardware-pilot-261012/full-p90-r9-261014'
HARDWARE_GROUPS = (('pairs with a\nlate-locking channel', '#d95f02'), ('coherent-core pairs', '#26828e'))


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


# --------------------------------------------------------------------------- CML2D (registry example)

def load_cml2d(spec, controls=(3.84, 3.86212, 3.89)):
    """Collective period-two CML, confirmation bank (544 held rows, M=32 of N=65,536, T=1000).

    Snapshots: the first confirmation seed at three fixed r values (as in three_systems); MTS, MPIs and
    z all come from those rows. The glyph is the mean of the 32 observed sensors over the first 100
    steps (a period-two proxy, not the global Q, whose lattice masters are not local)."""
    from research.inference.benchmarks.three_systems import load_cml
    data = load_cml()
    base = data['frame'].query('eligible').reset_index(drop=True)
    root = ROOT/'data/order-parameter-inference/cml2d/cml2d_confirmation_260911'
    eligibility = json.loads((root/'primary-analysis/eligibility.json').read_text())
    hashes = {r['row_id']: r['mpi_sha256'] for r in eligibility['sources']}
    with np.load(root/'primary-analysis/features.npz') as bank:
        rows = list(bank['row_id'])
        order = list(bank['spi_order'])
        columns = [pair_index(order, f.spi_a, f.spi_b) for f in spec.features]
        full = bank['z'][[rows.index(r) for r in base.row_id]].astype(float)
    frame = pd.DataFrame(dict(row_id=base.row_id, control=base.r, Q=base.Q_reference, q=base.q, seed=base.seed, cohort='all'))
    frame['mean-|r|'] = base.mean_abs_correlation.to_numpy()
    frame['mean-MPI'] = base.mean_PC1.to_numpy()
    for k, column in enumerate(columns):
        frame[f'z{k}'] = full[:, column]
    seed = int(base.seed.min())
    snapshots, checks = [], []
    with np.load(root/'primary/observations.npz') as observations:
        for r in controls:
            i = int(np.flatnonzero(base.seed.eq(seed) & np.isclose(base.r, r))[0])
            row_id = base.row_id[i]
            path, = (root/'primary/mpi/primary').glob(f"*-{row_id.replace('.', '-')}/spi_mpis.npz")
            check(sha(path) == hashes[row_id], f'CML MPI changed: {row_id}')
            with np.load(path) as m:
                mpis = {k: m[k] for k in order}
            contract = verify_ordered_row(mpis, order, full[i])
            check(contract['passes'], f'CML z row is not the ordered contract: {row_id}')
            edges = {k: edge_z(mpis[f.spi_a], mpis[f.spi_b]) for k, f in enumerate(spec.features)}
            x = observations[row_id]
            trace = x[:, :100].mean(0)
            snapshots.append(Snapshot(control=r, label=rf'$r={r:g}$', row_id=row_id, x=x,
                                      glyph=dict(t=np.arange(100), trace=trace, window=(0., 99.)),
                                      edges=edges, groups=np.zeros(32*31, int)))
            checks.append(dict(row_id=row_id, row_contract=contract))
    metrics = {key: rho(frame, key, 'all') for key in ['q', 'mean-|r|', 'mean-MPI']}
    provenance = dict(snapshot_checks=checks, scores_sha256=data['score_sha256'], note=load_cml2d.__doc__.split('\n\n', 1)[1].strip())
    return SystemData(spec, frame, snapshots, metrics, provenance, primary='all')


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
        ax.scatter(g['t'][::stride], wrap_cycles(row)[::stride], s=.35, lw=0, color=color,
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


def glyph_trace(ax, g, color=GLYPH_COLOR):
    ax.plot(g['t'], g['trace'], color=color, lw=.6)
    ax.set(xlim=g['window'])
    _bare(ax, ('bottom', 'left'))


def glyph_field(ax, g, color=None):
    ax.imshow(g['field'], origin='lower', cmap='viridis', vmin=0, vmax=1, interpolation='nearest', rasterized=True)
    _bare(ax, ())


GLYPHS = {'wrapped': glyph_wrapped, 'unwrapped': glyph_unwrapped, 'spiral': glyph_spiral,
          'synchrony': glyph_synchrony, 'lissajous': glyph_lissajous, 'raster': glyph_raster, 'trace': glyph_trace,
          'field': glyph_field}


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


def draw_heatmaps(axes, data):
    spec = data.spec
    start, count = spec.heatmap_window or (0, data.snapshots[0].x.shape[1])
    scaled, limit = robust_scaled([s.x for s in data.snapshots])
    from seaborn import color_palette
    cmap = color_palette('icefire', as_cmap=True)
    for ax, x in zip(axes, scaled, strict=True):
        m = x.shape[0]
        ax.imshow(x[:, start:start+count], origin='lower', aspect='auto', cmap=cmap, vmin=-limit, vmax=limit,
                  extent=(start, start+count, .5, m+.5), interpolation='antialiased', rasterized=True)
        if spec.communities:
            for k in range(1, spec.communities):
                ax.axhline(k*m/spec.communities+.5, color='white', lw=.4)
        _no_ticks(ax)
        _box(ax, .5)


def mark_boundary(ax, spec):
    if isinstance(spec.boundary, tuple):
        ax.axvspan(*spec.boundary, color='.5', alpha=.16, lw=0)
    elif spec.boundary is not None:
        ax.axvline(spec.boundary, color='.5', lw=.6, ls=':')


def _series(axis, data, column, color, marker, *, ms=2.4, lw=1.1, secondary=False, label=None):
    mean, lo, hi = curve(data.frame, column, data.primary)
    handle, = axis.plot(mean.index, mean, marker+'-', color=color, ms=ms, lw=lw, label=label)
    if data.frame.cohort.eq(data.primary).sum() > len(mean):
        axis.fill_between(mean.index, lo, hi, color=color, alpha=.15, lw=0)
    if secondary:
        for cohort, ls in data.spec.secondary_style.items():
            mean, _, _ = curve(data.frame, column, cohort)
            axis.plot(mean.index, mean, ls=ls, marker=marker, mfc='white', color=color, ms=ms*.85, lw=lw*.7, alpha=.8)
    return handle


def draw_curves(ax, data, *, legend=True):
    """Control against Q (left, black) and q (right, teal); bands are 10-90% across held instances."""
    spec = data.spec
    right = ax.twinx()
    handles = [_series(ax, data, 'Q', Q_COLOR, 'o', label='$Q$'), _series(right, data, 'q', Z_COLOR, 's', label='$q$')]
    if spec.Q_log:
        ax.set_yscale('log')
        ax.yaxis.set_major_locator(plt.LogLocator(base=10, numticks=4))
        ax.yaxis.set_minor_locator(plt.NullLocator())
    if spec.Q_limits:
        ax.set_ylim(spec.Q_limits)
    if spec.q_limits:
        right.set_ylim(spec.q_limits)
    mark_boundary(ax, spec)
    ax.set_ylabel(spec.Q_label, color=Q_COLOR, labelpad=1.5, fontsize=7)
    right.set_ylabel(spec.q_label, color=Z_COLOR, labelpad=1.5, fontsize=7)
    right.tick_params(axis='y', labelcolor=Z_COLOR, labelsize=6.5, pad=1.5)
    ax.tick_params(labelsize=6.5, pad=1.5)
    ax.set_xlabel(spec.control_label, labelpad=1.5, fontsize=7)
    if spec.control_ticks:
        ax.set_xticks(spec.control_ticks)
    _box(ax); _box(right)
    if legend:
        loc, anchor = spec.legend_loc if isinstance(spec.legend_loc, tuple) else (spec.legend_loc, None)
        right.legend(handles=handles, loc=loc, bbox_to_anchor=anchor, fontsize=6.5, handlelength=1.4, borderaxespad=.4)
    return right


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
        ax.plot(snapshot.control, value, 'o', ms=4.2, mfc='none', mec='.25', mew=.6, zorder=6, clip_on=False)
        px, py = ax.transAxes.inverted().transform(ax.transData.transform((snapshot.control, value)))
        top = y+hh+.012+gh
        if py < y:
            anchor, source = (.5, -.32), hax          # below the label under the heatmap
        elif py > top:
            anchor, source = (.5, 1.02), gax
        else:
            anchor, source = ((1., .5) if px > x+w else (0., .5)), hax
        fig.add_artist(ConnectionPatch(xyA=anchor, coordsA=source.transAxes, xyB=(snapshot.control, value),
                                       coordsB=ax.transData, color='.45', lw=.45, zorder=7))
    return glyph_axes, heat_axes


def draw_baseline(ax, data, key):
    """Q (black) and one simple baseline against control: boxed, no ticks."""
    spec = data.spec
    right = ax.twinx()
    _series(ax, data, 'Q', Q_COLOR, 'o', ms=1.4, lw=.8)
    _series(right, data, key, BASELINES[key], 's', ms=1.4, lw=.8)
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
    """z_ab against control on a fixed [-1, 1] range: boxed, no ticks, snapshot controls marked."""
    spec = data.spec
    f = spec.features[k]
    ax.axhline(0, color='.85', lw=.5, zorder=0)
    _series(ax, data, f'z{k}', Z_COLOR, 'o', ms=1.4, lw=.85, secondary=True)
    mark_boundary(ax, spec)
    for snapshot in data.snapshots:
        ax.plot(snapshot.control, 1.0, 'v', color='.45', ms=2.2, zorder=5)
    ax.set(ylim=(-1.1, 1.1))
    _no_ticks(ax)
    _box(ax, .5)
    reading = textwrap.wrap(f.reading, 24)
    ax.set_title(f.label, fontsize=6.3, pad=3+6.2*len(reading), linespacing=1.0)
    ax.text(.5, 1.03, '\n'.join(reading), transform=ax.transAxes, ha='center', va='bottom',
            fontsize=5.4, style='italic', color='.35', linespacing=1.0)
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


def draw_edge_plane(ax, data, k, i, *, limits=None, axis_labels=True):
    """SPI a against SPI b across all ordered channel pairs at one snapshot: boxed, no ticks."""
    spec, snapshot = data.spec, data.snapshots[i]
    a, b, z = snapshot.edges[k]
    order = np.argsort(snapshot.groups, kind='stable')[::-1]   # first-listed group drawn on top
    colors = np.array([c for _, c in spec.edge_groups])
    ax.scatter(a[order], b[order], s=1.8, lw=0, c=colors[snapshot.groups[order]], alpha=.6, rasterized=True)
    if limits is not None:
        ax.set(xlim=limits[0], ylim=limits[1])
    else:
        ax.margins(.06)
    ax.set_box_aspect(1)
    _no_ticks(ax)
    _box(ax, .5)
    ax.set_title(snapshot.label+'\n'+rf'$z={z:+.2f}$', fontsize=6, pad=2, linespacing=1.1)
    f = spec.features[k]
    if axis_labels:
        ax.set_xlabel(short_spi(f.spi_a), fontsize=6, labelpad=1)
        ax.set_ylabel(short_spi(f.spi_b), fontsize=6, labelpad=1)


SPI_NAMES = {'lcss': 'LCSS', 'te_kraskov_NN-4_DCE-AUTO_k-1_kt-1_l-1_lt-1': 'TE (KSG)',
             'coint_johansen_max_eig_stat_order-0_ardiff-10': 'Johansen max-eig.',
             'coint_johansen_trace_stat_order-1_ardiff-1': 'Johansen trace',
             'corr_pearson_tau-2': 'Pearson lag 2', 'corr_pearson_tau-20-sq': r'Pearson lag 20$^2$',
             'si_kernel_W-0.5_k-1': 'SI (kernel)', 'ids': 'IDS',
             'corr_pearson_tau-1': 'Pearson 2 ms', 'corr_pearson_tau-10': 'Pearson 20 ms',
             'pec_orth': 'Envelope corr.', 'pec_orth_abs': '|Envelope corr.|', 'corr_pearson_tau-5': 'Pearson lag 5'}


def short_spi(name):
    return SPI_NAMES.get(name, name)


def edge_legend(ax, spec, title=None):
    handles = [Line2D([], [], marker='o', ls='', color=c, ms=2.6, label=g) for g, c in spec.edge_groups]
    ax.legend(handles=handles, loc='center left', fontsize=5.8, handletextpad=.15, borderaxespad=0, frameon=False,
              title=title, title_fontsize=6, alignment='left', labelspacing=.3)
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

GEOMETRY = dict(width=3.55, left=.42, right=.42, title=.30, curves=2.30, curves_bottom=.30,
                inset_width=.62, inset_heat=.24, inset_glyph=.22, inset_glyph_square=.40,
                row_gap=.10, baseline_title=.14, baseline=.50, baseline_bottom=.16,
                feature_title=.34, feature=.62, feature_bottom=.16, edge_title=.13, edge=.66, edge_bottom=.16)
SQUARE_GLYPHS = ('spiral', 'lissajous', 'field')


def axes_at(fig, x, y, w, h, **kw):
    """Axes from inch coordinates measured from the figure's top-left corner."""
    fw, fh = fig.get_size_inches()
    return fig.add_axes([x/fw, 1-(y+h)/fh, w/fw, h/fh], **kw)


def headline_height():
    g = GEOMETRY
    return g['title']+g['curves']+g['curves_bottom']


def column_height(spec, edge_rows):
    g = GEOMETRY
    return (headline_height()+g['row_gap']+g['baseline_title']+g['baseline']+g['baseline_bottom']+g['row_gap']
            +g['feature_title']+g['feature']+g['feature_bottom']+len(edge_rows)*(g['edge_title']+g['edge']+g['edge_bottom']))


def panel_label(fig, x, y, text):
    if text:
        fw, fh = fig.get_size_inches()
        fig.text(x/fw, 1-y/fh, text, fontsize=9, fontweight='bold', va='top', ha='left')


def draw_headline(fig, x0, y0, data, glyph=None, *, label=None):
    """Title, boxed Q/q curves, and MTS+glyph insets inside the axes; returns the bottom (in)."""
    g, spec = GEOMETRY, data.spec
    glyph = glyph or spec.glyphs[0]
    left, width = x0+g['left'], g['width']-g['left']-g['right']
    fw, fh = fig.get_size_inches()
    fig.text(left/fw, 1-(y0+.03)/fh, spec.title, fontsize=7.5, fontweight='bold', va='top')
    fig.text(left/fw, 1-(y0+.16)/fh, spec.subtitle+rf"; $|\rho_s(q,Q)|={data.metrics['q']:.2f}$",
             fontsize=6, va='top', color='.3')
    panel_label(fig, x0, y0, label)
    y = y0+g['title']
    ax = axes_at(fig, left, y, width, g['curves'])
    right = draw_curves(ax, data)
    glyphs, heats = draw_insets(ax, right, data, glyph)
    return y+g['curves']+g['curves_bottom'], dict(curves=ax, right=right, glyphs=glyphs, heatmaps=heats)


def draw_column(fig, x0, y0, data, glyph=None, edge_rows=None, labels='abcd'):
    g, spec = GEOMETRY, data.spec
    edge_rows = spec.edge_features if edge_rows is None else edge_rows
    left, width = x0+g['left'], g['width']-g['left']-g['right']
    y, _ = draw_headline(fig, x0, y0, data, glyph, label=labels[0])
    y += g['row_gap']
    panel_label(fig, x0, y, labels[1])
    y += g['baseline_title']
    gap = .22
    for j, key in enumerate(BASELINES):
        draw_baseline(axes_at(fig, left+j*((width-gap)/2+gap), y, (width-gap)/2, g['baseline']), data, key)
    y += g['baseline']+g['baseline_bottom']+g['row_gap']
    panel_label(fig, x0, y, labels[2])
    y += g['feature_title']
    side, gap = g['feature'], (width-3*g['feature'])/2
    for k in range(3):
        xk = left+k*(side+gap)
        if k < len(spec.features):
            draw_feature(axes_at(fig, xk, y, side, side), data, k, ylabel=k == 0)
        else:
            cohort_legend(axes_at(fig, xk-.05, y, side+.35, side), spec)
    y += side+g['feature_bottom']
    for r, k in enumerate(edge_rows):
        panel_label(fig, x0, y, labels[3] if r == 0 else None)
        y += g['edge_title']
        size, gap = g['edge'], .08
        limits = edge_limits(data, k) if spec.edge_scale == 'shared' else None
        for i in range(3):
            draw_edge_plane(axes_at(fig, left+i*(size+gap), y, size, size), data, k, i, limits=limits, axis_labels=i == 0)
        legend = axes_at(fig, left+3*(size+gap)+.04, y, g['width']-g['left']-3*(size+gap), size)
        edge_legend(legend, spec, spec.features[k].label.replace('\n', ' '))
        y += size+g['edge_bottom']
    return y


def composite(systems, glyphs=None, edge_rows=None, gap=.08):
    """System columns side by side; `glyphs` maps name to glyph variant, `edge_rows` to feature indices."""
    style()
    glyphs, edge_rows = glyphs or {}, edge_rows or {}
    heights = [column_height(d.spec, edge_rows.get(d.spec.name, d.spec.edge_features)) for d in systems]
    width = len(systems)*GEOMETRY['width']+(len(systems)-1)*gap
    fig = plt.figure(figsize=(width, max(heights)))
    letters = 'abcdefghijklmnop'
    for c, data in enumerate(systems):
        labels = [letters[c+len(systems)*r] for r in range(4)]
        draw_column(fig, c*(GEOMETRY['width']+gap), 0, data, glyphs.get(data.spec.name), edge_rows.get(data.spec.name), labels)
    return fig


def save(fig, name, out=OUT, formats=('svg', 'pdf', 'png')):
    out.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        fig.savefig(out/f'{name}.{ext}', dpi=180 if ext == 'png' else 600, bbox_inches='tight', pad_inches=.03)
    plt.close(fig)
    return out/f'{name}.png'


def export_panels(data, out=OUT):
    """Every panel as its own SVG/PDF (+PNG preview), at the size it occupies in the composite."""
    style()
    g, spec, folder, written = GEOMETRY, data.spec, out/'panels'/data.spec.name, []
    width = g['width']-g['left']-g['right']

    def emit(fig, name):
        written.append(save(fig, name, folder))

    def single(w, h):
        fig = plt.figure(figsize=(w+.6, h+.5))
        return fig, axes_at(fig, .3, .2, w, h)

    for glyph in spec.glyphs:
        fig = plt.figure(figsize=(g['width'], headline_height()))
        draw_headline(fig, 0, 0, data, glyph)
        emit(fig, f'headline-{glyph}')
        square = glyph in SQUARE_GLYPHS
        gw, gh = (g['inset_glyph_square'],)*2 if square else (g['inset_width'], g['inset_glyph'])
        pairs = [single(gw, gh) for _ in data.snapshots]
        draw_glyphs([ax for _, ax in pairs], data, glyph)
        for i, (fig, _) in enumerate(pairs):
            emit(fig, f'glyph-{glyph}-{i+1}')
    fig = plt.figure(figsize=(g['width'], g['curves']+g['curves_bottom']+.2))
    draw_curves(axes_at(fig, g['left'], .1, width, g['curves']), data)
    emit(fig, 'curves')
    pairs = [single(g['inset_width'], g['inset_heat']) for _ in data.snapshots]
    draw_heatmaps([ax for _, ax in pairs], data)
    for i, (fig, _) in enumerate(pairs):
        emit(fig, f'heatmap-{i+1}')
    for key, slug in zip(BASELINES, ('mean-abs-r', 'mean-mpi')):
        fig, ax = single((width-.22)/2, g['baseline'])
        draw_baseline(ax, data, key)
        emit(fig, f'baseline-{slug}')
    for k in range(len(spec.features)):
        fig, ax = single(g['feature'], g['feature'])
        draw_feature(ax, data, k)
        emit(fig, f'feature-{k+1}')
        limits = edge_limits(data, k) if spec.edge_scale == 'shared' else None
        for i in range(len(data.snapshots)):
            fig, ax = single(g['edge'], g['edge'])
            draw_edge_plane(ax, data, k, i, limits=limits)
            emit(fig, f'edges-{k+1}-{i+1}')
    fig, ax = single(1.0, .5)
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
        axes = [axes_at(fig, 1.0+i*(size[0]+.15)+((size[0]-h)/2 if kind in SQUARE_GLYPHS else 0), y+.14, h if kind in SQUARE_GLYPHS else size[0], h)
                for i in range(3)]
        draw_glyphs(axes, data, kind)
        for ax, snapshot in zip(axes, data.snapshots):
            ax.set_title(snapshot.label, fontsize=6, pad=1)
        fw, fh = fig.get_size_inches()
        fig.text(.05/fw, 1-(y+.14+h/2)/fh, f'{data.spec.name}\n{kind}', fontsize=7, va='center')
        y += h+.28
    return fig


def compare_Q(datasets, titles):
    """Headline curves only, side by side, e.g. for alternative physical quantities."""
    style()
    g = GEOMETRY
    fig = plt.figure(figsize=(len(datasets)*g['width'], g['curves']*.8+.75))
    for i, (data, title) in enumerate(zip(datasets, titles)):
        ax = axes_at(fig, i*g['width']+g['left'], .3, g['width']-g['left']-g['right'], g['curves']*.8)
        draw_curves(ax, data)
        ax.set_title(title+rf"   $|\rho_s(q,Q)|={data.metrics['q']:.2f}$", fontsize=7)
    return fig


def receipt(datasets, out=OUT):
    out.mkdir(parents=True, exist_ok=True)
    record = {d.spec.name: dict(metrics=d.metrics, provenance=d.provenance) for d in datasets}
    record['source_sha256'] = sha(__file__)
    record['claims'] = ('q tracks Q across controls; no calibration, mechanism or superiority is claimed. '
                        'Bands are 10-90% across held instances (locking), not confidence intervals; hardware has one record per setting.')
    (out/'provenance.json').write_text(json.dumps(record, indent=2, default=float)+'\n')
    return out/'provenance.json'


# --------------------------------------------------------------------------- registry

LOCKING_FEATURES = (
    Feature('lcss', 'te_kraskov_NN-4_DCE-AUTO_k-1_kt-1_l-1_lt-1', 'LCSS\n× TE',
            'predictive dependence vs waveform similarity (candidate)'),
    Feature('coint_johansen_max_eig_stat_order-0_ardiff-10', 'coint_johansen_trace_stat_order-1_ardiff-1',
            'Johansen ×\nJohansen', 'short- vs long-history model sensitivity'),
    Feature('corr_pearson_tau-2', 'corr_pearson_tau-20-sq', 'Pearson lag 2\n× lag 20$^2$',
            'within-community lag-profile redistribution'),
)
HARDWARE_FEATURES = (
    Feature('si_kernel_W-0.5_k-1', 'ids', 'kernel SI\n× IDS', 'recruitment-associated profile reorganisation'),
    Feature('corr_pearson_tau-1', 'corr_pearson_tau-10', 'Pearson 2 ms\n× 20 ms', 'lag-profile / spectral organisation'),
)

SYSTEMS = {
    'locking': Spec(
        name='locking', title='2:1 cross-frequency locking', loader=load_locking,
        subtitle='M = N = 24, T = 1000, varying sensor noise; 168 held records',
        control_label=r'Cross-coupling $\gamma$', control_short=r'$\gamma$', Q_label=r'Locking index $Q$',
        boundary=.1017, features=LOCKING_FEATURES, edge_features=(0,), glyphs=('wrapped', 'spiral', 'lissajous'),
        glyph_titles={'wrapped': r'$\Phi$'}, heatmap_window=(0, 200), dt=.5, communities=3, edge_groups=LOCKING_GROUPS,
        control_ticks=(0, .1, .2), legend_loc='upper left', inset_boxes=((.06, .50), (.50, .10), (.76, .10)),
        Q_limits=(-.04, 1.06),
        note='M=N=24, T=1000, recording-specific sensor noise; 168 held records (8 seeds x 21 controls).'),
    'hardware': Spec(
        name='hardware', title='Electronic oscillators', loader=load_hardware,
        subtitle='M = N = 28, T = 5000 at 500 Hz; ER sweep, one recording per setting',
        control_label='Coupling-control voltage (V)', control_short='V', Q_label='Carrier frequency range (Hz)',
        Q_log=True, boundary=(.115, .12), features=HARDWARE_FEATURES, edge_features=(0,),
        glyphs=('wrapped', 'synchrony', 'raster'),
        glyph_titles={'wrapped': r'$\Delta\phi$', 'synchrony': '$R$', 'raster': r'$\Delta\phi$'},
        heatmap_window=(500, 100), dt=.002, edge_groups=HARDWARE_GROUPS, secondary_style={'SW': '--'},
        control_ticks=(0, .2, .4), legend_loc=('center left', (.0, .68)), edge_scale='panel', primary_label='ER (fit)',
        inset_boxes=((.04, .30), (.40, .34), (.70, .34)),
        note='M=N=28, T=5000 at 500 Hz; one record per setting; q fitted on all 29 ER records, SW projected.'),
    'cml2d': Spec(
        name='cml2d', title='Collective period-two CML', loader=load_cml2d,
        subtitle='M = 32 of N = 65,536 sites, T = 1000; confirmation seeds',
        control_label='Local nonlinearity $r$', control_short='$r$', Q_label='Period-two $Q$', boundary=3.86212,
        features=(Feature('pec_orth', 'pec_orth_abs', 'Envelope ×\n|envelope| (orth.)', 'signed/absolute transformation geometry'),
                  Feature('corr_pearson_tau-5', 'pec_orth_abs', 'Pearson lag 5\n× |envelope|', 'descriptive')),
        glyphs=('trace',), glyph_titles={'trace': r'$\bar x$'}, heatmap_window=(0, 100),
        edge_groups=(('channel pairs', '#26828e'),), control_ticks=(3.84, 3.865, 3.89), edge_scale='panel',
        inset_boxes=((.05, .12), (.52, .62), (.76, .62)),
        primary_label='all seeds', note='Registry example only; not part of the two-system composite.'),
}


def load(name, **changes):
    """Load a registered system; keyword changes override its Spec (e.g. Q, labels, glyph order)."""
    if name not in SYSTEMS:
        raise KeyError(f'Unknown system {name!r}; registered: {sorted(SYSTEMS)}')
    spec = replace(SYSTEMS[name], **changes)
    return spec.loader(spec)
