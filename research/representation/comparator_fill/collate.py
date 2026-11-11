"""Markdown tables of the comparator fill for the real-world results overview.

Reads results/representation/comparator-fill-261110/{stat,library}/*.json and prints
the rows added to Matrices 1-4. Run: python -m research.representation.comparator_fill.collate
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3] / 'results' / 'representation' / 'comparator-fill-261110'
DISCOVERY = ('chbmit', 'hmc', 'grabmyo', 'lanl3ss')
COHORTS = (('ad_ftd', 'AD_vs_control'), ('ad_ftd', 'FTD_vs_control'), ('ad_ftd', 'AD_vs_FTD'),
           ('lemon', 'old_vs_young'), ('propofol', 'moderate_vs_baseline'),
           ('propofol', 'mild_vs_baseline'), ('propofol', 'moderate_vs_recovery'),
           ('neurotycho', 'anaesthesia_vs_awake'))
# Row label -> key in the discovery files (underscore), the fMRI file (hyphen) and the cohort files.
TRANSFER_ROWS = (
    ('z, first PC only', 'z_pc1', 'z-pc1', 'z_pc1'),
    ('z, full rank', 'z_full', 'z-full', 'z_full'),
    ('Pearson + z', 'pearson_z', None, 'pearson_z'),
    ('per-channel spectra alone', 'spectra', None, 'spectral'),
    ('spectra + z', 'spectra_z', 'spectral-z', 'spectral_z'),
    ('per-channel univariate + z', 'univariate_z', None, None),
    ('Pearson + spectra', 'pearson_spectra', 'pearson-spectral', 'pearson_spectral'),
    ('Pearson + spectra + z', 'pearson_spectra_z', 'pearson-spectral-z', 'pearson_spectral_z'),
    ('mean-SPI (K edge means)', 'mean_spi', 'mean-spi', 'mean_spi'),
    ('mean-SPI + z', 'mean_spi_z', 'mean-spi-z', 'mean_spi_z'),
    ('tangent-space covariance', 'tangent', 'tangent', 'tangent'),
    ('per-channel catch22', 'catch22', 'catch22', 'catch22'),
)
COHORT_ROWS = (('z, first PC only', 'z_pc1'), ('z, full rank', 'z_full'), ('Pearson + z', 'pearson_z'),
               ('power + z', 'power_z'), ('relative spectra + z', 'spectral_z'),
               ('Pearson + spectra', 'pearson_spectral'), ('Pearson + spectra + z', 'pearson_spectral_z'),
               ('mean-SPI', 'mean_spi'), ('mean-SPI + z', 'mean_spi_z'), ('m (23 descriptors/SPI)', 'm'),
               ('m + z', 'm_z'), ('full aligned MPIs', 'aligned'), ('best single SPI (inner-fold)', 'single_spi'),
               ('tangent-space covariance', 'tangent'), ('per-channel catch22', 'catch22'))
LIBRARY = (('ROCKET', 'rocket'), ('MiniRocket (aeon, this fill)', 'minirocket'),
           ('MultiRocket-Hydra', 'multirocket_hydra'), ('QUANT', 'quant'))


def load(kind, name):
    return json.loads((ROOT / kind / f'{name}.json').read_text())


def f(x):
    if x is None:
        return '—'
    if 0.9995 <= x < 1:
        return '>.999'
    s = f'{x:.3f}'.replace('0.', '.', 1) if abs(x) < 1 else ('1.0' if x == 1 else f'{x:.3f}')
    return s.replace('-', '−')


def signed(x):
    return ('+' if x >= 0 else '') + f(x)


def interval(v, key='interval'):
    lo, hi = v[key]
    return f'{signed(v.get("auroc_difference", v.get("improvement")))} [{signed(lo)}, {signed(hi)}]'


def row(label, cells):
    return f'| {label} | ' + ' | '.join(cells) + ' |'


def transfer_sources():
    """Per transfer column: method -> metrics dict (brier, balanced_accuracy, auroc_pooled[, auroc])."""
    cols = {}
    nt = load('stat', 'neurotycho')['contrasts']['anaesthesia_vs_awake']['scores']
    cols['NT'] = {k: dict(v, auroc_pooled=v['auroc']) for k, v in nt.items()}
    cols['NT'].update({k: dict(v, auroc_pooled=v['auroc'])
                       for k, v in load('library', 'neurotycho')['anaesthesia_vs_awake']['scores'].items()})
    fm = load('stat', 'fmri')['roles']['confirmation_primary']['methods']
    cols['fMRI'] = dict(fm, **load('library', 'fmri')['confirmation_primary'])
    chb = load('stat', 'chbmit')
    lib = load('library', 'chbmit')
    cols['CHB'] = dict(chb['screen']['methods'], **lib['screen'])
    cols['Siena'] = dict(chb['siena']['methods'], **lib['siena'])
    for d in DISCOVERY[1:]:
        cols[d] = dict(load('stat', d)['screen']['methods'], **load('library', d)['screen'])
    return cols


def matrix1():
    cols = transfer_sources()
    order = ('NT', 'fMRI', 'CHB', 'Siena', 'hmc', 'grabmyo', 'lanl3ss')
    out = ['| Representation | NT r10 (BA pooled over 128 windows) | fMRI primary | CHB-MIT screen | Siena (post hoc) | HMC | GRABMyo | LANL |',
           '|---|---|---|---|---|---|---|---|']
    for label, disc, fmri, nt in TRANSFER_ROWS:
        keys = dict(NT=nt, fMRI=fmri, CHB=disc, Siena=disc, hmc=disc, grabmyo=disc, lanl3ss=disc)
        cells = []
        for c in order:
            m = cols[c].get(keys[c]) if keys[c] else None
            cells.append(f'{f(m["balanced_accuracy"])} / {f(m["brier"])}' if m else 'not run')
        out.append(row(label, cells))
    for label, key in (('z (PCA), now with Brier', 'z'), ('Pearson (r)', 'pearson'), ('m', 'm'), ('m + z', 'm_z'),
                       ('full aligned MPIs', 'aligned'), ('best single SPI', 'single_spi')):
        m = cols['NT'][key]
        out.append(row(f'{label} (NT r10 only)', [f'{f(m["balanced_accuracy"])} / {f(m["brier"])}'] + ['—'] * 6))
    for label, key in LIBRARY:
        out.append(row(label, [f'{f(cols[c][key]["balanced_accuracy"])} / {f(cols[c][key]["brier"])}'
                               if cols[c].get(key) else 'not run' for c in order]))
    return '\n'.join(out)


def matrix2():
    """Pooled AUROC on the transfer columns, including existing methods on the discovery screen (gap 2)."""
    cols = transfer_sources()
    existing = {d: load('stat', d)['screen']['existing_auroc_pooled'] for d in DISCOVERY}
    order = ('NT', 'fMRI', 'CHB', 'hmc', 'grabmyo', 'lanl3ss', 'Siena')
    out = ['| Representation | NT r10 (pooled over windows) | fMRI (mean within / pooled) | CHB-MIT screen | HMC | GRABMyo (one-vs-rest) | LANL | Siena |',
           '|---|---|---|---|---|---|---|---|']
    def cell(c, key, fkey):
        m = cols[c].get(fkey if c == 'fMRI' else key) if (fkey if c == 'fMRI' else key) else None
        if not m:
            return 'not run'
        return f'{f(m["auroc"])} / {f(m["auroc_pooled"])}' if c == 'fMRI' else f(m['auroc_pooled'])
    for label, disc, fmri, nt in TRANSFER_ROWS:
        out.append(row(label, [cell(c, nt if c == 'NT' else disc, fmri) for c in order]))
    for label, key in LIBRARY:
        out.append(row(label, [cell(c, key, key) for c in order]))
    names = (('z (PCA)', 'z'), ('Pearson', 'pearson'), ('m', 'm'), ('m + z', 'm_plus_z'), ('full aligned MPIs', 'aligned'),
             ('best single SPI', 'single_spi'), ('source baseline + z', 'baseline_plus_z'),
             ('per-channel univariate', 'univariate'), ('univariate RBF', 'univariate_rbf'), ('validity', 'validity'),
             ('InceptionTime', 'inceptiontime'), ('MiniRocket (original)', 'minirocket'), ('aligned raw encoder', 'aligned_encoder'))
    for label, key in names:
        out.append(row(f'{label} (existing fit)', ['see Matrix 2', 'see Matrix 2']
                       + [f(existing[d].get(key)) for d in DISCOVERY[0:1] + DISCOVERY[1:]] + ['see Matrix 2']))
    return '\n'.join(out)


def matrix3():
    out = ['| Representation | AD vs HC | FTD vs HC | AD vs FTD | LEMON | Propofol moderate vs baseline | mild vs baseline | moderate vs recovery | NT r10 transfer |',
           '|---|---|---|---|---|---|---|---|---|']
    stat = {(c, k): load('stat', c)['contrasts'][k] for c, k in COHORTS}
    lib = {(c, k): load('library', c)[k] for c, k in COHORTS}
    for label, key in (('check: z', 'z'), ('check: Pearson', 'pearson'), ('check: relative spectra', 'spectral'),
                       ('check: absolute power', 'power')) + COHORT_ROWS:
        out.append(row(label, [f(stat[ck]['scores'][key]['auroc']) if key in stat[ck]['scores'] else 'not run'
                               for ck in COHORTS]))
    for label, key in LIBRARY:
        out.append(row(label, [f(lib[ck]['scores'][key]['auroc']) for ck in COHORTS]))
    return '\n'.join(out)


def matrix4():
    stat = {(c, k): load('stat', c)['contrasts'][k]['contrasts'] for c, k in COHORTS}
    contrasts = list(stat[COHORTS[0]])
    out = ['| AUROC difference [95%] | AD vs HC | FTD vs HC | AD vs FTD | LEMON | Propofol moderate vs baseline | mild vs baseline | moderate vs recovery | NT r10 transfer |',
           '|---' * (len(COHORTS) + 1) + '|']
    for name in contrasts:
        out.append(row(name, [interval(stat[ck][name]) if name in stat[ck] else 'not run' for ck in COHORTS]))
    out += ['', '| Brier improvement [95%]; BA difference | CHB-MIT | Siena | HMC | GRABMyo | LANL | fMRI primary |', '|---|---|---|---|---|---|---|']
    chb = load('stat', 'chbmit')
    tables = dict(CHB=chb['screen']['contrasts'], Siena=chb['siena']['contrasts'],
                  **{d: load('stat', d)['screen']['contrasts'] for d in DISCOVERY[1:]})
    fm = {k.replace('-', '_').replace('z_pca', 'z').replace('spectral', 'spectra'): v
          for k, v in load('stat', 'fmri')['roles']['confirmation_primary']['contrasts'].items()}
    tables['fMRI'] = fm
    for name in tables['CHB']:
        cells = []
        for c in ('CHB', 'Siena', 'hmc', 'grabmyo', 'lanl3ss', 'fMRI'):
            v = tables[c].get(name)
            cells.append(f'{interval(v["brier"], "conditional_95_interval")}; {signed(v["balanced_accuracy"]["improvement"])}'
                         if v else 'not run')
        out.append(row(name, cells))
    return '\n'.join(out)


if __name__ == '__main__':
    for title, table in (('Matrix 1 rows', matrix1), ('Matrix 2 rows', matrix2), ('Matrix 3 rows', matrix3),
                         ('Matrix 4 rows', matrix4)):
        print(f'### {title}\n\n{table()}\n')
