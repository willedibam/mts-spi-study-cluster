"""Dispersion-sign correction (README, "Correction: dispersion signs"): old against corrected tables.

    python -m research.representation.character_axes.validation.correction --old DIR --new DIR
        [--check BASE CORR ...]

Writes NEW/correction_comparison.csv (one row per table, key and field, with old and new values) and prints
every changed verdict. --check takes pairs of window directories (original, corrected) and confirms that
level-0 windows, whose inputs the correction does not change, give the same statistics.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

# table: (key columns, compared fields)
TABLES = {
    'verdicts.csv': (['measure'], ['verdict', 'SR', 'worst_other', 'own_mean', 'own_d', 'reversed_at', 'insensitive_at',
                                   'other own knob/own', 'other own knob/own B1', 'SR incl. stage-2 knobs', 'level_aware']),
    'second_family.csv': (['measure', 'family'], ['verdict', 'SR', 'worst_other', 'own_B0', 'reversed_at']),
    'surrogates.csv': (['measure', 'knob', 'baseline'], ['real', 'surrogate', 'kept']),
    'corrections.csv': (['pair', 'correction'], ['own_d', 'cv_s/own', 'noise/own', 'usable']),
    'posthoc_strong_coupling.csv': (['measure', 'knob'], ['real', 'd', 'status', 'kept']),
    'bound.csv': (['cohort', 'pair'], ['d_raw', 'd_rel_corrected', 'd_rel_corrected_floor_0001', 'claim_stands']),
    'stage3_verdicts.csv': (['measure'], ['verdict', 'SR3', 'worst_other', 'own_B0', 'reversed_at']),
    'stage3_core.csv': (['readout'], ['SR_stage12_knobs', 'SR_with_stage3_knobs', 'worst_other', 'own_B0', 'status']),
}


def _same(a, b):
    if isinstance(a, (float, np.floating)) and isinstance(b, (float, np.floating)):
        return (np.isnan(a) and np.isnan(b)) or np.isclose(a, b, rtol=1e-6, atol=1e-12)
    return (pd.isna(a) and pd.isna(b)) or a == b


def compare(old_dir, new_dir):
    rows = []
    for name, (keys, fields) in TABLES.items():
        if not (Path(old_dir) / name).exists() or not (Path(new_dir) / name).exists():
            print(f'{name}: missing on one side, skipped'); continue
        old, new = pd.read_csv(Path(old_dir) / name), pd.read_csv(Path(new_dir) / name)
        m = old.merge(new, on=keys, how='outer', suffixes=(' old', ' new'), indicator=True)
        for _, r in m.iterrows():
            key = ' | '.join(str(r[k]) for k in keys)
            for f in fields:
                if f + ' old' not in m:
                    continue
                o, n = r[f + ' old'], r[f + ' new']
                rows.append(dict(table=name[:-4], key=key, field=f, old=o, new=n, changed=not _same(o, n)))
    return pd.DataFrame(rows)


def check_level0(base, corr):
    """Max |difference| of S between original and corrected level-0 windows (identical inputs)."""
    diffs = []
    for f in sorted(glob.glob(str(Path(corr) / '*-L0-*.npz'))):
        b = Path(base) / Path(f).name
        if not b.exists():
            continue
        with np.load(f) as x, np.load(b) as y:
            if list(x['spi']) != list(y['spi']):
                continue
            d = np.abs(x['S'].astype(np.float64) - y['S'].astype(np.float64))
            both = np.isnan(x['S']) & np.isnan(y['S'])
            diffs.append(np.nanmax(np.where(both, 0, d)))
    diffs = np.array(diffs)
    print(f'level-0 check {corr}: {len(diffs)} windows, {int((diffs == 0).sum())} identical, '
          f'max |dS| {diffs.max() if len(diffs) else np.nan:.3g}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--old', type=Path, required=True, help='tables before the correction')
    ap.add_argument('--new', type=Path, required=True, help='tables after it; the comparison is written here')
    ap.add_argument('--check', type=Path, nargs='*', default=[], help='pairs: original and corrected window directories')
    a = ap.parse_args()
    for base, corr in zip(a.check[::2], a.check[1::2]):
        check_level0(base, corr)
    t = compare(a.old, a.new)
    t.to_csv(a.new / 'correction_comparison.csv', index=False, float_format='%.4g')
    pd.set_option('display.width', 250, 'display.max_colwidth', 70, 'display.max_rows', 500)
    print(t[t.changed & t.field.isin(['verdict', 'usable', 'claim_stands', 'status'])].to_string(index=False))


if __name__ == '__main__':
    main()
