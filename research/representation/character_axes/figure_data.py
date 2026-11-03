"""Compact arrays for the character-axes figures (run on Gadi; figures.ipynb plots them).

Per cohort contrast: unit-level core readouts by class, the paired effect of every z coordinate, and for the
top z pairs the channel-pair maps of both statistics in each class (unit means), so a feature pair can be
traced back to the channel pairs that move it.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from mtsspi.character_axes import CORE, core_scores, edge_profiles

COHORTS = dict(neurotycho=('character-transfer-261031/neurotycho-primary', 1, 0, 'awake', 'anaesthesia'),
               propofol=('character-axes-261102/propofol', 2, 0, 'baseline', 'moderate'))


def unit_keys(bank, cohort):
    g = bank['group'].astype(str)
    return np.array([f'{a}-{r}' for a, r in zip(g, bank['role'].astype(str))]) if cohort == 'neurotycho' else g


def main(root, schema_path, out):
    schema = json.load(open(schema_path))
    names_z = np.array([f"{s['spi_a']}|{s['spi_b']}" for s in schema])
    out.mkdir(parents=True, exist_ok=True)
    for cohort, (rel, pos, neg, neg_name, pos_name) in COHORTS.items():
        bank = np.load(root / rel / 'bank.npz')
        spi = [str(s) for s in bank['spi_order']]
        M = int(bank['M'][0])
        profiles = edge_profiles(bank['aligned'], M, len(spi))
        core, core_names = core_scores(profiles, spi)
        units, y = unit_keys(bank, cohort), bank['y']
        common = sorted(set(units[y == pos]) & set(units[y == neg]))
        mean = lambda v, u, c: np.nanmean(v[(units == u) & (y == c)], 0)
        core_units = np.stack([[mean(core, u, c) for u in common] for c in (neg, pos)])        # (2, U, C)
        z = np.asarray(bank['z'], dtype=np.float64)
        diff = np.stack([mean(z, u, pos) - mean(z, u, neg) for u in common])                     # (U, K)
        with np.errstate(invalid='ignore', divide='ignore'):
            d = np.nanmean(diff, 0) / np.nanstd(diff, 0, ddof=1)
        agree = np.mean(np.sign(diff) == np.sign(np.nanmean(diff, 0)), 0)
        z_class = np.stack([np.nanmean(z[y == c], 0) for c in (neg, pos)])
        ok = np.isfinite(d) & (agree == 1)
        top = np.argsort(-np.abs(np.where(ok, d, 0)))[:12]
        off = ~np.eye(M, dtype=bool)
        maps = np.full((len(top), 2, 2, M, M), np.nan)                                          # pair, spi a/b, class
        for t, k in enumerate(top):
            for s, name in enumerate(schema[k][key] for key in ('spi_a', 'spi_b')):
                for ci, c in enumerate((neg, pos)):
                    maps[t, s, ci][off] = np.nanmean(profiles[y == c, spi.index(name)], 0)
        np.savez_compressed(out / f'{cohort}.npz', core_names=np.array(core_names),
                            core_roles=np.array([CORE[n][0] for n in core_names]), core_units=core_units,
                            units=np.array(common), classes=np.array([neg_name, pos_name]), d=d, agree=agree,
                            z_class=z_class, names_z=names_z, top=top, maps=maps, M=M)
        print(cohort, len(common), 'units; top pairs:', [(names_z[k], round(float(d[k]), 2)) for k in top[:5]], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--schema', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    a = parser.parse_args()
    main(a.root, a.schema, a.out)
