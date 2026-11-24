"""Build catalogue.json and catalogue.md from entries.py and committed atlas outputs (no pyspi, seconds).

    python -m research.representation.atlas.pair_catalogue.build

Numbers attached per pair, each with its source:
  reliability: between-window edge-profile reliability in awake eyes-closed, 2-s and 10-s arms
    (results/representation/atlas-discovery-261112/neurotycho-{short,long}-reliability-all.json);
  z0, null SD: wake z and eyes-open null SD from the assumptions run (minimal pairs) or the 42 named pairs,
    2-s and 10-s (pooled) arms, raw pyspi; the axis screen (pyspi-free, 10 s, 32 or all channels) where screened;
  upstream: the pyspi PRs (entries.UPSTREAM) that unbundle or change an SPI of the pair;
  matched: the 2-s per-window joint-IAAFT net, drug - awake, mean over 19 sessions and sessions sharing its sign
    (results/representation/atlas-matched-screen-261119/matched-screen-deltas.npz; normal-scored construction).
Ranking: score = interp/3 x measurability; measurability = best arm's min partner reliability x ceiling factor
(z0 > .98: .4; > .95: .7) x null_ok. Exclusions are scored 0 and listed separately.
"""
import json
import re
from pathlib import Path

import numpy as np

from research.representation.atlas.pair_catalogue.entries import ENTRIES, GUIDE_PROPOSALS, OUTSIDE_BANK, STATUS, UPSTREAM, UPSTREAM_NOTES

HERE = Path(__file__).parent
ROOT = HERE.parents[3]
R = ROOT / 'results/representation'
DISC = R / 'atlas-discovery-261112'


def load():
    spis = json.loads((R / 'atlas-movers-261117/movers.json').read_text())['spis']
    rel = {arm: {k: v['reliability'] for k, v in json.loads((DISC / f'neurotycho-{arm}-reliability-all.json').read_text())['reference_state'].items()}
           for arm in ('short', 'long')}
    classes = {c['cls']: [tuple(p) for p in c['pairs']] for c in json.loads((DISC / 'assumption-classes.json').read_text())['classes']}
    z = {}
    for arm in ('short', 'long'):
        for c in json.loads((DISC / f'neurotycho-{arm}-assumptions.json').read_text())['classes']:
            for p in c['pairs']:
                z.setdefault(frozenset((p['a'], p['b'])), {})[arm] = dict(z0=p['z0'], null_sd=p['null_sd'], src='assumptions')
        for p in json.loads((DISC / f'neurotycho-{arm}-pairs.json').read_text())['pairs']:
            d = z.setdefault(frozenset((p['a'], p['b'])), {})
            if arm not in d and p.get('reference_z') is not None:
                d[arm] = dict(z0=p['reference_z'], null_sd=p.get('eyes_open_null_sd'), src='named pairs')
    axis = {}
    for arm in ('long', 'short'):
        s = json.loads((R / f'atlas-axis-screen-261119/score-{arm}.json').read_text())
        for fam, F in s['families'].items():
            for row in [x for k in ('rows', 'screened_no_reliability', 'top_not_screened') for x in F.get(k, []) if isinstance(x, dict) and 'spis' in x]:
                key = frozenset(row['spis'])
                if len(key) == 2 and (key, arm) not in axis:
                    axis[(key, arm)] = dict(pair=row['pair'], family=fam, z0=row.get('z0'), null_sd=row.get('null_sd'), delta=row.get('mean'),
                                            t3=row.get('t3'), sd_multiple=row.get('sd_multiple'), axis=bool(row.get('axis')))
    m = np.load(R / 'atlas-matched-screen-261119/matched-screen-deltas.npz')
    mspis = [str(x) for x in m['spis']]
    a, b = np.triu_indices(len(mspis), 1)
    midx = {frozenset((mspis[i], mspis[j])): c for c, (i, j) in enumerate(zip(a, b))}
    return spis, rel, classes, z, axis, (m['Dn'], m['usable'], midx)


def pair_record(p, rel, z, axis, matched, new):
    A, B = p
    rec = dict(a=A, b=B, reliability={arm: [None if x in new else (round(rel[arm][x], 3) if x in rel[arm] else None) for x in p]
                                     for arm in ('short', 'long')})
    key = frozenset(p)
    zz = z.get(key, {})
    rec['z0'] = {arm: (round(zz[arm]['z0'], 3) if arm in zz else None) for arm in ('short', 'long')}
    rec['null_sd'] = {arm: (round(zz[arm]['null_sd'], 4) if arm in zz and zz[arm]['null_sd'] is not None else None) for arm in ('short', 'long')}
    ax = {arm: axis.get((key, arm)) for arm in ('long', 'short')}
    rec['axis_screen'] = {arm: ({k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()} if d else None) for arm, d in ax.items()}
    Dn, usable, midx = matched
    if key in midx:
        c = midx[key]
        x = Dn[:, c]
        mean = float(np.nanmean(x))
        rec['matched_2s'] = dict(net=round(mean, 4), same_sign=int(np.sum(np.sign(x) == np.sign(mean))), sessions=int(np.isfinite(x).sum()), usable=bool(usable[c]))
    else:
        rec['matched_2s'] = None
    return rec


def measurability(rec, null_ok, tier):
    """Best arm's min partner reliability x ceiling factor x null_ok. Higher-order pairs read beyond IAAFT carry the
    surrogate's saturation mechanics in the net, so their ceiling factor is milder (.7 / .9 against .4 / .7)."""
    best, arm_used = 0., None
    for arm in ('long', 'short'):
        vals = [v for v in rec['reliability'][arm] if v is not None]
        if vals and min(vals) > best:
            best, arm_used = min(vals), arm
    z0s = [v for v in rec['z0'].values() if v is not None] + [d['z0'] for d in rec['axis_screen'].values() if d and d.get('z0') is not None]
    z0 = max(abs(v) for v in z0s) if z0s else None
    hi, mid = (.7, .9) if tier == 'higher-order' else (.4, .7)
    ceil = hi if z0 is not None and z0 > .98 else mid if z0 is not None and z0 > .95 else 1.
    return round(best * ceil * null_ok, 3), arm_used, z0


def build():
    spis, rel, classes, z, axis, matched = load()
    bank = set(spis)
    explicit = {frozenset(p) for e in ENTRIES for p in e['pairs']}
    claimed = set()
    out = []
    for e in ENTRIES:
        new = set(e.get('new', []))
        for p in e['pairs']:
            for x in p:
                assert x in bank or x in new, f'{e["id"]}: {x} not in the bank'
        members = [tuple(p) for p in e['pairs']]
        for c in e['classes']:
            assert c in classes, f'{e["id"]}: unknown class {c}'
            claimed.add(c)
            members += [p for p in classes[c] if frozenset(p) not in explicit]
        recs = [pair_record(p, rel, z, axis, matched, new) for p in members]
        for r in recs:
            r['upstream'] = [u['pr'] for u in UPSTREAM['prs'] if u['spis'] and any(re.search(u['spis'], x) for x in (r['a'], r['b']))]
        meas, arm, z0 = measurability(recs[0], e['null_ok'], e['tier'])
        score = 0. if e.get('exclusion') else round(e['interp'] / 3 * meas, 3)
        out.append({**{k: v for k, v in e.items() if k != 'pairs'}, 'representative': recs[0], 'members': recs, 'n_members': len(recs),
                    'measurability': meas, 'reliability_arm': arm, 'z0_max': z0, 'score': score, 'status': STATUS.get(e['id'], 'excluded' if e.get('exclusion') else 'raw'),
                    'upstream': UPSTREAM_NOTES.get(e['id']), 'upstream_members': sum(bool(r['upstream']) for r in recs)})
    rest = []
    for c, ps in classes.items():
        if c not in claimed:
            rest.append(dict(cls=c, members=[pair_record(p, rel, z, axis, matched, set()) for p in ps], reading='literal only (no entry)'))
    out.sort(key=lambda e: (-e['score'], e['id']))
    for i, e in enumerate(out, 1):
        e['rank'] = i if e['score'] > 0 else None
    doc = dict(source='research/representation/atlas/pair_catalogue/build.py', bank='NeuroTycho r13 atlas, 272 SPIs (configs/pyspi/atlas/benchmarked_p90_pcorr_symbolic.yaml)',
               ranking=__doc__.split('Ranking: ')[1].strip(), entries=out, uncatalogued_classes=rest, guide_proposals=GUIDE_PROPOSALS, outside_bank=OUTSIDE_BANK, upstream=UPSTREAM,
               identities='identities.json (identities.py)')
    (HERE / 'catalogue.json').write_text(json.dumps(doc, indent=1) + '\n')
    return doc


def fmt_rel(rec):
    f = lambda v: '–' if v is None else f'{v:.2f}'
    return ' / '.join(f'{f(rec["reliability"][arm][0])},{f(rec["reliability"][arm][1])}' for arm in ('short', 'long'))


def fmt_z0(rec):
    f = lambda v: '–' if v is None else f'{v:.2f}'
    ax = rec['axis_screen'].get('long')
    s = f'{f(rec["z0"]["short"])} / {f(rec["z0"]["long"])}'
    return s + (f' (ax {ax["z0"]:.2f})' if ax and ax.get('z0') is not None else '')


def fmt_matched(rec):
    m = rec['matched_2s']
    return '–' if not m else f'{m["net"]:+.3f} ({m["same_sign"]}/{m["sessions"]})'


if __name__ == '__main__':
    doc = build()
    print(f'{len(doc["entries"])} entries, {sum(e["n_members"] for e in doc["entries"])} pairs; '
          f'{len(doc["uncatalogued_classes"])} uncatalogued classes')
    for e in doc['entries']:
        r = e['representative']
        print(f'{str(e["rank"]):>4} {e["score"]:.3f} {e["id"]:18s} interp {e["interp"]} meas {e["measurability"]:.2f} ({e["reliability_arm"]}) '
              f'rel {fmt_rel(r)} z0 {fmt_z0(r)} matched {fmt_matched(r)}')
