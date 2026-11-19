"""Axis screen post hoc checks (labelled post hoc in result.md; not in the frozen rules).

degeneracy: real - IAAFT session delta (drug - awake eyes-closed) of every linear-linear pair. The prereg assumed
  spectrum-final IAAFT reproduces every linear statistic; this measures where it does not.
reliability: change in edge-profile reliability (drug - awake eyes-closed) per statistic, real against IAAFT, by agent
  and animal; eyes-open and sleep for comparison.

  python -m research.representation.atlas.axis_screen.posthoc <screen-<arm>.npz> <manifest.json> <out.json>
"""
import sys, json
import numpy as np
from scipy.stats import ttest_1samp
from research.representation.atlas.axis_screen import score as S


def main(path, manifest, output):
    D, keys, gi, I = S.load(path, manifest)
    unit, state = I['unit'], I['state']
    ag, an = dict(zip(unit, I['agent'])), dict(zip(unit, I['animal']))
    groups = [str(g) for g in D['groups']]
    ses = sorted(u for u in set(unit) if ag[u] in S.DRUGS and f'{u}__drug' in groups and f'{u}__{S.REF}' in groups)
    slp = sorted(u for u in set(unit) if f'{u}__sleep' in groups and f'{u}__{S.REF}' in groups)
    animals = sorted({an[u] for u in ses}); A = np.array([an[u] for u in ses]); AG = np.array([ag[u] for u in ses])

    Z = D['Z'].astype(float)
    net = Z[:, 0] - np.nanmean(Z[:, 1:4], 1)
    sd = lambda x: np.array([np.nanmean(x[(unit == u) & (state == 'drug')]) - np.nanmean(x[(unit == u) & (state == S.REF)]) for u in ses])
    lin = [k for k in keys if k not in S.HIGHER]
    deg = []
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            if keys[a] in lin and keys[b] in lin:
                deg.append(dict(pair=f'{keys[a]}|{keys[b]}', net_delta=float(sd(net[:, a, b]).mean()), mean_abs_net=float(np.nanmean(np.abs(net[:, a, b])))))
    deg.sort(key=lambda r: -abs(r['net_delta']))
    per_stat = {k: max(abs(r['net_delta']) for r in deg if k in r['pair'].split('|')) for k in lin}

    rel, names = D['rel'], keys + ['ksg']
    g = lambda u, s: groups.index(f'{u}__{s}')
    dr = lambda units, s, vs: np.array([[np.mean([rel[g(u, s), v, k] - rel[g(u, S.REF), v, k] for v in vs]) for k in range(len(names))] for u in units])
    real, iaa, eo, sl = dr(ses, 'drug', (0,)), dr(ses, 'drug', (1, 2, 3)), dr(ses, 'awake_eo', (0,)), dr(slp, 'sleep', (0,))
    relrows = []
    for k, n in enumerate(names):
        nt = real[:, k] - iaa[:, k]
        am = np.array([nt[A == a].mean() for a in animals])
        relrows.append(dict(stat=n, awake_reliability=float(np.mean([rel[g(u, S.REF), 0, k] for u in ses])), real=float(real[:, k].mean()),
                            iaaft=float(iaa[:, k].mean()), net=float(nt.mean()), net_t3=float(ttest_1samp(am, 0).statistic) if am.std() > 0 else None,
                            net_animals_negative=int((am < 0).sum()), net_agents={a: float(nt[AG == a].mean()) for a in S.DRUGS},
                            real_agents={a: float(real[AG == a, k].mean()) for a in S.DRUGS},
                            eyes_open=float(eo[:, k].mean()), sleep=float(sl[:, k].mean())))
    out = dict(degeneracy=dict(median_abs=float(np.median([abs(r['net_delta']) for r in deg])), per_statistic_max=per_stat, pairs=deg),
               reliability=relrows)
    json.dump(out, open(output, 'w'), indent=1)
    print('degeneracy median', out['degeneracy']['median_abs'], {k: round(v, 3) for k, v in per_stat.items()})
    for r in relrows:
        print(f"{r['stat']:10s} awake {r['awake_reliability']:.2f} real {r['real']:+.3f} iaaft {r['iaaft']:+.3f} net {r['net']:+.3f} t3 {r['net_t3']} "
              f"agents {' '.join(f'{v:+.3f}' for v in r['real_agents'].values())}")


if __name__ == '__main__':
    main(*sys.argv[1:4])
