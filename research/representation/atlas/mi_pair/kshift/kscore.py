"""Score the kshift side-by-side rescoring (kz.py output) with each study's own estimand, units and null scale.

    python kscore.py claar  KZ_PRIMARY KZ_IND KZ_LATE STORED_DIR OUT_DIR    (STORED_DIR has claar_z_{ns,ind,late}.npz)
    python kscore.py kauvar KZ STORED.npz OUT_DIR
    python kscore.py prios  KZ STORED.npz OUT_DIR [need]

1. Gate: the k4 column against the stored study z over every window, real and surrogate separately.
2. Per estimator, a view in the study's stored format (claar `ns`, Kauvar `asis`, PRIOS `Z[:, None]`) is written to OUT_DIR and
   the study's frozen scorer is run on it unchanged (claar_sum/claar_ind_sum/claar_late_sum, kv_sum, human/nulls). The k4 view
   must reprint the frozen primary lines. Verdict words those scripts print for kshift or k2te are descriptive, not verdicts.
   claar_ind_sum.py and claar_late_sum.py hard-code the k4 null SD .0026; the table below gives each estimator's own.
3. Table (robustness-result.md format): per estimator, mean unit Δ, Δ in its own null SD, R = Δ_e / Δ_k4, units negative,
   one-sided test, units whose sign agrees with k4, and Δ split into real-z and surrogate-z changes. Plus the mean subtracted
   shift offset (k4 − kshift over pairs) by state, real and surrogate.
"""
import sys, os, json, subprocess, numpy as np
from scipy.stats import binom, ttest_1samp
HERE = os.path.dirname(os.path.abspath(__file__)); MP = os.path.join(HERE, '..'); PY = sys.executable
EST = ('k4', 'kshift', 'k2te'); SECOND = 'sub-569073_ses-20210416'


def load(fn):
    d = np.load(fn); return d['Z'], d['O'], json.loads(str(d['rows']))


def gate(Z, stored):
    gr, gs = np.nanmax(np.abs(Z[:, 0, 0] - stored[:, 0])), np.nanmax(np.abs(Z[:, 0, 1:] - stored[:, 1:]))
    print(f'gate: {len(Z)} windows, max |k4 z − stored| real {gr:.2e}, surrogate {gs:.2e} ->', 'PASS' if max(gr, gs) < 1e-6 else 'FAIL', flush=True)


def frozen(script, *args):
    print(f'\n$ {os.path.relpath(script, MP)} ' + ' '.join(os.path.basename(a) for a in args), flush=True)
    print(subprocess.run([PY, script, *args], capture_output=True, text=True, cwd=os.path.dirname(script)).stdout, flush=True)


def table(name, units, real, surr, nullfn, test='sign'):
    """units(res) -> (labels, Δ per unit); nullfn(res) -> split-half per unit. real/surr: (W, est) z."""
    print(f'\n## {name}\n| Estimator | mean Δ | own null SD | Δ / null SD | R | units < 0 | test | sign = k4 | Δ real z | Δ surr z |')
    print('|---|---|---|---|---|---|---|---|---|---|'); base = None
    for e, est in enumerate(EST):
        if not np.isfinite(real[:, e]).any(): continue
        res = real[:, e] - surr[:, e]; lab, d = units(res); _, dr = units(real[:, e]); nl = nullfn(res)
        sd = np.std(nl, ddof=1) if len(nl) > 1 else np.nan; n, neg = len(d), int((d < 0).sum())
        if base is None: base = d
        t = (f'sign p {binom.sf(neg - 1, n, .5):.4f}' if test == 'sign' else f't({n - 1}) {ttest_1samp(d, 0).statistic:+.2f}, '
             f'p {ttest_1samp(d, 0, alternative="less").pvalue:.4f}')
        R = d.mean() / base.mean() if est != 'k4' else 1.0
        print(f'| {est} | {d.mean():+.4f} | {sd:.4f} | {d.mean() / sd:+.1f} | {R:.2f} | {neg}/{n} | {t} | '
              f'{int((np.sign(d) == np.sign(base)).sum())}/{n} | {dr.mean():+.4f} | {dr.mean() - d.mean():+.4f} |')
        if est == 'kshift': print('| | per unit (k4 → kshift): ' + ', '.join(f'{l} {a:+.4f}→{b:+.4f}' for l, a, b in zip(lab, base, d)) + ' |')


def offsets(O, state, labels):
    print('\nmean subtracted shift offset (k4 − kshift, mean over pairs), real / surrogate mean:')
    for s in labels: m = state == s; print(f'  {s}: {O[m, 0].mean():+.4f} / {O[m, 1:].mean():+.4f} (n {m.sum()})')


def view(out, Z, rows, key, e, wrap=False):
    v = Z[:, e][:, None] if wrap else Z[:, e]; np.savez(out, rows=json.dumps(rows), **{key: v}); return out


def claar(kp, ki, kl, sd, od):
    (ZA, OA, rA), (ZI, OI, rI), (ZL, OL, rL) = load(kp), load(ki), load(kl)
    for Z, f in ((ZA, 'claar_z_ns'), (ZI, 'claar_z_ind'), (ZL, 'claar_z_late')): print(f, end=' '); gate(Z, np.load(f'{sd}/{f}.npz')['ns'])
    for e, est in enumerate(EST):
        if not np.isfinite(ZA[:, e]).any(): continue
        a, i, l = (view(f'{od}/claar_{k}_{est}.npz', Z, r, 'ns', e) for k, Z, r in (('ns', ZA, rA), ('ind', ZI, rI), ('late', ZL, rL)))
        print(f'\n===== frozen scorers on the {est} view' + ('' if est == 'k4' else ' (descriptive)'))
        frozen(f'{MP}/claar/claar_sum.py', a); frozen(f'{MP}/claar/claar_ind_sum.py', a, i); frozen(f'{MP}/claar/claar_late_sum.py', a, i, l)
    g = lambda R, k: np.array([r[k] for r in R])
    sA, stA, tA, biA, dA = g(rA, 'session'), g(rA, 'state'), g(rA, 'start_s'), g(rA, 'burst'), g(rA, 'delta_fraction')
    sI, biI, dI, sL = g(rI, 'session'), g(rI, 'burst'), g(rI, 'delta_fraction'), g(rL, 'session')
    P = [s for s in dict.fromkeys(sA) if s != SECOND]
    aw = lambda s: (sA == s) & (stA == 'awake')

    def split(res):
        out = []
        for s in P:
            w = np.flatnonzero(aw(s)); w = w[np.argsort(tA[w])]; h = len(w) // 2; out.append(res[w[-h:]].mean() - res[w[:h]].mean())
        return np.array(out)
    prim = lambda res: (P, np.array([res[(sA == s) & (stA == 'iso')].mean() - res[aw(s)].mean() for s in P]))
    table('Claar primary: isoflurane − awake, 17 mice (frozen: ≥ 13/17 negative and mean beyond −2 null SD)', prim,
          ZA[:, :, 0], ZA[:, :, 1:].mean(2), split)
    tgt = {s: (sI == s) & (biI >= np.percentile(biA[aw(s)], 5)) & (dI >= np.percentile(dA[aw(s)], 95)) for s in P}
    Q = [s for s in P if tgt[s].sum() >= 3]
    rres = lambda Z: Z[:, :, 0] - Z[:, :, 1:].mean(2)
    RA, RI, RL = rres(ZA), rres(ZI), rres(ZL); zA, zI, zL = ZA[:, :, 0], ZI[:, :, 0], ZL[:, :, 0]
    print('\n## Claar Addendum A: induction target − awake (frozen: ≥ k negative, mean below −2 null SD; null SD = primary split half)')
    print('| Estimator | mean Δ_ind | Δ / own null SD | R | mice < 0 | sign p | sign = k4 | Δ real z | Δ surr z |'); print('|---|---|---|---|---|---|---|---|---|')
    L = [s for s in P if (sL == s).sum() >= 3]; base = {}
    for e, est in enumerate(EST):
        if not np.isfinite(RA[:, e]).any(): continue
        d = np.array([RI[tgt[s], e].mean() - RA[aw(s), e].mean() for s in Q]); dr = np.array([zI[tgt[s], e].mean() - zA[aw(s), e].mean() for s in Q])
        sd = np.std(split(RA[:, e]), ddof=1); base.setdefault('i', d); n, neg = len(d), int((d < 0).sum())
        print(f'| {est} | {d.mean():+.4f} | {d.mean() / sd:+.1f} | {d.mean() / base["i"].mean():.2f} | {neg}/{n} | {binom.sf(neg - 1, n, .5):.4f} | '
              f'{int((np.sign(d) == np.sign(base["i"])).sum())}/{n} | {dr.mean():+.4f} | {dr.mean() - d.mean():+.4f} |')
    print('\n## Claar Addendum B: late − early awake (frozen: drift explains if mean ≤ Δ_ind/2)')
    print('| Estimator | mean Δ_late | Δ_late / Δ_ind (same estimator) | mice < 0 | sign = k4 |'); print('|---|---|---|---|---|')
    for e, est in enumerate(EST):
        if not np.isfinite(RA[:, e]).any(): continue
        d = np.array([RL[sL == s, e].mean() - RA[aw(s), e].mean() for s in L]); di = np.mean([RI[tgt[s], e].mean() - RA[aw(s), e].mean() for s in Q])
        base.setdefault('l', d)
        print(f'| {est} | {d.mean():+.4f} | {d.mean() / di:+.2f} | {int((d < 0).sum())}/{len(d)} | {int((np.sign(d) == np.sign(base["l"])).sum())}/{len(d)} |')
    offsets(np.concatenate([OA, OI, OL]), np.concatenate([stA, np.where(np.isin(np.arange(len(rI)), np.flatnonzero(np.any([tgt[s] for s in Q], 0))), 'induction target', 'induction other'), ['late'] * len(rL)]),
            ('awake', 'iso', 'induction target', 'induction other', 'late'))


def kauvar(kz, st, od):
    Z, O, rows = load(kz); gate(Z, np.load(st)['asis'])
    for e, est in enumerate(EST):
        if np.isfinite(Z[:, e]).any():
            print(f'\n===== frozen scorer on the {est} view' + ('' if est == 'k4' else ' (descriptive)')); frozen(f'{MP}/kauvar/kv_sum.py', view(f'{od}/kv_{est}.npz', Z, rows, 'asis', e))
    g = lambda k: np.array([r[k] for r in rows]); sub, stt, t = g('session'), g('state'), g('start_s'); S = list(dict.fromkeys(sub))
    S = [s for s in S if ((sub == s) & (stt == 'reference')).sum() >= 10 and ((sub == s) & (stt == 'infusion')).sum() >= 10]
    u = lambda res: (S, np.array([res[(sub == s) & (stt == 'infusion')].mean() - res[(sub == s) & (stt == 'reference')].mean() for s in S]))

    def split(res):
        out = []
        for s in S:
            w = np.flatnonzero((sub == s) & (stt == 'reference')); w = w[np.argsort(t[w])]; h = len(w) // 2; out.append(res[w[-h:]].mean() - res[w[:h]].mean())
        return np.array(out)
    table('Kauvar: infusion − reference, 7 subjects (frozen ABSENT: |mean| within 2 null SD and CI lower end > −.010). R is undefined: Δ(k4) ≈ 0',
          u, Z[:, :, 0], Z[:, :, 1:].mean(2), split)
    for e, est in enumerate(EST):
        if np.isfinite(Z[:, e]).any():
            d = u(Z[:, e, 0] - Z[:, e, 1:].mean(1))[1]; b = np.random.default_rng(0).choice(d, (10000, len(d))).mean(1)
            print(f'{est}: subject-bootstrap 95% CI [{np.percentile(b, 2.5):+.4f}, {np.percentile(b, 97.5):+.4f}]')
    offsets(O, stt, ('reference', 'infusion', 'postinfusion'))


def prios(kz, st, od, need=20):
    Z, O, rows = load(kz); gate(Z, np.load(st)['Z'][:, 0])
    for e, est in enumerate(EST):
        if np.isfinite(Z[:, e]).any():
            print(f'\n===== frozen null-scale script on the {est} view' + ('' if est == 'k4' else ' (descriptive)')); frozen(f'{MP}/human/nulls.py', view(f'{od}/prios_{est}.npz', Z, rows, 'Z', e, wrap=True))
    subj, stt = np.array([r['subject'] for r in rows]), np.array([r['state'] for r in rows])
    S = [s for s in np.unique(subj) if min(((subj == s) & (stt == q)).sum() for q in ('awake', 'anaes')) >= need]
    u = lambda res: ([s[4:] for s in S], np.array([res[(subj == s) & (stt == 'anaes')].mean() - res[(subj == s) & (stt == 'awake')].mean() for s in S]))

    def split(res):
        out = []
        for s in S:
            a = res[(subj == s) & (stt == 'awake')]; h = len(a) // 2; out.append(a[:h].mean() - a[h:2 * h].mean())
        return np.array(out)
    table('PRIOS: propofol − awake (frozen REPLICATES: mean < 0, one-sided t p < .05, at most one subject > 0)', u, Z[:, :, 0], Z[:, :, 1:].mean(2), split, test='t')
    for e, est in enumerate(EST):
        if np.isfinite(Z[:, e]).any():
            d = u(Z[:, e, 0] - Z[:, e, 1:].mean(1))[1]; p = ttest_1samp(d, 0, alternative='less').pvalue
            print(f'{est}: frozen rule applied (descriptive for kshift/k2te):', 'REPLICATES' if d.mean() < 0 and p < .05 and (d > 0).sum() <= 1 else 'not met')
    offsets(O, stt, ('awake', 'anaes'))


if __name__ == '__main__':
    a = sys.argv; os.makedirs(a[-1] if a[1] != 'prios' or len(a) == 5 else a[-2], exist_ok=True)
    if a[1] == 'claar': claar(*a[2:7])
    elif a[1] == 'kauvar': kauvar(*a[2:5])
    elif a[1] == 'prios': prios(*a[2:5], *(int(x) for x in a[5:6]))
