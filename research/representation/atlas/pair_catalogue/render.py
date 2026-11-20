"""Render catalogue.md from catalogue.json and identities.json (run after build.py).

    python -m research.representation.atlas.pair_catalogue.render
"""
import json
from pathlib import Path

HERE = Path(__file__).parent

INTRO = """# Pair catalogue: z(A, B) coordinates with a clean reading of dependence character (20 November 2026)

Interpretive map over the r13 NeuroTycho atlas (272 SPIs; 222 valid in every window), built to choose the next computation. Nothing here is a new effect: numbers are copied from committed outputs by [`build.py`](build.py), the Gaussian relations are checked numerically by [`identities.py`](identities.py) ([`identities.json`](identities.json)), and the full pair-level map, every one of the 584 minimal pairs plus 15 non-minimal ones, is [`catalogue.json`](catalogue.json). Reading rules: [`docs/interpreting-spi-pairs.md`](../../../../docs/interpreting-spi-pairs.md). Curated readings: [`entries.py`](entries.py). Regenerate with `python -m research.representation.atlas.pair_catalogue.build && python -m research.representation.atlas.pair_catalogue.render`.

## How to read this

- **Entry:** one assumption difference (a class of minimal pairs from [`assumptions.py`](../assumptions.py)) or one principled non-minimal pair with a known Gaussian relation. The representative pair comes first; `catalogue.json` lists every member pair with its own numbers.
- **Tier and null:**
  - *higher-order:* both statistics have a Gaussian expectation fixed by second-order structure, so real − joint spectrum-final IAAFT reads the departure the flexible member targets.
  - *linear-spectral:* both are fixed by the sample cross-spectra, so IAAFT is degenerate; these need the zero-phase and stationary Gaussian nulls with the state's cross-spectrum.
  - *linear-static:* fixed by the zero-lag covariance; read raw with strength controls.
- **Ranking:** score = interpretability/3 × measurability. Measurability is the better arm's smaller partner reliability, times a ceiling factor and a null-validity factor. The ceiling factor is .4 or .7 at z₀ > .98 or > .95 (.7 or .9 for higher-order pairs, whose IAAFT net carries the surrogate's saturation mechanics). Exclusions score 0.
- **Columns:**
  - rel: wake between-window edge-profile reliability of A and B, 2-s / 10-s arms.
  - z₀: wake z in the 2-s / 10-s pyspi bank (ax: pyspi-free axis screen, 10 s).
  - matched: the 2-s per-window joint-IAAFT net, drug − awake, mean over 19 sessions (sessions sharing its sign). It comes from the normal-scored construction of [`movers/matched-result.md`](../movers/matched-result.md); it is degenerate (≈ 0 by construction) for linear pairs.
- **Status:** *untested* (no z computed), *raw* (stored pyspi, no surrogate), *beyond IAAFT* (a net exists), *axis* (passed the frozen axis-screen gates), *failed* (failed them, reason in brackets).

## Principles found while building it

1. **Static-statistic principle (verified).** On z-scored channels, the expectation of any time-exchangeable bivariate statistic is a fixed function of r under a Gaussian process. The set includes the rank correlations, the MI estimators, dCor, IDS, the L1, L2, cosine and Canberra distances and the barycentre moments. So every pair from that set is a principled non-minimal pair beyond IAAFT, and its reading is the feature the more flexible member weights:
   - rank correlations: concordance;
   - kNN MI: the full KL departure;
   - dCor: non-monotone dependence;
   - L1/L2: kurtosis of x − y;
   - Canberra: per-sample scale-free agreement.

   The Gaussian network check gives z(r², dCor²) .988, z(r, Spearman) .998 and z(L1, L2) .995. Maxima are the exception. Chebyshev and the barycentre maximum also depend on T and on the spectrum of x ∓ y (max/RMS 2.97 for white noise against 2.31 at AR .98, T = 200), which joint IAAFT fixes but r alone does not.
2. **Map the partner through its Gaussian relation before correlating.** This removes saturation from z₀, which is then 1 minus estimator noise under the null. The maps:
   - ranks: arcsin;
   - dCor: Székely et al. 2007, Thm 7;
   - MI: −½ log(1 − r²);
   - envelopes: closed forms in |γ|².

   The same map gives a per-edge level readout (excess = H − f(L)) beside z.
3. **The envelope's Gaussian partner is |γ|² = r² + corr(x, H[y])², not r² (verified).** For proper complex Gaussian analytic signals, corr(\\|a\\|², \\|b\\|²) = \\|γ\\|². Envelope amplitude and log power are closed functions of |γ|², and PLV is (π/4)|γ| ₂F₁(½, ½; 2; |γ|²) per frequency. So z(pec, r²) mixes in lag, and z(pec, |γ|²) reads amplitude co-modulation beyond linear coupling. IAAFT keeps γ exactly, because γ is a linear functional of the cross-periodogram.
4. **Canberra is gain-invariant on z-scored windows (verified by synthetic, this session).** The excess of Canberra over its Gaussian value f_G(r):

   | Synthetic | Excess |
   |---|---|
   | Pure shared gain | −.004 |
   | Separate per-channel gains | −.001 |
   | Correlation falling with amplitude (amplitude-linked decoupling) | −.120 |
   | Correlation rising with amplitude (gain over a fixed floor) | +.100 |
   | Static quadratic | −.011 |
   | After normal-scoring, shared gain | −.061 |

   So on z-scored windows the sign of the excess says how correlation depends on amplitude, while a pure scale mixture leaves it unmoved, unlike the MI anchor. This is the first readout in the atlas that could separate "one gain on an unchanged correlation" from "correlation that changes with amplitude", the sub-question the anchor's route-2 reading leaves open. Normal-scoring (the atlas's IAAFT construction) destroys the invariance.
5. **The null follows the estimator, not only the statistic.** Spectrum-final IAAFT keeps the periodogram and cross-periodogram exactly, so a frequency-smoothed (Daniell) coherence is degenerate under it. Multitaper and zero-padded lagged statistics are not (axis screen, post hoc), so a multitaper member makes real − IAAFT also respond to within-window spectral non-stationarity.
"""


def esc(text):
    return str(text).replace('|', '\\|')


def short(text, n=150):
    text = (text or '').split(';')[0]
    return text if len(text) <= n else text[:n - 1].rsplit(' ', 1)[0] + '…'


def f2(v):
    return '–' if v is None else f'{v:.2f}'


def rel(r):
    return ' / '.join(f'{f2(r["reliability"][a][0])}, {f2(r["reliability"][a][1])}' for a in ('short', 'long'))


def z0(r):
    s = f'{f2(r["z0"]["short"])} / {f2(r["z0"]["long"])}'
    ax = r['axis_screen'].get('long')
    return s + (f' (ax {ax["z0"]:.2f})' if ax and ax.get('z0') is not None else '')


def matched(r):
    m = r['matched_2s']
    return '–' if not m else f'{m["net"]:+.3f} ({m["same_sign"]}/{m["sessions"]})'


def pair(r):
    return f'`{r["a"]}` · `{r["b"]}`'


def status(e):
    return e['status']


def row(*cells):
    return '| ' + ' | '.join(esc(c) for c in cells) + ' |'


def main():
    d = json.loads((HERE / 'catalogue.json').read_text())
    ident = json.loads((HERE / 'identities.json').read_text())
    E = d['entries']
    L = [INTRO]

    L.append('## Shortlist: what to compute next\n')
    L.append('Ordered for the next computation: the gain-discriminating and envelope readouts first (new, cheap, pyspi-free), then the extreme-value family that already moves beyond IAAFT, then calibration of the existing MinCovDet axis, then the ceiling form pairs with mapped partners, then the linear tier. Row 0 is the anchor, recomputed as the calibration reference in every run. Contrast = drug − awake eyes-closed per session, animal-level t(3), eyes-open and within-awake split-half controls, as in the axis screen.\n')
    L.append('| # | Entry | Pair(s) | Reading | Contrast to compute | Null |')
    L.append('|---|---|---|---|---|---|')
    sl = sorted([e for e in E if e.get('shortlist')], key=lambda e: e['shortlist']['rank'])
    for e in sl:
        L.append(row(e['shortlist']['rank'], e['title'], pair(e['representative']), short(e['character'] or e['literal'], 110), e['shortlist']['contrast'], short(e['shortlist']['null'], 120)))
    L.append('')
    L.append('Expected compute: rows 1–4 and 6–7 are pyspi-free statistics on the anchor\'s windows with the screen\'s IAAFT draws or a Gaussian FT surrogate. That is the axis-screen pipeline (0.23 kSU for 33 SPIs in both arms), so about 0.1–0.3 kSU expected and 0.5 kSU worst case on Gadi. Row 8 needs a Daniell estimator added to the same pipeline. Calibration synthetics (rows 1, 2, 5, 8) are local and take minutes.\n')

    U = d['upstream']
    L.append(f"## Upstream pyspi changes (checked {U['checked']})\n")
    L.append(f"Pin {U['pin']}, upstream main {U['upstream_main']}. The atlas config is a frozen copy of `benchmarked_p90`, so no r13 number here changes and nothing leaves the bank until the main session re-syncs it. {U['principle']}\n")
    L.append('| PR | State | What | Impact on the catalogue |')
    L.append('|---|---|---|---|')
    for u in U['prs']:
        L.append(row(u['pr'], u['state'], u['what'], u['impact']))
    L.append(f"\nPending upstream: {U['pending']}.\n")
    L.append('| Entry | Members affected | Note |')
    L.append('|---|---|---|')
    for e in E:
        if e.get('upstream'):
            L.append(row(e['title'], f"{e['upstream_members']}/{e['n_members']}", e['upstream']))
    L.append('\nCandidate pairs from new upstream SPIs (not yet entries):\n')
    for c in U['candidates']:
        L.append(f'- {c}')
    L.append('')

    L.append('## Ranked catalogue\n')
    L.append('| Rank | Entry | Level · tier | Representative pair | Reads | rel 2 s / 10 s | z₀ 2 s / 10 s | matched 2-s net | Status | Score |')
    L.append('|---|---|---|---|---|---|---|---|---|---|')
    for e in E:
        if e['rank'] is None:
            continue
        r = e['representative']
        L.append(row(e['rank'], f"[{e['title']}](#{e['id']})", f"{e['level']} · {e['tier']}", pair(r), short(e['character'] or e['literal'], 120), rel(r), z0(r), matched(r), status(e), f"{e['score']:.2f}"))
    L.append('')

    L.append('## Readings\n')
    for e in [x for x in E if x['rank'] is not None]:
        r = e['representative']
        L.append(f'### <a id="{e["id"]}"></a>{e["rank"]}. {e["title"]}\n')
        L.append(f'- **Pairs:** {"; ".join(pair(m) for m in e["members"][:4])}{" (+" + str(e["n_members"] - 4) + " more in the JSON)" if e["n_members"] > 4 else ""}.')
        L.append(f'- **What differs:** {e["differs"]}.')
        L.append(f'- **Character reading:** {e["character"] or "none earned"}.')
        L.append(f'- **Literal reading:** {e["literal"]}.')
        L.append(f'- **Other consequences:** {"; ".join(e["consequences"]) or "–"}. **Main confound:** {e["confound"]}.')
        L.append(f'- **Null:** {e["null"]}.')
        L.append(f'- **Gaussian relation:** {e["gaussian_relation"]}.')
        L.append(f'- **Measurability:** reliability {rel(r)}; z₀ {z0(r)}; null SD {f2(r["null_sd"]["short"]) if r["null_sd"]["short"] is None else format(r["null_sd"]["short"], ".3f")} / {f2(r["null_sd"]["long"]) if r["null_sd"]["long"] is None else format(r["null_sd"]["long"], ".3f")}; matched 2-s net {matched(r)}.')
        L.append(f'- **Validation:** {e["validation"]}. **Status:** {e["status"]}.')
        L.append(f'- **Atlas so far:** {e["atlas"]}.')
        L.append(f'- **Calibration needed:** {e["calibration"]}.' + ('' if e.get('upstream') else '\n'))
        if e.get('upstream'):
            L.append(f'- **Upstream pyspi:** {e["upstream"]}\n')

    L.append('## Exclusions (scored 0)\n')
    L.append('| Entry | Representative pair | Reason |')
    L.append('|---|---|---|')
    for e in E:
        if e['rank'] is None:
            L.append(row(e['title'], pair(e['representative']), e.get('exclusion') or 'interpretability 0'))
    L.append('')

    L.append('## Gaussian relations: numerical check ([`identities.json`](identities.json))\n')
    L.append('| Relation | Check | Largest deviation |')
    L.append('|---|---|---|')
    rk = ident['rank_correlations']
    L.append(f'| Spearman = (6/π) arcsin(ρ/2) | ρ .1–.9, n 20,000 | {max(abs(x["spearman"] - x["spearman_formula"]) for x in rk):.3f} |')
    L.append(f'| Kendall = (2/π) arcsin ρ | ρ .1–.9, n 4,000 | {max(abs(x["kendall"] - x["kendall_formula"]) for x in rk):.3f} (sampling) |')
    dc = ident['distance_correlation']
    L.append(f'| dCor² Thm 7 (Székely et al. 2007) | ρ .1–.9, n 2,500 | {max(abs(x["dcor2"] - x["formula"]) for x in dc):.3f} |')
    ev = ident['envelope']
    L.append(f'| corr(\\|a\\|², \\|b\\|²) = \\|γ\\|² | lags 0–10, n 400,000 | {max(abs(x["power_corr"] - x["power_formula"]) for x in ev):.3f} |')
    L.append(f'| corr(\\|a\\|, \\|b\\|) = (π/4)(₂F₁(−½,−½;1;\\|γ\\|²) − 1)/(1 − π/4) | same | {max(abs(x["amp_corr"] - x["amp_formula"]) for x in ev):.3f} |')
    L.append(f'| corr(log\\|a\\|², log\\|b\\|²) = 6 Li₂(\\|γ\\|²)/π² | same | {max(abs(x["logpower_corr"] - x["logpower_formula"]) for x in ev):.3f} |')
    pl = ident['phase_locking']
    L.append(f'| PLV = (π/4)\\|γ\\| ₂F₁(½,½;2;\\|γ\\|²) | \\|γ\\| .2–.95 | {max(abs(x["plv"] - x["formula"]) for x in pl):.4f} |')
    xi = ident['chatterjee_xi']
    L.append(f'| Chatterjee ξ = (3/π) arcsin((1 + ρ²)/2) − ½ | ρ 0–.95, n 20,000 | {max(abs(x["xi"] - x["formula"]) for x in xi):.3f}; SD at T 200 {min(x["sd_T200"] for x in xi):.3f}–{max(x["sd_T200"] for x in xi):.3f} |')
    ds = ident['difference_signal']
    L.append(f'| mean\\|d\\| / rms(d) = √(2/π) | T 200/1000, AR 0–.98 | {max(abs(x["l1_over_rms"] - x["sqrt_2_over_pi"]) for x in ds):.3f} (finite-T bias at AR .98) |')
    L.append(f'| max\\|d\\| / rms(d) depends on T and spectrum | same | {min(x["max_over_rms"] for x in ds):.2f}–{max(x["max_over_rms"] for x in ds):.2f} |')
    g = ident['gaussian_network_z']
    L.append(f'| z under a 16-channel Gaussian network | 10 reps, T 1000 | r²·dCor² {g["r2|dcor2"]["mean"]:.3f}, r·Spearman {g["r|spearman"]["mean"]:.3f}, L1·L2 {g["l1|l2"]["mean"]:.3f}, L∞·L2 {g["linf|l2"]["mean"]:.2f} |')
    L.append('\nThe GC = 2 TE relation (Barnett, Barrett & Seth 2009) is definitional in pyspi: `gc_gaussian` is the Gaussian TE estimator in nats (source, `infotheory.py`). Geweke\'s spectral identity relates `sgc` integrated over frequency to time-domain GC with the true marginal restricted model, not to `gc_gaussian` with history 1, so it gives no fixed edgewise map in this bank (and `sgc` reliability is .01–.18 at 2 s).\n')

    L.append('## Proposals for the reading guide (owned by the main session)\n')
    for p in d['guide_proposals']:
        L.append(f'- {p}')
    L.append('')
    L.append('## Limits\n')
    L.append('- **Scores are judgements.** Interpretability is curated, and measurability uses wake reliability and wake z₀ of the representative pair only, so read the ranks as an ordering for the next computation, not as evidence.')
    L.append('- **New statistics have no numbers yet.** z₀ and null SD exist only for minimal pairs and named pairs in the stored banks; non-minimal and pyspi-free pairs (|γ|², normal-score Pearson, mapped partners) are untested here.')
    L.append('- **Synthetic checks are idealised.** The Canberra and identity checks are iid or AR(1), long-T synthetics; finite-T, multichannel, realistic-spectrum calibration is part of the next step.')
    L.append('- **The matched net comes from one construction.** It is normal-scored, so it is not the null Canberra needs.')
    (HERE / 'catalogue.md').write_text('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
