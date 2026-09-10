# Collective period-doubling CML: bounded staged protocol

User-approved next benchmark; no Ising/Vicsek work or monitor restart implied.
Preserve all previous results. Source: https://arxiv.org/html/nlin/0605004,
Eqs1–4, Fig3/Table1. Synchronous square periodic lattice, f(x)=r*x*(1-x),
g=.2; numerical thermodynamic r_c=3.86212(12), not an analytic solution.

## Physics and observation, fixed before results

- Smoke L64, r3.84/3.89, one seed, burn2k/record4k.
- Size gate L64/128/256, seven r anchors in committed config, two seeds,
  burn20k/record42k. All-step global means; float64 M64 dispersed/contiguous
  sensor traces only for first2k steps; no full field history.
- Q=mean(abs(xbar(2t+1)-xbar(2t))) on the following40k steps, disjoint from
  every observation prefix. Retain eight blocks and full scalar mean trace.
  Large-N/long-time averages, not microscopic absolute differences.
- Independent dynamic/sensor RNG streams. Same seed's controls are paired;
  nested M8/16/32/64 sensor identities and T prefixes are not new replicates.
  Contiguous rectangles 2x4,4x4,4x8,8x8; no target-selected location.
- Numerical tests: independent roll oracle; streaming time/site alignment;
  exact constructed Q; RNG replay; invariant interval; no overwrite.
- Physics review: global period branches, finite-N background, block/start
  sensitivity and sensor dynamics. Longer burns/preordered starts and larger
  L only on anchors that resolve a specific uncertainty, before selecting N.
  Do not infer adequate sampling solely from a larger lattice.

## Conditional p90 development

First choose one adequate N using physics only. Freeze a local control grid,
independent development/evaluation seeds, M32/T1000 dispersed observation;
test a bounded primary bank with all289 p90 SPIs and unified_ordered_v3 Pearson
z (41616 features). Target-blind training-only finite/SD gates, imputation and
centering; frozen PC1 is the initial hypothesis, not a target-selected component.
If geometry is not adequately one-dimensional, report it rather than search
for the PC with greatest Q association. State exact geometry/row gates before
extraction. Evaluation is exploratory unless a genuinely untouched confirmation
stage is separately sealed. Report simple raw baselines without superiority
being a success requirement. No mandatory within-control/time tracking.

If useful, add nested M8/16 and T100/500 arms with the same transform, plus
contiguous sensitivity. M64 only to resolve measured observation insufficiency;
it is not a default grid factor. No supervised decoder or new embedding-method
search in this pass. Append positive, null or blocked results to the comparison
notebook through its builder, preserving prior sections and statuses.

## Resources and stopping

Initial self-imposed ceiling500 SU for physics plus a bounded p90 pilot; scale
only after measured timings and scientific review, not merely available cores.
No new physics beyond the one intercept. No automatic recurring reminder.
Source/config and archive identities must be retained; immutable job snapshots
avoid modifying other tasks' live cluster checkout. Record job IDs and actual
costs. Stop a failed physics/observability gate and report, without modifying
the generator or choosing sensors to force a favourable result.

## First physics review and bounded follow-up

Size gate reproduces stable ordered-side Q~.334 and decreasing disordered
finite-size floor. At r=.866, near-boundary block/seed variation remains.
Convergence config compares random/preordered starts at L128/256 after200k
burn plus402k record, and four L512 short size anchors. This is a targeted
convergence/precision check, not a blanket million-site expansion.
Two already-generated L256 endpoint views (seed26091101,r3.84/3.89,M32T1000)
are a p90 runtime/validity smoke only; no q fitting or benchmark scores.

Execution: local smoke2 and physics42 complete; all7 numerical/export tests
pass. Pinned published-base source d619b02658cbfb704c6450b17f50e01818c66428 on
`codex/cml2d-period-doubling`; unrelated local commits remain unpushed.
Isolated remote worktree `/scratch/ql44/we2614/cml2d-source-d619b02` avoids
changing another task's live checkout. Convergence job178709040 requests
12cores48GB/30min (maximum12SU). P90 smoke178709102 requests4cores16GB/1h,
2workers (maximum8SU); input archive SHA
`5b9221b6c299111d4c990b99cbd7fcc5d04cef15f9b83592184e485db6076eff`.
No recurring reminder created. Notebook section7 currently contains physics,
with explicit absence of a SPI result until a primary analysis exists.

## Primary pilot specification (prepared; submission conditional on gates)

Use L256 (N65536): initial size audit resolves the disordered noise floor while
avoiding the additional relaxation cost of L512. Larger-L anchors remain
physical sensitivities. For the primary bank, extend the future reference to
one million steps uniformly across controls, after200k burn and2k observation;
this targets the measured temporal-reference uncertainty, not larger observed M.
Nine r controls in `cml2d-period-doubling-primary.yaml`, eight fresh seeds:
26091111–14 development;26091115–18 evaluation.72 masters; initial p90 only
M32/T1000 dispersed. All physical outcomes used to plan these settings are from
the separate physics scouts. This is exploratory held-seed evaluation, not
prospective independent confirmation of a previously frozen CML2D coordinate.

Freeze before extraction: unified_ordered_v3 Pearson; all289 p90 SPIs;
development feature finite fraction>=.99 and SD>=.05; median imputation and
centering only, no per-feature whitening. PC1 is fixed a priori, sign anchored
to its largest-magnitude loading. Geometry gate EVR1>=.20, EVR1/EVR2>=1.5,
minimum leave-development-seed-out loading cosine>=.8. Row eligibility selected
missingness<=.05, excluded fraction<=.10 and at least2 retained records per
role/control/view/M/T cell. Failed geometry/eligibility is reported, not repaired
by target-selected components, sensor locations or lower thresholds.
Primary Q_reference is the common disjoint million-step scalar; Q_window is
a separate matched-window physical diagnostic. Report both associations plus
within-control association, raw mean-absolute-correlation and observed-sample
order baselines. No requirement to beat either baseline. Report no precise
thermodynamic boundary or instantaneous-tracking claim.
