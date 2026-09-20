# Candidate evaluation against the original objective

## Decision

Updated after the completed finite-system follow-up: the suite is sufficient. Stuart–Landau, large 2D collective-period-doubling CML and Rössler are strong complementary choices; Desai–Zwanzig is a strong canonical-order alternative, and confirmed fullN32 TASEP adds stochastic non-synchronization physics. Miller–Huse remains supporting canonical evidence and Kuramoto the accessible baseline. Vicsek is not required. Prioritize physical clarity, validation and coverage of distinct observation regimes, not the largest correlation or the number of models. The old initial rankings below are qualified by current completed outcomes. No reminder is active.

The user authorized focused progression but then requested comparative evaluation before committing to Vicsek. Temporal tracking is a possible extension, not a new eligibility requirement for the original across-transition demonstration. Do not allow this extension to recast completed evidence as inadequate.

## Requirements retained

- Recognizable model with an established independently defined physical order parameter and a studied control-parameter boundary. A local intercept suffices.
- Clear finite-run order contrast; a sharp continuous transition or dynamical bifurcation qualifies. A first-order discontinuity is not mandatory.
- Genuine nondegenerate multivariate dynamics on both sides, without modifying the model merely to rescue the representation.
- M=N at small size is ideal, not mandatory. Large physical N and M<=32 partial observations are legitimate, with finite-size/time and sampling qualifications.
- Target/control-blind representation fitting, followed by validation against Q. Inference of a latent order-related coordinate is legitimate terminology; numerical calibration, equation discovery and instantaneous state recovery are separate, stronger claims.
- Variable M,T comparability is the methodological motivation; superiority to purpose-built statistics and direct clinical interpretation are not required.
- Preserve prior Ising-family implementation exclusion. Literature assessment does not authorize reinstating those simulations.

## Portfolio assessment

| System | Established target and local boundary | Fitness and decision |
|---|---|---|
| Stuart–Landau population | Complex collective order Z; magnitude and variability at locking-to-unsteady boundary | Best existing small-full-observation example. Fine M=N=32 confirmation aligns latent/physical steepest intervals; distinguish temporal dynamics of Z from its mean magnitude. Core. |
| Desai–Zwanzig | Population first moment; symmetry-breaking noise-driven pitchfork | Strong existing partial-observation example (N12000, M32), fine boundary recovery. Continuous, finite-size-shifted, not a first-order jump. Core. |
| Miller–Huse | Sign magnetization; coupling-driven symmetry breaking | Strong chaotic-lattice example with time-varying microscopic channels in either phase; existing N16384/M8–32 evidence. Strict confirmation gate failure and row-exclusion sensitivity remain disclosed. Core. |
| Kuramoto | Phase coherence; synchronization onset | Most accessible positive control; less dramatic and less complementary to Stuart–Landau. Existing full-catalogue result retrospective. Baseline. |
| 2D collective-period-doubling CML | Published even/odd spatial-mean order at a studied coupling/map-parameter boundary | Completed independent frozen confirmation, rho=.883, plus paired M/T results; large physical lattice, dispersed sparse observations. Core option. SmallL6/8 and contiguous-patch failures remain separate. |
| Rössler pair | Mean angular-frequency mismatch at chaotic phase entrainment | All six physical state coordinates,32-fresh-seed frozen confirmation rho=.848; localized sampled contrast, not exact asymptotic locking. Strong full-observation option. |
| Open TASEP | Exact stationary-density phase diagram; finite-size density crossover | FullN32 confirmed rho=.746; nativeN64 pilot rho=.813. Rounded finite-N boundary, but physically diverse and useful. Retrospective unchangedN32→N64 model also passes rho=.782; not fresh confirmation or universal size invariance. |
| Vicsek | Polarization; collective ordering transition | Highly recognizable, distinct active-matter physics. Existing N32768 cohort has finite-run contrast and episodes, but partial-observation informativeness is untested. Moving particles, fixed bins and projection choices matter. Optional, no automatic extraction. |
| Exact existing 1D Kaneko ring | Persistence at a specific published antiferromagnetic-order boundary; no canonical scalar spans our old epsilon=.3 sweep | Not generally disqualified as a model family. The old sweep is a regime-diagnostic stress test; the separate persistence target is history-dependent and less suitable for arbitrary windows. Do not retrofit it. |

Existing numerical results and their limitations are in the executed `notebooks/inference/order-parameter-benchmark-comparison.ipynb` and `docs/context/order-parameter-benchmarks.md`. Sizes above are tested designs, not universal minimum sizes or guarantees for arbitrary sensor layouts.

## Strong alternatives, and why they do not force expansion

| Candidate | Physics fit | Practical assessment |
|---|---|---|
| 2D synchronously coupled logistic maps, collective period doubling | Explicit temporal order parameter and studied transition line | Now completed and independently confirmed; see portfolio above. Not the old1D ring. |
| Periodically driven kinetic Ising | Cycle-averaged magnetization across a dynamic phase transition | Excellent formal fit and recognizable family; real stochastic dynamics, not static snapshots. Binary observations and slow response require care. Implementation remains excluded pending explicit revision. |
| Blume–Capel | Magnetization across first-order segment of temperature/crystal-field diagram | Excellent familiar sharp-transition physics. Use a specified local stochastic dynamics, not algorithm-dependent accelerated-Monte-Carlo trajectories. Discrete inputs/metastability and prior exclusion reduce current priority. |
| Potts q>4 in 2D | Majority-state magnetization across discontinuous temperature transition | Canonical first-order benchmark. Nominal categories cannot innocently be treated as ordered real numbers for all SPIs; one-hot/aggregate observation needs justification. Not a simpler drop-in replacement. |
| Inertial noisy Kuramoto | Coherence across a first-order surface | Strong if an actual jump is required; redundant with existing oscillator demonstrations. Metastable escape times increase exponentially with N, so very large N is not a shortcut. |
| ZGB with CO desorption | CO coverage across low-/high-coverage first-order line | Established catalytic model; distinct mechanism, stochastic dynamics. Existing scouts show severe start dependence. Desorption avoids CO absorption but does not solve mixing or make all patches informative. Reserve. |
| Open TASEP | Bulk density across alpha=beta<1/2 low-/high-density boundary | Now completed and confirmed at fullN32, with supportive N64; see portfolio above. Finite-N rounding and physical mixing remain relevant. |
| Hamiltonian mean-field rotators | Magnetization across homogeneous/clustered transition | Established and low-N simulation feasible, but ordinary equilibrium transition is continuous and slow quasistationary relaxation complicates the protocol. Does not clearly improve on the current suite. |
| Stochastic spatial Schlögl model | Concentration across chemical bistability/first-order transition | Genuine additional canonical candidate. Well-mixed model has one dynamic species, so it is not itself MTS; coupled spatial compartments require a specified reaction–diffusion model and separate volume/site-count controls. Interesting reserve, not turnkey. |
| Montbrio–Pazo–Roxin QIF network | Population rate/voltage across a saddle-node activity boundary | Recognizable neuroscience bridge but not automatically a canonical symmetry-breaking order parameter. Quiescent cells and divergent raw QIF voltage complicate observations. Not needed to justify the method. |

### Precisely specified CML alternative

Marcq–Chaté–Manneville use the 2D nearest-neighbor synchronous lattice with f(x)=r*x*(1-x). At g=.2, the collective period-1/period-2 boundary is r_c=3.86212(12). With xbar(t) the spatial mean, the published order parameter is Q=<|xbar(2t+1)-xbar(2t)|>. Individual sites remain chaotic on both sides. The critical study uses L32–128 (N1024–16384); its illustrative bifurcation diagram uses L1024. Thus a million-site run is precedent, not a minimum. This is a defensible optional across-boundary example, not evidence of fixed-control switching or small-M SPI recovery. Preserve synchronous updates and unit-step sampling; an even stride can erase the period-two distinction.

## Selection principles

### Focused follow-ups after finite-system confirmation

Keep the primary positive portfolio; stop pursuing the current smallCML, L96 and HR arms as positive exhibits, while retaining concise negative results. Do not homogenize all existing control grids: common counts are not common physical resolution. For a future selected benchmark, roughly21 controls plus a physics-justified denser local bracket is a useful default, not a mandate; use32 independent confirmation seed clusters when precision is needed. Preserve original grids/roles rather than retrospectively recasting them. Existing Rössler21, TASEP21, fineSL19 and confirmedCML17 grids need no cosmetic rerun. Add seeds to narrow uncertainty; add control points to localize a boundary. Both must follow physical time/burn-in convergence, and neither alone creates a sharp finite-system transition.

The highest-value completed no-extraction follow-up is the unchanged N32 TASEP model applied to N64 cached features. All168rows pass, held84rho=.782 [.726,.867]; all source model arrays/sign/scale are identical. This is retrospective empirical size transfer (N and M change together), not fresh confirmation or arbitrary-size invariance. Evidence and reproduction are in the finite-full-observation report and notebook. Additional paired-T work is optional because SL/CML already address variable observation sizes; no blanket T or larger-M bank was launched.

Resource scoping at T1000/full289 p90: matched TASEP median times are18.09min atM32 and60.39min atM64 (168datasets each); N32fresh confirmation median18.98min (672datasets). Pair-count planning, t(M)=60.39min×M(M−1)/(64×63), gives approximately2.47h/M100,9.93h/M200,16.30h/M256,65.31h/M512. These are unvalidated extrapolations, not predicted runtimes or upper bounds. Two measured sizes imply a descriptive exponent1.74 but cannot establish asymptotic scaling. Larger M may affect covariance/model-estimator validity at fixedT, not just cost. Peak single-worker memory near24.4GiB atM64 and some joint-estimator workspaces prevent a trustworthy RAM extrapolation from MPI-array size. Before a scientifically justified large-M bank, run a small representative timing/memory/validity scout; do not treat available SU as sufficient evidence of feasibility. No resource-only farm or reminder is active.

### Additional untried full-observation contenders

| Candidate | Established quantity and boundary | Scope and priority |
|---|---|---|
| Chua circuit, full three-state observation | Single-spiral to double-scroll crisis; characteristic residence/intermittency time has published critical scaling as capacitance/control is varied | Best genuinely new physics-only candidate: recognisable electronic circuit, dynamic signals on both sides. Exact equation/cut and residence classifier still need full-source replication; rare-event horizons and only three undirected MPI entries are real risks. No SPI success assumed. |
| Classic Lorenz system, full three-state observation | Periodic-to-intermittent-chaotic transition near r166.07 for the standard sigma10,b8/3 cut; laminar-duration/coherence diagnostics | Very well-established route to chaos, distinct from Lorenz96. It is not automatically sharper in a finite window; long laminar episodes and periodic-side feature validity need early checks. Second physics-only candidate, not another automatic full farm. |
| Inertial Kuramoto / damped driven pendula | Canonical coherence and hysteretic first-order synchronization in the coupling/inertia plane | Strong previously identified reserve, not newly discovered. Full observation of n rotators is feasible in principle; recording all dynamical states gives2n scalar phase/velocity coordinates. Small-n jump/basin behavior requires validation; redundancy with existing oscillator examples lowers priority. |

Primary sources: [Chua crisis and characteristic-time scaling, PRE52,2268(1995)](https://doi.org/10.1103/PhysRevE.52.2268); [Manneville–Pomeau Lorenz intermittency, PhysicaD1,219–226(1980)](https://doi.org/10.1016/0167-2789(80)90013-5), with [original1979 report](https://doi.org/10.1016/0375-9601(79)90255-X); [Tanaka–Lichtenberg–Oishi inertial transition, PRL78,2104(1997)](https://doi.org/10.1103/PhysRevLett.78.2104) and [finite-size study](https://arxiv.org/abs/1406.3724). These are literature-grounded candidates, not reproduced protocols or positive results. Avoid building an ensemble of independent copies or adding drive/duplicate channels simply to inflate M. Hénon–Heiles escape has a clear energy threshold but escaping/nonstationary records make it a less clean fit; Josephson voltage steps are physically compelling but a suitable nondegenerate small-array observation protocol is not yet established here. Neither outranks the first two for this task.

Sharpness, representational recoverability, and informativeness of a particular small observation are separate questions. No literature result guarantees z recovery from arbitrary M32 subsets. Dispersed sites are a reasonable default for spatially representative coverage; contiguous patches test local sensing. Neither universally wins. A larger N sharpens some asymptotic contrasts while increasing mixing time or reducing the representativeness of a fixed patch.

First-order systems are not automatically cleaner: branch choice, nucleation, hysteresis and measurement horizon can complicate their Q. Conversely, a continuous transition is not a weak benchmark merely because it lacks a jump. For this proof of concept, preserve the existing three distinct physical mechanisms plus Kuramoto; add at most one system only for a specific missing capability, not to maximize count or choose the most flattering result.

## Literature supporting the alternatives

- [Collective logistic-map transition](https://arxiv.org/html/nlin/0605004), Eqs2–4, Fig3 and Table1.
- [Kinetic Ising dynamic order and finite-size scaling](https://arxiv.org/abs/cond-mat/9803127).
- [Blume–Capel phase diagram](https://arxiv.org/abs/1612.02138).
- [Potts information-flow study](https://pubmed.ncbi.nlm.nih.gov/36071118/).
- [Inertial noisy Kuramoto phase diagram and escape times](https://arxiv.org/abs/1309.0035).
- [ZGB-desorption metastability](https://arxiv.org/abs/cond-mat/0506271).
- [Open TASEP density phases and correlation dynamics](https://pmc.ncbi.nlm.nih.gov/articles/PMC5240335/).
- [Original HMF dynamics](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.52.2361).
- [Schlögl original reaction models](https://doi.org/10.1007/BF01379769) and [stochastic/volume caveats](https://pmc.ncbi.nlm.nih.gov/articles/PMC2838355/).
- [Exact QIF macrodynamics](https://arxiv.org/abs/1506.06581).
- [Exact-ring persistence boundary](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.87.052905).
