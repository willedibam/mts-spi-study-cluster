# Introduction rewrite (working draft, 2026-10-20)

Rewrite of [introduction_draft.md](introduction_draft.md) as a funnel introduction. It was revised once after an adversarial referee pass and an independently written example-first alternative. Citation keys reuse [references.bib](references.bib) where an entry exists; new keys are listed in the source table below. Bracketed `[ALIGN]` items must be matched to frozen results before submission. Framing constraints come from [premise-and-character-axes.md](../docs/context/premise-and-character-axes.md).

**Title candidates.** The context file advises against "mechanistic similarity", because SPI–SPI does not identify mechanism.
- *How, not how strongly: the organisation of dependence in multivariate time series*
- *Beyond coupling strength: representing how dependence is organised across a system*

---

## Introduction

Many of the properties that distinguish complex systems are properties of how their parts interact. Cognition, a climate teleconnection and systemic financial risk are not attributes of a single cortical region, ocean basin or asset [@bassett2017network; @tsonis2004architecture; @mantegna1999hierarchical]. Multivariate time series (MTS), simultaneous recordings of many interacting components, are the main empirical access to those interactions. Many scientific questions about such systems are therefore comparisons of MTS: a young brain against an ageing one, a patient against a control, a system before and after an intervention, or two systems that may share an organising principle. Part of what distinguishes them lies in individual channels, but much of it lies in the dependence among channels [@friston2011functional; @runge2019inferring].

The prevailing way to make dependence comparable is to summarise it. A pairwise statistic, such as a correlation, coherence, Granger-causal or transfer-entropy estimate, or phase-locking index, is computed for every pair of channels. The resulting matrix of coupling strengths then stands in for the recording in all downstream analysis [@pereda2005nonlinear; @bastos2016tutorial]. When channels correspond across recordings, these matrices are compared entry by entry or through their geometry, as in connectome fingerprinting [@finn2015functional], dynamic functional connectivity [@hutchison2013dynamic] and Riemannian classification [@barachant2012multiclass; @dadi2019benchmarking]. When channels do not correspond, the matrix is abstracted further into mean coupling, edge-weight distributions or network topology [@bullmore2009complex; @rubinov2010complex], which brings its own sensitivities to network size and density [@vanwijk2010comparing; @tantardini2019comparing]. Multilayer approaches keep several such matrices, typically one per frequency band, as the layers of one network [@brookes2016multilayer; @buldu2018frequency]. Machine learning takes one of three routes:
- it infers an adjacency of interaction strengths [@wu2020connecting; @jin2024gnn4ts];
- it discards cross-channel dependence by treating channels independently [@nie2023patchtst; @han2024capacity];
- or it learns dependence end to end, in coordinates that have no stated meaning [@liu2024itransformer; @woo2024moirai].

Across these traditions the primitive is the same: how strongly pairs of components are coupled, under one or a few chosen notions of coupling. Each further abstraction trades information for comparability, and none can restore what the first summary removed, because any function of two identical strength matrices is identical.

Hundreds of pairwise statistics exist because dependence has many facets. It can be linear or nonlinear, instantaneous or lagged, directed or symmetric, carried by phase or by amplitude, and confined to one frequency band or spread across several. Choosing a statistic fixes, before any data are seen, the sense in which components are taken to interact; the matrix then reports how strongly they interact in that sense. Pearson correlation measures linear, zero-lag association. The phase-lag index discards zero-phase coupling by design, to suppress volume conduction [@nolte2004identifying; @stam2007phase]. Transfer entropy depends on the history length it is given [@schreiber2000measuring; @wibral2013measuring]. Systematic comparisons find that conclusions about network organisation, structure–function correspondence, individual identification and behavioural prediction can change substantially with the statistic [@smith2011network; @wang2014systematic; @liu2025benchmarking]. The *pyspi* library extended the highly comparative approach of univariate time-series analysis [@fulcher2013highly; @fulcher2017hctsa] to pairwise dependence. Its more than two hundred statistics of pairwise interactions (SPIs) make the choice of statistic empirical rather than conventional [@cliff2023unifying], and across tasks the best and worst statistics differ widely in performance [@cliff2023unifying; @bryant2024extracting]. `[ALIGN: quote the published spread; the "10% null vs 99%" anecdote is not a published figure.]`

Choosing well does not change what is represented. A pairwise statistic compresses the relationship between two time series, a joint process with form, timing and spectral content, into one number. That map is many-to-one: relationships of very different form can share a correlation coefficient [@anscombe1973graphs], and correlation summarises dependence completely only for elliptical distributions [@embrechts2002correlation]. Consider two recordings from the same sensors whose zero-lag correlation matrices are identical entry for entry (Fig. 1). In one, components move together without delay. In the other, they follow one another at delays that differ from pair to pair. Every analysis built on that matrix, whether entry-wise, spectral or topological, returns the same answer for both. A change in *how* components interact is therefore not invisible to a strength representation; it is aliased. A fixed statistic registers at most a change in its value, which cannot be distinguished from a change in coupling strength. A better statistic changes which differences are aliased, not whether some are. Such differences occur in real systems. Between wakefulness and slow-wave sleep, networks defined by zero-lag correlation are largely preserved, while the lag structure of propagated activity reorganises [@mitra2015propagated]. Axonal conduction delays vary systematically across the human cortex [@lemarechal2022atlas] and continue to shorten into adulthood [@vanblooijs2023developmental], so any process that alters conduction unevenly changes how delays are distributed across pairs. Synchronisation transitions reorganise the observed dependence between components while the coupling law itself stays fixed [@arenas2008synchronization]. Statements that interactions became more variably delayed, or changed from intermittent to locked, map onto experiments more directly than a list of edges that strengthened or weakened. Neither kind of statement identifies mechanism on its own [@mehler2018lure; @reid2019advancing].

The character of dependence has not been ignored, but it has been interrogated one hypothesis at a time.
- *Nonlinearity.* Surrogate-data tests ask whether a dependence exceeds what a linear Gaussian process could produce [@theiler1992testing; @prichard1994generating; @schreiber2000surrogate]. Comparisons of linear with nonlinear redundancies [@palus1995testing], or of mutual information with its Gaussian counterpart, quantify nonlinear interdependence in electroencephalography [@breakspear2002detection] and functional magnetic resonance imaging [@hlinka2011functional]. Contrasts between linear and nonlinear dependence networks have been used to characterise climate dynamics and financial crises [@donges2009complex; @haluszczynski2017linear].
- *Timing.* Cross-correlograms [@perkel1967neuronal], lag-structure analyses [@mitra2014lag] and delay-resolved transfer entropy [@wibral2013measuring] target delays.
- *Phase versus amplitude.* Phase- and amplitude-coupling patterns have been contrasted directly [@engel2013intrinsic; @siems2020dissociated; @mostame2020phase].
- *Redundancy versus synergy.* Information decomposition separates redundant from synergistic interaction [@luppi2022synergistic].

Several of these would detect the difference in Fig. 1, but only if the analyst already suspected timing. Agreement between statistics has itself been studied in two ways. Analytically, it appears as fixed relations that hold under Gaussianity and whose violation signals non-Gaussian coupling [@nolte2020mathematical]. Empirically, it has been measured and averaged over recordings to characterise the statistics [@cliff2023unifying; @liu2025benchmarking], not to describe an individual recording. A vector of per-statistic summaries would also separate the two recordings of Fig. 1, but only as apparently stronger lag-tolerant coupling. Any recording-wide change in gain or signal-to-noise ratio would move all of its entries together. What such summaries discard is which channel pairs score jointly high or low across statistics. That joint organisation is where the two recordings differ, and to our knowledge it has not been used to represent a recording.

Fig. 1 shows where to look. For every channel pair, compute the zero-lag correlation and a lag-tolerant similarity, and treat each statistic's values over all pairs as its *edge profile*. In the synchronous recording the two profiles agree. In the delayed recording they diverge, and more so as delays vary more across pairs. The relationship between the profiles identifies the difference as one of timing. We generalise this by inverting the role of the statistic library: rather than selecting the best statistic, we measure how statistics relate within a recording. For SPIs $a$ and $b$ computed on an $M$-channel recording, $z_{ab}$ is the correlation between their edge profiles across the recording's $M(M-1)$ channel pairs. The vector of all $z_{ab}$ is the SPI–SPI representation. A coordinate $z_{ab}$ is high when two statistics order channel pairs alike. It falls when the property that distinguishes them varies across pairs independently of shared strength. Pearson and Spearman correlation, for example, separate when monotone nonlinearity varies across pairs. The representation therefore targets how the character of dependence is organised across a system; a change applied uniformly to every pair registers only weakly. Because $z_{ab}$ is unchanged by positive affine rescaling of either profile, recording-wide shifts in a statistic's level or spread do not register as change. Coupling strength can still affect $z$ through the shape of the profiles. For linear Gaussian data, many coupling measures are fixed functions of coherency [@nolte2020mathematical], so the spread of strength across pairs alone determines part of $z$. We therefore test each reading against multivariate linear surrogates that preserve that spread [@prichard1994generating]. `[ALIGN: required analysis, not yet run.]` The construction also requires heterogeneity across pairs in both profiles, and it becomes ill-conditioned when a statistic saturates. Each coordinate is indexed by two statistics with documented assumptions, which proposes a candidate interpretation. As with any multivariate readout [@haufe2014interpretation], we accept that interpretation only after perturbation, surrogate and transfer tests. Comparable second-order constructions compare neural representations [@kriegeskorte2008representational; @kornblith2019similarity] and the layers of multiplex networks [@nicosia2015measuring; @dedomenico2015structural]. Applied between structural and functional connectivity, the same across-edge correlation tracks brain development [@baum2020development]. Here the layers are alternative notions of dependence evaluated on one system. The representation's dimension does not depend on the number of channels or samples, so recordings of different sizes share a coordinate system. Their positions are comparable when sampling is matched to each system's timescales.

`[ALIGN: the demonstrations below are current repository results, worded at their verified strength. The baseline sentence presupposes comparisons that must exist before submission.]`
- *Generators.* Where character is set by construction, SPI–SPI coordinates track nonlinearity and lag organisation that strength summaries leave ambiguous.
- *Electronic oscillators.* In a network of 28 coupled oscillators, the representation separates recordings at matched mean correlation by the organisation of their lag and spectral profiles, an empirical counterpart of Fig. 1. At the locking transition, mean correlation rises only modestly. Calibration traces the change in the leading coordinate to the cessation of phase slips in a subset of channels, and the frozen interpretation transfers to a second wiring topology.
- *Intracranial recordings.* A lag-profile coordinate fixed on a five-participant pilot replicates across 19 further participants. We show that it follows 8–24 Hz spectral content, and we explain why lag profiles reorganise with it.

Each reading is compared with channel spectra, band-resolved coherence and per-statistic summaries. It is calibrated with perturbations and surrogates rather than read from loadings. SPI–SPI introduces no new estimator; it draws on the field's accumulated methodology. We close by outlining targeted measures that would place each channel pair on named axes of character rather than of strength. Examples are linear to nonlinear, fixed to variable lag, and phase- to amplitude-mediated.

---

## Sources and why each is needed

Status key: **V** = verified this session (title/venue/year/DOI); **R** = already verified in repository context or `references.bib`; **M** = from memory with high confidence, DOI to check before submission.

| Key | Reference | Role in the argument | Status |
|---|---|---|---|
| bassett2017network | Bassett & Sporns, *Nat. Neurosci.* 20:353 (2017) | Function arises from interactions (brain) | R |
| tsonis2004architecture | Tsonis & Roebber, *Physica A* 333:497 (2004), 10.1016/j.physa.2003.10.045 | Same, climate | M |
| mantegna1999hierarchical | Mantegna, *Eur. Phys. J. B* 11:193 (1999), 10.1007/s100510050929 | Same, finance | M |
| friston2011functional | Friston, *Brain Connect.* 1:13 (2011), 10.1089/brain.2011.0008 | Dependence as the object of comparison | M |
| runge2019inferring | Runge et al., *Nat. Commun.* 10:2553 (2019), 10.1038/s41467-019-10105-3 | Cross-domain (Earth system) dependence inference | M |
| pereda2005nonlinear | Pereda, Quian Quiroga & Bhattacharya, *Prog. Neurobiol.* 77:1 (2005), 10.1016/j.pneurobio.2005.10.003 | Diversity of coupling measures | M |
| bastos2016tutorial | Bastos & Schoffelen, *Front. Syst. Neurosci.* 9:175 (2016), 10.3389/fnsys.2015.00175 | Measures and interpretational pitfalls | M |
| finn2015functional | Finn et al., *Nat. Neurosci.* 18:1664 (2015), 10.1038/nn.4135 | Entry-wise matrix comparison | M |
| hutchison2013dynamic | Hutchison et al., *NeuroImage* 80:360 (2013), 10.1016/j.neuroimage.2013.05.079 | Windowed matrix trajectories | M |
| barachant2012multiclass | Barachant et al., *IEEE TBME* 59:920 (2012), 10.1109/TBME.2011.2172210 | Covariance geometry | M |
| dadi2019benchmarking | Dadi et al., *NeuroImage* 192:115 (2019), 10.1016/j.neuroimage.2019.02.062 | Tangent-space connectome benchmark | M |
| bullmore2009complex | Bullmore & Sporns, *Nat. Rev. Neurosci.* 10:186 (2009), 10.1038/nrn2575 | Topological abstraction | M |
| rubinov2010complex | Rubinov & Sporns, *NeuroImage* 52:1059 (2010), 10.1016/j.neuroimage.2009.10.003 | Graph summaries | M |
| vanwijk2010comparing | van Wijk, Stam & Daffertshofer, *PLoS ONE* 5:e13701 (2010), 10.1371/journal.pone.0013701 | Size/density confounds | V |
| tantardini2019comparing | Tantardini et al., *Sci. Rep.* 9:17557 (2019), 10.1038/s41598-019-53708-y | Network comparison with/without node correspondence | V |
| brookes2016multilayer | Brookes et al., *NeuroImage* 132:425 (2016), 10.1016/j.neuroimage.2016.02.045 | Several coupling matrices as layers (pre-empts "only one notion") | V |
| buldu2018frequency | Buldú & Porter, *Netw. Neurosci.* 2:418 (2018), 10.1162/netn_a_00033 | Same | V |
| wu2020connecting | Wu et al., KDD 2020 (MTGNN), 10.1145/3394486.3403118 | Learned adjacency of strengths | M |
| jin2024gnn4ts | Jin et al., GNNs for time series survey (2024) | Same | R |
| nie2023patchtst | Nie et al., ICLR 2023 (PatchTST) | Channel independence | R |
| han2024capacity | Han, Ye & Zhan, *IEEE TKDE* 36:7129 (2024), 10.1109/TKDE.2024.3400008 | Channel-independent vs dependent trade-off | V |
| liu2024itransformer | Liu et al., ICLR 2024 (iTransformer) | End-to-end cross-variate dependence | V |
| woo2024moirai | Woo et al., ICML 2024, PMLR 235 (Moirai) | Any-variate encoder: fixed dimension without named axes | V |
| nolte2004identifying | Nolte et al., *Clin. Neurophysiol.* 115:2292 (2004), 10.1016/j.clinph.2004.04.029 | Measure with an explicit prior | M |
| stam2007phase | Stam, Nolte & Daffertshofer, *Hum. Brain Mapp.* 28:1178 (2007), 10.1002/hbm.20346 | Same (PLI) | M |
| schreiber2000measuring | Schreiber, *PRL* 85:461 (2000) | Transfer entropy | R |
| wibral2013measuring | Wibral et al., *PLoS ONE* 8:e55809 (2013) | History/delay dependence of TE | R |
| smith2011network | Smith et al., *NeuroImage* 54:875 (2011), 10.1016/j.neuroimage.2010.08.063 | Statistic changes conclusions | M |
| wang2014systematic | Wang et al., *Front. Neurosci.* 8:405 (2014), 10.3389/fnins.2014.00405 | 42 measures benchmarked | V |
| liu2025benchmarking | Liu, Luppi, …, Misic, *Nat. Methods* (2025), 10.1038/s41592-025-02704-4 | 239 statistics; inter-statistic agreement averaged (precedent) | V |
| cliff2023unifying | Cliff et al., *Nat. Comput. Sci.* 3:883 (2023) | pyspi; inter-SPI agreement averaged (precedent) | R |
| bryant2024extracting | Bryant et al., *PLoS Comput. Biol.* 20:e1012692 (2024) | Systematic comparison of representations | R |
| fulcher2013highly | Fulcher, Little & Jones, *J. R. Soc. Interface* 10:20130048 (2013) | Highly comparative paradigm | V (DOI not checked) |
| fulcher2017hctsa | Fulcher & Jones, *Cell Syst.* 5:527 (2017) | Same | R |
| anscombe1973graphs | Anscombe, *Am. Stat.* 27:17 (1973), 10.1080/00031305.1973.10478966 | Many-to-one scalar summaries | M |
| embrechts2002correlation | Embrechts, McNeil & Straumann, in *Risk Management* (CUP, 2002) 176–223, 10.1017/CBO9780511615337.008 | Correlation is a complete summary only for elliptical laws | V |
| mitra2015propagated | Mitra et al., *eLife* 4:e10781 (2015), 10.7554/eLife.10781 | Zero-lag networks preserved while lag structure reorganises (wake vs slow-wave sleep): the cleanest empirical case of character changing at preserved strength structure | V |
| lemarechal2022atlas | Lemaréchal et al., *Brain* 145:1653 (2022), 10.1093/brain/awab362 | Delays vary across cortex (heterogeneity across pairs) | V |
| vanblooijs2023developmental | van Blooijs et al., *Nat. Neurosci.* 26:537 (2023), 10.1038/s41593-023-01272-0 | Transmission speed develops into adulthood; also the source of the repository's CCEP data | V |
| arenas2008synchronization | Arenas et al., *Phys. Rep.* 469:93 (2008), 10.1016/j.physrep.2008.09.002 | Reorganisation at a fixed coupling law | M |
| mehler2018lure | Mehler & Kording, arXiv:1812.03363 | Strength maps do not license mechanism | V |
| reid2019advancing | Reid et al., *Nat. Neurosci.* 22:1751 (2019), 10.1038/s41593-019-0510-4 | Same | M |
| theiler1992testing | Theiler et al., *Physica D* 58:77 (1992) | Surrogate nonlinearity tests | R |
| prichard1994generating | Prichard & Theiler, *PRL* 73:951 (1994) | Multivariate surrogates | R |
| schreiber2000surrogate | Schreiber & Schmitz, *Physica D* 142:346 (2000), 10.1016/S0167-2789(00)00043-9 | Surrogate review | M |
| palus1995testing | Paluš, *Physica D* 80:186 (1995) | Linear vs nonlinear redundancies | R |
| breakspear2002detection | Breakspear & Terry, *Clin. Neurophysiol.* 113:735 (2002) | Nonlinear interdependence in EEG | V (DOI not checked) |
| hlinka2011functional | Hlinka et al., *NeuroImage* 54:2218 (2011), 10.1016/j.neuroimage.2010.08.042 | Linear correlation captures most fMRI dependence (counterweight) | V |
| donges2009complex | Donges et al., *Eur. Phys. J. Spec. Top.* 174:157 (2009) | Linear vs nonlinear climate networks (two-measure contrast precedent) | V (DOI not checked) |
| haluszczynski2017linear | Haluszczynski et al., *PRE* 96:062315 (2017), 10.1103/PhysRevE.96.062315 | Linear vs nonlinear market correlations across crises (two-measure state descriptor precedent) | V |
| nolte2020mathematical | Nolte et al., *Front. Neurosci.* 14:577574 (2020), 10.3389/fnins.2020.577574 | Under Gaussianity, measures are functions of coherency; departures signal non-Gaussianity. **The most dangerous precedent** and the source of the required null | V |
| perkel1967neuronal | Perkel, Gerstein & Moore, *Biophys. J.* 7:419 (1967) | Cross-correlogram: the transcript's bespoke example | M |
| mitra2014lag | Mitra et al., *J. Neurophysiol.* 111:2374 (2014), 10.1152/jn.00804.2013 | Lag structure changes with state | V |
| engel2013intrinsic | Engel et al., *Neuron* 80:867 (2013) | Phase vs envelope coupling modes | V (DOI not checked) |
| siems2020dissociated | Siems & Siegel, *NeuroImage* 209:116538 (2020) | Phase/amplitude dissociation | R |
| mostame2020phase | Mostame & Sadaghiani, *NeuroImage* 219:117051 (2020) | Phase/amplitude patterns compared directly | V |
| luppi2022synergistic | Luppi et al., *Nat. Neurosci.* 25:771 (2022), 10.1038/s41593-022-01070-0 | A non-strength axis of interaction type | V |
| haufe2014interpretation | Haufe et al., *NeuroImage* 87:96 (2014), 10.1016/j.neuroimage.2013.10.067 | Loadings are not explanations | V |
| kriegeskorte2008representational | Kriegeskorte, Mur & Bandettini, *Front. Syst. Neurosci.* 2:4 (2008) | Second-order similarity analogue | R |
| kornblith2019similarity | Kornblith et al., ICML 2019 (CKA) | Same, ML | R |
| nicosia2015measuring | Nicosia & Latora, *PRE* 92:032805 (2015), 10.1103/PhysRevE.92.032805 | Inter-layer correlations: closest formal analogue | V |
| dedomenico2015structural | De Domenico et al., *Nat. Commun.* 6:6864 (2015) | Layer redundancy | R |
| baum2020development | Baum et al., *PNAS* 117:771 (2020), 10.1073/pnas.1912034117 | Same across-edge correlation operation, between modalities: a reviewer will raise it | V (pages from memory) |

**Removed after review:**
- **Reshef 2011 and Kinney & Atwal 2014.** Equitability is contested: mutual information is self-equitable, and the impossibility result concerns R²-equitability. The aliasing argument needs only the many-to-one map.
- **Price et al. 2017.** It reports *uniform* evoked slowing, which z does not target, and the delays are not interregional conduction delays.

**Held back for Methods or Discussion:**
- Granger 1969 and Kraskov 2004.
- Condition-dependent character, as related-work or Discussion support. All were verified by the precedent search:
  - Mitra et al. 2016, *PNAS*: hippocampal–cortical direction reverses in sleep.
  - Lee et al. 2013, *Anesthesiology*: anaesthetics selectively inhibit frontal-to-parietal feedback.
  - Stankovski et al. 2016, *Phil. Trans. A*: anaesthesia alters coupling functions.
  - Andrzejak et al. 2012, *PRE*: nonlinear dependence is greater in focal iEEG.
  - Motlaghian et al. 2022, *HBM*: nonlinear FNC differs in schizophrenia.
  - Gatica et al. 2021, *Brain Connect.*: redundancy dominance increases with age.
- Earlier multi-measure comparisons:
  - Dauwels et al. 2010, *NeuroImage*: about 30 synchrony measures in Alzheimer's disease.
  - Quian Quiroga et al. 2002, *PRE*.
  - Hlinka et al. 2014, *Clim. Dyn.*
- James, Barnett & Crutchfield 2016, *PRL*: transfer entropy conflates pairwise and synergistic dependence.
- Battiston et al. 2020 on higher-order interactions (SPI–SPI is restricted to pairs).
- Nguyen et al. 2025, *PRR*.
- Power and source-mixing confounds: Muthukumaraswamy & Singh 2011; Colclough et al. 2015/2016.
- Any-variate EEG encoders: BIOT (NeurIPS 2023) and LaBraM (ICLR 2024), unverified this session.
- DeepSets (Zaheer et al. 2017).

---

## Framing decisions

**1. Aliased, not blind.**
- The transcript claims strength representations could *never* see a change in character. That is false: Pearson does respond to nonlinear coupling, but it registers the change as lower strength.
- The defensible and sharper claim is non-identifiability: under one statistic, character is confounded with strength. The entry-wise matched pair of Fig. 1 makes this a construction rather than an assertion.

**2. Organisation, not uniform change.**
- The referee's most important point: z is a correlation *across channel pairs*, so it responds to character that varies across pairs or co-varies with strength. A uniform change, such as every coupling slowing by the same amount, mostly does not register.
- Motivating examples therefore have to concern heterogeneity: delays that differ from pair to pair, or a subset of channels that stops slipping. Uniform age-related slowing does not qualify.
- This makes the title's "organisation" accurate rather than decorative.

**3. Why z rather than many statistics.**
- Aliasing within one statistic argues only for using many statistics, and the per-statistic summary vector already does that. Your record shows those summaries recover character targets under supervision.
- The case for z is the joint argument in paragraph 5:
  - per-statistic summaries discard which pairs score jointly;
  - they misattribute timing to strength;
  - they move together under recording-wide nuisances.
- Without this step, a referee will call z unmotivated.

**4. The historical arc is a ladder of abstraction, not a chronology.**
- Publication history does not match the proposed sequence (quantitative representation → comparison of strength measures → beyond covariance → topology → character):
  - Granger causality dates from 1969 and transfer entropy from 2000.
  - Nonlinear interdependence was reviewed in 2005 [@pereda2005nonlinear], before graph theory became standard (around 2009).
  - Systematic benchmarking of measures postdates topology (2011–2025).
- The rewrite therefore orders the work by what each representation retains. "The next phase" survives as the next level of representation, not the next era.

**5. "Bakes in a how" in one precise sentence.** "Choosing a statistic fixes, before any data are seen, the sense in which components are taken to interact." This keeps the transcript's insight without the false claim that covariance *assumes* linearity.

**6. Lead with timing.**
- The repository's real-data evidence is lag- and spectral-profile organisation. Monotone nonlinearity is not established: hardware Pearson–Spearman z≈1 throughout.
- Hlinka et al. found that linear correlation captures most fMRI dependence.
- Keep nonlinearity for the generator pedagogy.

**7. The spectral objection is the main risk.**
- For a narrowband signal, lagged correlation ≈ zero-lag correlation × cos(2πfτ). At τ = 31.25 ms the sign flips near 16 Hz, so the CCEP coordinate largely indexes 8–24 Hz power (ρ = −.968).
- A referee will call SPI–SPI a roundabout multi-band coherence descriptor.
- The draft pre-empts this by naming spectral content and promising baselines: channel spectra, band-resolved coherence and per-statistic summaries. Those comparisons must exist before submission.

**8. Interpretability is a procedure, not a property.** "Interpretable by construction" was removed. Coordinates *propose* interpretations, and perturbation, surrogate and transfer tests accept them. This matches repository practice: the PLV×envelope z = .992 was reproduced by fixed harmonic waveforms, so a large loading alone was misleading.

**9. Comparability across M and T is a consequence, and conditional.** The intro adds the caveat that positions are comparable only under sampling matched to timescales. Fixed lags mean different things at different oscillation frequencies, and reference choice moved one CCEP participant from −.789 to +.188.

**10. Transcript errors to fix before reuse.**
- **The Δt example is inverted.** The faster connection must have the shorter delay.
- **"Degraded and noisy, so more linear" confuses an estimator effect with coupling.** Noise makes nonlinearity undetectable; it does not make the coupling linear.
- **Kuramoto-type transitions do not "rely on a change in the coupling".** Strength crosses a threshold under a fixed coupling law.
- **The "spaghetti" IQ example is hypothetical.**
- **"First general-purpose method" is not established.**

## Points to formalise next (highest value first)

1. **Build Fig. 1.** The introduction now rests on it.
   - *Proposed construction (by the alternative-framing agent; not yet built or checked).* Start from a jointly Gaussian process with separable cross-covariance $\Sigma^*_{jk}\rho(h)$ and shift each channel by its own delay $\tau_j$. Its zero-lag correlation matrix becomes $\Sigma^* \circ R$, with $R_{jk} = \rho(\tau_j - \tau_k)$. Setting $\Sigma^* = \Sigma \oslash R$ therefore reproduces a target $\Sigma$ exactly, provided the result is positive semidefinite, which is plausible for modest lags.
   - *Contrast.* The synchronous recording uses $\Sigma$ directly.
   - *Readout.* The lag-tolerant profile is $\Sigma \oslash R$ in the delayed recording, so z(zero-lag, lag-tolerant) drops below 1 as the $\tau_j$ spread.
   - *Status.* The hardware matched-mean contrast is only the empirical analogue, matched on mean rather than entry for entry. Without this construction, paragraph 4 is rhetoric.
2. **A linear-Gaussian null for z: the most important new analysis, and potentially the formal definition of character.**
   - *Why it is needed.* Nolte et al. 2020 show that, under Gaussianity, coupling measures are fixed functions of coherency. Part of every z_ab is therefore predicted by how coupling strength is spread across pairs, with no non-Gaussian character at all.
   - *What to compute.* Generate multivariate phase-randomised (Prichard–Theiler) surrogates that preserve auto- and cross-spectra, and report z − E[z | linear-Gaussian surrogate].
   - *What it does for the argument.* This turns "character" from a gloss into a residual. It answers the strength-bleed objection, and it ties the method to an established null rather than to a new one.
   - *Caveat.* Linear lag and spectral organisation, which the current real-data results show, survives phase randomisation. That organisation is captured by the surrogate expectation, not the residual. The surrogate therefore separates *linear* lag/spectral organisation from *non-Gaussian* character. The paper needs to say which of the two each finding is.
3. **Operational definitions.**
   - *Strength:* the level and scale of a statistic's edge profile.
   - *Character organisation:* structure in the joint behaviour, across edges, of statistics with different assumptions.
   - State that the two are not orthogonal; at weak coupling, character is undefined in practice.
4. **An invariance proposition.**
   - *Invariant to:* positive affine maps of each profile, and common permutations of edges (channel relabelling).
   - *Dimension:* independent of M and T.
   - *Not invariant to:* physical coupling strength in general, nor to uniform character changes beyond curvature effects.
5. **Domain of validity.** z requires cross-edge heterogeneity in both profiles and is ill-conditioned near saturation. The intro now states this in one clause; Methods should quantify it.
6. **The baseline triad.** Channel spectra, band-resolved coherence and the per-statistic mean vector, run on the oscillator and CCEP data. Without these, the demonstrations paragraph overclaims.

## Alternative opening (example-first)

An independently written alternative opens with the Fig. 1 pair instead of the broad funnel. Its first paragraph:

> Consider two recordings from the same sensors whose matrices of zero-lag correlation are identical, entry for entry. In the first, every pair of components moves together without delay. In the second, the same correlation values arise because components follow one another at delays that differ from pair to pair. Any analysis computed from the correlation matrix alone, whether it compares entries, eigenstructure or network topology, returns the same answer for both, although the two systems interact in different ways.

**Gains.** The problem is shown within four sentences, and the example threads through the whole introduction.

**Losses.** The cross-domain motivation and the ladder of abstraction go. The paper reads narrower, and the entire opening depends on Fig. 1 existing.

**Recommendation.** Keep the funnel for Nature Computational Science, with the example at its pivot (paragraph 4), as above. The example-first opening suits a NeurIPS/ICLR version, where reviewers reward a crisp failure case early.
