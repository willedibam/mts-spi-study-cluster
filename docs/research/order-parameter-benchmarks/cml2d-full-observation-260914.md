# Small-lattice full-observation CML diagnostic

## Question and fixed design

User-requested diagnostic: does an unsupervised SPI–SPI coordinate track the finite-system period-alternation observable when every lattice site is observed? This changes the physical system size, not merely the observation layout. A failure at L=6 or L=8 does not invalidate the existing L=256 dispersed-observation result. The thermodynamic transition at r=3.86212(12) is only a reference marker for these small lattices, not their assumed transition location.

Use the unchanged synchronous periodic-square logistic update, g=0.2, all 17 controls from the completed large-system confirmation, 200,000 burn steps, 2,000 recorded observation steps followed by a disjoint 1,000,000-step Q reference. Two arms: L=6, M=N=36 and L=8, M=N=64; primary T=1000. All sites are recorded once in fixed row-major order, without noise injection, temporal decimation, sensor selection or preprocessing changes. All 289 p90 SPIs are attempted. No extra M,T grid completion, longer-reference or finer-control experiment is launched under this diagnostic.

Eight development seeds 260914001–008 and 32 evaluation seeds 260914101–132 at every control and size give 680 recordings per arm, 1,360 total. Controls are paired within seed; uncertainty resamples seed clusters, not sites or controls. Seeds do not select the best-looking attractor. Fixed figure examples use the first evaluation seed at r=3.84, 3.86212 and 3.89 and the first 100 input steps. Lattice images are the saved final fields, explicitly later than the illustrated input.

## Separate inference and transfer questions

Primary: independently fit the existing target-blind hygiene and PC1 procedure within each L using development seeds only, then freeze it for evaluation. Keep the same minimum feature validity 0.99, variance threshold 0.05, row missingness limit 0.05 and one-coordinate geometry gates (EVR >=0.2, PC1/PC2 >=1.5, leave-seed loading cosine >=0.8). A display-only sign may use development Q as in the existing analysis; no target/control selects the component or representation. Report failed geometry or row gates explicitly, without searching alternative components. This is a prespecified held-out diagnostic, not a retroactive claim of thermodynamic criticality.

Secondary: apply the original L=256 frozen model unchanged to the same recordings. A transfer failure is not a failure of the arm-specific inference test. Never renormalise the transferred q within an arm. Independent fits have distinct coordinate units, and are not claims of the same coordinate across physical sizes.

For each analysis, at most 10% of evaluation rows may exceed selected-feature missingness, with at least 24 of 32 evaluation rows retained per control. For the new fit also report development coverage, requiring at least six of eight per control. Seal geometry and eligibility before endpoint reporting. Report the planned/retained denominators even if raw constancy or extraction fails; no fabricated or silently dropped cases. Do not plot a successful q tracking result if its gates fail; show the failure and the physical curve instead.

Endpoints: held-out q–Q_reference Spearman with 2,000 seed-bootstrap resamples, control-mean association, sampled steepest-slope interval (descriptive, not a critical-point estimator), matched-window Q and within-control association as secondary diagnostics. Retain mean-absolute-correlation and same-window sampled Q baselines. For full observation, sampled Q is exactly Q_window, so it is a transparent direct baseline, not a competing hidden-state estimator. Inspect Q blocks, half-reference differences, input channel variance and redundancy. A finite-N absolute-difference floor or small-system attractor change must not be labelled thermodynamic ordering.

## Predictions recorded before simulation

Expect more rounding, possible displacement from the large-lattice transition and stronger initial-condition/attractor effects. The ordering between L=6 and L=8 is not assumed monotonic. Full observation removes spatial subsampling error but does not restore large-system physics. It may improve recovery of each small system's own Q, while increased channel redundancy or periodic behaviour may reduce SPI feature validity or make the large-system direction uninformative. No numerical correlation or clean transition is predicted.

## Execution and evidence

Use an isolated pinned source branch, shared-source compute workers, immutable inputs, and result corpora on gdata because Scratch inode headroom is limited. Run a two-record p90 smoke (one per size), then a representative node batch before selecting homogeneous many-core production farms from measured runtime and memory. Use finite PBS dependencies, not a recurring reminder. Existing large-system data and figures remain unchanged. Results and job identifiers will be recorded here; notebook subsections belong under the two-dimensional CML section.
