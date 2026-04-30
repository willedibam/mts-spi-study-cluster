# Handoff: CML sweep animation in SPI-SPI PCA space

Build two animations (α-sweep and ε-sweep) of CML parameter trajectories in a 2D (or 3D) PCA space defined by CML anchor regimes. Feature = SPI-SPI correlation vector per MTS.

## Inputs

Both use [configs/pyspi-v2/benchmarked_config.yaml](configs/pyspi-v2/benchmarked_config.yaml) (292 SPIs) at M=10, T=1000. Same SPI config → feature columns align.

| Dataset | Path | Count | Notes |
|---|---|---|---|
| **Sweep** | `data/embeddings/cml_param_sweep_260423/` | 2,440 | `cml-alpha-sweep/` (61α × 20 inst, ε=0.3), `cml-eps-sweep/` (61ε × 20 inst, α=1.75). Variant slugs: `a1p40`…`a2p00`, `e0p00`…`e0p60`. Config: [configs/generate/embeddings/cml-param-sweep.yaml](configs/generate/embeddings/cml-param-sweep.yaml). |
| **Anchors** | `data/embeddings/proof_benchmarked_260424/` | 1,000 | 20 classes × 50 instances. Config: [configs/generate/embeddings/proof-benchmarked.yaml](configs/generate/embeddings/proof-benchmarked.yaml). Only the 10 CML classes enter PCA + animation. |

**CML classes** (filter to these for PCA + display):
`brownian-defect, chaotic-traveling-wave, defect-turbulence, fdstc, frozen-chaos, pattern-selection, sti-i, sti-ii, traveling-wave, traveling-wave-kaneko67`

Non-CML (5 VAR + 2 Kuramoto + 2 noise + 1 wave-1d) are in the anchor npz for *other* analyses — do not project or display in this animation.

## Pipeline

### 1. Feature extraction (laptop, after both datasets present)
```bash
python -m src.process_features \
  --data-path data/embeddings/proof_benchmarked_260424 \
  --metric pearson,spearman --var-threshold 1e-6 --output-dir features/

python -m src.process_features \
  --data-path data/embeddings/cml_param_sweep_260423 \
  --metric pearson,spearman --var-threshold 1e-6 --output-dir features/
```
Produces 4 npz under `features/`. See [src/process_features.py](src/process_features.py) `cache_path()` for exact names. **Gate**: `np.array_equal(anchor['pairs'], sweep['pairs'])` must be True before proceeding. If False, SPI-order drift upstream — diagnose, don't paper over.

### 2. PCA fit (CML-only anchors)
```python
CML = {"brownian-defect","chaotic-traveling-wave","defect-turbulence","fdstc",
       "frozen-chaos","pattern-selection","sti-i","sti-ii",
       "traveling-wave","traveling-wave-kaneko67"}
cml_mask = np.isin(anchor['y'], list(CML))       # 500 samples
scaler = StandardScaler().fit(anchor['X'][cml_mask])
pca = PCA(n_components=2, random_state=0).fit(scaler.transform(anchor['X'][cml_mask]))

Z_anchor = pca.transform(scaler.transform(anchor['X'][cml_mask]))
Z_sweep  = pca.transform(scaler.transform(sweep['X']))
print(pca.explained_variance_ratio_.cumsum())  # info only, not a gate
```
If PC1+PC2 variance looks too low (<~25%), refit with `n_components=3` and produce 3D animations. User is fine either way.

### 3. Sweep indexing
```python
alpha_mask = sweep['y'] == "cml-alpha-sweep"
eps_mask   = sweep['y'] == "cml-eps-sweep"

def parse(slug):  # "a1p40" -> 1.40, "e0p30" -> 0.30
    return float(slug[1:].replace('p', '.'))

alpha_vals = np.array([parse(v) for v in sweep['variant'][alpha_mask]])
eps_vals   = np.array([parse(v) for v in sweep['variant'][eps_mask]])
instances  = sweep['instance']  # 0..19
# Reshape: Z_by_alpha of shape (61, 20, n_pcs) indexed by α-grid-position × instance.
```

### 4. Animation (one per axis)

Build with `matplotlib.animation.FuncAnimation`. Per frame at parameter value `v`:
- **Background**: 500 CML anchor points in PC space, colored by class, with legend.
- **Current cloud**: 20 instance points at `v`, larger markers, distinct color (e.g. black-edged).
- **Centroid trail**: one point per *past* frame over the sliding window `[max(0, f-10), f)`. Values = `Z_by_alpha[f-10:f].mean(axis=1)` (mean across 20 instances per past frame). Linear alpha fade (newest ~0.8 → oldest ~0.1). Centroid-only; not a fading cloud.
- **Title**: `"α = {v:.2f}  (ε = 0.3 fixed)"` (and ε equivalent for the other animation).
- Fixed axes throughout the animation (no zoom-jump per frame). Compute xlim/ylim once from `np.concatenate([Z_anchor, Z_sweep])`.

Save MP4 via ffmpeg, gif fallback via Pillow. Target fps=5. ~12 s per animation.

No scaffold to copy from — write directly. ~80 lines of plain matplotlib in a notebook cell. Don't over-abstract.

## Decisions

- Metric for animation: **pearson**. (Spearman is computed and saved but not animated.)
- Trail: **centroid-only**, sliding 10-frame window, linear alpha fade.
- Directed SPIs: **symmetric** (default; no `--split-directed`).
- `--var-threshold 1e-6` vs `--var-threshold 0`: which to use? 
- PCA fit on CML-only (10 classes × 50 instances = 500 samples).
- Non-CML anchors are *not* shown in the animation.

## Output expectations

Single notebook, e.g. `notebooks/embeddings/proof.ipynb`, producing:
- One static figure per axis: anchor scatter + full sweep trajectory with α/ε colormap.
- `alpha_sweep_pearson.mp4`, `eps_sweep_pearson.mp4` (or `.gif` fallback).
- or follow with user's commands.

## Style constraints from user (CLAUDE.md)

- No sycophancy. Don't agree by default; say when uncertain.
- No speculative abstractions. Plain PCA, plain FuncAnimation, minimal code.
- Surgical edits. If existing code does the right thing, don't rewrite it.

## Things I'm genuinely uncertain about (flag rather than guess)

- PC1+PC2 explained variance at ~42k features × 500 samples is unknown until you run it. If <~25%, use 3D — simple pivot, user already cleared it.
- Some SPIs had partial NaN at certain (α, ε) during sweep compute (documented, 10-20 per dataset). `process_features` replaces non-finite cells with 0 before correlation ([src/process_features.py:221](src/process_features.py#L221)). Acceptable; flag in notebook output if >5% of cells are non-finite.
