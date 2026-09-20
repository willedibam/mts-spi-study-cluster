from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap

from src.cml_order_parameter import spatial_power_distribution, temporal_spectral_entropy
from src.generators.dynamical import generate_cml_logistic
from src.spi_spi_analysis import fit_feature_transform


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "figures"
DATA = ROOT / "data/embeddings/cml_param_sweep_260508/cml-alpha-sweep"
FEATURES = ROOT / "features/order_parameter/cml.npz"

EXAMPLES = (
    ("a1p60", 1.60, "spatially ordered"),
    ("a1p71", 1.71, "transition region"),
    ("a2p00", 2.00, "spatiotemporal chaos"),
)

BLUE = "#31688E"
ORANGE = "#D97732"
PURPLE = "#76528B"
GREEN = "#2A9D8F"
GREY = "#5C6670"


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 240,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _example_path(variant: str) -> Path:
    return DATA / f"M20_T1000_I0_{variant}"


def _reconstruct(variant: str) -> tuple[np.ndarray, np.ndarray]:
    path = _example_path(variant)
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    params = dict(meta["generator"]["resolved_params"])
    params["return_full_lattice"] = True
    observed, full = generate_cml_logistic(
        M=int(meta["M"]),
        T=int(meta["T"]),
        rng=np.random.default_rng(int(meta["generator"]["seed"])),
        **params,
    )
    saved = np.load(path / "timeseries.npy")
    if not np.allclose(observed, saved, rtol=0.0, atol=4e-8):
        raise RuntimeError(f"reconstructed observation does not match {path}")
    return saved.astype(float), full.astype(float)


def _save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def full_system_heatmaps(examples: list[tuple[np.ndarray, np.ndarray]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.7, 4.35), constrained_layout=True, sharey=True)
    for column, (ax, ((_, alpha, regime), (_, full))) in enumerate(zip(axes, zip(EXAMPLES, examples))):
        image = ax.imshow(
            full.T,
            aspect="auto",
            origin="lower",
            extent=(0, 1000, 0, 100),
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
            interpolation="nearest",
        )
        ax.add_patch(Rectangle((0, 40), 1000, 20, fill=False, lw=1.7, edgecolor=ORANGE))
        ax.set_title(rf"$\alpha={alpha:.2f}$  |  {regime}")
        ax.set_xlabel("map iteration")
        if column == 0:
            ax.set_ylabel("site on the physical ring")
    colorbar = fig.colorbar(image, ax=axes, location="right", shrink=0.83, pad=0.015)
    colorbar.set_label(r"state $x_i(t)$")
    fig.suptitle(
        r"Quadratic CML: exact $N=100$ physical rings at fixed coupling $\epsilon=0.3$",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.01,
        r"Orange outlines the central $M=20$ sites shown to SPI–SPI; panels are independent realizations.",
        ha="center",
        fontsize=9,
    )
    _save(fig, "cml-01-full-system")


def observed_mts_heatmaps(examples: list[tuple[np.ndarray, np.ndarray]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.7), constrained_layout=True, sharey=True)
    for column, (ax, ((_, alpha, regime), (observed, _))) in enumerate(zip(axes, zip(EXAMPLES, examples))):
        image = ax.imshow(
            observed.T,
            aspect="auto",
            origin="lower",
            extent=(0, 1000, 1, 20),
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
            interpolation="nearest",
        )
        ax.set_title(rf"$\alpha={alpha:.2f}$  |  {regime}")
        ax.set_xlabel("map iteration")
        if column == 0:
            ax.set_ylabel("observed site")
    colorbar = fig.colorbar(image, ax=axes, location="right", shrink=0.82, pad=0.015)
    colorbar.set_label(r"input $x_i(t)$")
    fig.suptitle(
        r"What SPI–SPI receives: $M=20$, $T=1000$ contiguous observations",
        fontsize=13,
        fontweight="bold",
    )
    _save(fig, "cml-02-observed-mts")


def spectral_coordinate(examples: list[tuple[np.ndarray, np.ndarray]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.3, 3.75), constrained_layout=True, sharex=True, sharey=True)
    for column, (ax, ((_, alpha, regime), (observed, _))) in enumerate(zip(axes, zip(EXAMPLES, examples))):
        probabilities = spatial_power_distribution(observed)
        frequencies = 2.0 * np.fft.rfftfreq(observed.shape[1])[1:]
        q_selected = float(probabilities[[2, 3]].sum())
        ax.bar(frequencies, probabilities, width=0.075, color=BLUE, alpha=0.82)
        ax.bar(frequencies[[2, 3]], probabilities[[2, 3]], width=0.075, color=ORANGE)
        ax.set_title(rf"$\alpha={alpha:.2f}$  |  $Q_{{\rm sel}}={q_selected:.2f}$")
        ax.set_xlabel(r"spatial wavenumber $k/\pi$")
        if column == 0:
            ax.set_ylabel("fraction of non-DC power")
    fig.suptitle(
        r"Operational state coordinate: $Q_{\rm sel}$ is power in the frozen $k/\pi=0.3,0.4$ modes",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.01,
        r"It summarizes spatial organization in the observed window; it is not a canonical thermodynamic order parameter.",
        ha="center",
        fontsize=9,
    )
    _save(fig, "cml-03-spectral-coordinate")


def _load_feature_data() -> dict[str, np.ndarray]:
    with np.load(FEATURES, allow_pickle=True) as archive:
        return {name: archive[name] for name in archive.files}


def spi_spi_pipeline(artifact: dict[str, np.ndarray]) -> None:
    path = _example_path("a1p71")
    observed = np.load(path / "timeseries.npy").astype(float)
    mpi_names = ("cov_EmpiricalCovariance", "corr_pearson_tau-1", "mi_gaussian")
    mpi_titles = ("covariance", "lag-1 correlation", "Gaussian MI")
    with np.load(path / "spi_mpis.npz") as archive:
        mpis = [archive[name].astype(float) for name in mpi_names]
    paths = artifact["dataset_paths"].astype(str)
    row = int(np.flatnonzero(np.char.endswith(paths, "M20_T1000_I0_a1p71"))[0])
    spi_names = artifact["spi_order"].astype(str)
    vector = artifact["X_sym"][row].astype(float)
    z_matrix = np.full((len(spi_names), len(spi_names)), np.nan, dtype=float)
    triangle = np.triu_indices(len(spi_names), k=1)
    z_matrix[triangle] = vector
    z_matrix[(triangle[1], triangle[0])] = vector

    fig = plt.figure(figsize=(15.8, 4.0), constrained_layout=True)
    grid = fig.add_gridspec(1, 6, width_ratios=(1.35, 0.82, 0.82, 0.82, 1.25, 1.35))
    ax_input = fig.add_subplot(grid[0, 0])
    ax_input.imshow(observed.T, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-1, vmax=1)
    ax_input.set(title=r"input $X$", xlabel="time", ylabel="20 sites")
    ax_input.set_xticks([])

    for column, (matrix, title) in enumerate(zip(mpis, mpi_titles), start=1):
        ax = fig.add_subplot(grid[0, column])
        finite = matrix[np.isfinite(matrix)]
        bound = max(abs(np.quantile(finite, 0.02)), abs(np.quantile(finite, 0.98)), 1e-12)
        ax.imshow(matrix, cmap="RdBu_r", vmin=-bound, vmax=bound, interpolation="nearest")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("target site")
        if column == 1:
            ax.set_ylabel("source site")
        ax.set_xticks([])
        ax.set_yticks([])

    ax_z = fig.add_subplot(grid[0, 4])
    image = ax_z.imshow(z_matrix, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    ax_z.set(title=r"SPI–SPI matrix $Z$", xlabel="SPI", ylabel="SPI")
    ax_z.set_xticks([])
    ax_z.set_yticks([])
    colorbar = fig.colorbar(image, ax=ax_z, fraction=0.046, pad=0.03)
    colorbar.set_label("Pearson agreement")

    ax_vector = fig.add_subplot(grid[0, 5])
    ax_vector.imshow(vector[:, None], aspect="auto", origin="lower", cmap="RdBu_r", vmin=-1, vmax=1)
    ax_vector.set(xlabel="one feature vector", ylabel="SPI-pair index")
    ax_vector.set_xticks([])
    ax_vector.set_yticks([0, vector.size - 1], labels=["1", f"{vector.size:,}"])
    ax_vector.set_title(r"upper triangle $z(X)$" + f"\n{vector.size:,} features")

    fig.suptitle(
        r"One MTS $\rightarrow$ 289 interaction matrices $\rightarrow$ SPI–SPI agreement $z(X)$ $\rightarrow$ PC1 $q$",
        fontsize=13,
        fontweight="bold",
    )
    _save(fig, "cml-04-spi-spi-pipeline")


def _residualize(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=float).copy()
    for group in np.unique(groups):
        member = groups == group
        result[member] -= result[member].mean()
    return result


def _abs_spearman(first: np.ndarray, second: np.ndarray) -> float:
    return float(abs(spearmanr(first, second).statistic))


def fit_coordinates(artifact: dict[str, np.ndarray]) -> dict[str, object]:
    alpha = np.asarray(
        [float(str(value).replace("a", "").replace("p", ".")) for value in artifact["variant"]]
    )
    instance = artifact["instance"].astype(int)
    alpha_index = np.rint((alpha - 1.6) * 100).astype(int)
    development = (instance <= 9) & (alpha_index % 2 == 0)
    confirmation = (instance >= 10) & (alpha_index % 2 == 1)

    selected_power: list[float] = []
    entropy: list[float] = []
    for dataset_path in artifact["dataset_paths"].astype(str):
        field = np.load(ROOT / dataset_path / "timeseries.npy")
        selected_power.append(float(spatial_power_distribution(field)[[2, 3]].sum()))
        entropy.append(temporal_spectral_entropy(field))
    target = np.asarray(selected_power)
    entropy_values = np.asarray(entropy)

    values = artifact["X_sym"]
    transform = fit_feature_transform(
        values[development],
        np.repeat("sym", values.shape[1]),
        minimum_valid_fraction=1.0,
        variance_threshold=1e-8,
        block_balanced=False,
    )
    fitted = transform.transform(values[development])
    all_values = transform.transform(values)
    pca = PCA(n_components=10, svd_solver="randomized", random_state=0)
    fitted_pca = pca.fit_transform(fitted)
    all_pca = pca.transform(all_values)
    isomap = Isomap(n_neighbors=15, n_components=1, eigen_solver="arpack").fit(fitted_pca)

    coordinates = {
        "PC1": all_pca[:, 0],
        "PCA10 → Isomap": isomap.transform(all_pca)[:, 0],
        "temporal entropy": entropy_values,
    }
    metrics: dict[str, dict[str, float]] = {}
    for name, coordinate in coordinates.items():
        metrics[name] = {
            "overall": _abs_spearman(coordinate[confirmation], target[confirmation]),
            "within_alpha": _abs_spearman(
                _residualize(coordinate[confirmation], alpha[confirmation]),
                _residualize(target[confirmation], alpha[confirmation]),
            ),
            "alpha": _abs_spearman(coordinate[confirmation], alpha[confirmation]),
        }
    return {
        "alpha": alpha,
        "instance": instance,
        "development": development,
        "confirmation": confirmation,
        "target": target,
        "coordinates": coordinates,
        "metrics": metrics,
        "transform": transform,
        "pca": pca,
    }


def coordinate_tracking(result: dict[str, object]) -> None:
    alpha = result["alpha"]
    confirmation = result["confirmation"]
    target = result["target"]
    coordinates = result["coordinates"]
    metrics = result["metrics"]
    pc1 = coordinates["PC1"].copy()
    isomap = coordinates["PCA10 → Isomap"].copy()
    for coordinate in (pc1, isomap):
        sign = np.sign(spearmanr(coordinate[confirmation], target[confirmation]).statistic) or 1.0
        coordinate *= sign

    held = pd.DataFrame(
        {
            "alpha": alpha[confirmation],
            "Q": target[confirmation],
            "PC1": pc1[confirmation],
            "Isomap": isomap[confirmation],
        }
    )
    summaries = held.groupby("alpha").agg(["mean", "sem"])

    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.15), constrained_layout=True)
    colors = {"Q": GREEN, "PC1": PURPLE}
    labels = {"Q": r"spectral coordinate $Q_{\rm sel}$", "PC1": r"SPI–SPI PC1 $q$"}
    for name in ("Q", "PC1"):
        means = summaries[(name, "mean")].to_numpy()
        sem = summaries[(name, "sem")].to_numpy()
        low, high = means.min(), means.max()
        scale = high - low
        scaled = (means - low) / scale
        interval = 1.96 * sem / scale
        axes[0].fill_between(summaries.index, scaled - interval, scaled + interval, color=colors[name], alpha=0.14)
        axes[0].plot(summaries.index, scaled, "-o", ms=3.5, lw=1.5, color=colors[name], label=labels[name])
    axes[0].set(
        xlabel=r"control parameter $\alpha$",
        ylabel="independently rescaled cell mean",
        ylim=(-0.08, 1.08),
        title="Held realizations across the sweep",
    )
    axes[0].legend(frameon=False, fontsize=8.5)

    axes[1].scatter(
        pc1[confirmation],
        target[confirmation],
        c=alpha[confirmation],
        cmap="viridis",
        norm=Normalize(alpha[confirmation].min(), alpha[confirmation].max()),
        s=15,
        alpha=0.55,
        edgecolors="none",
    )
    axes[1].set(
        xlabel=r"oriented development-fitted PC1 $q$",
        ylabel=r"contemporaneous $Q_{\rm sel}$",
        title=rf"Held realizations: $|\rho_S|={metrics['PC1']['overall']:.3f}$",
    )
    scalar = plt.cm.ScalarMappable(
        norm=Normalize(alpha[confirmation].min(), alpha[confirmation].max()), cmap="viridis"
    )
    colorbar = fig.colorbar(scalar, ax=axes[1], pad=0.02)
    colorbar.set_label(r"$\alpha$")

    methods = ["PC1", "PCA10 → Isomap", "temporal entropy"]
    positions = np.arange(len(methods))
    width = 0.34
    overall = [metrics[name]["overall"] for name in methods]
    within = [metrics[name]["within_alpha"] for name in methods]
    axes[2].barh(positions - width / 2, overall, height=width, color=BLUE, label="overall")
    axes[2].barh(positions + width / 2, within, height=width, color=ORANGE, label=r"within $\alpha$")
    axes[2].set_yticks(positions, labels=["SPI–SPI PC1", "SPI–SPI Isomap", "temporal entropy"])
    axes[2].invert_yaxis()
    axes[2].set_xlim(0, 1)
    axes[2].set_xlabel(r"held $|\rho_S(\cdot,Q_{\rm sel})|$")
    axes[2].set_title(r"Compression check: blue overall; orange within $\alpha$")
    for row, (first, second) in enumerate(zip(overall, within)):
        axes[2].text(first + 0.015, row - width / 2, f"{first:.3f}", va="center", fontsize=8)
        axes[2].text(second + 0.015, row + width / 2, f"{second:.3f}", va="center", fontsize=8)

    fig.suptitle(
        r"A development-fitted SPI–SPI PC1 represents the window's current spectral organization",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.29,
        -0.015,
        r"Bands are 95% normal intervals for cell means; signs and curve ranges are oriented only for display.",
        ha="center",
        fontsize=9,
    )
    _save(fig, "cml-05-pc1-tracking")


def pc1_loadings(artifact: dict[str, np.ndarray], result: dict[str, object]) -> None:
    transform = result["transform"]
    pca = result["pca"]
    weights = pca.components_[0]
    order = np.argsort(np.abs(weights))[-14:]
    feature_indices = transform.keep_indices[order]
    spi_a = artifact["feature_spi_a"][: artifact["X_sym"].shape[1]].astype(str)
    spi_b = artifact["feature_spi_b"][: artifact["X_sym"].shape[1]].astype(str)

    def compact(name: str) -> str:
        replacements = {
            "corr_pearson": "Pearson corr.",
            "corr_spearman": "Spearman corr.",
            "corr_kendall": "Kendall corr.",
            "dtw_constraint-sakoe-chiba_radius-auto": "DTW (Sakoe–Chiba)",
            "dtw_constraint-itakura": "DTW (Itakura)",
            "mi_gaussian": "Gaussian MI",
            "cov_EmpiricalCovariance": "empirical covariance",
        }
        return replacements.get(name, name.replace("_", " "))

    labels = [
        textwrap.fill(f"{compact(spi_a[index])}  ×  {compact(spi_b[index])}", width=50)
        for index in feature_indices
    ]
    values = weights[order]
    colors = [ORANGE if value < 0 else BLUE for value in values]
    fig, ax = plt.subplots(figsize=(10.4, 6.7), constrained_layout=True)
    ax.barh(np.arange(len(values)), values, color=colors)
    ax.axvline(0, color=GREY, lw=0.8)
    ax.set_yticks(np.arange(len(values)), labels=labels, fontsize=8.3)
    ax.set_xlabel("PC1 loading after development-set standardization")
    ax.set_title(
        rf"Largest SPI-pair contributions to $q$  |  PC1 explains {pca.explained_variance_ratio_[0]:.1%}",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.99,
        -0.01,
        "Loadings describe a distributed linear coordinate; they are not unique feature-importance scores.",
        ha="right",
        fontsize=9,
        color=GREY,
    )
    _save(fig, "cml-06-pc1-loadings")


def main() -> None:
    _style()
    examples = [_reconstruct(variant) for variant, _, _ in EXAMPLES]
    artifact = _load_feature_data()
    result = fit_coordinates(artifact)
    full_system_heatmaps(examples)
    observed_mts_heatmaps(examples)
    spectral_coordinate(examples)
    spi_spi_pipeline(artifact)
    coordinate_tracking(result)
    pc1_loadings(artifact, result)
    print("wrote six CML figures to", OUT)
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
