"""Timing triplet: zero-lag r, maximum cross-correlation and phase-aligned correlation P.

With ``S_jk(f) = |S_jk(f)| exp(i phi_jk(f))`` and ``N_jk = sqrt(int S_jj int S_kk)``:

    r_jk     = int |S_jk| cos(phi_jk) / N_jk                     no realignment
    xcorr_jk = max_h int |S_jk| cos(phi_jk + 2 pi f h) / N_jk    one delay (linear phase)
    P_jk     = int |S_jk| / N_jk                                 any phase (all-pass filter)

so ``r <= xcorr <= P`` for every pair. Helpers for
``notebooks/timing-triplet_261020.ipynb``; pyspi supplies r, max cross-correlation
and coherence magnitude through ``spis/timing_triplet.yaml``, and P is computed
here from Welch cross-spectra. Nothing here runs the full catalogue.
"""
from __future__ import annotations

import zlib
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy import signal

from mtsspi.compute import run_pyspi
from mtsspi.generators import band_delay_cross_spectrum, generate_band_delayed_mts
from research.cases.delay_alias.delay_alias import profile, save, separation, style, z  # noqa: F401

HERE = Path(__file__).resolve().parent
SPI_CONFIG = HERE / "spis" / "timing_triplet.yaml"
SEED_BASE = 20261020
NPERSEG = 256  # Welch segment length for P: 155 half-overlapping segments at T = 20,000

R = "cov_EmpiricalCovariance"  # zero-lag Pearson on z-scored data
XC = "xcorr_max_sig-True"
CM = "cohmag_multitaper_mean_fs-1_fmin-0_fmax-0-5"
P = "P"
STATS = (R, XC, P, CM)
SHORT = {R: "r", XC: "xcorr", P: "P", CM: "cohmag"}
TRIPLET = ((R, XC), (XC, P), (R, P))
PAIRS = TRIPLET + ((XC, CM), (CM, P))
FAMILIES = ("synchronous", "delayed", "dispersive")
COLORS = {"synchronous": "0.55", "delayed": "#1b6ca8", "dispersive": "#d55e00"}


def pair_name(a: str, b: str) -> str:
    return f"{SHORT[a]}–{SHORT[b]}"


# --------------------------------------------------------------------------- estimators

def two_sided_weights(n_bins: int, n: int) -> np.ndarray:
    """Multiplicity of each one-sided frequency bin on the two-sided grid of length ``n``."""
    w = np.full(n_bins, 2.0)
    w[0] = 1.0
    if n % 2 == 0:
        w[-1] = 1.0
    return w


def welch_cross_spectra(data: np.ndarray, nperseg: int = NPERSEG) -> np.ndarray:
    """Hann-window Welch spectral matrices ``S[f, j, k]`` (one-sided bins, 50% overlap, unscaled)."""
    x = np.asarray(data, float)
    starts = range(0, len(x) - nperseg + 1, nperseg // 2)
    segs = np.stack([x[s: s + nperseg] for s in starts])
    segs = segs - segs.mean(axis=1, keepdims=True)
    X = np.fft.rfft(segs * signal.get_window("hann", nperseg)[None, :, None], axis=1)
    return np.einsum("sfj,sfk->fjk", X, X.conj()) / len(segs)


def phase_aligned_corr(data: np.ndarray, nperseg: int = NPERSEG) -> np.ndarray:
    """P_jk = sum_f |S_jk(f)| / sqrt(sum_f S_jj(f) sum_f S_kk(f)) from Welch cross-spectra.

    The supremum over phase-only realignments is a population quantity: on raw
    periodograms every pair can be phase-aligned perfectly and P is about pi/4 for
    independent white noise. Averaging over segments restricts the realignment to
    phases that are smooth on the scale 1/nperseg, and leaves a positive bias of
    order sqrt(pi / (4 K)) for K effective segments at zero coherence.
    """
    S = welch_cross_spectra(data, nperseg)
    w = two_sided_weights(S.shape[0], nperseg)
    power = np.einsum("f,fjj->j", w, S).real
    return np.einsum("f,fjk->jk", w, np.abs(S)) / np.sqrt(np.outer(power, power))


def lagged_corr(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Zero-lag correlation and pyspi's ``xcorr_max_sig-True`` for every pair, in numpy.

    Biased normalisation over lags |h| <= T // 4; lags with |r(h)| <= 1.96/sqrt(T)
    are dropped before the maximum, and a pair with none left scores 0 (pyspi
    ``CrossCorrelation(statistic="max", sigonly=True)``).
    """
    T = len(data)
    x = (data - data.mean(0)) / data.std(0)
    n = 1 << int(np.ceil(np.log2(2 * T)))
    X = np.fft.rfft(x, n, axis=0)
    c = np.fft.irfft(np.einsum("fj,fk->fjk", X, X.conj()), n, axis=0) / T
    lag_max = T // 4
    c = np.concatenate([c[: lag_max + 1], c[n - lag_max:]])
    keep = np.abs(c) > 1.96 / np.sqrt(T)
    xmax = np.where(keep, c, -np.inf).max(axis=0)
    return c[0], np.where(keep.any(axis=0), xmax, 0.0)


# --------------------------------------------------------------------------- population

def ladder_from_spectrum(S: np.ndarray, n: int, lag_max: int) -> dict:
    """r, max cross-correlation over |h| <= lag_max, P and cohmag from a one-sided ``S[f, j, k]`` on an n-point grid."""
    w = two_sided_weights(S.shape[0], n)
    power = np.einsum("f,fjj->j", w, S).real
    N = np.sqrt(np.outer(power, power))
    c = np.fft.irfft(S, n, axis=0) * n / N
    window = np.concatenate([c[: lag_max + 1], c[n - lag_max:]])
    lags = np.r_[0: lag_max + 1, -lag_max:0]
    auto = np.einsum("fjj->fj", S).real
    msc = np.abs(S) ** 2 / (auto[:, :, None] * auto[:, None, :])
    return {"r": c[0], "xcorr": window.max(axis=0), "lag": lags[window.argmax(axis=0)],
            "P": np.einsum("f,fjk->jk", w, np.abs(S)) / N, "cohmag": msc.mean(axis=0)}


def population(params: dict, family: str, n: int = 2**16) -> dict:
    """Exact ladder of one family on an n-point frequency grid, plus its internals.

    ``cohmag`` is the unweighted mean of population magnitude-squared coherence over
    the one-sided grid on [0, 1/2], pyspi's definition without estimation bias.
    """
    T = params["T"]
    _, it = generate_band_delayed_mts(**{**params, "T": 1}, family=family, rng=0, return_internals=True)
    out = ladder_from_spectrum(band_delay_cross_spectrum(it, np.fft.rfftfreq(n)), n, T // 4)
    out["internals"] = it
    return out


def band_mismatch(internals) -> np.ndarray:
    """|(tau_slow_j - tau_slow_k) - (tau_fast_j - tau_fast_k)|: the pair's disagreement in delay between bands."""
    d = internals.delays
    diff = [lag[:, None] - lag[None, :] for lag in d]
    return np.abs(diff[0] - diff[1]).astype(float)


def random_ladder_check(n_trials: int, rng, n: int = 2**14) -> pd.DataFrame:
    """Random spectra, coherence and (non-linear) phase: check r <= xcorr <= P <= W <= 1.

    W = sqrt(int |S_jk|^2 / S_kk / int S_jj) is the supremum over every linear
    filter of k (gain and phase), one rung above P. Also returns the largest
    correlation reached by 200 random all-pass realignments, which must not exceed P.
    """
    f = np.fft.rfftfreq(n)
    w = two_sided_weights(len(f), n)
    rows = []
    for trial in range(n_trials):
        def bump():
            return 1e-3 + sum(rng.uniform(0.2, 1) * np.exp(-0.5 * ((f - rng.uniform(0, 0.5)) / rng.uniform(0.01, 0.15)) ** 2)
                              for _ in range(rng.integers(1, 6)))
        sjj, skk = bump(), bump()
        coh = rng.uniform(0, 1) * np.exp(-0.5 * ((f - rng.uniform(0, 0.5)) / rng.uniform(0.02, 0.4)) ** 2)
        phase = -2 * np.pi * f * rng.integers(-20, 21) + rng.normal(0, 3) * np.sin(2 * np.pi * f * rng.uniform(0, 6))
        phase[[0, -1]] = 0.0  # a real pair has odd phase: 0 at DC and Nyquist
        sjk = coh * np.sqrt(sjj * skk) * np.exp(1j * phase)
        S = np.empty((len(f), 2, 2), complex)
        S[:, 0, 0], S[:, 1, 1], S[:, 0, 1], S[:, 1, 0] = sjj, skk, sjk, sjk.conj()
        lad = ladder_from_spectrum(S, n, n // 4)
        norm = np.sqrt((w * sjj).sum() * (w * skk).sum())
        W = np.sqrt((w * np.abs(sjk) ** 2 / skk).sum() / (w * sjj).sum())
        # Zero-lag correlation after filtering k by the all-pass exp(i theta(f)).
        theta = (rng.normal(0, 2, (200, 1)) * np.sin(2 * np.pi * f * rng.uniform(0, 8, (200, 1)))
                 + 2 * np.pi * f * rng.integers(-30, 30, (200, 1)))
        theta[:, [0, -1]] = 0.0
        allpass = (w * np.abs(sjk) * np.cos(phase - theta)).sum(axis=1).max() / norm
        rows.append(dict(r=lad["r"][0, 1], xcorr=lad["xcorr"][0, 1], P=lad["P"][0, 1], W=W, allpass_best=allpass))
    return pd.DataFrame(rows)


def narrowband_scan(params: dict, kappas, n: int = 2**17) -> pd.DataFrame:
    """Population z(xcorr, P) and mean xcorr/P as both bands narrow by ``kappa``.

    Bands are band-pass, ((0.05, 0.05 kappa), (0.25, 0.15 kappa)), so each keeps
    its centre while its relative bandwidth shrinks.
    """
    rows = []
    for k in kappas:
        p = {**params, "bands": ((0.05, 0.05 * k), (0.25, 0.15 * k))}
        for fam in ("delayed", "dispersive"):
            pop = population(p, fam, n)
            rows.append(dict(kappa=k, family=fam, z_xcorr_P=z(pop["xcorr"], pop["P"]),
                             z_r_xcorr=z(pop["r"], pop["xcorr"]),
                             mean_ratio=float((profile(pop["xcorr"]) / profile(pop["P"])).mean())))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- measurement

def recording_rng(label: str, rep: int, seed: int = 0) -> np.random.Generator:
    """Independent process noise per (label, replicate, seed); stable across notebooks."""
    return np.random.default_rng([SEED_BASE, zlib.crc32(label.encode()), seed, rep])


def band_makers(params: dict, families=FAMILIES, suffix: str = "") -> dict:
    """{label: make(rng) -> (T, M) data} for band-delay families sharing ``params``."""
    return {f"{fam}{suffix}": (lambda rng, fam=fam: generate_band_delayed_mts(family=fam, rng=rng, **params))
            for fam in families}


def statistics(data: np.ndarray, nperseg: int = NPERSEG) -> dict:
    """pyspi subset plus P, as similarity matrices keyed by STATS."""
    found = run_pyspi(data, config_path=SPI_CONFIG, normalise=False).matrices
    return {R: found[R], XC: found[XC], P: phase_aligned_corr(data, nperseg), CM: found[CM]}


def measure(makers: dict, n_rep: int, seed: int = 0, nperseg: int = NPERSEG):
    """Statistics for ``n_rep`` recordings per maker.

    Returns ``(frame, mats)``: one frame row per recording and statistic pair with
    its z, and ``mats[label]`` of shape ``(n_rep, len(STATS), M, M)``.
    """
    rows, mats = [], {}
    for label, make in makers.items():
        stack = []
        for rep in range(n_rep):
            m = statistics(make(recording_rng(label, rep, seed)), nperseg)
            for a, b in PAIRS:
                rows.append(dict(family=label, rep=rep, pair=pair_name(a, b), z=z(m[a], m[b])))
            stack.append(np.stack([m[s] for s in STATS]))
        mats[label] = np.stack(stack)
    return pd.DataFrame(rows), mats


def phase_randomise(data: np.ndarray, rng) -> np.ndarray:
    """Multivariate Fourier surrogate: one random phase per frequency, shared by all channels.

    Preserves every sample auto- and cross-periodogram, hence the circular
    cross-correlation at every lag (Prichard and Theiler 1994).
    """
    T = len(data)
    X = np.fft.rfft(data, axis=0)
    psi = rng.uniform(0, 2 * np.pi, X.shape[0])
    psi[0] = 0.0
    if T % 2 == 0:
        psi[-1] = 0.0
    return np.fft.irfft(X * np.exp(1j * psi)[:, None], T, axis=0)


def numpy_triplet(data: np.ndarray, nperseg: int = NPERSEG) -> dict:
    r0, xmax = lagged_corr(data)
    return {R: r0, XC: xmax, P: phase_aligned_corr(data, nperseg)}


# --------------------------------------------------------------------------- figure

def plot_fig(mats: dict, frame: pd.DataFrame, params: dict, z_pop: dict):
    """One-panel figure: xcorr against P per channel pair (pure delay vs dispersive), z by family inset."""
    ix, ip = STATS.index(XC), STATS.index(P)
    mismatch = profile(band_mismatch(population(params, "dispersive")["internals"]))
    first = frame[frame.rep == 0].set_index(["family", "pair"]).z
    cmap = ListedColormap(plt.get_cmap("Oranges")(np.linspace(0.35, 1.0, 256)))

    fig, ax = plt.subplots(figsize=(3.5, 3.2), constrained_layout=True)
    lim = (0, 0.72)
    ax.plot(lim, lim, color="0.8", lw=0.8, zorder=0)
    xd, yd = profile(mats["delayed"][0, ip]), profile(mats["delayed"][0, ix])
    ax.scatter(xd, yd, s=7, color=COLORS["delayed"], lw=0, zorder=2)
    xs, ys = profile(mats["dispersive"][0, ip]), profile(mats["dispersive"][0, ix])
    order = np.argsort(mismatch)
    vmax = float(mismatch.max())
    sc = ax.scatter(xs[order], ys[order], s=7, c=mismatch[order], cmap=cmap, vmin=0, vmax=vmax, lw=0, zorder=3)
    ax.set(xlim=lim, ylim=lim, aspect="equal", xticks=np.arange(0, 0.8, 0.2), yticks=np.arange(0, 0.8, 0.2),
           xlabel=r"phase-aligned correlation $P_{jk}$",
           ylabel=r"max. cross-correlation $\mathrm{max}_h\, r_{jk}(h)$")
    handles = [Line2D([], [], ls="", marker="o", ms=3.5, mew=0, color=c) for c in (COLORS["delayed"], cmap(0.55))]
    zd, zs = first[("delayed", pair_name(XC, P))], first[("dispersive", pair_name(XC, P))]
    ax.legend(handles, [f"pure delay, $z = {zd:.2f}$", f"dispersive, $z = {zs:.2f}$"],
              loc="lower right", handletextpad=0.1, borderaxespad=0.1)
    cb = fig.colorbar(sc, ax=ax, shrink=0.5, aspect=16, pad=0.02, anchor=(0, 0.15))
    cb.set_label(r"$|\Delta^{\mathrm{slow}}_{jk}-\Delta^{\mathrm{fast}}_{jk}|$ (samples)", fontsize=8)
    cb.set_ticks(np.arange(0, vmax + 1, 10))
    cb.ax.tick_params(labelsize=7)
    cb.outline.set_visible(False)

    ins = ax.inset_axes([0.14, 0.63, 0.35, 0.34])  # above the diagonal, which xcorr <= P leaves empty
    styles = {pair_name(R, XC): ("o", (0, (1, 1.5))), pair_name(XC, P): ("s", "-"), pair_name(R, P): ("^", (0, (3, 1.5)))}
    xs_f = np.arange(len(FAMILIES))
    for pair, (marker, ls) in styles.items():
        sub = frame[frame.pair == pair]
        for i, fam in enumerate(FAMILIES):
            v = sub[sub.family == fam].z
            ins.scatter(np.full(len(v), i), v, s=1.5, lw=0, alpha=0.4, color=COLORS[fam], zorder=3)
        ins.plot(xs_f, [z_pop[(fam, pair)] for fam in FAMILIES], color="k", lw=0.7, ls=ls,
                 marker=marker, ms=2.6, mfc="white", mew=0.6, zorder=2, label=pair.replace("xcorr", "xc"))
    ins.set_ylim(0, 1.08)
    ins.set_xlim(-0.35, len(FAMILIES) - 0.65)
    ins.set_xticks(xs_f, ["sync", "delay", "disp"])
    ins.set_yticks([0, 0.5, 1])
    ins.tick_params(labelsize=6.5, length=2, pad=1)
    ins.set_ylabel("$z$", fontsize=7.5, labelpad=0)
    ins.legend(fontsize=5.8, loc="lower left", handlelength=1.8, handletextpad=0.3, borderaxespad=0.1,
               labelspacing=0.15, bbox_to_anchor=(0.0, 0.0))
    return fig
