"""Selected-statistic calibration of the frozen hardware pilot (no pyspi import)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, hilbert, sosfiltfilt

FS = 500
TRIM = 250
I, J = np.triu_indices(28, 1)
SI = "si_kernel_W-0.5_k-1"
BANDS = {58: (40, 75), 116: (90, 140), 174: (150, 195)}
DETAIL_STEPS = (21, 22, 23, 24, 26, 28, 35, 42, 49, 56, 63, 70, 77, 84, 87)
WEAK = np.array([11, 17, 20, 23])  # Prior exploratory selection at setting 23.


def edge_corr(a, b, floor=1e-8):
    """No imputation; a numerical spread floor is not a reliability threshold."""
    a, b = np.asarray(a), np.asarray(b)
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return float("nan")
    if min(a.std(), b.std()) <= floor:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def lag_matrix(x, tau):
    """Symmetric Pearson lag matrix; matches the archived r7 definition."""
    if tau == 0:
        return np.corrcoef(x)
    m = len(x)
    cross = np.corrcoef(np.vstack((x[:, tau:], x[:, :-tau])))[:m, m:]
    return .5 * (cross + cross.T)


def lag_covariance(x, tau):
    a, b = (x, x) if tau == 0 else (x[:, tau:], x[:, :-tau])
    a, b = a - a.mean(axis=1, keepdims=True), b - b.mean(axis=1, keepdims=True)
    cross = a @ b.T / a.shape[1]
    return .5 * (cross + cross.T)


def band_signal(x, limits):
    return sosfiltfilt(butter(4, limits, fs=FS, btype="bandpass", output="sos"), x, axis=1)


def phase_properties(x, limits):
    y = band_signal(x, limits)
    h = hilbert(y, axis=1)[:, TRIM:-TRIM]
    u = h / np.maximum(abs(h), 1e-12)
    cross = u @ u.conj().T / u.shape[1]
    t = np.arange(u.shape[1]) / FS
    tc = t - t.mean()
    frequency = np.unwrap(np.angle(u), axis=1) @ tc / (tc @ tc) / (2 * np.pi)
    return dict(signal=y[:, TRIM:-TRIM], unit=u, plv=abs(cross),
                frequency=frequency, offset=np.angle(cross[:, 0]),
                R1=float(abs(u.mean(axis=0)).mean()),
                R2=float(abs((u ** 2).mean(axis=0)).mean()))


def load_inputs(run):
    run = Path(run)
    rows = json.loads((run / "manifest.json").read_text())["rows"]
    with np.load(run / "observations.npz") as bank:
        observations = {r["step"]: bank[r["name"]] for r in rows}
    names = [SI, "ids", "corr_pearson_tau-1", "corr_pearson_tau-10",
             "corr_kendall_tau-10", "cov_EmpiricalCovariance", "corr_spearman_tau-1"]
    matrices, paths = {}, {}
    for row in rows:
        path, = (run / "mpis").glob(f"*/????-{row['name']}/spi_mpis.npz")
        with np.load(path) as bank:
            matrices[row["step"]] = {key: bank[key] for key in names}
        paths[row["step"]] = path
    with np.load(run / "analysis/features.npz") as bank:
        scores = bank["scores"].copy()
    return dict(rows=rows, x=observations, mpi=matrices, paths=paths, scores=scores)


def _lag_summary(x):
    a, b = [lag_matrix(x, tau)[I, J] for tau in (1, 10)]
    deletion = [edge_corr(a[(I != k) & (J != k)], b[(I != k) & (J != k)]) for k in range(28)]
    return dict(z=edge_corr(a, b), mean_pearson=float(lag_matrix(x, 0)[I, J].mean()),
                sd_short=float(a.std()), sd_long=float(b.std()),
                delete_min=min(deletion), delete_max=max(deletion))


def analyse_measurements(data, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    physical, harmonics, power, reconstruction, sensitivities = [], [], [], [], []
    phase, components = {}, {}
    maximum_error = 0.
    for index, row in enumerate(data["rows"]):
        step, x, mpi = row["step"], data["x"][row["step"]], data["mpi"][row["step"]]
        for tau in (1, 10):
            error = np.max(abs(lag_matrix(x, tau)[I, J] - mpi[f"corr_pearson_tau-{tau}"][I, J]))
            maximum_error = max(maximum_error, float(error))
        p = phase_properties(x, BANDS[116])
        physical.append(dict(step=step, voltage=step * .005, pc1=data["scores"][index, 0],
                             mean_R116=p["R1"], mean_PLV116=float(p["plv"][I, J].mean()),
                             frequency_range=float(np.ptp(p["frequency"])),
                             z_si_ids=edge_corr(mpi[SI][I, J], mpi["ids"][I, J]),
                             z_lag=edge_corr(mpi["corr_pearson_tau-1"][I, J], mpi["corr_pearson_tau-10"][I, J]),
                             z_leading=edge_corr(mpi["corr_pearson_tau-1"][I, J], mpi["corr_kendall_tau-10"][I, J]),
                             mean_pearson=float(mpi["cov_EmpiricalCovariance"][I, J].mean())))
        if step not in DETAIL_STEPS:
            continue
        phase[step] = {116: p, **{f: phase_properties(x, limits) for f, limits in BANDS.items() if f != 116}}
        raw = x[:, TRIM:-TRIM]
        normalizer = np.outer(raw.std(axis=1), raw.std(axis=1))
        components[step] = {f: lag_covariance(v["signal"], 0) / normalizer for f, v in phase[step].items()}
        pred = {tau: sum(np.cos(2 * np.pi * np.median(phase[step][f]["frequency"]) * tau / FS) * g
                         for f, g in components[step].items()) for tau in (1, 10)}
        variants = {"Raw": raw, "58 Hz": phase[step][58]["signal"], "116 Hz": p["signal"],
                    "Three bands": sum(v["signal"] for v in phase[step].values()),
                    "Remove 58 Hz": raw - phase[step][58]["signal"]}
        for name, y in variants.items():
            summary = _lag_summary(y)
            half = y.shape[1] // 2
            summary.update({f"half_{n}": _lag_summary(v)["z"] for n, v in enumerate((y[:, :half], y[:, half:]), 1)})
            reconstruction.append(dict(step=step, variant=name, **summary))
        prediction = dict(step=step, variant="Cosine prediction", z=edge_corr(pred[1][I, J], pred[10][I, J]))
        for tau in (1, 10):
            actual = lag_matrix(raw, tau)[I, J]
            prediction[f"r_tau{tau}"] = edge_corr(actual, pred[tau][I, J])
            prediction[f"rmse_tau{tau}"] = float(np.sqrt(np.mean((actual - pred[tau][I, J]) ** 2)))
        reconstruction.append(prediction)
        for f, v in phase[step].items():
            frac = v["signal"].var(axis=1) / raw.var(axis=1)
            harmonics.append(dict(step=step, frequency=f, R1=v["R1"], R2=v["R2"],
                                  mean_plv=float(v["plv"][I, J].mean()), median_power_fraction=float(np.median(frac)),
                                  phase_lock_2to1=float(np.median(abs((phase[step][116]["unit"] * phase[step][58]["unit"].conj() ** 2).mean(axis=1)))),
                                  group_positive=int((np.cos(v["offset"]) > 0).sum()),
                                  **{f"profile_sd_tau{tau}": float((lag_covariance(v["signal"], tau) / normalizer)[I, J].std()) for tau in (1, 10)}))
            for k in range(28):
                power.append(dict(step=step, band=f, channel=k + 1, power_fraction=frac[k],
                                  phase_to_channel1=v["offset"][k], frequency=v["frequency"][k]))
        if step in (21, 23, 24, 26, 63, 70):
            for limits in ((80, 160), (90, 140), (100, 130)):
                props = phase_properties(x, limits)
                sensitivities.append(dict(step=step, check="116-Hz frequency range", band=str(limits),
                                          value=float(np.ptp(props["frequency"]))))
        if step in (63, 70):
            reference_group = np.cos(phase[step][58]["offset"]) > 0
            for limits in ((40, 75), (45, 70), (50, 65)):
                u = phase_properties(x, limits)["unit"]
                for label, v in (("full", u), ("first half", u[:, :2250]), ("second half", u[:, 2250:])):
                    offset = np.angle((v * v[0].conj()).mean(axis=1))
                    group = np.cos(offset) > 0
                    sensitivities.append(dict(step=step, check="58-Hz group size", band=str(limits), window=label,
                                              value=int(group.sum()), membership_mismatches=int((group != reference_group).sum())))
    assert maximum_error < 1e-11, maximum_error
    frames = dict(physical=pd.DataFrame(physical), harmonics=pd.DataFrame(harmonics), power=pd.DataFrame(power),
                  reconstruction=pd.DataFrame(reconstruction), sensitivities=pd.DataFrame(sensitivities))
    for name, frame in frames.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    result = dict(**frames, phase=phase, components=components, replay_max_error=maximum_error)
    return result


def analytic_profiles(phi, signs, amplitude=.24, third=.075, frequency=116.):
    """Infinite-time correlations of a phase-locked three-component waveform."""
    delta, parity = phi[:, None] - phi, np.outer(signs, signs)
    modes = [np.cos(delta), amplitude ** 2 * parity * np.cos(delta / 2),
             third ** 2 * parity * np.cos(1.5 * delta)]
    frequencies = np.array([frequency, frequency / 2, 1.5 * frequency])
    return {tau: sum(np.cos(2 * np.pi * f * tau / FS) * mode for f, mode in zip(frequencies, modes))
            / (1 + amplitude ** 2 + third ** 2) for tau in (0, 1, 10)}


def control_signal(phi, signs, amplitude=.24, third=.075, coherent=28, frequency=116., samples=5000):
    time = np.arange(samples) / FS
    psi = 2 * np.pi * frequency * time + phi[:, None]
    psi = psi + 2 * np.pi * .35 * time * (np.arange(28) >= coherent)[:, None]
    return (np.cos(psi) + amplitude * signs[:, None] * np.cos(psi / 2)
            + third * signs[:, None] * np.cos(1.5 * psi)), psi


def calibrate_controls(output):
    """A declared small factorial, with no fitted axis or outcome-ranked controls."""
    records = []
    seeds = range(26101301, 26101309)
    for seed in seeds:
        rng = np.random.default_rng(seed)
        base = rng.normal(size=28)
        base = (base - base.mean()) / base.std()
        signs = np.ones(28)
        signs[rng.permutation(28)[:14]] = -1
        for family, levels in (("offset", (.1, .2, .3, .4, .6, .8)),
                               ("amplitude", (0., .05, .1, .2, .24, .3, .4)),
                               ("coherent", (16, 20, 24, 26, 28))):
            for level in levels:
                spread = level if family == "offset" else .2
                a = level if family == "amplitude" else .24
                coherent = int(level) if family == "coherent" else 28
                phi = spread * base
                # Scale both weaker waveform components together in this intervention.
                third = .075 * a / .24 if family == "amplitude" else .075
                x, psi = control_signal(phi, signs, a, third=third, coherent=coherent)
                profiles = [lag_matrix(x, tau)[I, J] for tau in (0, 1, 10)]
                units = np.exp(1j * psi)
                records.append(dict(seed=seed, family=family, level=level, z=edge_corr(profiles[1], profiles[2]),
                                    mean_pearson=float(profiles[0].mean()), mean_R116=float(abs(units.mean(axis=0)).mean()),
                                    mean_PLV116=float(abs(units @ units.conj().T / units.shape[1])[I, J].mean()),
                                    sd_short=float(profiles[1].std()), sd_long=float(profiles[2].std())))
    frame = pd.DataFrame(records)
    frame.to_csv(Path(output) / "controls.csv", index=False)
    # Independent formula check uses periodic correlation, for which the identity is exact.
    phi = .2 * np.linspace(-np.sqrt(3), np.sqrt(3), 28)
    signs = np.tile([-1., 1.], 14)
    x, _ = control_signal(phi, signs)
    x = x - x.mean(axis=1, keepdims=True)
    denominator = np.outer(x.std(axis=1), x.std(axis=1))
    analytic = analytic_profiles(phi, signs)
    errors = []
    for tau in (0, 1, 10):
        cross = x @ np.roll(x, tau, axis=1).T / x.shape[1] / denominator
        errors.append(float(np.max(abs(.5 * (cross + cross.T) - analytic[tau]))))
    assert max(errors) < 1e-10, errors
    # Harmonic amplitude changes neither prescribed carrier phases nor their coordination.
    spread = frame.query("family == 'amplitude'").groupby("seed").mean_R116.agg(lambda v: v.max() - v.min()).max()
    assert spread < 1e-12
    return frame, max(errors)


def write_receipt(data, measurements, control_error, output, run):
    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    receipt = dict(source="Frozen pyspi 3.0.0.r7 bank; all new calculations are separate NumPy/SciPy diagnostics",
                   selected_lags_samples=[1, 10], fs=FS, crop_samples_each_end=TRIM, bands_hz=BANDS,
                   cached_lag_replay_max_abs_error=measurements["replay_max_error"],
                   analytic_periodic_control_max_abs_error=control_error,
                   control_seeds=list(range(26101301, 26101309)),
                   source_sha256={str(p.relative_to(Path(run))): sha(p) for p in data["paths"].values()},
                   manifest_sha256=sha(Path(run) / "manifest.json"),
                   observations_sha256=sha(Path(run) / "observations.npz"),
                   featurebank_sha256=sha(Path(run) / "analysis/features.npz"),
                   helper_sha256=sha(__file__))
    (Path(output) / "verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def _plot_tools():
    import matplotlib.pyplot as plt
    from research.empirical.hardware_pilot.plot_hardware_pilot import style, box
    style()
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white"})
    return plt, box


def _save(fig, output, name):
    import matplotlib.pyplot as plt
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(Path(output) / f"{name}.{suffix}", dpi=180 if suffix == "png" else 600, bbox_inches="tight")
    plt.close(fig)


def plot_locking(data, measured, output):
    import seaborn as sns
    plt, box = _plot_tools()
    steps = (21, 23, 24, 26)
    fig = plt.figure(figsize=(10.4, 7.7), constrained_layout=True)
    grid = fig.add_gridspec(3, 4, height_ratios=(1.15, 1.35, 1.8))
    scaled = []
    for step in steps:
        x = data["x"][step]
        q25, q75 = np.quantile(x, [.25, .75], axis=1)
        scaled.append((x - np.median(x, axis=1, keepdims=True)) / (q75 - q25)[:, None])
    limit = np.quantile(abs(np.concatenate([v.ravel() for v in scaled])), .99)
    heat_axes = []
    core = np.array([k for k in range(28) if k not in WEAK])
    for k, (step, x) in enumerate(zip(steps, scaled)):
        ax = fig.add_subplot(grid[0, k]); heat_axes.append(ax)
        im = ax.imshow(x[:, 500:600], origin="lower", aspect="auto", interpolation="nearest",
                       extent=(1, 1.2, .5, 28.5), cmap=sns.color_palette("icefire", as_cmap=True), vmin=-limit, vmax=limit)
        ax.set(title=f"Setting {step} · {step * .005:.3f} V", xlabel="Time (s)", ylabel="Channel", yticks=[1, 14, 28])
        ax.tick_params(direction="out", length=2)
        ax = fig.add_subplot(grid[1, k])
        u = measured["phase"][step][116]["unit"]
        reference = u[core].mean(axis=0)
        relative = np.unwrap(np.angle(u * (reference / abs(reference)).conj()), axis=1)
        relative = (relative - relative[:, :1]) / (2 * np.pi)
        t = .5 + np.arange(u.shape[1]) / FS
        ax.plot(t[::4], relative[core, ::4].T, color=".7", lw=.5)
        ax.plot(t[::4], relative[WEAK, ::4].T, color="#d95f02", lw=.85, alpha=.8)
        ax.set(xlabel="Time (s)", ylabel="Relative phase change (cycles)", xlim=(.5, 9.5), ylim=(-6, .3), yticks=[-6, -3, 0])
        if k == 0:
            ax.plot([], [], color="#d95f02", label="Channels 12, 18, 21, 24")
            ax.plot([], [], color=".7", label="Other 24 channels")
            ax.legend(fontsize=6, loc="lower left")
        ax.tick_params(direction="out", length=2)
    fig.colorbar(im, ax=heat_axes, shrink=.7, pad=.01, label="Robust-scaled voltage")
    p = measured["physical"].query("14 <= step <= 35")
    ax = fig.add_subplot(grid[2, :2]); twin = ax.twinx()
    a, = ax.semilogy(p.voltage, p.frequency_range, "o-", color="#222222", label="Frequency range")
    b, = twin.plot(p.voltage, p.pc1, "s-", color="#26828e", label="Frozen full-p90 PC1")
    ax.axvspan(.115, .120, color=".85", zorder=-2)
    ax.set(xlabel="Control voltage (V)", ylabel="116 Hz frequency range (Hz)", xlim=(.067, .178), ylim=(5e-5, 7))
    twin.set_ylabel("Frozen PC1 score", color="#26828e")
    twin.tick_params(axis="y", colors="#26828e")
    ax.legend([a, b], [a.get_label(), b.get_label()], loc="center left", fontsize=7)
    ax = fig.add_subplot(grid[2, 2:])
    ax.plot(p.voltage, p.z_si_ids, "o-", color="#76528e", label="Kernel SI × IDS")
    ax.plot(p.voltage, p.z_lag, "s-", color="#d08b22", label="Pearson, 2 × 20 ms")
    ax.axvspan(.115, .120, color=".85", zorder=-2)
    ax.set(xlabel="Control voltage (V)", ylabel="Full-record edge agreement", xlim=(.067, .178), ylim=(-1.05, 1.05))
    ax.legend(fontsize=7, loc="upper right")
    fig.suptitle("Measured recruitment into frequency locking", fontsize=12)
    _save(fig, output, "locking-composite")


def plot_decomposition(measured, output):
    plt, box = _plot_tools()
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.25))
    table = measured["reconstruction"].pivot(index="step", columns="variant", values="z")
    for variant, color, marker, label in (("Raw", "#26828e", "s", "Measured"),
                                          ("Cosine prediction", "#d95f02", ".", "Three-band cosine prediction"),
                                          ("116 Hz", "#777777", None, "116 Hz component only")):
        axes[0].plot(table.index * .005, table[variant], color=color, marker=marker, lw=1.1, label=label)
    axes[0].set(xlabel="Control voltage (V)", ylabel="Lag-profile agreement", ylim=(-1.05, 1.05), title="A  A spectral explanation")
    axes[0].legend(fontsize=6, loc="lower right")
    colors = {58: "#d95f02", 116: "#31688e", 174: "#76528e"}
    for ax, tau, title in zip(axes[1:], (1, 10), ("B  Short lag: 2 ms", "C  Long lag: 20 ms")):
        for frequency, group in measured["harmonics"].groupby("frequency"):
            ax.semilogy(group.step * .005, group[f"profile_sd_tau{tau}"], "o-", color=colors[frequency], label=f"{frequency} Hz")
        ax.set(xlabel="Control voltage (V)", ylabel="SD across channel pairs", ylim=(.001, .2), title=title)
        ax.legend(fontsize=7)
    for ax in axes:
        for voltage in (.315, .350):
            ax.axvline(voltage, color=".8", ls=":", lw=.6, zorder=-1)
        box(ax)
    _save(fig, output, "spectral-explanation")


def plot_matched(data, measured, output):
    plt, box = _plot_tools()
    fig, axes = plt.subplots(2, 3, figsize=(8.7, 6.1))
    steps, scatters = (63, 70), {}
    for step in steps:
        raw = data["x"][step][:, TRIM:-TRIM]
        scatters[step] = [lag_matrix(raw, tau)[I, J] for tau in (1, 10)]
    limits = []
    for k in (0, 1):
        v = np.concatenate([scatters[s][k] for s in steps]); pad = .06 * np.ptp(v)
        limits.append((v.min() - pad, v.max() + pad))
    for row, step in enumerate(steps):
        props = measured["phase"][step]
        group = np.cos(props[58]["offset"]) > 0
        counts = (int(group.sum()), int((~group).sum()))
        for column, frequency in enumerate((58, 116)):
            u = props[frequency]["unit"]
            alignment = (u @ u.conj().T / u.shape[1]).real
            ax = axes[row, column]
            im = ax.imshow(alignment, origin="lower", vmin=-1 if frequency == 58 else .78, vmax=1,
                           cmap="RdBu_r" if frequency == 58 else "viridis", interpolation="nearest",
                           extent=(.5, 28.5, .5, 28.5), rasterized=True)
            ax.set(xlabel="Channel", ylabel="Channel", xticks=[1, 14, 28], yticks=[1, 14, 28],
                   title=f"Setting {step}: {frequency} Hz" + (f" · groups {counts[0]}:{counts[1]}" if frequency == 58 else ""))
            if row == 1:
                fig.colorbar(im, ax=axes[:, column], shrink=.6, pad=.02, label="Phase alignment (cosine)")
            box(ax)
        ax = axes[row, 2]
        within = group[I] == group[J]
        for mask, color, label in ((within, "#31688e", "Same 58 Hz group"), (~within, "#d95f02", "Opposite 58 Hz groups")):
            ax.scatter(scatters[step][0][mask], scatters[step][1][mask], s=7, color=color, alpha=.65, linewidths=0, rasterized=True, label=label)
        z = edge_corr(*scatters[step])
        ax.set(xlabel="Pearson, 2 ms", ylabel="Pearson, 20 ms", xlim=limits[0], ylim=limits[1],
               title=f"Setting {step}: z = {z:.3f}")
        if row == 0:
            ax.legend(fontsize=6, loc="upper left")
        box(ax)
    fig.suptitle("Different spectral organisation at matched mean Pearson strength", fontsize=11)
    _save(fig, output, "matched-strength-organisation")


def plot_controls(frame, output):
    plt, box = _plot_tools()
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.2))
    choices = (("offset", "Carrier phase-offset SD (rad)", "A  Stationary phase offsets"),
               ("amplitude", "58 Hz amplitude (116 Hz = 1)", "B  Weaker waveform components"),
               ("coherent", "Channels at the same carrier frequency", "C  Coherent-group recruitment"))
    for ax, (family, xlabel, title) in zip(axes, choices):
        summary = frame.query("family == @family").groupby("level").z.agg(["mean", "min", "max"])
        ax.fill_between(summary.index, summary["min"], summary["max"], color="#26828e", alpha=.18, linewidth=0)
        ax.plot(summary.index, summary["mean"], "s-", color="#26828e")
        ax.set(xlabel=xlabel, ylabel="Lag-profile agreement", title=title, ylim=(-1.05, 1.05))
        box(ax)
    axes[1].text(.03, .96, "Carrier phases fixed\nMean $R_{116}\\approx0.980$", transform=axes[1].transAxes, va="top", fontsize=7)
    _save(fig, output, "controlled-calibration")
