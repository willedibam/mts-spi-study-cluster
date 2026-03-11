from __future__ import annotations

import importlib
import os
import warnings

import numpy as np
from scipy import stats
import yaml

try:
    from spectral_connectivity.transforms import prepare_time_series
except ImportError:
    def prepare_time_series(time_series: np.ndarray, axis: str = "signals") -> np.ndarray:
        if axis != "signals":
            raise ImportError(
                "spectral_connectivity is missing prepare_time_series; "
                "upgrade the package or ensure axis='signals'."
            )
        return time_series[:, np.newaxis, :]

import pandas as pd

from pyspi.base import Undirected, Signed, parse_bivariate, parse_multivariate
from pyspi.statistics import basic as _basic
from pyspi.statistics import distance as _distance
from pyspi.statistics import spectral as _spectral
from pyspi.statistics import infotheory as _infotheory
from pyspi import calculator as _calculator


# PySPI still references np.NaN in some paths; keep compatibility on NumPy 2.
if not hasattr(np, "NaN"):
    np.NaN = np.nan


_DTW_SAKOE_LINEAR_FRAC = float(os.getenv("PYSPI_DTW_SAKOE_LINEAR_FRAC", "0.10"))
_DTW_SAKOE_SQRT_COEFF = float(os.getenv("PYSPI_DTW_SAKOE_SQRT_COEFF", "1.5"))
_DTW_SAKOE_MIN_RADIUS = int(os.getenv("PYSPI_DTW_SAKOE_MIN_RADIUS", "10"))


def _ensure_time_series_3d(z: np.ndarray) -> np.ndarray:
    """Ensure time-series array follows (n_time, n_trials, n_signals)."""
    if z.ndim == 2:
        return prepare_time_series(z, axis="signals")
    return z


def _auto_sakoe_radius(length: int) -> int:
    if length <= 1:
        return 1
    linear = int(np.ceil(max(0.0, _DTW_SAKOE_LINEAR_FRAC) * length))
    sqrt_scaled = int(np.ceil(max(0.0, _DTW_SAKOE_SQRT_COEFF) * np.sqrt(length)))
    radius = min(linear, sqrt_scaled)
    radius = max(1, _DTW_SAKOE_MIN_RADIUS, radius)
    return min(radius, length - 1)


def _patch_multivariate():
    def patched(self, data):
        try:
            res = data.spectral_mv[self.key]
            freq = data.spectral_mv["freq"]
        except (AttributeError, KeyError):
            z = np.transpose(data.to_numpy(squeeze=True))
            z = _ensure_time_series_3d(z)
            m = _spectral.sc.Multitaper(z, sampling_frequency=self._fs)
            conn = _spectral.sc.Connectivity.from_multitaper(m)
            try:
                res = getattr(conn, self.measure)()
            except TypeError:
                res = self._get_statistic(conn)

            freq = conn.frequencies
            try:
                data.spectral_mv[self.key] = res
            except AttributeError:
                data.spectral_mv = {"freq": freq, self.measure: res}
        return res, freq

    _spectral.NonparametricSpectralMultivariate._get_cache = patched


def _patch_bivariate():
    def patched(self, data, i, j):
        key = (self.measure, i, j)
        try:
            res = data.spectral_bv[key]
            freq = data.spectral_bv["freq"]
        except (KeyError, AttributeError):
            z = np.transpose(data.to_numpy(squeeze=True)[[i, j]])
            z = _ensure_time_series_3d(z)
            m = _spectral.sc.Multitaper(z, sampling_frequency=self._fs)
            conn = _spectral.sc.Connectivity.from_multitaper(m)
            try:
                res = getattr(conn, self.measure)()
            except TypeError:
                res = self._get_statistic(conn)

            freq = conn.frequencies
            try:
                data.spectral_bv[key] = res
            except AttributeError:
                data.spectral_bv = {"freq": freq, key: res}
        return res, freq

    _spectral.NonparametricSpectralBivariate._get_cache = patched


_patch_multivariate()
_patch_bivariate()


def _patch_cross_correlation():
    original_bivariate = _basic.CrossCorrelation.bivariate

    @parse_bivariate
    def safe_bivariate(self, data, i=None, j=None):
        try:
            return original_bivariate(self, data, i=i, j=j)
        except IndexError:
            if not getattr(self, "_sigonly", False):
                raise
            previous = self._sigonly
            self._sigonly = False
            try:
                return original_bivariate(self, data, i=i, j=j)
            finally:
                self._sigonly = previous

    _basic.CrossCorrelation.bivariate = safe_bivariate


_patch_cross_correlation()


def _patch_dynamic_time_warping():
    if getattr(_distance.DynamicTimeWarping, "_dtaidistance_c_patch", False):
        return

    try:
        from dtaidistance import dtw as _dtw_c
        from dtaidistance.exceptions import CythonException
    except Exception:
        return

    base_init = _distance.TimeWarping.__init__

    def patched_init(
        self,
        global_constraint=None,
        sakoe_chiba_radius=None,
        sakoe_chiba_ratio=None,
    ):
        if sakoe_chiba_radius is not None and sakoe_chiba_ratio is not None:
            raise ValueError("Set only one of sakoe_chiba_radius or sakoe_chiba_ratio.")
        if sakoe_chiba_radius is not None:
            sakoe_chiba_radius = int(sakoe_chiba_radius)
            if sakoe_chiba_radius < 1:
                raise ValueError("sakoe_chiba_radius must be >= 1.")
        if sakoe_chiba_ratio is not None:
            sakoe_chiba_ratio = float(sakoe_chiba_ratio)
            if sakoe_chiba_ratio <= 0:
                raise ValueError("sakoe_chiba_ratio must be > 0.")

        base_init(self, global_constraint=global_constraint)
        self._sakoe_chiba_radius = sakoe_chiba_radius
        self._sakoe_chiba_ratio = sakoe_chiba_ratio
        self._warned_itakura_fallback = False
        self._warned_c_fallback = False

        if global_constraint == "sakoe_chiba":
            if sakoe_chiba_radius is not None:
                self.identifier += f"_radius-{sakoe_chiba_radius}"
            elif sakoe_chiba_ratio is not None:
                self.identifier += f"_ratio-{sakoe_chiba_ratio:.4g}"
            else:
                self.identifier += "_radius-auto"

    def _resolve_radius(self, n: int) -> int:
        if self._sakoe_chiba_radius is not None:
            return min(self._sakoe_chiba_radius, max(1, n - 1))
        if self._sakoe_chiba_ratio is not None:
            ratio_radius = int(np.ceil(self._sakoe_chiba_ratio * n))
            return min(max(1, ratio_radius), max(1, n - 1))
        return _auto_sakoe_radius(n)

    @parse_bivariate
    def patched_bivariate(self, data, i=None, j=None):
        z = data.to_numpy(squeeze=True)
        x = np.ascontiguousarray(z[i], dtype=np.double)
        y = np.ascontiguousarray(z[j], dtype=np.double)
        constraint = self._global_constraint
        n = min(len(x), len(y))

        if constraint == "itakura":
            if not self._warned_itakura_fallback:
                warnings.warn(
                    "DynamicTimeWarping(itakura) falls back to tslearn; "
                    "dtaidistance C backend does not support itakura.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._warned_itakura_fallback = True
            return _distance.tslearn.metrics.dtw(x, y, global_constraint="itakura")

        radius = None
        window = None
        if constraint == "sakoe_chiba":
            # tslearn radius r matches dtaidistance window r+1.
            radius = _resolve_radius(self, n)
            window = radius + 1

        try:
            kwargs = {"use_c": True}
            if window is not None:
                kwargs["window"] = window
            return _dtw_c.distance(x, y, **kwargs)
        except (CythonException, ValueError):
            if not self._warned_c_fallback:
                warnings.warn(
                    "dtaidistance C backend unavailable for DynamicTimeWarping; "
                    "falling back to tslearn.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._warned_c_fallback = True
            if constraint == "sakoe_chiba":
                return _distance.tslearn.metrics.dtw(
                    x,
                    y,
                    global_constraint="sakoe_chiba",
                    sakoe_chiba_radius=radius,
                )
            return _distance.tslearn.metrics.dtw(x, y)

    _distance.DynamicTimeWarping.__init__ = patched_init
    _distance.DynamicTimeWarping.bivariate = patched_bivariate
    _distance.DynamicTimeWarping._dtaidistance_c_patch = True


_patch_dynamic_time_warping()


def _patch_lagged_correlation():
    if hasattr(_basic, "LaggedCorrelation"):
        return

    class LaggedCorrelation(Undirected, Signed):
        name = "Lagged correlation"
        labels = ["basic", "linear", "undirected", "temporal"]

        def __init__(self, estimator="pearson", tau=None, max_tau=None, squared=False):
            est = str(estimator).lower()
            if est not in {"pearson", "spearman", "kendall"}:
                raise ValueError(f"Unknown estimator: {estimator}")
            if max_tau is not None:
                raise ValueError("max_tau is only supported in config expansion; use tau.")
            if tau is None:
                raise ValueError("LaggedCorrelation requires tau.")

            self._estimator = est
            self._squared = bool(squared)
            if self._squared:
                self.issigned = lambda: False
                self.labels = LaggedCorrelation.labels + ["unsigned"]
                suffix = "-sq"
            else:
                self.labels = LaggedCorrelation.labels + ["signed"]
                suffix = ""
            self._tau = int(tau)
            if self._tau < 0:
                raise ValueError("tau must be >= 0.")
            self.identifier = f"corr_{est}_tau-{self._tau}{suffix}"

        def _corr(self, x, y):
            x = np.asarray(x).reshape(-1)
            y = np.asarray(y).reshape(-1)
            if x.size < 2 or y.size < 2:
                return np.nan
            if self._estimator == "pearson":
                return stats.pearsonr(x, y).correlation
                # return np.corrcoef(x, y)[0, 1]
            if self._estimator == "spearman":
                return stats.spearmanr(x, y).correlation
            if self._estimator == "kendall":
                return stats.kendalltau(x, y).correlation
            raise ValueError(f"Unknown estimator: {self._estimator}")

        def _lagged_corr(self, x, y, tau):
            if tau == 0:
                return self._corr(x, y)
            if tau >= x.size:
                return np.nan
            return self._corr(x[tau:], y[:-tau])

        def _symmetric_lagged_corr(self, x, y, tau):
            forward = self._lagged_corr(x, y, tau)
            backward = self._lagged_corr(y, x, tau)
            if np.isnan(forward):
                return backward
            if np.isnan(backward):
                return forward
            return 0.5 * (forward + backward)

        @parse_bivariate
        def bivariate(self, data, i=None, j=None):
            x, y = data.to_numpy()[[i, j]]
            value = self._symmetric_lagged_corr(x, y, self._tau)
            return value**2 if self._squared else value

    LaggedCorrelation.__module__ = _basic.__name__
    _basic.LaggedCorrelation = LaggedCorrelation


_patch_lagged_correlation()


def _expand_lagged_correlation_configs(configs):
    expanded = []
    for params in configs or []:
        if "max_tau" in params:
            if "tau" in params:
                raise ValueError("LaggedCorrelation config cannot set both tau and max_tau.")
            max_tau = int(params["max_tau"])
            if max_tau < 1:
                raise ValueError("max_tau must be >= 1.")
            base = {key: value for key, value in params.items() if key != "max_tau"}
            for tau in range(1, max_tau + 1):
                entry = dict(base)
                entry["tau"] = tau
                expanded.append(entry)
        else:
            expanded.append(dict(params))
    return expanded


def _patch_calculator_lagged_correlation_configs():
    if getattr(_calculator.Calculator._load_yaml, "_lagged_expansion", False):
        return

    def patched(self, document):
        print("Loading configuration file: {}".format(document))

        with open(document) as f:
            yf = yaml.load(f, Loader=yaml.FullLoader)

            # Instantiate the SPIs
            for module_name in yf:
                print("*** Importing module {}".format(module_name))
                module = importlib.import_module(module_name, _calculator.__package__)
                for fcn in yf[module_name]:
                    deps = yf[module_name][fcn].get("dependencies")
                    if deps is not None:
                        all_deps_met = all(
                            _calculator.Calculator._optional_dependencies.get(dep, False)
                            for dep in deps
                        )
                        if not all_deps_met:
                            current_base_spi = yf[module_name][fcn]
                            print(
                                "Optional dependencies: {} not met. Skipping {} SPI(s):".format(
                                    deps, len(current_base_spi.get("configs"))
                                )
                            )
                            for params in current_base_spi.get("configs"):
                                print(
                                    f"*SKIPPING SPI: {module_name}.{fcn}(x,y,{params})..."
                                )
                                self._excluded_spis.append([f"{fcn}(x,y,{params})", deps])
                            continue
                    try:
                        configs = yf[module_name][fcn].get("configs")
                        if fcn == "LaggedCorrelation" and configs is not None:
                            configs = _expand_lagged_correlation_configs(configs)
                        for params in configs:
                            print(
                                f"[{self.n_spis}] Adding SPI {module_name}.{fcn}(x,y,{params})"
                            )
                            spi = getattr(module, fcn)(**params)
                            self._spis[spi.identifier] = spi
                            print(
                                'Succesfully initialised SPI with identifier "{}" and labels {}'.format(
                                    spi.identifier, spi.labels
                                )
                            )
                    except TypeError:
                        print(f"[{self.n_spis}] Adding SPI {module_name}.{fcn}(x,y)...")
                        spi = getattr(module, fcn)()
                        self._spis[spi.identifier] = spi
                        print(
                            'Succesfully initialised SPI with identifier "{}" and labels {}'.format(
                                spi.identifier, spi.labels
                            )
                        )

    patched._lagged_expansion = True
    _calculator.Calculator._load_yaml = patched


_patch_calculator_lagged_correlation_configs()


# ---------------------------------------------------------------------------
# Vectorized multivariate patches (1-5)
# Replace M×M bivariate loops with batch numpy/scipy operations.
# ---------------------------------------------------------------------------

def _patch_spearman_multivariate():
    """SpearmanR: scipy.stats.spearmanr on full (M, T) matrix → (M, M) at once."""
    @parse_multivariate
    def multivariate(self, data):
        Z = data.to_numpy(squeeze=True)  # (M, T)
        rho, _ = stats.spearmanr(Z, axis=1)
        if Z.shape[0] == 2:
            # spearmanr returns scalar for 2 variables
            rho = np.array([[1.0, rho], [rho, 1.0]])
        if self._squared:
            rho = rho ** 2
        np.fill_diagonal(rho, np.nan)
        return rho

    _basic.SpearmanR.multivariate = multivariate


_patch_spearman_multivariate()


def _patch_kendall_multivariate():
    """KendallTau: pandas .corr(method='kendall') on (T, M) DataFrame → (M, M)."""
    @parse_multivariate
    def multivariate(self, data):
        Z = data.to_numpy(squeeze=True)  # (M, T)
        df = pd.DataFrame(Z.T)
        tau = df.corr(method="kendall").values
        if self._squared:
            tau = tau ** 2
        np.fill_diagonal(tau, np.nan)
        return tau

    _basic.KendallTau.multivariate = multivariate


_patch_kendall_multivariate()


def _patch_lagged_correlation_multivariate():
    """LaggedCorrelation: np.corrcoef on time-shifted arrays, symmetrized."""
    @parse_multivariate
    def multivariate(self, data):
        Z = data.to_numpy(squeeze=True)  # (M, T)
        M, T = Z.shape
        tau = self._tau

        if tau == 0 or tau >= T:
            if tau >= T:
                return np.full((M, M), np.nan)
            # tau == 0: plain correlation
            if self._estimator == "pearson":
                C = np.corrcoef(Z)
            elif self._estimator == "spearman":
                C, _ = stats.spearmanr(Z, axis=1)
                if M == 2:
                    C = np.array([[1.0, C], [C, 1.0]])
            elif self._estimator == "kendall":
                C = pd.DataFrame(Z.T).corr(method="kendall").values
            else:
                raise ValueError(f"Unknown estimator: {self._estimator}")
            if self._squared:
                C = C ** 2
            np.fill_diagonal(C, np.nan)
            return C

        # tau > 0: forward = corr(x[tau:], y[:-tau]), backward = corr(y[tau:], x[:-tau])
        # Symmetric average: 0.5 * (forward + backward)
        Z_lead = Z[:, tau:]    # x[tau:]
        Z_lag = Z[:, :-tau]    # y[:-tau]  (when computing corr(x_i[tau:], y_j[:-tau]))

        if self._estimator == "pearson":
            # Stack [Z_lead; Z_lag] and compute full (2M, 2M) correlation
            stacked = np.vstack([Z_lead, Z_lag])  # (2M, T-tau)
            C_full = np.corrcoef(stacked)  # (2M, 2M)
            # Forward: C_full[i, M+j] = corr(x_i[tau:], y_j[:-tau])
            # Backward: C_full[M+i, j] = corr(x_i[:-tau], y_j[tau:])
            forward = C_full[:M, M:]
            backward = C_full[M:, :M]
        elif self._estimator == "spearman":
            stacked = np.vstack([Z_lead, Z_lag])
            rho, _ = stats.spearmanr(stacked, axis=1)
            if stacked.shape[0] == 2:
                rho = np.array([[1.0, rho], [rho, 1.0]])
            forward = rho[:M, M:]
            backward = rho[M:, :M]
        elif self._estimator == "kendall":
            stacked = np.vstack([Z_lead, Z_lag])
            df = pd.DataFrame(stacked.T)
            C_full = df.corr(method="kendall").values
            forward = C_full[:M, M:]
            backward = C_full[M:, :M]
        else:
            raise ValueError(f"Unknown estimator: {self._estimator}")

        # Symmetric average
        C = 0.5 * (forward + backward)
        if self._squared:
            C = C ** 2
        np.fill_diagonal(C, np.nan)
        return C

    _basic.LaggedCorrelation.multivariate = multivariate


_patch_lagged_correlation_multivariate()


def _patch_cross_correlation_multivariate():
    """CrossCorrelation: precompute all pairwise xcorr, then apply statistic.

    Matches original signal.correlate normalization exactly.
    Avoids redundant per-pair caching overhead.
    """
    from scipy.signal import fftconvolve

    @parse_multivariate
    def multivariate(self, data):
        Z = data.to_numpy(squeeze=True)  # (M, T)
        M, T = Z.shape
        result = np.full((M, M), np.nan)

        stds = Z.std(axis=1)
        stds[stds == 0] = 1.0
        quarter = T // 4

        for i in range(M):
            for j in range(i + 1, M):
                # Exact match of original: signal.correlate(x, y, "full")
                # fftconvolve(x, y[::-1]) == correlate(x, y)
                r_ij = fftconvolve(Z[i], Z[j, ::-1], mode="full")
                r_ij = r_ij / (stds[i] * stds[j] * (T - 1))
                # Truncate to T/4 around center
                r_ij = r_ij[T - 1 - quarter : T - 1 + quarter]

                if self._sigonly:
                    N = len(r_ij) // 2
                    if N > 0:
                        threshold = 1.96 / np.sqrt(N)
                        fwd = np.where(np.abs(r_ij[N:]) <= threshold)[0]
                        fzf = fwd[0] if len(fwd) > 0 else len(r_ij) - N
                        bwd = np.where(np.abs(r_ij[:N]) <= threshold)[0]
                        fzr = bwd[-1] if len(bwd) > 0 else 0
                        r_ij = r_ij[N - fzr : N + fzf]

                if self._statistic == "max":
                    val = np.max(r_ij ** 2) if self._squared else np.max(r_ij)
                elif self._statistic == "mean":
                    val = np.mean(r_ij ** 2) if self._squared else np.mean(r_ij)
                else:
                    val = np.max(r_ij ** 2) if self._squared else np.max(r_ij)

                result[i, j] = val
                result[j, i] = val

        return result

    _basic.CrossCorrelation.multivariate = multivariate


_patch_cross_correlation_multivariate()


def _patch_dtw_multivariate():
    """DTW: dtaidistance.dtw.distance_matrix for batch computation."""
    try:
        from dtaidistance import dtw as _dtw_c
    except ImportError:
        return

    @parse_multivariate
    def multivariate(self, data):
        Z = data.to_numpy(squeeze=True)  # (M, T)
        M = Z.shape[0]
        T = Z.shape[1]
        series = [np.ascontiguousarray(Z[i], dtype=np.double) for i in range(M)]

        constraint = self._global_constraint

        if constraint == "itakura":
            # dtaidistance doesn't support itakura; fall back to bivariate loop
            A = np.full((M, M), np.nan)
            for i in range(M):
                for j in range(i + 1, M):
                    d = _distance.tslearn.metrics.dtw(
                        series[i], series[j], global_constraint="itakura"
                    )
                    A[i, j] = d
                    A[j, i] = d
            return A

        kwargs = {"use_c": True, "compact": False}
        if constraint == "sakoe_chiba":
            n = min(len(s) for s in series)
            radius = self._resolve_radius(n) if hasattr(self, '_resolve_radius') else _auto_sakoe_radius(n)
            kwargs["window"] = radius + 1

        try:
            dm = _dtw_c.distance_matrix(series, **kwargs)
        except Exception:
            # Fallback: try without C
            kwargs["use_c"] = False
            dm = _dtw_c.distance_matrix(series, **kwargs)

        # distance_matrix returns full (M, M) with 0 on diagonal and inf for not-computed
        # It only computes upper triangle; mirror it
        dm = np.array(dm)
        # Make symmetric (lower triangle might be 0 or inf)
        mask_upper = np.triu(np.ones((M, M), dtype=bool), k=1)
        dm_sym = np.where(mask_upper, dm, dm.T)
        np.fill_diagonal(dm_sym, np.nan)
        return dm_sym

    _distance.DynamicTimeWarping.multivariate = multivariate


_patch_dtw_multivariate()


# ---------------------------------------------------------------------------
# Vectorized info-theoretic patches (7): Gaussian estimator replacements
# Replace JIDT Gaussian MI/JE/CE/TLMI with pure numpy analytical formulas.
# Only applies to estimator="gaussian". KSG/kernel still use JIDT.
# ---------------------------------------------------------------------------

def _patch_gaussian_mutual_info():
    """MI_gaussian: MI(X;Y) = -0.5 * ln(1 - r^2) where r = Pearson correlation."""
    @parse_multivariate
    def multivariate(self, data):
        Z = data.to_numpy(squeeze=True)  # (M, T)
        R = np.corrcoef(Z)  # (M, M) Pearson correlation
        r2 = np.clip(R ** 2, 0, 1 - 1e-15)
        MI = -0.5 * np.log(1 - r2)
        np.fill_diagonal(MI, np.nan)
        return MI

    # Only patch instances with gaussian estimator — done at compute time
    _infotheory.MutualInfo._gaussian_multivariate = multivariate


def _patch_gaussian_joint_entropy():
    """JE_gaussian: H(X,Y) = ln(2*pi*e) + 0.5*ln(var_x) + 0.5*ln(var_y) + 0.5*ln(1 - r^2)."""
    @parse_multivariate
    def multivariate(self, data):
        Z = data.to_numpy(squeeze=True)  # (M, T)
        M = Z.shape[0]
        R = np.corrcoef(Z)
        variances = np.var(Z, axis=1, ddof=0)
        log_var = np.log(np.maximum(variances, 1e-300))
        r2 = np.clip(R ** 2, 0, 1 - 1e-15)

        # H(X_i, X_j) = ln(2*pi*e) + 0.5*ln(var_i) + 0.5*ln(var_j) + 0.5*ln(1 - r_ij^2)
        JE = (np.log(2 * np.pi * np.e)
              + 0.5 * log_var[:, None]
              + 0.5 * log_var[None, :]
              + 0.5 * np.log(1 - r2))
        np.fill_diagonal(JE, np.nan)
        return JE

    _infotheory.JointEntropy._gaussian_multivariate = multivariate


def _patch_gaussian_conditional_entropy():
    """CE_gaussian: H(Y|X) = H(X,Y) - H(X).
    H(X) = 0.5*ln(2*pi*e*var_x).
    CE is directed: result[i,j] = H(j|i).
    """
    @parse_multivariate
    def multivariate(self, data):
        Z = data.to_numpy(squeeze=True)  # (M, T)
        M = Z.shape[0]
        R = np.corrcoef(Z)
        variances = np.var(Z, axis=1, ddof=0)
        log_var = np.log(np.maximum(variances, 1e-300))
        r2 = np.clip(R ** 2, 0, 1 - 1e-15)

        # H(X_i) = 0.5 * ln(2*pi*e*var_i)
        H_marginal = 0.5 * np.log(2 * np.pi * np.e * np.maximum(variances, 1e-300))

        # H(X_j, X_i) = ln(2*pi*e) + 0.5*ln(var_j) + 0.5*ln(var_i) + 0.5*ln(1 - r^2)
        JE = (np.log(2 * np.pi * np.e)
              + 0.5 * log_var[:, None]
              + 0.5 * log_var[None, :]
              + 0.5 * np.log(1 - r2))

        # CE[i, j] = H(j | i) = H(i, j) - H(i)
        CE = JE - H_marginal[:, None]
        np.fill_diagonal(CE, np.nan)
        return CE

    _infotheory.ConditionalEntropy._gaussian_multivariate = multivariate


def _patch_gaussian_time_lagged_mi():
    """TLMI_gaussian: MI between x[:-1] and y[1:], using Gaussian formula."""
    @parse_multivariate
    def multivariate(self, data):
        Z = data.to_numpy(squeeze=True)  # (M, T)
        M, T = Z.shape
        Z_src = Z[:, :-1]   # sources: x_i(t)
        Z_tgt = Z[:, 1:]    # targets: y_j(t+1)

        # Stack and compute full correlation
        stacked = np.vstack([Z_src, Z_tgt])  # (2M, T-1)
        R = np.corrcoef(stacked)  # (2M, 2M)

        # Cross-block: R[i, M+j] = corr(x_i[:-1], y_j[1:])
        r_cross = R[:M, M:]
        r2 = np.clip(r_cross ** 2, 0, 1 - 1e-15)
        TLMI = -0.5 * np.log(1 - r2)
        np.fill_diagonal(TLMI, np.nan)
        return TLMI

    _infotheory.TimeLaggedMutualInfo._gaussian_multivariate = multivariate


def _apply_gaussian_info_patches():
    """Install vectorized multivariate for Gaussian info-theoretic SPIs.

    For Gaussian estimator: uses pure numpy (no JIDT/JVM needed).
    For other estimators (kraskov, kernel): falls back to original JIDT-based method.
    """
    _patch_gaussian_mutual_info()
    _patch_gaussian_joint_entropy()
    _patch_gaussian_conditional_entropy()
    _patch_gaussian_time_lagged_mi()

    for cls in (_infotheory.MutualInfo, _infotheory.JointEntropy,
                _infotheory.ConditionalEntropy, _infotheory.TimeLaggedMutualInfo):
        if not hasattr(cls, '_gaussian_multivariate'):
            continue
        # Both orig_mv and gaussian_mv are already @parse_multivariate-wrapped.
        # The dispatch function should NOT be wrapped again — just delegate.
        orig_mv = cls.multivariate
        gaussian_mv = cls._gaussian_multivariate

        def make_dispatch(orig, fast):
            def dispatched(self, data, inplace=True):
                if getattr(self, '_estimator', None) == 'gaussian':
                    return fast(self, data, inplace=inplace)
                return orig(self, data, inplace=inplace)
            return dispatched

        cls.multivariate = make_dispatch(orig_mv, gaussian_mv)


_apply_gaussian_info_patches()


_PARALLEL_CALC = None  # module-level for fork-based sharing


def _fork_compute_spi(spi_key):
    """Worker function for fork-based parallel SPI computation.

    Runs in a forked child process. Accesses _PARALLEL_CALC via inherited
    memory (copy-on-write). Each child gets its own address space, so
    spectral cache writes don't race.
    """
    spi = _PARALLEL_CALC._spis[spi_key]
    data = _PARALLEL_CALC.dataset
    try:
        S = spi.multivariate(data)
        np.fill_diagonal(S, np.nan)
        return spi_key, S, None
    except Exception as err:
        return spi_key, np.nan, str(err)


def _patch_parallel_compute():
    """
    Patch Calculator.compute() to run SPIs in parallel using fork-based
    multiprocessing.

    Controlled by PYSPI_N_JOBS environment variable (default: 1 = sequential).

    Fork-based parallelism avoids GIL and pickling issues:
      - child processes inherit parent memory (copy-on-write)
      - SPI objects don't need to be picklable (only spi_key strings
        and numpy arrays cross the pipe)
      - each child process has its own address space (no cache races)

    JIDT (infotheory) SPIs run sequentially in the main process because
    JPype's JVM does not survive fork().
    """
    import multiprocessing as _mp
    import time as _time

    from colorama import Fore
    from tqdm import tqdm

    from pyspi.calculator import inspect_calc_results

    _original_compute = _calculator.Calculator.compute

    def parallel_compute(self):
        if not hasattr(self, "_dataset"):
            raise AttributeError(
                "Dataset not loaded yet. Please initialise with load_dataset."
            )

        n_jobs = int(os.getenv("PYSPI_N_JOBS", "1"))
        if n_jobs <= 1:
            return _original_compute(self)

        spi_keys = list(self.spis.keys())

        # JIDT SPIs use JPype/JVM which doesn't survive fork()
        # Gaussian info-theoretic SPIs are patched to use pure numpy,
        # so they can safely run in forked workers.
        def _needs_jidt(spi):
            if "infotheory" not in spi.__class__.__module__:
                return False
            return getattr(spi, '_estimator', None) != 'gaussian'

        jidt_keys = [k for k in spi_keys if _needs_jidt(self._spis[k])]
        fork_keys = [k for k in spi_keys if k not in jidt_keys]
        n_workers = min(n_jobs, len(fork_keys))

        print(
            f"[pyspi-parallel] {len(fork_keys)} SPIs via {n_workers} "
            f"fork workers, {len(jidt_keys)} JIDT SPIs sequential"
        )

        t0 = _time.time()

        # Fork-based parallel for non-JIDT SPIs
        global _PARALLEL_CALC
        _PARALLEL_CALC = self

        ctx = _mp.get_context("fork")
        with ctx.Pool(n_workers) as pool:
            pbar = tqdm(
                pool.imap_unordered(_fork_compute_spi, fork_keys),
                total=len(fork_keys),
                desc="SPIs (parallel)",
            )
            for spi_key, result, err in pbar:
                if err is not None:
                    warnings.warn(
                        f'Caught error for SPI "{spi_key}": {err}'
                    )
                self._table[spi_key] = result
                pbar.set_description(f"Done: {spi_key}")

        _PARALLEL_CALC = None

        # Sequential for JIDT SPIs (need JVM in main process)
        if jidt_keys:
            pbar = tqdm(jidt_keys, desc="SPIs (JIDT)")
            for spi_key in pbar:
                pbar.set_description(f"JIDT: {spi_key}")
                try:
                    S = self._spis[spi_key].multivariate(self.dataset)
                    np.fill_diagonal(S, np.nan)
                    self._table[spi_key] = S
                except Exception as err:
                    warnings.warn(
                        f'Caught {type(err).__name__} for SPI "{spi_key}": {err}'
                    )
                    self._table[spi_key] = np.nan

        elapsed = _time.time() - t0
        print(
            Fore.GREEN
            + f"\nCalculation complete. Time taken: {elapsed:.4f}s"
        )
        inspect_calc_results(self)

    _calculator.Calculator.compute = parallel_compute


_patch_parallel_compute()
