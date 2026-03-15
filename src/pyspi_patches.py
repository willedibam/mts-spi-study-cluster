from __future__ import annotations

import importlib
import os
import warnings

import numpy as np
from scipy import stats
from scipy.spatial import cKDTree
from scipy.special import digamma, gammaln
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

from pyspi.base import Undirected, Signed, parse_bivariate, parse_multivariate, parse_univariate
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


# CrossCorrelation: no multivariate patch.
# The sigonly truncation is direction-dependent (a pyspi quirk where the
# IndexError fallback in safe_bivariate makes results depend on pair ordering).
# Vectorizing would change output values. The bivariate loop with caching is
# already reasonably fast for this SPI (only 1 config in our setup).


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
# CrossPairwiseDistance: custom SPI — lagged Euclidean distance
# For pair (i,j): compute Euclidean distance at each lag t in 0..tau,
# symmetric (min of fwd and bwd directions), report min or mean over t.
# tau=0 returns identical value to pdist_euclidean (L2 norm, not RMSE).
# ---------------------------------------------------------------------------

def _patch_cross_pairwise_distance():
    if hasattr(_distance, "CrossPairwiseDistance"):
        return

    class CrossPairwiseDistance(Undirected):
        name = "Cross pairwise distance"
        labels = ["distance", "nonlinear", "undirected", "temporal"]

        def __init__(self, metric="euclidean", tau=1, statistic="min"):
            if metric != "euclidean":
                raise ValueError(f"Unsupported metric: {metric!r}. Only 'euclidean' supported.")
            if int(tau) < 0:
                raise ValueError("tau must be >= 0.")
            stat = str(statistic).lower()
            if stat not in {"min", "mean"}:
                raise ValueError(f"statistic must be 'min' or 'mean', got: {statistic!r}")
            self._metric = metric
            self._tau = int(tau)
            self._statistic = stat
            self.identifier = f"xpdist_{metric}_tau-{self._tau}_{stat}"

        @staticmethod
        def _dist(a, b):
            """L2 (Euclidean) distance. At tau=0 matches pdist_euclidean exactly.
            WARNING: at tau>0 vectors shrink (length T-t), so L2 decreases with
            lag regardless of similarity — min will spuriously prefer the largest lag.
            Use RMSE (commented below) to normalise across lags.
            """
            return np.sqrt(np.sum((a - b) ** 2))
            # return np.sqrt(np.mean((a - b) ** 2))  # RMSE: length-normalised, lag-comparable

        def _cross_dist(self, x, y):
            """Compute Euclidean distance at each lag 0..tau (symmetric), return min or mean."""
            tau = self._tau
            per_lag = np.empty(tau + 1)
            per_lag[0] = self._dist(x, y)
            for t in range(1, tau + 1):
                if t >= len(x):
                    per_lag[t] = np.inf
                    continue
                fwd = self._dist(x[t:], y[:-t])
                bwd = self._dist(y[t:], x[:-t])
                per_lag[t] = min(fwd, bwd)
            if self._statistic == "min":
                return float(np.min(per_lag))
            finite = per_lag[np.isfinite(per_lag)]
            return float(np.mean(finite)) if finite.size > 0 else np.nan

        @parse_bivariate
        def bivariate(self, data, i=None, j=None):
            Z = data.to_numpy(squeeze=True)
            return self._cross_dist(Z[i], Z[j])

        @parse_multivariate
        def multivariate(self, data):
            Z = data.to_numpy(squeeze=True)  # (M, T)
            M = Z.shape[0]
            A = np.full((M, M), np.nan)
            for i in range(M):
                for j in range(i + 1, M):
                    d = self._cross_dist(Z[i], Z[j])
                    A[i, j] = d
                    A[j, i] = d
            return A

    CrossPairwiseDistance.__module__ = _distance.__name__
    _distance.CrossPairwiseDistance = CrossPairwiseDistance


_patch_cross_pairwise_distance()


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
        variances = np.var(Z, axis=1, ddof=1)
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
        variances = np.var(Z, axis=1, ddof=1)
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


# ---------------------------------------------------------------------------
# KSG (Kraskov) MI estimator — pure numpy/scipy, no JIDT
# ---------------------------------------------------------------------------

def _ksg_mi_pair(x, y, k, w, tree_x, tree_y):
    """KSG Estimator 1 MI for a single pair.

    Args:
        x, y    : 1-D float arrays, length N (z-scored)
        k       : number of nearest neighbours
        w       : Theiler window (exclude |j-i| <= w from kNN search)
        tree_x  : prebuilt cKDTree for x (shape N×1), reused across pairs
        tree_y  : prebuilt cKDTree for y (shape N×1), reused across pairs

    Returns:
        MI estimate (float, clamped to 0)
    """
    N = len(x)
    xy = np.column_stack([x, y])
    tree_xy = cKDTree(xy)

    if w == 0:
        # Vectorized path: query k+1 neighbours (index 0 = self), take k-th distance
        dists, _ = tree_xy.query(xy, k=k + 1, p=np.inf)
        eps = dists[:, k]  # (N,) — strictly positive for k >= 1

        # Count marginal neighbours strictly within eps (open ball), subtract self.
        # KSG uses strict < comparison; cKDTree uses <=, so shrink radius slightly.
        eps_strict = eps * (1.0 - 1e-10)
        nx_lists = tree_x.query_ball_point(x.reshape(-1, 1), eps_strict, p=np.inf)
        ny_lists = tree_y.query_ball_point(y.reshape(-1, 1), eps_strict, p=np.inf)
        n_x = np.array([len(lst) - 1 for lst in nx_lists], dtype=np.float64)
        n_y = np.array([len(lst) - 1 for lst in ny_lists], dtype=np.float64)
    else:
        # Per-point loop path: over-query joint tree then exclude Theiler window
        n_query = min(k + 2 * w + 2, N)
        dists_all, idx_all = tree_xy.query(xy, k=n_query, p=np.inf)

        eps = np.empty(N)
        n_x = np.empty(N)
        n_y = np.empty(N)

        for i in range(N):
            # Filter out indices within Theiler window
            valid = np.abs(idx_all[i] - i) > w
            valid[0] = False  # always exclude self (index 0 is i itself)
            d_valid = dists_all[i][valid]

            if len(d_valid) < k:
                # Fallback: expand search exhaustively
                all_dists = np.max(np.abs(xy - xy[i]), axis=1)
                all_dists[max(0, i - w): i + w + 1] = np.inf
                all_dists[i] = np.inf
                d_valid = np.sort(all_dists)
                d_valid = d_valid[np.isfinite(d_valid)]

            e = d_valid[k - 1] if len(d_valid) >= k else np.inf
            eps[i] = e

            # Count marginal neighbours within eps
            ix = tree_x.query_ball_point([[x[i]]], e, p=np.inf)[0]
            iy = tree_y.query_ball_point([[y[i]]], e, p=np.inf)[0]
            # Exclude temporal neighbours from marginal counts too
            n_x[i] = sum(1 for j in ix if abs(j - i) > w and j != i)
            n_y[i] = sum(1 for j in iy if abs(j - i) > w and j != i)

    mi = digamma(k) - np.mean(digamma(n_x + 1) + digamma(n_y + 1)) + digamma(N)
    return float(mi)


def _patch_kraskov_mutual_info():
    @parse_multivariate
    def kraskov_multivariate(self, data):
        Z = data.to_numpy(squeeze=True)  # (M, T), already z-scored by pyspi.Data
        M, N = Z.shape
        k = int(self._prop_k)

        raw_w = getattr(self, '_dyn_corr_excl', None)
        if raw_w is None:
            w = 0
        elif isinstance(raw_w, str):
            warnings.warn(
                f"KSG Theiler window: _dyn_corr_excl='{raw_w}' unsupported, using w=0."
            )
            w = 0
        else:
            w = int(raw_w)

        # Build M marginal trees once; reuse for all M*(M-1)/2 pairs
        marginal_trees = [cKDTree(Z[i].reshape(-1, 1)) for i in range(M)]

        result = np.full((M, M), np.nan)
        for i in range(M):
            for j in range(i + 1, M):
                mi = _ksg_mi_pair(Z[i], Z[j], k, w,
                                  marginal_trees[i], marginal_trees[j])
                result[i, j] = result[j, i] = mi
        return result

    _infotheory.MutualInfo._kraskov_multivariate = kraskov_multivariate


def _apply_kraskov_info_patches():
    """Install KSG MI multivariate patch and extend MutualInfo dispatch."""
    _patch_kraskov_mutual_info()

    # MutualInfo.multivariate already has gaussian dispatch; extend it
    prev_mv = _infotheory.MutualInfo.multivariate
    kraskov_mv = _infotheory.MutualInfo._kraskov_multivariate

    def dispatched(self, data, inplace=True):
        if getattr(self, '_estimator', None) == 'kraskov':
            return kraskov_mv(self, data, inplace=inplace)
        return prev_mv(self, data, inplace=inplace)

    _infotheory.MutualInfo.multivariate = dispatched


_apply_kraskov_info_patches()


# ---------------------------------------------------------------------------
# KSG (Kraskov) TLMI — time-shifted MI via pure numpy/scipy
# ---------------------------------------------------------------------------

def _apply_kraskov_tlmi_patch():
    """Patch TimeLaggedMutualInfo to use KSG estimator for kraskov variant.

    TLMI(i→j) = MI(x_i[:-1], x_j[1:])  — same KSG formula on time-shifted data.
    """

    @parse_multivariate
    def kraskov_multivariate(self, data):
        Z = data.to_numpy(squeeze=True)  # (M, T)
        M, T = Z.shape
        k = int(self._prop_k)

        raw_w = getattr(self, '_dyn_corr_excl', None)
        if raw_w is None:
            w = 0
        elif isinstance(raw_w, str):
            warnings.warn(
                f"KSG Theiler window: _dyn_corr_excl='{raw_w}' unsupported, using w=0."
            )
            w = 0
        else:
            w = int(raw_w)

        Z_src = Z[:, :-1]  # x_i(t)
        Z_tgt = Z[:, 1:]   # x_j(t+1)

        # Build marginal trees for both src and tgt views
        src_trees = [cKDTree(Z_src[i].reshape(-1, 1)) for i in range(M)]
        tgt_trees = [cKDTree(Z_tgt[j].reshape(-1, 1)) for j in range(M)]

        result = np.full((M, M), np.nan)
        for i in range(M):
            for j in range(M):
                if i == j:
                    continue
                mi = _ksg_mi_pair(Z_src[i], Z_tgt[j], k, w,
                                  src_trees[i], tgt_trees[j])
                result[i, j] = mi
        return result

    _infotheory.TimeLaggedMutualInfo._kraskov_multivariate = kraskov_multivariate

    # Extend the existing dispatch (which already handles gaussian)
    prev_mv = _infotheory.TimeLaggedMutualInfo.multivariate
    kraskov_mv = _infotheory.TimeLaggedMutualInfo._kraskov_multivariate

    def dispatched(self, data, inplace=True):
        if getattr(self, '_estimator', None) == 'kraskov':
            return kraskov_mv(self, data, inplace=inplace)
        return prev_mv(self, data, inplace=inplace)

    _infotheory.TimeLaggedMutualInfo.multivariate = dispatched


_apply_kraskov_tlmi_patch()


# ---------------------------------------------------------------------------
# CrossmapEntropy pure-numpy bivariate (gaussian/kozachenko) — skip jp.JArray
# ---------------------------------------------------------------------------

def _apply_crossmap_entropy_patch():
    """Patch CrossmapEntropy.bivariate to skip jp.JArray for gaussian/kozachenko.

    The original directly calls self._entropy_calc.setObservations(jp.JArray(...)).
    For gaussian/kozachenko, our KLEntropyCalculator/patched gaussian calc
    accepts plain numpy arrays.
    """
    orig_bivariate = _infotheory.CrossmapEntropy.bivariate

    @parse_bivariate
    def numpy_bivariate(self, data, i=None, j=None):
        src, targ = data.to_numpy(squeeze=True)[[i, j]]
        k = self._history_length
        targ_future = targ[k:]
        src_past = np.expand_dims(src[k - 1: -1], axis=1)
        for idx in range(2, k):
            src_past = np.append(
                src_past, np.expand_dims(src[k - idx: -idx], axis=1), axis=1
            )

        joint = np.concatenate([src_past, np.expand_dims(targ_future, axis=1)], axis=1)

        self._entropy_calc.initialise(joint.shape[1])
        self._entropy_calc.setObservations(joint)
        H_xy = self._entropy_calc.computeAverageLocalOfObservations()

        self._entropy_calc.initialise(src_past.shape[1])
        self._entropy_calc.setObservations(src_past)
        H_y = self._entropy_calc.computeAverageLocalOfObservations()

        return H_xy - H_y

    def dispatched(self, data, i=None, j=None):
        if getattr(self, '_estimator', None) in ('gaussian', 'kozachenko'):
            return numpy_bivariate(self, data, i=i, j=j)
        return orig_bivariate(self, data, i=i, j=j)

    _infotheory.CrossmapEntropy.bivariate = dispatched


_apply_crossmap_entropy_patch()


# ---------------------------------------------------------------------------
# Kozachenko-Leonenko entropy estimator — pure numpy/scipy, no JIDT
# ---------------------------------------------------------------------------

class GaussianEntropyCalculator:
    """Drop-in replacement for JIDT's EntropyCalculatorMultiVariateGaussian.

    Formula: H = 0.5 * d * log(2πe) + 0.5 * log|Σ|
    where Σ = sample covariance (ddof=1, matching JIDT).
    """

    def __init__(self):
        self._d = None
        self._obs = None

    def initialise(self, d):
        self._d = int(d)
        self._obs = None

    def setObservations(self, data):
        self._obs = np.asarray(data, dtype=np.float64)
        if self._obs.ndim == 1:
            self._obs = self._obs.reshape(-1, 1)

    def setProperty(self, key, value):
        pass

    def computeAverageLocalOfObservations(self):
        X = self._obs
        N, d = X.shape
        cov = np.cov(X, rowvar=False, ddof=1)
        if d == 1:
            log_det = np.log(max(float(cov), 1e-300))
        else:
            sign, log_det = np.linalg.slogdet(cov)
            if sign <= 0:
                log_det = -np.inf
        return float(0.5 * d * np.log(2 * np.pi * np.e) + 0.5 * log_det)


class KLEntropyCalculator:
    """Drop-in replacement for JIDT's EntropyCalculatorMultiVariateKozachenko.

    Uses L2 (Euclidean) norm with k=1, matching JIDT's implementation.
    Formula: H = ψ(N) - ψ(1) + log(c_d) + (d/N) * Σ log(ε_i)
    where c_d = π^(d/2) / Γ(d/2 + 1) (volume of unit L2 ball).
    """

    def __init__(self):
        self._d = None
        self._obs = None

    def initialise(self, d):
        self._d = int(d)
        self._obs = None

    def setObservations(self, data):
        self._obs = np.asarray(data, dtype=np.float64)
        if self._obs.ndim == 1:
            self._obs = self._obs.reshape(-1, 1)

    def setProperty(self, key, value):
        pass  # Ignore JIDT-specific properties

    def computeAverageLocalOfObservations(self):
        X = self._obs
        N, d = X.shape
        tree = cKDTree(X)
        dists, _ = tree.query(X, k=2, p=2)  # k=1 NN (index 0 = self)
        eps = dists[:, 1]
        log_cd = (d / 2.0) * np.log(np.pi) - gammaln(d / 2.0 + 1)
        return float(
            digamma(N) - digamma(1) + log_cd + (d / N) * np.sum(np.log(eps))
        )


def _numpy_delay_embedding(x, dim):
    """Numpy equivalent of JIDT's MatrixUtils.makeDelayEmbeddingVector.

    For a 1D array x of length T, returns (T - dim + 1, dim) matrix
    where row t = [x[t], x[t-1], ..., x[t-dim+1]].
    For dim=0, returns an empty (T+1, 0) matrix (matching JIDT convention).
    """
    x = np.asarray(x).ravel()
    T = len(x)
    if dim == 0:
        return np.empty((T + 1, 0))
    return np.column_stack([x[dim - 1 - lag: T - lag] for lag in range(dim)])


def _apply_kozachenko_patches():
    """Patch kozachenko/gaussian-estimator info SPIs to use pure numpy entropy.

    Intercepts _getcalc to return numpy calculators for gaussian/kozachenko,
    avoiding JVM entirely. Also patches CausalEntropy/DirectedInfo to avoid
    JIDT MatrixUtils.
    """

    # --- 1) Intercept _getcalc to return numpy calculators ---
    _orig_getcalc = _infotheory.JIDTBase._getcalc

    def patched_getcalc(self, measure):
        est = getattr(self, '_estimator', None)
        if measure == "entropy":
            if est == 'kozachenko':
                return KLEntropyCalculator()
            if est in ('gaussian', 'kraskov'):
                # kraskov's default entropy calc is also gaussian in JIDT
                return GaussianEntropyCalculator()
        if measure == "MutualInfo" and est in ('gaussian', 'kraskov'):
            # MI calculators are not used — our patches override multivariate()
            # directly. Return a dummy that won't crash.
            return GaussianEntropyCalculator()
        return _orig_getcalc(self, measure)

    _infotheory.JIDTBase._getcalc = patched_getcalc

    # --- 2) Patch _compute_joint_entropy to skip jp.JArray for kozachenko ---
    @parse_bivariate
    def _koz_compute_joint_entropy(self, data, i, j):
        if not hasattr(data, 'joint_entropy'):
            data.joint_entropy = {}
        key = self._getkey()
        if key not in data.joint_entropy:
            data.joint_entropy[key] = np.full(
                (data.n_processes, data.n_processes), -np.inf
            )
        if data.joint_entropy[key][i, j] == -np.inf:
            x, y = data.to_numpy()[[i, j]]
            joint = np.concatenate([x, y], axis=1)
            self._entropy_calc.initialise(2)
            self._entropy_calc.setObservations(joint)
            val = self._entropy_calc.computeAverageLocalOfObservations()
            data.joint_entropy[key][i, j] = val
            data.joint_entropy[key][j, i] = val
        return data.joint_entropy[key][i, j]

    # --- 3) Patch _compute_entropy to skip jp.JArray for kozachenko ---
    @parse_univariate
    def _koz_compute_entropy(self, data, i=None):
        if not hasattr(data, 'entropy'):
            data.entropy = {}
        key = self._getkey()
        if key not in data.entropy:
            data.entropy[key] = np.full((data.n_processes,), -np.inf)
        if data.entropy[key][i] == -np.inf:
            x = np.squeeze(data.to_numpy()[i])
            self._entropy_calc.initialise(1)
            self._entropy_calc.setObservations(x)
            data.entropy[key][i] = self._entropy_calc.computeAverageLocalOfObservations()
        return data.entropy[key][i]

    # --- 4) Patch _compute_conditional_entropy to skip jp.JArray ---
    def _koz_compute_conditional_entropy(self, X, Y):
        XY = np.concatenate([X, Y], axis=1)
        self._entropy_calc.initialise(XY.shape[1])
        self._entropy_calc.setObservations(XY)
        H_XY = self._entropy_calc.computeAverageLocalOfObservations()
        self._entropy_calc.initialise(Y.shape[1])
        self._entropy_calc.setObservations(Y)
        H_Y = self._entropy_calc.computeAverageLocalOfObservations()
        return H_XY - H_Y

    # --- 5) Patch CausalEntropy._compute_causal_entropy (uses JIDT MatrixUtils) ---
    def _koz_compute_causal_entropy(self, src, targ):
        src = np.squeeze(src)
        targ = np.squeeze(targ)
        causal_entropy = 0
        for i in range(1, self._n + 1):
            Yp = _numpy_delay_embedding(targ, i - 1)[:-1]
            Xp = _numpy_delay_embedding(src, i)
            XYp = np.concatenate([Yp, Xp], axis=1)
            Yf = np.expand_dims(targ[i - 1:], 1)
            causal_entropy += self._compute_conditional_entropy(Yf, XYp)
        return causal_entropy

    # --- 6) Patch DirectedInfo._compute_entropy_rates (uses JIDT MatrixUtils) ---
    def _koz_compute_entropy_rates(self, targ):
        targ = np.squeeze(targ)
        entropy_rate_sum = 0
        for i in range(1, self._n + 1):
            Yi = _numpy_delay_embedding(targ, i)
            self._entropy_calc.initialise(i)
            self._entropy_calc.setObservations(Yi)
            entropy_rate_sum += self._entropy_calc.computeAverageLocalOfObservations() / i
        return entropy_rate_sum

    # --- 7) Gaussian pure-numpy helpers for _compute_conditional_entropy etc. ---
    # These complement the kozachenko helpers above, allowing CausalEntropy,
    # DirectedInfo, StochasticInteraction, CrossmapEntropy with gaussian estimator
    # to run without JIDT/JVM.  Formula: H(Z) = 0.5*d*log(2πe) + 0.5*log|Σ|
    # where Σ = sample covariance (ddof=1, matching JIDT).

    def _gaussian_entropy_from_data(data_2d):
        """Compute Gaussian entropy from (N, d) array. Returns scalar."""
        N, d = data_2d.shape
        cov = np.cov(data_2d, rowvar=False, ddof=1)
        if d == 1:
            log_det = np.log(max(float(cov), 1e-300))
        else:
            sign, log_det = np.linalg.slogdet(cov)
            if sign <= 0:
                log_det = -np.inf
        return 0.5 * d * np.log(2 * np.pi * np.e) + 0.5 * log_det

    def _gauss_compute_conditional_entropy(self, X, Y):
        XY = np.concatenate([X, Y], axis=1)
        return _gaussian_entropy_from_data(XY) - _gaussian_entropy_from_data(Y)

    @parse_univariate
    def _gauss_compute_entropy(self, data, i=None):
        if not hasattr(data, 'entropy'):
            data.entropy = {}
        key = self._getkey()
        if key not in data.entropy:
            data.entropy[key] = np.full((data.n_processes,), -np.inf)
        if data.entropy[key][i] == -np.inf:
            x = np.squeeze(data.to_numpy()[i])
            data.entropy[key][i] = _gaussian_entropy_from_data(x.reshape(-1, 1))
        return data.entropy[key][i]

    @parse_bivariate
    def _gauss_compute_joint_entropy(self, data, i, j):
        if not hasattr(data, 'joint_entropy'):
            data.joint_entropy = {}
        key = self._getkey()
        if key not in data.joint_entropy:
            data.joint_entropy[key] = np.full(
                (data.n_processes, data.n_processes), -np.inf
            )
        if data.joint_entropy[key][i, j] == -np.inf:
            x, y = data.to_numpy()[[i, j]]
            joint = np.concatenate([x, y], axis=1)
            val = _gaussian_entropy_from_data(joint)
            data.joint_entropy[key][i, j] = val
            data.joint_entropy[key][j, i] = val
        return data.joint_entropy[key][i, j]

    # Gaussian CausalEntropy: same logic as kozachenko but uses gaussian entropy
    def _gauss_compute_causal_entropy(self, src, targ):
        src = np.squeeze(src)
        targ = np.squeeze(targ)
        causal_entropy = 0
        for i in range(1, self._n + 1):
            Yp = _numpy_delay_embedding(targ, i - 1)[:-1]
            Xp = _numpy_delay_embedding(src, i)
            XYp = np.concatenate([Yp, Xp], axis=1)
            Yf = np.expand_dims(targ[i - 1:], 1)
            causal_entropy += self._compute_conditional_entropy(Yf, XYp)
        return causal_entropy

    # Gaussian DirectedInfo entropy rates: same logic, gaussian entropy
    def _gauss_compute_entropy_rates(self, targ):
        targ = np.squeeze(targ)
        entropy_rate_sum = 0
        for i in range(1, self._n + 1):
            Yi = _numpy_delay_embedding(targ, i)
            entropy_rate_sum += _gaussian_entropy_from_data(Yi) / i
        return entropy_rate_sum

    # --- Install dispatching wrappers on each method ---
    # Dispatch to kozachenko or gaussian pure-numpy paths,
    # otherwise fall through to original JIDT.
    orig_cje = _infotheory.JIDTBase._compute_joint_entropy
    orig_ce = _infotheory.JIDTBase._compute_entropy
    orig_cce = _infotheory.JIDTBase._compute_conditional_entropy

    def dispatch_cje(self, *args, **kwargs):
        est = getattr(self, '_estimator', None)
        if est == 'kozachenko':
            return _koz_compute_joint_entropy(self, *args, **kwargs)
        if est == 'gaussian':
            return _gauss_compute_joint_entropy(self, *args, **kwargs)
        return orig_cje(self, *args, **kwargs)

    def dispatch_ce(self, *args, **kwargs):
        est = getattr(self, '_estimator', None)
        if est == 'kozachenko':
            return _koz_compute_entropy(self, *args, **kwargs)
        if est == 'gaussian':
            return _gauss_compute_entropy(self, *args, **kwargs)
        return orig_ce(self, *args, **kwargs)

    def dispatch_cce(self, *args, **kwargs):
        est = getattr(self, '_estimator', None)
        if est in ('kozachenko', 'gaussian'):
            if est == 'kozachenko':
                return _koz_compute_conditional_entropy(self, *args, **kwargs)
            return _gauss_compute_conditional_entropy(self, *args, **kwargs)
        return orig_cce(self, *args, **kwargs)

    _infotheory.JIDTBase._compute_joint_entropy = dispatch_cje
    _infotheory.JIDTBase._compute_entropy = dispatch_ce
    _infotheory.JIDTBase._compute_conditional_entropy = dispatch_cce

    # CausalEntropy and DirectedInfo: dispatch their JIDT-dependent methods
    orig_causal = _infotheory.CausalEntropy._compute_causal_entropy
    orig_rates = _infotheory.DirectedInfo._compute_entropy_rates

    def dispatch_causal(self, src, targ):
        est = getattr(self, '_estimator', None)
        if est == 'kozachenko':
            return _koz_compute_causal_entropy(self, src, targ)
        if est == 'gaussian':
            return _gauss_compute_causal_entropy(self, src, targ)
        return orig_causal(self, src, targ)

    def dispatch_rates(self, targ):
        est = getattr(self, '_estimator', None)
        if est == 'kozachenko':
            return _koz_compute_entropy_rates(self, targ)
        if est == 'gaussian':
            return _gauss_compute_entropy_rates(self, targ)
        return orig_rates(self, targ)

    _infotheory.CausalEntropy._compute_causal_entropy = dispatch_causal
    _infotheory.DirectedInfo._compute_entropy_rates = dispatch_rates


_apply_kozachenko_patches()


# ---------------------------------------------------------------------------
# Transfer Entropy — pure numpy for gaussian (Granger) & kraskov (KSG CMI)
# Only for fixed-embedding configs. Auto-embedding falls through to JIDT.
# ---------------------------------------------------------------------------

class _DummyTECalculator:
    """Dummy TE calculator for gaussian/kraskov fixed-embedding.

    Accepts setProperty/initialise calls without JVM. Properties are stored
    but unused — our numpy bivariate bypasses this calculator entirely.
    """

    def __init__(self):
        self._props = {}

    def setProperty(self, key, value):
        self._props[key] = value

    def initialise(self):
        pass

    def setObservations(self, src, targ):
        pass

    def computeAverageLocalOfObservations(self):
        return np.nan


def _te_build_embeddings(src, targ, k_history, k_tau, l_history, l_tau):
    """Build delay embeddings for transfer entropy computation.

    Returns (Y_future, Y_past, X_past) aligned arrays.

    Y_past[t] = [targ[t], targ[t - k_tau], ..., targ[t - (k-1)*k_tau]]
    X_past[t] = [src[t], src[t - l_tau], ..., src[t - (l-1)*l_tau]]
    Y_future[t] = targ[t + 1]

    All arrays trimmed to the same valid time range.
    """
    T = len(src)
    # Maximum lookback needed
    max_lookback = max((k_history - 1) * k_tau, (l_history - 1) * l_tau)
    start = max_lookback
    end = T - 1  # need t+1 for Y_future

    if start >= end:
        return None, None, None

    # Y_future: targ[start+1 : end+1]
    Y_future = targ[start + 1: end + 1].reshape(-1, 1)

    # Y_past: for each t in [start, end), columns are targ[t], targ[t-k_tau], ...
    Y_past_cols = []
    for lag_idx in range(k_history):
        lag = lag_idx * k_tau
        Y_past_cols.append(targ[start - lag: end - lag])
    Y_past = np.column_stack(Y_past_cols)

    # X_past: for each t in [start, end), columns are src[t], src[t-l_tau], ...
    X_past_cols = []
    for lag_idx in range(l_history):
        lag = lag_idx * l_tau
        X_past_cols.append(src[start - lag: end - lag])
    X_past = np.column_stack(X_past_cols)

    return Y_future, Y_past, X_past


def _gaussian_te_bivariate(src, targ, k_history, k_tau, l_history, l_tau):
    """Gaussian TE (Granger causality) via log-determinant ratio.

    TE(X→Y) = H(Y_f | Y_p) - H(Y_f | Y_p, X_p)
            = 0.5 * [log|Σ(Y_f,Y_p)| - log|Σ(Y_p)| - log|Σ(Y_f,Y_p,X_p)| + log|Σ(Y_p,X_p)|]
    """
    Y_f, Y_p, X_p = _te_build_embeddings(src, targ, k_history, k_tau, l_history, l_tau)
    if Y_f is None:
        return np.nan

    def _slogdet(data):
        N, d = data.shape
        cov = np.cov(data, rowvar=False, ddof=1)
        if d == 1:
            return np.log(max(float(cov), 1e-300))
        sign, ld = np.linalg.slogdet(cov)
        return ld if sign > 0 else -np.inf

    YfYp = np.concatenate([Y_f, Y_p], axis=1)
    YfYpXp = np.concatenate([Y_f, Y_p, X_p], axis=1)
    YpXp = np.concatenate([Y_p, X_p], axis=1)

    te = 0.5 * (_slogdet(YfYp) - _slogdet(Y_p) - _slogdet(YfYpXp) + _slogdet(YpXp))
    return float(te)


def _kraskov_te_bivariate(src, targ, k_history, k_tau, l_history, l_tau, k_nn, w):
    """Kraskov TE via Frenzel-Pompe CMI estimator (KSG for conditional MI).

    TE(X→Y) = CMI(Y_f; X_p | Y_p)
            = ψ(k) + mean[ψ(n_Yp + 1) - ψ(n_{YfYp} + 1) - ψ(n_{XpYp} + 1)]

    Uses Chebyshev (L∞) norm in joint (Y_f, X_p, Y_p) space.
    """
    Y_f, Y_p, X_p = _te_build_embeddings(src, targ, k_history, k_tau, l_history, l_tau)
    if Y_f is None:
        return np.nan

    N = len(Y_f)

    # Joint space: (Y_f, X_p, Y_p)
    joint = np.concatenate([Y_f, X_p, Y_p], axis=1)
    tree_joint = cKDTree(joint)

    # Subspaces for counting
    YfYp = np.concatenate([Y_f, Y_p], axis=1)
    XpYp = np.concatenate([X_p, Y_p], axis=1)

    tree_YfYp = cKDTree(YfYp)
    tree_XpYp = cKDTree(XpYp)
    tree_Yp = cKDTree(Y_p)

    if w == 0:
        # Vectorized: find k-th neighbor in joint space
        dists, _ = tree_joint.query(joint, k=k_nn + 1, p=np.inf)
        eps = dists[:, k_nn]
        eps_strict = eps * (1.0 - 1e-10)

        # Count neighbors in each subspace within eps (Chebyshev)
        n_YfYp = np.array([len(lst) - 1 for lst in
                           tree_YfYp.query_ball_point(YfYp, eps_strict, p=np.inf)],
                          dtype=np.float64)
        n_XpYp = np.array([len(lst) - 1 for lst in
                           tree_XpYp.query_ball_point(XpYp, eps_strict, p=np.inf)],
                          dtype=np.float64)
        n_Yp = np.array([len(lst) - 1 for lst in
                         tree_Yp.query_ball_point(Y_p, eps_strict, p=np.inf)],
                        dtype=np.float64)
    else:
        # Per-point with Theiler window exclusion
        n_query = min(k_nn + 2 * w + 2, N)
        dists_all, idx_all = tree_joint.query(joint, k=n_query, p=np.inf)

        n_YfYp = np.empty(N)
        n_XpYp = np.empty(N)
        n_Yp = np.empty(N)

        for i in range(N):
            valid = np.abs(idx_all[i] - i) > w
            valid[0] = False
            d_valid = dists_all[i][valid]

            if len(d_valid) < k_nn:
                all_dists = np.max(np.abs(joint - joint[i]), axis=1)
                all_dists[max(0, i - w): i + w + 1] = np.inf
                all_dists[i] = np.inf
                d_valid = np.sort(all_dists)
                d_valid = d_valid[np.isfinite(d_valid)]

            e = d_valid[k_nn - 1] if len(d_valid) >= k_nn else np.inf
            e_strict = e * (1.0 - 1e-10)

            lsts_YfYp = tree_YfYp.query_ball_point(YfYp[i], e_strict, p=np.inf)
            lsts_XpYp = tree_XpYp.query_ball_point(XpYp[i], e_strict, p=np.inf)
            lsts_Yp = tree_Yp.query_ball_point(Y_p[i], e_strict, p=np.inf)

            n_YfYp[i] = sum(1 for j in lsts_YfYp if abs(j - i) > w and j != i)
            n_XpYp[i] = sum(1 for j in lsts_XpYp if abs(j - i) > w and j != i)
            n_Yp[i] = sum(1 for j in lsts_Yp if abs(j - i) > w and j != i)

    # Frenzel-Pompe CMI formula
    te = float(digamma(k_nn) + np.mean(
        digamma(n_Yp + 1) - digamma(n_YfYp + 1) - digamma(n_XpYp + 1)
    ))
    return te


def _apply_transfer_entropy_patches():
    """Patch TransferEntropy for gaussian/kraskov with fixed embedding.

    Auto-embedding configs fall through to JIDT (require AIS search).
    Kernel and symbolic estimators always fall through to JIDT.
    """

    # --- 1) Extend _getcalc for TransferEntropy ---
    _prev_getcalc = _infotheory.JIDTBase._getcalc

    def te_getcalc(self, measure):
        if measure == "TransferEntropy":
            est = getattr(self, '_estimator', None)
            if est in ('gaussian', 'kraskov'):
                return _DummyTECalculator()
        return _prev_getcalc(self, measure)

    _infotheory.JIDTBase._getcalc = te_getcalc

    # --- 2) Patch TransferEntropy.__init__ to store embedding params ---
    _orig_te_init = _infotheory.TransferEntropy.__init__

    def patched_te_init(self, auto_embed_method=None, k_search_max=None,
                        tau_search_max=None, k_history=1, k_tau=1,
                        l_history=1, l_tau=1, **kwargs):
        _orig_te_init(self, auto_embed_method=auto_embed_method,
                      k_search_max=k_search_max, tau_search_max=tau_search_max,
                      k_history=k_history, k_tau=k_tau,
                      l_history=l_history, l_tau=l_tau, **kwargs)
        # Store for numpy path
        self._auto_embed_method = auto_embed_method
        self._k_history = k_history
        self._k_tau = k_tau
        self._l_history = l_history
        self._l_tau = l_tau

    _infotheory.TransferEntropy.__init__ = patched_te_init

    # --- 3) Dispatch bivariate ---
    _orig_te_bivariate = _infotheory.TransferEntropy.bivariate

    def _resolve_theiler(self, data, i, j):
        """Resolve Theiler window for TE, matching JIDT's AUTO logic."""
        raw_w = getattr(self, '_dyn_corr_excl', None)
        if raw_w is None:
            return 0
        if raw_w == "AUTO":
            if not hasattr(data, 'theiler'):
                from pyspi import utils
                z = data.to_numpy()
                M = data.n_processes
                theiler = -np.ones((M, M))
                for _i in range(M):
                    for _j in range(_i + 1, M):
                        theiler[_i, _j] = 2 * np.dot(
                            utils.acf(z[_i]), utils.acf(z[_j])
                        )
                        theiler[_j, _i] = theiler[_i, _j]
                data.theiler = theiler
            return int(data.theiler[i, j])
        return int(raw_w)

    _infotheory.TransferEntropy._resolve_theiler = _resolve_theiler

    @parse_bivariate
    def numpy_te_bivariate(self, data, i=None, j=None):
        src, targ = data.to_numpy(squeeze=True)[[i, j]]

        if self._estimator == 'gaussian':
            return _gaussian_te_bivariate(
                src, targ, self._k_history, self._k_tau,
                self._l_history, self._l_tau
            )
        elif self._estimator == 'kraskov':
            k_nn = int(self._prop_k)
            w = self._resolve_theiler(data, i, j)
            return _kraskov_te_bivariate(
                src, targ, self._k_history, self._k_tau,
                self._l_history, self._l_tau, k_nn, w
            )
        return np.nan

    def dispatched_te_bivariate(self, data, i=None, j=None, verbose=False):
        est = getattr(self, '_estimator', None)
        auto = getattr(self, '_auto_embed_method', None)
        if est in ('gaussian', 'kraskov') and auto is None:
            return numpy_te_bivariate(self, data, i=i, j=j)
        return _orig_te_bivariate(self, data, i=i, j=j, verbose=verbose)

    _infotheory.TransferEntropy.bivariate = dispatched_te_bivariate


_apply_transfer_entropy_patches()


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
            est = getattr(spi, '_estimator', None)
            if est not in ('gaussian', 'kraskov', 'kozachenko'):
                return True
            # TE with auto-embedding still needs JIDT
            if isinstance(spi, _infotheory.TransferEntropy):
                return getattr(spi, '_auto_embed_method', None) is not None
            return False

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
