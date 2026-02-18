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

from pyspi.base import Undirected, Signed, parse_bivariate
from pyspi.statistics import basic as _basic
from pyspi.statistics import distance as _distance
from pyspi.statistics import spectral as _spectral
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
