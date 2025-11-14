from __future__ import annotations

import numpy as np
import numpy as np

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

from pyspi.statistics import spectral as _spectral


def _ensure_time_series_3d(z: np.ndarray) -> np.ndarray:
    """Ensure time-series array follows (n_time, n_trials, n_signals)."""
    if z.ndim == 2:
        return prepare_time_series(z, axis="signals")
    return z


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
