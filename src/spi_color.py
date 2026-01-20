from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ColorScale:
    vmin: float
    vmax: float
    center: float | None
    cmap: str


def infer_spi_color_scale(
    spi_name: str,
    data_min: float,
    data_max: float,
    labels: list[str] | None = None,
) -> ColorScale:
    """
    Infer sensible heatmap bounds/cmap for an SPI matrix based on its name, labels, and value range.

    Prefer labels from configs/meta (unsigned/signed/squared), otherwise fall back to name patterns.
    """
    name = spi_name.lower()
    label_set = {lbl.lower() for lbl in labels} if labels else set()

    dmin = float(data_min)
    dmax = float(data_max)
    if not np.isfinite(dmin):
        dmin = 0.0
    if not np.isfinite(dmax):
        dmax = 0.0

    corr_keywords = ("corr", "spearman", "kendall")
    nonneg_keywords = (
        "-sq",
        "sq_",
        "pdist",
        "dcorr",
        "hsic",
        "gwtau",
        "mi_",
        "mi-",
        "tlmi",
        "te_",
        "te-",
        "di_",
        "cce_",
        "ce_",
        "je_",
        "si_",
        "xcorr-sq",
        "dcorrx",
        "bary",
        "coh",
        "plv",
        "pli",
        "wpli",
        "dtf",
        "dcoh",
        "pdcoh",
        "gpdcoh",
        "sgc",
        "gd",
        "pec",
    )

    is_unsigned = ("unsigned" in label_set) or any(k in name for k in nonneg_keywords)
    is_signed = "signed" in label_set

    # Correlation-like SPIs (bounded)
    if any(k in name for k in corr_keywords) and "-sq" not in name and "sq_" not in name:
        return ColorScale(vmin=-1.0, vmax=1.0, center=0.0, cmap="coolwarm")

    # Covariance/precision: symmetric, clamp within [-1,1] if applicable
    if name.startswith("cov") or name.startswith("prec"):
        limit = max(abs(dmin), abs(dmax), 1e-6)
        limit = min(1.0, limit) if limit <= 1.0 else limit
        return ColorScale(vmin=-limit, vmax=limit, center=0.0, cmap="coolwarm")

    # Label-driven non-negative
    if is_unsigned and not is_signed:
        vmax = dmax if dmax > 0 else 1.0
        return ColorScale(vmin=0.0, vmax=vmax, center=None, cmap="magma")

    # Signed (label-driven) -> symmetric
    if is_signed:
        limit = max(abs(dmin), abs(dmax))
        return ColorScale(vmin=-limit, vmax=limit, center=0.0, cmap="coolwarm")

    # Name-driven non-negative families
    if any(k in name for k in nonneg_keywords):
        vmax = dmax if dmax > 0 else 1.0
        return ColorScale(vmin=0.0, vmax=vmax, center=None, cmap="magma")

    # Fallbacks based on data sign
    if dmin < 0 < dmax:
        limit = max(abs(dmin), abs(dmax))
        return ColorScale(vmin=-limit, vmax=limit, center=0.0, cmap="coolwarm")
    if dmax <= 0:
        return ColorScale(vmin=dmin, vmax=0.0, center=None, cmap="magma")
    return ColorScale(vmin=0.0, vmax=dmax, center=None, cmap="magma")
