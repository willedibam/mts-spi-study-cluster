"""Directional share of lagged correlation, with reproducible across-window energies.

Usage: python analyze.py /path/to/neurotycho-long /path/to/state_scores.csv
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def lag_components(x, lag):
    """Upper-triangle even (edge-centred) and odd components of lagged Pearson."""
    past, future = x[:-lag], x[lag:]
    past = (past - past.mean(0)) / past.std(0)
    future = (future - future.mean(0)) / future.std(0)
    corr = past.T @ future / len(past)
    pairs = np.triu_indices(corr.shape[0], 1)
    forward, reverse = corr[pairs], corr.T[pairs]
    even = (forward + reverse) / 2
    return even - even.mean(), (forward - reverse) / 2


def energies(profiles):
    # Distinct-window products estimate the part aligned across repeated windows.
    profiles = np.asarray(profiles)
    n, edges = profiles.shape
    gram = profiles @ profiles.T / edges
    across = (gram.sum() - np.trace(gram)) / (n * (n - 1))
    return gram.diagonal(), across


def dispersed_sites(n_channels):
    return np.rint(np.linspace(0, n_channels - 2, 32)).astype(int)


def state_scores(windows, lag):
    parts = [lag_components(x, lag) for x in windows]
    even, stable_even = energies([v[0] for v in parts])
    odd, stable_odd = energies([v[1] for v in parts])
    total = stable_even + stable_odd
    return dict(window_share=np.mean(odd / (even + odd)), odd=odd.mean(),
                even=even.mean(), stable_odd=stable_odd, stable_even=stable_even,
                stable_share=stable_odd / total if total > 0 else np.nan)


def validate():
    rng = np.random.default_rng(261118)
    x = rng.normal(size=(3000, 5))
    x[:, 1] = np.roll(x[:, 0], 2) + .2 * x[:, 1]
    s, a = lag_components(x, 2)
    sr, ar = lag_components(x[::-1], 2)
    np.testing.assert_allclose(sr, s, atol=1e-12)
    np.testing.assert_allclose(ar, -a, atol=1e-12)
    z = np.corrcoef(np.r_[s+a, s-a], np.r_[s-a, s+a])[0, 1]
    np.testing.assert_allclose((1-z)/2, a @ a / (a @ a + s @ s))
    repeated, across = energies([a, a, a])
    np.testing.assert_allclose(repeated, across)
    # The same magnitude of delay with opposite orientation across two windows.
    np.testing.assert_allclose(energies([a, -a])[1], -across)


def main(bank, output):
    validate()
    manifest = json.loads((bank / "manifest.json").read_text())
    assert (manifest["fs"], manifest["T"]) == (100, 1000)
    by_unit = defaultdict(lambda: defaultdict(list))
    for row in manifest["rows"]:
        by_unit[row["unit"]][row["state"]].append(row)
    scores = []
    with np.load(bank / "windows.npz") as windows:
        for unit, states in sorted(by_unit.items()):
            wake = states["awake_ec"]
            idx = dispersed_sites(wake[0]["M"])
            power = []
            for row in wake:
                x = windows[row["id"]][:, idx].astype(float)
                power.append(np.abs(np.fft.rfft(x / x.std(0), axis=0))**2)
            reference = np.sqrt(np.mean(power, axis=0))
            reference[0] = 0
            for state, rows in sorted(states.items()):
                assert all(r["M"] == wake[0]["M"] for r in rows)
                arrays = [windows[r["id"]].astype(float) for r in rows]
                selected = [x[:, idx] for x in arrays]
                bipolar = [x[:, idx + 1] - x[:, idx] for x in arrays]
                phase_only = [np.fft.irfft(reference * np.exp(1j * np.angle(np.fft.rfft(x, axis=0))),
                                           n=len(x), axis=0) for x in selected]
                views = (("CAR", selected, (1, 2, 4)), ("bipolar", bipolar, (1, 2, 4)),
                         ("all CAR", arrays, (2,)), ("wake amplitude", phase_only, (2,)))
                for montage, view, lags in views:
                    for lag in lags:
                        result = state_scores(view, lag)
                        scores.append(dict(unit=unit, animal=rows[0]["animal"], agent=rows[0]["agent"],
                                           state=state, lag_ms=10*lag, montage=montage, n=len(rows), **result))
    pd.DataFrame(scores).to_csv(output, index=False, float_format="%.10g")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bank", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    main(args.bank, args.output)
