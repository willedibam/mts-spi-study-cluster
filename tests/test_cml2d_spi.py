"""The primary coordinate must not depend on evaluation rows or physical targets."""
import json
import numpy as np

from scripts import analyze_cml2d_spi as analysis


def test_fit_is_target_and_evaluation_blind(tmp_path, monkeypatch):
    rng = np.random.default_rng(18)
    rows = [dict(row_id=f"row{i}", r=3.84 + .01 * (i % 3), L=16, N=256,
                 seed=i // 3, M=4, T=20, view="dispersed",
                 role="development" if i < 12 else "evaluation",
                 Q_reference=float(i % 3), Q_window=float(i % 3)) for i in range(24)]
    z = np.outer(np.tile([-1., 0., 1.], 8), rng.normal(size=40))
    z += rng.normal(scale=.02, size=z.shape)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "manifest.json").write_text("{}")
    np.savez_compressed(corpus / "observations.npz",
                        **{r['row_id']: rng.normal(size=(4, 20)) for r in rows})
    monkeypatch.setattr(analysis, "assemble", lambda *args: (rows, z, ["a", "b"], []))
    first = tmp_path / "first"
    analysis.run(corpus, tmp_path, first)
    for row in rows:
        row["Q_reference"] = float(rng.normal())
        row["Q_window"] = float(rng.normal())
    z[12:] = rng.normal(size=z[12:].shape)
    second = tmp_path / "second"
    analysis.run(corpus, tmp_path, second)
    with np.load(first / "model.npz", allow_pickle=False) as a, np.load(second / "model.npz", allow_pickle=False) as b:
        for key in a.files:
            np.testing.assert_array_equal(a[key], b[key])
    with np.load(first / "features.npz", allow_pickle=False) as archive:
        assert archive["row_id"].tolist() == [r["row_id"] for r in rows]
    assert json.loads((first / "geometry.json").read_text())["passes_one_coordinate_gate"]
    # A new observation arm must retain the original centre and score units,
    # even if its own development rows have a systematic distribution shift.
    z += 3
    third = tmp_path / "frozen"
    analysis.run(corpus, tmp_path, third, frozen=first)
    with np.load(first / "model.npz", allow_pickle=False) as a, np.load(third / "model.npz", allow_pickle=False) as b:
        for key in a.files:
            np.testing.assert_array_equal(a[key], b[key])
        expected = ((z[:, a['keep']] - a['center']) @ a['component']) / a['score_scale']
    import pandas as pd
    np.testing.assert_allclose(pd.read_csv(third / "scores.csv").q, expected)
    assert json.loads((first / "summary.json").read_text())["display_sign"] == json.loads((third / "summary.json").read_text())["display_sign"]
