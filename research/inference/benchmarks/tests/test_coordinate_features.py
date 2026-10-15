"""Presentation filters must not refit or silently reinterpret PC1."""
import numpy as np
import pandas as pd

from research.inference.benchmarks.coordinate_features import ranking, selections


def test_ranking_keeps_signed_coefficients_and_catalogue_coordinates():
    names=['corr_pearson_tau-1','corr_pearson_tau-2','ids','dcorr']
    ranked=ranking(names,np.array([0,2,5]),np.array([-.2,.1,.2]))
    assert ranked.feature_index.tolist()==[0,5,2]  # deterministic original-order tie
    assert ranked.weight.tolist()==[-.2,.2,.1]
    assert ranked.both_correlation.tolist()==[True,False,False]
    assert ranked.loc[1,['spi_a','spi_b']].tolist()==['ids','dcorr']


def test_filtered_display_preserves_weights_and_rejects_only_display_redundancy():
    table=pd.DataFrame(dict(feature_index=[0,1,2,3],rank=[1,2,3,4],weight=[.4,-.3,.2,.1],
                            both_correlation=[True,False,False,False]))
    z=np.array([[.2,.1,.1,.1],[.3,.4,.4,.5],[.4,np.nan,np.nan,np.nan]])
    original=table.copy(deep=True)
    full,reduced=selections({'ranks':table,'z':z})
    assert full.feature_index.tolist()==[0,1,2,3]
    assert reduced.feature_index.tolist()==[1,3]
    assert reduced.weight.tolist()==[-.3,.1]
    pd.testing.assert_frame_equal(table,original)
    # The API never consumes physical outcomes or changes the original feature bank.
    np.testing.assert_equal(z[:,1],z[:,2])


def test_notebook_exports_without_external_tex(monkeypatch, tmp_path):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.texmanager import TexManager
    from research.inference.benchmarks import three_systems

    def no_tex(*args, **kwargs):
        raise AssertionError('The coordinate notebook must not invoke TeX')

    monkeypatch.setattr(TexManager, '_run_checked_subprocess', no_tex)
    monkeypatch.setattr(three_systems, 'OUT', tmp_path)
    # Also recover from a previous notebook having enabled external TeX globally.
    with mpl.rc_context({'text.usetex': True}):
        three_systems.style()
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.plot([0, 1], [0, 1])
        ax.set(title='SI × IDS → agreement', xlabel=r'$z_{ab}$', ylabel=r'Mean $|r_{ij}|$')
        three_systems.save(fig, 'native-text')
    for suffix in ('png', 'svg', 'pdf'):
        assert (tmp_path / f'native-text.{suffix}').stat().st_size > 0
