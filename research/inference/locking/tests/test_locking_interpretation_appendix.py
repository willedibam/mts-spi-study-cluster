"""Frozen-prefix notebook assembly and label-blind family-sensitivity invariants."""
import ast
import hashlib
from pathlib import Path
import nbformat
import numpy as np

from research.inference.benchmarks.build_lean_interpretation_appendix import SOURCE, TARGET, build
from research.inference.locking.locking_feature_sensitivity_261007 import feature_masks, project


def test_family_masks_are_catalogue_defined():
    names = ['cov_EmpiricalCovariance', 'corr_pearson_tau-2', 'xcorr_max_sig-True',
             'cohmag_multitaper_mean', 'plv_multitaper_mean', 'te_kraskov']
    no_lag, temporal, a, b = feature_masks(names)
    assert temporal.tolist() == [False, True, True, True, True, False]
    assert all(no_lag[k] == (a[k] not in (1,2) and b[k] not in (1,2)) for k in range(len(a)))
    assert no_lag.sum() == 6


def test_projection_uses_only_development_data():
    rng = np.random.default_rng(71)
    x = rng.normal(size=(24,9))
    x[:,0] = 3
    x[0,1] = np.nan
    fit = np.arange(24) < 16
    altered = x.copy()
    altered[~fit] = rng.normal(size=(8,9))*100
    for standard in (False, True):
        q, _, keep, w, transformed, _ = project(x, fit, standard)
        q2, _, keep2, w2, _, _ = project(altered, fit, standard)
        assert not keep[0] and not keep[1]
        np.testing.assert_array_equal(keep, keep2)
        np.testing.assert_allclose(q[fit], q2[fit], atol=1e-12)
        np.testing.assert_allclose(w, w2, atol=1e-12)
        # Transition attribution is additive for a fixed PC, not an importance theorem.
        delta = transformed[12:16].mean(0)-transformed[:4].mean(0)
        np.testing.assert_allclose((w*delta).sum(), q[12:16].mean()-q[:4].mean(), atol=1e-12)


def test_copy_preserves_original_cells_and_outputs():
    before = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    result, original, extra = build()
    nbformat.validate(result)
    assert result.cells[:len(original.cells)] == original.cells
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == before
    markdown = '\n'.join(c.source for c in extra if c.cell_type=='markdown')
    for required in ['Resonant oscillator', 'NeuroTycho', 'retrospective', '.041', 'not a thermodynamic',
                     'mean absolute Pearson', 'cointegration']:
        assert required in markdown
    assert r'\[' not in markdown and r'\]' not in markdown
    for c in extra:
        if c.cell_type=='code':
            ast.parse(c.source)


def test_delivered_copy_is_executed_and_preserves_prefix():
    import json
    original = nbformat.read(SOURCE,4)
    result = nbformat.read(TARGET,4)
    expected=[c.source for c in original.cells]
    if 'font_fallback_repair' in result.metadata:
        expected=[s.replace("'font.serif': ['Computer Modern Roman']", "'font.serif': ['Computer Modern Roman', 'DejaVu Serif']") for s in expected]
    assert [c.source for c in result.cells[:len(original.cells)]] == expected
    if 'typography_refresh' in result.metadata:
        digest=hashlib.sha256(json.dumps(result.cells[:len(original.cells)],sort_keys=True).encode()).hexdigest()
        assert digest==result.metadata.typography_refresh.preserved_prefix_sha256
    else:
        assert result.cells[:len(original.cells)] == original.cells
    extra = result.cells[len(original.cells):]
    images = []
    for cell in extra:
        if cell.cell_type == 'code':
            assert cell.execution_count is not None
            for output in cell.outputs:
                assert output.output_type != 'error'
                if 'image/png' in output.get('data',{}):
                    images.append(output)
    assert len(images) == 18


def test_gallery_ranking_and_spearman():
    from scripts.spi_feature_gallery import column_rho, ranking_indices
    from scipy.stats import spearmanr
    x=np.array([[1,4,1],[2,1,1],[3,3,1],[4,2,1]],float)
    rho=column_rho(x,np.arange(4))
    np.testing.assert_allclose(rho[:2],[1,spearmanr(x[:,1],np.arange(4)).statistic])
    assert np.isnan(rho[2])
    rankings=ranking_indices(np.array([.1,-.9,.2]),rho)
    assert rankings['weight'].tolist()==[1,2,0]
    assert rankings['rho'].tolist()==[0,1,2]


def test_correlation_endpoint_definition():
    from scripts.spi_feature_gallery import correlation_endpoint
    for name in ['corr_pearson_tau-20-sq','spearmanr','kendalltau-sq','xcorr-sq_max_sig-True',
                 'cov_EmpiricalCovariance','cov-sq_EmpiricalCovariance']:
        assert correlation_endpoint(name)
    for name in ['dcorr','pec','cohmag_multitaper_mean','cov_LedoitWolf','prec_EmpiricalCovariance','te_kraskov']:
        assert not correlation_endpoint(name)


def test_m24_edge_profiles_use_all_directions_and_label_blind_selection():
    import pandas as pd
    from research.inference.locking.plot_locking_m24_edge_profiles import community_masks,choose_records
    i,j=np.where(~np.eye(24,dtype=bool));masks=community_masks(i,j)
    assert [int(m.sum()) for m in masks]==[56,56,56,128,128,128]
    np.testing.assert_array_equal(np.sum(masks,axis=0),np.ones(552))
    rows=pd.DataFrame([dict(row_id=f'{g}-{s}',control=g,eta=eta,system='random-noise',role='evaluation',
                           comparison_eligible=True,Q=s,z_PC1=s) for g in [0.,.1,.14] for s,eta in [(1,.3),(2,.59),(3,.85)]])
    selected=[r.row_id for r in choose_records(rows)]
    changed=rows.assign(Q=-rows.Q,z_PC1=rows.z_PC1*100)
    assert selected==[r.row_id for r in choose_records(changed)]
    assert selected==['0.0-2','0.1-2','0.14-2']


def test_m24_edge_profile_replay():
    import json
    from scripts.spi_feature_gallery import DEFAULT_OUT
    audit=json.loads((DEFAULT_OUT/'edge-profile-audit.json').read_text())
    assert audit['panels']==15 and audit['ordered_edges_per_panel']==552
    assert audit['max_replay_error']<1e-6


def test_correlation_composition_preserves_global_ranking():
    import pandas as pd
    from scripts.spi_feature_gallery import DEFAULT_OUT,correlation_endpoint
    all_rows=pd.read_csv(DEFAULT_OUT/'all-feature-rankings.csv')
    pairs=pd.read_csv(DEFAULT_OUT/'correlation-pair-rankings.csv')
    others=pd.read_csv(DEFAULT_OUT/'top20-at-most-one-correlation.csv')
    both=all_rows.spi_a.map(correlation_endpoint)&all_rows.spi_b.map(correlation_endpoint)
    assert len(pairs)==both.sum()
    assert pairs.rank_weight.is_monotonic_increasing
    np.testing.assert_array_equal(others.feature_index,all_rows[~both].sort_values('rank_weight').head(20).feature_index)
    np.testing.assert_allclose(others.weight,all_rows.set_index('feature_index').loc[others.feature_index,'weight'])


def test_composite_limits_preserve_uncertainty_and_original_values():
    from research.inference.locking.plot_locking_composites_261008 import independent_limits
    mean=np.array([5.,6.,8.]);low=np.array([1.,4.,5.]);high=np.array([7.,9.,13.])
    originals=[a.copy() for a in (mean,low,high)]
    lo,hi=independent_limits(mean,low,high)
    assert lo<1 and hi>13
    for a,b in zip((mean,low,high),originals):np.testing.assert_array_equal(a,b)
    lo,hi=independent_limits([3],[3],[3]);assert lo<3<hi


def test_snapshot_selection_does_not_use_q_or_Q():
    import pandas as pd
    from research.inference.locking.plot_locking_composites_261008 import choose_records
    rows=pd.DataFrame([dict(row_id=f'{g}-{s}',seed=s,control=g,role='evaluation',comparison_eligible=True,
                           eta=eta,Q_lock=s/10,z_PC1=s) for g in [0.,.1,.2] for s,eta in [(8,.3),(9,.61),(10,.8)]])
    expected=[r.row_id for r in choose_records(rows,'eta',.6)]
    changed=rows.assign(Q_lock=100-rows.Q_lock,z_PC1=-100*rows.z_PC1)
    assert [r.row_id for r in choose_records(changed,'eta',.6)]==expected
    assert all(r.seed==8 for r in choose_records(rows))


def test_latex_typography_preserves_math_and_escapes_prose():
    from scripts.benchmark_latex_style import latex_text, latex_style
    import matplotlib as mpl
    original=r'SPI_name × TE · 90%: $z_{ab}$ → $Q$'
    expected=r'SPI\_name $\times$ TE · 90\%: $z_{ab}$ $\rightarrow$ $Q$'
    assert latex_text(original)==expected
    assert latex_text(expected)==expected
    with mpl.rc_context():
        latex_style()
        assert mpl.rcParams['text.usetex']
        assert mpl.rcParams['font.serif']==['Computer Modern Roman','DejaVu Serif']


def test_latex_has_native_fallback_without_font_warning(caplog):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.texmanager import TexManager
    from scripts.benchmark_latex_style import latex_style
    with mpl.rc_context():
        latex_style()
        assert 'DejaVuSerif' in font_manager.findfont(font_manager.FontProperties(family='serif'),fallback_to_default=False)
        # No Times/Palatino switch: TeX's serif remains Computer Modern.
        # Asserted as the absence of a font package: the Computer Modern
        # package matplotlib loads differs between releases (type1ec, type1cm).
        source = TexManager._get_tex_source('test',9)
        assert not any(package in source for package in ('mathptmx', 'mathpazo'))
        fig,ax=plt.subplots(figsize=(2,2));ax.set_title(r'LaTeX $z_{ab}$');fig.canvas.draw();plt.close(fig)
    assert not any('findfont' in record.message for record in caplog.records)


def test_delivered_notebook_has_no_stale_computer_modern_warnings():
    result=nbformat.read(TARGET,4)
    assert not any('findfont: Generic family' in o.get('text','') for c in result.cells for o in c.get('outputs',[]))
