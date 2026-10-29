"""Composite order-parameter figure: data handling must replay archives and never refit or relabel."""
from dataclasses import replace
import shutil

import numpy as np
import pandas as pd
import pytest

from research.inference.benchmarks import composite_figures as cf
from research.inference.benchmarks import hardware_baselines as hb


def test_locking_trajectory_mirrors_the_archived_generator():
    from research.inference.locking.cross_frequency_locking import simulate, STEP
    kwargs = dict(burn=5., reference=10., samples=40)
    raw, truth = simulate(.1, 3, **kwargs)
    mirror, mirror_truth, mean_field = cf.locking_trajectory(.1, 3, **kwargs, every_fine=25)
    np.testing.assert_array_equal(mirror, raw)
    assert mirror_truth == truth
    # With every_fine = STEP/dt the mean-field record lands exactly on the observed samples.
    np.testing.assert_allclose(mean_field['t'], STEP*np.arange(1, 41))
    assert mean_field['phi'].shape == (40,)


def test_pair_index_matches_the_unified_builder():
    from mtsspi.spi_spi_contract import build_unified_feature_values
    rng = np.random.default_rng(0)
    order = ['a', 'b', 'c', 'd', 'e']
    mpis = {k: rng.normal(size=(6, 6)) for k in order}
    z, _, _ = build_unified_feature_values(mpis, order)
    for i, a in enumerate(order):
        for b in order[i+1:]:
            assert abs(z[cf.pair_index(order, a, b)]-cf.edge_z(mpis[a], mpis[b])[2]) < 1e-6
            assert cf.pair_index(order, a, b) == cf.pair_index(order, b, a)


def test_ordered_contract_check_rejects_the_legacy_symmetrized_contract():
    from mtsspi.spi_spi_contract import build_unified_feature_values
    rng = np.random.default_rng(1)
    order = ['directed', 'other', 'symmetric']
    s = rng.normal(size=(7, 7))
    mpis = {'directed': rng.normal(size=(7, 7)), 'other': rng.normal(size=(7, 7)), 'symmetric': s+s.T}
    ordered, _, _ = build_unified_feature_values(mpis, order)
    assert cf.verify_ordered_row(mpis, order, ordered)['passes']
    legacy = ordered.astype(float).copy()
    legacy[cf.pair_index(order, 'directed', 'other')] = cf.upper_symmetrized_z(mpis['directed'], mpis['other'])
    assert not cf.verify_ordered_row(mpis, order, legacy)['passes']
    # Symmetric pairs are identical under both contracts, so only directed pairs discriminate.
    sym = mpis['symmetric']
    assert abs(cf.upper_symmetrized_z(sym, sym*2+1)-cf.edge_z(sym, sym*2+1)[2]) < 1e-12


def test_mean_mpi_pc1_is_fitted_on_er_rows_only():
    rng = np.random.default_rng(2)
    means = rng.normal(size=(10, 4))
    fit = np.r_[np.ones(6, bool), np.zeros(4, bool)]
    first, _, _ = hb.mean_mpi_pc1(means, fit)
    changed = means.copy()
    changed[~fit] += 100
    second, _, _ = hb.mean_mpi_pc1(changed, fit)
    np.testing.assert_allclose(first[fit], second[fit])
    # An SPI undefined on any fit record is dropped, never imputed from projected records.
    missing = means.copy()
    missing[0, 3] = np.nan
    np.testing.assert_allclose(hb.mean_mpi_pc1(missing, fit)[0], hb.mean_mpi_pc1(means[:, :3], fit)[0])


def test_mean_abs_r_uses_every_ordered_entry():
    x = np.random.default_rng(3).normal(size=(5, 200))
    r = np.corrcoef(x)
    assert abs(hb.mean_abs_r(x)-abs(r[np.triu_indices(5, 1)]).mean()) < 1e-12


def test_steepest_reports_step_location_and_share():
    part = pd.DataFrame(dict(voltage=[0, .1, .2, .3], step=[0, 20, 40, 60], value=[0., .1, .9, 1.]))
    result = hb.steepest(part, 'value')
    assert result['largest_step'] == '20→40' and result['steepest_slope'] == '20→40'
    assert abs(result['largest_step_share']-.8) < 1e-12


def test_locking_edge_groups_partition_ordered_pairs():
    groups = cf.locking_groups(8)
    assert groups.shape == (24*23,)
    assert np.bincount(groups).tolist() == [2*64, 8*7, 8*7, 24*23-128-2*56]


def test_late_channels_finds_the_slipping_minority():
    t = np.arange(5000)/500
    rng = np.random.default_rng(4)
    # Four channels wobble in phase (PLV with the core ~J0(.9) = .81); the core stays coherent overall.
    phases = np.r_[np.zeros(24), np.full(4, .9)][:, None]*np.sin(2*np.pi*.3*t)
    x = np.sin(2*np.pi*116*t+phases)+.05*rng.normal(size=(28, 5000))
    late, reference = cf.late_channels(x)
    assert late.tolist() == [24, 25, 26, 27] and reference < 24


SNR_SCORES = cf.ROOT/'results/order-parameter-inference/cross-frequency-locking-snr-261006/scores.csv'


@pytest.mark.skipif(not SNR_SCORES.exists(), reason='locking scores not local')
def test_locking_snapshot_rule_matches_the_archived_edge_profile_selection():
    from research.inference.locking.plot_locking_m24_edge_profiles import choose_records
    scores = pd.read_csv(SNR_SCORES)
    original = [r.row_id for r in choose_records(scores)]
    assert [r.row_id for r in cf.choose_locking_records(scores, (0., .10, .14))] == original
    assert cf.choose_locking_records(scores, (.20,))[0].row_id == 'xfreq21-random-noise-g200-s215'


def test_z_goes_to_the_emptiest_corner():
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2)
    points = []
    for ax in axes:
        a, b = np.random.default_rng(0).random((2, 200))*.5+.5   # everything in the upper right
        ax.set(xlim=(0, 1), ylim=(0, 1)); points.append((a, b))
    assert cf.emptiest_corner(axes, points) != 'upper right'
    plt.close(fig)


def test_registry_names_and_glyphs_resolve():
    for name, spec in cf.SYSTEMS.items():
        assert spec.name == name and spec.glyphs and all(g in cf.GLYPHS for g in spec.glyphs)
        assert len(spec.inset_boxes) == 3 and (spec.edge_groups or not spec.features)
    with pytest.raises(KeyError):
        cf.load('not-a-system')


def synthetic_system(spec, *, replicated=True, features=None):
    rng = np.random.default_rng(5)
    controls = np.repeat(np.linspace(0, .2, 6), 3 if replicated else 1)
    frame = pd.DataFrame(dict(control=controls, Q=np.tanh(20*controls)+.01, q=np.tanh(20*controls)+.1*rng.normal(size=len(controls)),
                              cohort='held', trajectory=np.tile(np.arange(3 if replicated else 1), 6)))
    frame['mean-|r|'] = rng.normal(size=len(frame))
    frame['mean-MPI'] = rng.normal(size=len(frame))
    frame['z0'] = np.tanh(10*controls-1)
    frame['z1'] = -frame.z0
    frame['z2'] = .5*frame.z0
    groups = cf.locking_groups(2)
    snapshots = []
    for c in (0., .1, .2):
        t = np.linspace(0, 10, 300)
        phase = (1-5*c)*t[None]
        glyph = dict(t=t, phase=phase, lissajous=(np.cos(t), np.cos(2*t)), synchrony=np.full(300, .5+c),
                     raster=np.angle(np.exp(1j*np.vstack([phase]*6))), raster_marked=2, window=(0., 10.),
                     trace=.6+.2*np.cos(np.pi*np.arange(300)), lattice=rng.random((16, 16)), sensors=np.arange(0, 256, 9))
        edges = {k: (rng.normal(size=30), rng.normal(size=30), .1*k) for k in (0, 1, 2)}
        snapshots.append(cf.Snapshot(control=c, label=rf'$c={c}$', row_id=f'r{c}', x=rng.normal(size=(6, 80)),
                                     glyph=glyph, edges=edges, groups=groups))
    changes = dict(heatmap_window=(0, 40), control_ticks=None, boundary=.1, Q_limits=None)
    if features is not None:
        changes['features'] = features
    return cf.SystemData(replace(spec, **changes), frame, snapshots, {'q': .9, 'mean-|r|': .1, 'mean-MPI': .2}, {},
                         primary='held')


def all_systems():
    main = synthetic_system(replace(cf.SYSTEMS['locking'], communities=3, glyphs=('wrapped', 'spiral', 'lissajous')),
                            features=cf.LOCKING_FEATURES)
    stack = [synthetic_system(replace(cf.SYSTEMS['hardware'], Q_log=False), replicated=False, features=cf.HARDWARE_FEATURES)]
    stack += [synthetic_system(cf.SYSTEMS[name]) for name in ('stuart-landau', 'kuramoto', 'cml2d')]
    return main, stack


def no_tex(monkeypatch):
    from matplotlib.texmanager import TexManager

    def blocked(*args, **kwargs):
        raise AssertionError('The native renderer must not invoke TeX')

    monkeypatch.setattr(TexManager, '_run_checked_subprocess', blocked)
    original = cf.style
    monkeypatch.setattr(cf, 'style', lambda tex=None, font=None: original(tex=False, font=font))


def test_composites_and_panels_export_without_external_tex(monkeypatch, tmp_path):
    no_tex(monkeypatch)
    main, stack = all_systems()
    for spread in cf.SPREADS:
        cf.save(cf.composite_stacked(main, stack[:3], spread=spread), f'stacked-{spread}', tmp_path)
    shown = cf.display_readouts(stack[1])
    means = shown.frame.groupby('control')[['q (display)', 'mean-|r| (display)', 'mean-MPI (display)']].mean()
    np.testing.assert_allclose(means.min(), 0, atol=1e-12)
    np.testing.assert_allclose(means.max(), 1, atol=1e-12)
    for glyph in main.spec.glyphs:
        cf.save(cf.composite([main, main], glyphs={'locking': glyph}), f'composite-{glyph}', tmp_path)
    written = cf.export_panels(main, tmp_path)
    for data in stack:
        written += cf.export_panels(data, tmp_path, stacked=True)
    for suffix in ('png', 'svg', 'pdf'):
        assert (tmp_path/f'stacked-band.{suffix}').stat().st_size > 0
    names = {p.stem for p in written}
    assert {'curves-trajectories', 'curves-band', 'heatmap-1', 'baseline-mean-abs-r', 'feature-3', 'edges-3-3',
            'panel-synchrony', 'panel-lattice', 'glyph-lattice-3'} <= names
    assert all(p.with_suffix('.svg').exists() and p.with_suffix('.pdf').exists() for p in written)


@pytest.mark.skipif(not (shutil.which('latex') and shutil.which('dvipng')), reason='LaTeX unavailable')
def test_every_registered_string_compiles_under_usetex(tmp_path):
    main, stack = all_systems()
    for data in [main, *stack]:
        assert data.spec.glyphs
    cf.save(cf.composite_stacked(main, stack[:3]), 'tex-stacked', tmp_path, formats=('pdf',))
    cf.save(cf.composite_stacked(main, stack[:3], edge_rows=(0, 1, 2)), 'tex-edges', tmp_path, formats=('pdf',))
    cf.save(cf.composite_stacked(main, [stack[0], stack[3], stack[2]], glyphs={'cml2d': 'trace'}), 'tex-alt', tmp_path,
            formats=('pdf',))
    cf.save(cf.glyph_gallery([main, *stack]), 'tex-gallery', tmp_path, formats=('pdf',))
    assert (tmp_path/'tex-stacked.pdf').stat().st_size > 0


def test_spread_modes_draw_paths_or_bands(monkeypatch):
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    no_tex(monkeypatch)
    cf.style()
    data = synthetic_system(cf.SYSTEMS['stuart-landau'])
    for spread, paths, bands in [('trajectories', 3, 0), ('band', 0, 1), ('both', 3, 1), ('none', 0, 0)]:
        fig, ax = plt.subplots()
        cf._series(ax, data, 'q', 'k', 'o', spread=spread)
        assert len(ax.lines) == paths+1
        assert sum(isinstance(c, PolyCollection) for c in ax.collections) == bands
        plt.close(fig)
    with pytest.raises(ValueError):
        cf._series(plt.subplots()[1], data, 'q', 'k', 'o', spread='bars')


def test_stacked_layout_balances_heights_and_honours_zero_curve_height():
    assert cf.stack_height('synchrony', 0) == pytest.approx(cf.stack_height('synchrony')-cf.STACK['curves'])
    main, stack = all_systems()
    rows = main.spec.edge_features
    kinds = [d.spec.glyphs[0] for d in stack[:3]]
    fixed = sum(cf.stack_height(k, 0) for k in kinds)
    left = cf.column_height(main.spec, rows)
    curves = max(cf.STACK['curves'], (left-fixed)/3)
    assert fixed+3*curves >= left-1e-12


def test_font_rendition_persists_until_changed():
    import matplotlib.pyplot as plt
    try:
        cf.style(tex=False, font='helvetica')
        cf.style(tex=False)
        assert cf._font == 'helvetica' and plt.rcParams['font.family'] == ['sans-serif']
        assert r'\usepackage{helvet}' in plt.rcParams['text.latex.preamble']
        cf.style(tex=False, font='helvetica-sansmath')
        assert 'newtxsf' in plt.rcParams['text.latex.preamble']
        with pytest.raises(ValueError):
            cf.style(font='comic')
    finally:
        cf.style(tex=False, font='cm')
    assert plt.rcParams['font.family'] == ['serif'] and 'helvet' not in plt.rcParams['text.latex.preamble']


def test_export_figure_writes_every_rendition_and_resets_font(monkeypatch, tmp_path):
    no_tex(monkeypatch)
    main, stack = all_systems()
    out = cf.export_figure(main, stack[:3], tmp_path)
    for font in cf.FONTS:
        assert all((out/font/f'figure.{ext}').stat().st_size > 0 for ext in ('svg', 'pdf', 'png'))
        assert {p.name for p in (out/font/'panels').iterdir()} == {d.spec.name for d in [main, *stack[:3]]}
    assert cf._font == 'cm' and (out/'provenance.json').exists() and '| e |' in (out/'README.md').read_text()


@pytest.mark.skipif(not (shutil.which('latex') and shutil.which('dvipng')), reason='needs latex and dvipng')
def test_helvetica_renditions_compile_under_usetex(tmp_path):
    main, stack = all_systems()
    try:
        for font in ('helvetica', 'helvetica-sansmath'):
            cf.style(font=font)
            cf.save(cf.composite_stacked(main, stack[:3]), f'tex-{font}', tmp_path, formats=('pdf',))
            assert (tmp_path/f'tex-{font}.pdf').stat().st_size > 0
    finally:
        cf.style(font='cm')
