"""Presentation assembly and physical snapshot invariants."""
import ast
import hashlib
import json
import re
from pathlib import Path
import numpy as np
import nbformat

from scripts.build_lean_order_parameter_notebook import build, SOURCE
from scripts.lean_benchmark_figures import snapshots
from scripts import kuramoto_full_observation_260916 as kur


def test_lean_notebook_scope_and_source_preservation():
    before=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    notebook=build()
    nbformat.validate(notebook)
    markdown='\n'.join(c.source for c in notebook.cells if c.cell_type=='markdown')
    for name in ('Kuramoto','Stuart','Miller','Kaneko','logistic CML','Rössler'):
        assert name in markdown
    for forbidden in ('Desai','Vicsek',r'\[',r'\]'):
        assert forbidden not in markdown
    for cell in notebook.cells:
        if cell.cell_type=='code':
            ast.parse(cell.source)
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==before


def test_stuart_landau_trace_is_magnitude(tmp_path):
    folder=tmp_path/'data/order_parameter/lean_snapshots_260916'
    folder.mkdir(parents=True)
    np.savez(folder/'stuart-landau-0.725.npz', X=np.ones((32,1000)), trace=np.array([3+4j]))
    loaded=snapshots(tmp_path,'stuart-landau',[.725])[0]
    np.testing.assert_array_equal(loaded['trace'],[5])


def test_uniform_foreword_and_headlines_precede_snapshots():
    notebook=build()
    starts=[i for i,c in enumerate(notebook.cells)
            if c.cell_type=='markdown' and re.match(r'## [1-6]\. ',c.source)]
    assert len(starts)==6
    for system,start in enumerate(starts):
        stop=starts[system+1] if system<5 else len(notebook.cells)
        text=notebook.cells[start].source
        labels=['**System**','**Params**','**Phase diagram paper:**',
                '**Order-param paper:**','**Control-parameter sweep:**','**Order-parameter','**Results:**']
        positions=[text.index(label) for label in labels]
        assert positions==sorted(positions)
        source='\n'.join(c.source for c in notebook.cells[start:stop] if c.cell_type=='code')
        snapshot=min((source.find(token) for token in ['composite(', 'export_snapshots(']
                      if token in source),default=-1)
        assert snapshot>0
        assert source.index('dual_tracking(')<snapshot
        if system in (2,3):
            assert source.index('size_tracking(')<snapshot
        if system==1:
            assert source.index('T_ALPHA')<snapshot
        if system==4:
            assert source.index('Independent paired M,T confirmation')<snapshot
            assert source.index('Sampling-layout sensitivity')<snapshot


def test_kuramoto_export_orientation_and_seed_role(monkeypatch):
    from types import SimpleNamespace
    def fake(**kwargs):
        x=np.tile(np.sin(np.arange(1000)/10)[:,None],(1,32))
        a=SimpleNamespace(r_full=np.full(1000,.5),r_observed=np.full(1000,.5),
                          r_full_future=np.full(10000,.7))
        return x,a
    monkeypatch.setattr(kur,'generate_kuramoto_order_parameter',fake)
    for seed,role in [(260916008,'development'),(260916009,'evaluation')]:
        x,row=kur.simulate((1.,seed))
        assert x.shape==(32,1000) and row['role']==role
        assert np.isclose(row['Q_reference'],.7)
        assert row['reference_half_difference']==0
