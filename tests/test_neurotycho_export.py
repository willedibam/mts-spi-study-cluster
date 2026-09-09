from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np

from scripts.export_neurotycho_views import main
from src.run_external_corpus import ExternalCorpusConfig, load_inventory, load_timeseries


def test_export_obeys_actual_external_corpus_reader(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path/'configs/external').mkdir(parents=True)
    (tmp_path/'configs/pyspi').mkdir()
    catalogue=tmp_path/'configs/pyspi/benchmarked_p90.yaml'
    catalogue.write_text('test fixture')
    source=tmp_path/'input';source.mkdir()
    x=np.random.default_rng(93).normal(size=(1,16,2000)).astype('float32')
    np.savez_compressed(source/'20110112.npz',x=x)
    record=dict(quality=dict(accepted=True),array_row=0,archive='20110112',session='Session1',
                target=0,window=0,start=100000,animal='George')
    (source/'20110112.json').write_text(json.dumps(dict(usable=True,records=[record])))
    output=tmp_path/'output'
    main(SimpleNamespace(input=source,output=output,name='test',remote=tmp_path/'remote',exclude_manifest=None))
    config=ExternalCorpusConfig.from_file(tmp_path/'configs/external/test.yaml')
    config=replace(config,archive=output/'views.npz',pyspi_config=catalogue)
    entries=load_inventory(config)
    assert [(e.M,e.T) for e in entries]==[(16,2000),(8,1000)]
    assert entries[0].labels==('George','state-0')
    for entry in entries:
        actual,_=load_timeseries(config,entry)
        expected=x[0,:entry.M,-entry.T:].T.astype(float)
        expected=(expected-expected.mean(0))/expected.std(0)
        np.testing.assert_array_equal(actual,expected)
