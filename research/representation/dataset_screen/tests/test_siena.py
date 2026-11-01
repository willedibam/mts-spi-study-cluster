import numpy as np
import pytest
from research.representation.dataset_screen.siena import clocks,annotations,relative_events,montage,bipolar,NODES


def test_siena_ambiguous_clocks_do_not_become_positive_labels():
    text=('Seizure n 1\nFile name: PN10-2.edf\nRegistration start time: 09.30.15\n'
          'Seizure start time: 11.40.13\nSeizure end time: 11.41.04 opure 11.40.43\n')
    item=annotations(text)[0]
    all_events,eligible=relative_events(item,clocks('09.30.15')[0],3*3600)
    assert not eligible
    assert all_events==[(7798,7849)]
    assert clocks('1 6.49.25')==clocks('16.49.25')


def test_siena_midnight_and_out_of_record_annotations():
    item=annotations('Seizure n 1\nFile name: PN00-1.edf\nRegistration start time: 23.50.00\n'
                     'Seizure start time: 00.01.00\nSeizure end time: 00.02.00\n')[0]
    assert relative_events(item,clocks('23.50.00')[0],1200)==([(660,720)],[(660,720)])
    with pytest.raises(ValueError,match='outside EDF'):
        relative_events(item,clocks('23.50.00')[0],700)
    with pytest.raises(ValueError,match='start clocks'):
        relative_events(item,clocks('23.49.00')[0],1200)


def test_siena_uses_named_monopoles_in_the_existing_bipolar_order():
    old={'T7':'T3','T8':'T4','P7':'T5','P8':'T6'}
    names=['EKG EKG']+['EEG '+old.get(n,n) for n in reversed(NODES)]+['1']
    x=np.random.default_rng(3).normal(size=(128,len(NODES)))
    stored=np.c_[np.zeros(128),x[:,::-1],np.ones(128)]
    actual=stored[:,montage(names)]
    np.testing.assert_array_equal(actual,x)
    b=bipolar(actual)
    assert b.shape==(128,16) and np.linalg.matrix_rank(b)==14
    np.testing.assert_allclose(b[:,:4].sum(1),b[:,4:8].sum(1))
    with pytest.raises(ValueError,match='absent'):
        montage([n.replace('EEG O1','1') for n in names])
