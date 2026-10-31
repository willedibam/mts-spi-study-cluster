import numpy as np
import pytest

from research.representation.dataset_screen.prepare import (
    split_groups, parse_sleep, parse_seizures, control_windows, bipolar_emg, univariate, epoch, structural_split)


def test_subject_partitions_reserve_confirmation_and_ignore_input_order():
    people = [f'p{i:02d}' for i in range(43)]
    a = split_groups(people, 24, 8)
    assert a == split_groups(people[::-1], 24, 8)
    assert [len(a[k]) for k in ['source', 'screen', 'confirmation_reserved']] == [24, 8, 11]
    assert not set(a['source']) & set(a['screen'])
    assert not (set(a['source']) | set(a['screen'])) & set(a['confirmation_reserved'])


def test_structural_configurations_stay_separate_with_both_mechanisms():
    from research.representation.dataset_screen.common import folds
    split = structural_split()
    assert [len(split[r]) for r in ['source','screen','confirmation_reserved']] == [9,4,4]
    assert sorted(sum([split[r] for r in ['source','screen','confirmation_reserved']], [])) == list(range(1,18))
    for role in ['source','screen','confirmation_reserved']:
        assert {int(s >= 10) for s in split[role]} == {0,1}
    groups = np.repeat(split['source'],10)
    y = (groups >= 10).astype(int)
    for train, valid in folds(y, groups):
        assert set(groups[train]).isdisjoint(groups[valid])
        assert set(y[train]) == set(y[valid]) == {0,1}


def test_sleep_annotations_use_elapsed_time_not_midnight_clock():
    text = ('Date, Time, Recording onset, Duration, Annotation, Linked channel\n'
            '01.01.01, 23.59.30, 0, 30, Sleep stage N2,\n'
            '02.01.01, 00.00.00, 30, 30, Sleep stage R,\n'
            '02.01.01, 00.00.03, 33, 0, Lights off, EEG F4-A1\n')
    assert parse_sleep(text) == {0: [0.], 1: [30.]}


def test_seizure_parser_checks_counts_and_supports_numbered_events():
    text = ('File Name: chb01_03.edf\nNumber of Seizures in File: 2\n'
            'Seizure 1 Start Time: 100 seconds\nSeizure 1 End Time: 140 seconds\n'
            'Seizure 2 Start Time: 600 seconds\nSeizure 2 End Time: 650 seconds\n')
    assert parse_seizures(text)[0]['events'] == [(100., 140.), (600., 650.)]
    assert parse_seizures(text.replace('chb01_03.edf', 'chb02_16+.edf'))[0]['name'] == 'chb02_16+.edf'
    assert parse_seizures(text.replace('chb01_03.edf', 'chb17a_03.edf'))[0]['name'] == 'chb17a_03.edf'
    with pytest.raises(ValueError, match='unexpected EEG filename'):
        parse_seizures(text.replace('chb01_03.edf', '../chb01_03.edf'))
    with pytest.raises(ValueError, match='inconsistent seizure'):
        parse_seizures(text.replace('File: 2', 'File: 3'))


def test_control_epochs_do_not_overlap_events_margins_or_each_other():
    events = [(500, 550), (700, 750), (2400, 2450)]
    starts = control_windows(events, 3600, 4)
    assert len(starts) == 4
    for i, start in enumerate(starts):
        assert 10 <= start and start + 20 <= 3590
        assert all(start + 20 <= a - 300 or start >= b + 300 for a, b in events)
        assert all(abs(start - other) >= 20 for other in starts[:i])


def test_bipolar_emg_uses_names_and_ignores_unused_channels():
    names = [*[f'F{i}' for i in range(1, 17)], 'U1', *[f'W{i}' for i in range(1, 13)], 'U2']
    values = np.tile(np.arange(len(names)), (10, 1)).astype(float)
    values[:, names.index('U1')] = 1e9
    values[:, names.index('U2')] = -1e9
    answer = bipolar_emg(values, names)
    np.testing.assert_array_equal(answer[:, :8], np.full((10, 8), -8))
    np.testing.assert_array_equal(answer[:, 8:], np.full((10, 6), -6))
    order = np.random.default_rng(1).permutation(len(names))
    np.testing.assert_array_equal(answer, bipolar_emg(values[:, order], [names[i] for i in order]))


def test_amplitude_summary_survives_while_spi_input_is_scale_normalized():
    x = np.random.default_rng(2).normal(size=(512, 4))
    a, b = univariate(x), univariate(7*x)
    np.testing.assert_allclose(b[4:8] - a[4:8], np.log(7))
    np.testing.assert_allclose((x-x.mean(0))/x.std(0), (7*x-(7*x).mean(0))/(7*x).std(0))
    with pytest.raises(ValueError, match='flat'):
        univariate(np.ones((512, 4)))


def test_epoch_keeps_contiguous_duration_with_filter_context():
    class Raw:
        info = {'sfreq': 256}
        n_times = 256 * 60
        def get_data(self, picks, start, stop):
            t = np.arange(start, stop) / 256
            return np.array([np.sin(2*np.pi*(i+2)*t) for i in picks])
    x = epoch(Raw(), 10, 30, [0, 1], 128, .3, 55)
    assert x.shape == (3840, 2) and np.isfinite(x).all()
    assert np.all(x.std(0) > .5)
