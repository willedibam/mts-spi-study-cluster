import io
import pickle

import numpy as np
import pytest

from scripts.audit_bridge_transfer_data import NumpyOnlyUnpickler, unicode_field


def test_restricted_numpy_and_index_reader_agree():
    names = np.array(['B3_T27_N_R1', 'B3_T28_M1_R2'])
    payload = pickle.dumps({'lms_file_name': names, 'scale': np.arange(8.)}, protocol=4)
    restored = NumpyOnlyUnpickler(io.BytesIO(payload)).load()
    np.testing.assert_array_equal(unicode_field(payload, 'lms_file_name'), restored['lms_file_name'])


def test_restricted_reader_rejects_other_globals():
    with pytest.raises(pickle.UnpicklingError, match='Unsupported global'):
        NumpyOnlyUnpickler(io.BytesIO(b'cbuiltins\neval\n.')).load()
