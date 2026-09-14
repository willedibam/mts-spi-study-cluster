"""Opt-in, process-local memory repair for the pinned spectral backend.

This preserves the existing all-frequency-pairs PSI formula, including the
pyspi dispatch behaviour. It does not introduce a different PSI estimator.
"""
from contextlib import contextmanager
import hashlib
import inspect

import spectral_connectivity.connectivity as backend


ORIGINAL_SHA256 = "6dd9b62e91cdbbeb6632324ea4fa7d2942b2b246012d8f4fe3e171859cc9955b"


def inner_combination_linear_memory(data, axis=-3):
    """sum(i<j, conj(data[i])*data[j]), with linear temporary storage.

    Distributivity permits summing each j against the preceding prefix. Only
    floating-point summation order changes. NaNs propagate as in the original
    all-pairs sum. The supported backend inputs are floating/complex coherency.
    """
    xp = backend.xp
    ordered = xp.moveaxis(data, axis, 0)
    if ordered.shape[0] < 2:
        # The pinned implementation raises here; preserve that boundary.
        raise IndexError("All-pairs inner combination needs at least two entries")
    prefix = xp.cumsum(ordered[:-1].conjugate(), axis=0)
    prefix *= ordered[1:]
    return prefix.sum(axis=0)


@contextmanager
def bounded_psi_memory():
    """Patch only this process and restore even on failure; reject new backends."""
    original = backend._inner_combination
    digest = hashlib.sha256(inspect.getsource(original).encode()).hexdigest()
    if digest != ORIGINAL_SHA256:
        raise RuntimeError(f"Unvalidated spectral backend: {digest}")
    backend._inner_combination = inner_combination_linear_memory
    try:
        yield {"original_function_sha256": digest,
               "replacement_function_sha256": hashlib.sha256(
                   inspect.getsource(inner_combination_linear_memory).encode()).hexdigest(),
               "semantics": "Same all-pairs sum, different floating-point summation order; no PSI band/dispatch changes"}
    finally:
        backend._inner_combination = original
