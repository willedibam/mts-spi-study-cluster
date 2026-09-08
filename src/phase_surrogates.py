"""Fourier phase controls: exact spectra, explicitly circular correlations."""
import numpy as np


def phase_surrogate(raw, seed, shared):
    raw = np.asarray(raw, dtype=float)
    if raw.ndim != 2 or not np.isfinite(raw).all() or len(raw) < 3:
        raise ValueError('expected finite T by M data')
    coefficients = np.fft.rfft(raw, axis=0)
    rng = np.random.default_rng(seed)
    width = 1 if shared else raw.shape[1]
    phase = np.exp(1j*rng.uniform(-np.pi,np.pi,(len(coefficients),width)))
    phase[0] = 1
    if len(raw)%2 == 0:
        phase[-1] = rng.choice([-1.,1.],size=width)
    return np.fft.irfft(coefficients*phase,n=len(raw),axis=0)


def pooled_autospectrum(raw, bins=32):
    """Fixed frequency-band powers, pooled over channels; no cross-products."""
    raw = np.asarray(raw)
    power = abs(np.fft.rfft(raw,axis=0))**2/len(raw)
    if len(power)-1 < bins:
        raise ValueError('too few frequencies for requested bands')
    bands = np.log1p(np.stack([power[ix].mean(axis=0)
        for ix in np.array_split(np.arange(1,len(power)),bins)]))
    def summaries(values):
        return np.concatenate([values.mean(axis=-1),values.std(axis=-1),
            *np.quantile(values,[.1,.5,.9],axis=-1)])
    return np.concatenate([summaries(bands),summaries(raw.mean(axis=0)[None])])


def spectrum_checks(raw, transformed, shared):
    a,b = np.fft.rfft(raw,axis=0),np.fft.rfft(transformed,axis=0)
    power_error = float(np.max(abs(abs(a)-abs(b)))/max(1.,abs(a).max()))
    mean_error = float(abs(raw.mean(axis=0)-transformed.mean(axis=0)).max())
    assert power_error < 1e-12 and mean_error < 1e-12
    cross_error = float(np.max(abs(a[:,:,None]*a[:,None,:].conj()-
                                      b[:,:,None]*b[:,None,:].conj()))/max(1.,abs(a).max()**2))
    if shared:
        assert cross_error < 1e-12
    return dict(relative_amplitude_error=power_error,mean_error=mean_error,
                relative_cross_periodogram_error=cross_error)
