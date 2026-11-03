"""Which SPIs does multivariate IAAFT actually preserve in NeuroTycho?

IAAFT keeps auto- and cross-spectra, so spectral SPIs (coherence, spectral GC, dcoh/gpdcoh) should match real data
window by window; if they do not, residuals real − surrogate on those SPIs are surrogate artefacts. Per SPI and
variant: median over windows of the edge-profile correlation real vs surrogate, and the window-level shift of the
edge-mean value in units of its across-window SD in the real data. Run on Gadi (jobs/surrogate_fidelity.pbs).
"""
import argparse
import json
from pathlib import Path

import numpy as np

from mtsspi.character_axes import edge_profiles, profile_correlation
from research.representation.character_axes.surrogate_pairs import load


def main(real_root, surrogate_roots, output):
    bank, _, _, keys = load(real_root)
    spi = [str(s) for s in bank['spi_order']]
    real = edge_profiles(bank['aligned'], int(bank['M'][0]), len(spi))
    m_real = np.nanmean(real, -1)
    sd = np.nanstd(m_real, 0)
    report = dict(spi=spi, variants={})
    for root in surrogate_roots:
        sbank, _, _, skeys = load(root)
        assert np.array_equal(skeys, keys)
        surr = edge_profiles(sbank['aligned'], int(sbank['M'][0]), len(spi))
        r = np.nanmedian(profile_correlation(real.transpose(1, 0, 2), surr.transpose(1, 0, 2)), 1)
        with np.errstate(invalid='ignore', divide='ignore'):
            shift = np.nanmedian(np.abs(m_real - np.nanmean(surr, -1)), 0) / sd
        report['variants'][root.name] = dict(profile_r=r.tolist(), mean_shift_sd=shift.tolist())
        print(root.name, flush=True)
        for fam in ('coh', 'sgc', 'dcoh', 'pdcoh', 'gpdcoh', 'icoh', 'psi', 'prec', 'prec-sq', 'cov', 'corr_spearman', 'ids', 'te_', 'gc_', 'mi_'):
            ix = [k for k, s in enumerate(spi) if s.startswith(fam)]
            if ix:
                print(f'  {fam:14s} n={len(ix):3d} profile r median {np.nanmedian(r[ix]):.3f} min {np.nanmin(r[ix]):.3f}'
                      f'  mean shift/sd median {np.nanmedian(shift[ix]):.2f} max {np.nanmax(shift[ix]):.2f}', flush=True)
        for k in np.argsort(r)[:12]:
            print(f'  worst {spi[k][:60]:60s} r {r[k]:.3f} shift {shift[k]:.2f}', flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=1) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--real', type=Path, required=True)
    p.add_argument('--surrogates', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    main(a.real, a.surrogates, a.output)
