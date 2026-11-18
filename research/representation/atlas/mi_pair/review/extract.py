import sys, io, numpy as np
p = sys.argv[1]
b = np.load(p)
K = len(b['spi_order'])
A = b['aligned'].reshape(len(b['y']), K, -1)
out = {k: b[k] for k in ('z','y','group','role','row_id','spi_order','pearson','spectral','M','T') if k in b.files}
with np.errstate(all='ignore'):
    out['spi_mean'] = np.nanmean(A, 2).astype(np.float32)
    out['spi_sd'] = np.nanstd(A, 2).astype(np.float32)
    # split-half edge profile reliability proxy not possible without time halves; skip
buf = io.BytesIO(); np.savez(buf, **out); sys.stdout.buffer.write(buf.getvalue())
