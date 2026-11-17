"""Stream one Claar NWB's stimulus-free EEG stretches (exact compressed chunks via curl ranges) and save them at 100 Hz.

Per stretch: valid channels, µV, atlas `continuous` chain (notch 60, HP 0.5, LP 40, resample_poly to 100 Hz), BEFORE
common-average reference; source-rate flat-run mask (atlas `flat_runs`) reduced to 100 Hz; running speed on the
100-Hz grid. Windows are chosen later (select_claar.py). Rules: claar-prereg.md, pre-data clarifications.
    python extract_claar.py ASSET_ID OUT.npz
"""
import sys, json, time, zlib, subprocess, collections
import numpy as np, h5py
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, '/Users/wedi0306/Code/mts-spi-study-cluster/.claude/worktrees/real-world-apps-9b53a1')
from research.representation.atlas.prepare import continuous, flat_runs
from curlfile import CurlFile

NOMINAL_FS, MAINS = 2500, 60.0           # true rate 2500.002 Hz (1 ppm); nominal keeps resample_poly at 1/25


def curl_range(url, a, b, tries=6):
    for k in range(tries):
        out = subprocess.run(['curl', '-s', '--max-time', '300', '-r', f'{a}-{b}', url], capture_output=True).stdout
        if len(out) == b - a + 1:
            return out
        time.sleep(2 ** k)
    raise IOError(f'range {a}-{b}: got {len(out)} bytes after {tries} tries')


class SafeCurl(CurlFile):
    """CurlFile with length-checked retried reads and a bounded LRU block cache (metadata only)."""
    MAXB = 64
    def __init__(self, url):
        super().__init__(url); self.cache = collections.OrderedDict()
    def _blk(self, i):
        if i in self.cache:
            self.cache.move_to_end(i); return self.cache[i]
        a = i * self.B; b = min(self.size, a + self.B) - 1; self.nreq += 1
        self.cache[i] = curl_range(self.url, a, b)
        while len(self.cache) > self.MAXB: self.cache.popitem(last=False)
        return self.cache[i]


def stretches(st, t0, t1, a0, a1):
    """[(state, read_lo, read_hi, elig_lo, elig_hi)] in seconds; eligible ranges ≥ 10 s."""
    st = np.sort(st); out = [('awake', t0, st[0] - 1, t0 + 30, st[0] - 5)]
    for p, q in zip(st, np.r_[st[1:], t1]):
        lo, hi = max(p + 5, a0 + 60), min(q - 1 if q < t1 else q, a1)
        if hi - lo >= 10:
            out.append(('iso', max(p + 2, lo - 10), min(q - 0.5 if q < t1 else q, hi + 10), lo, hi))
    return out


if __name__ == '__main__':
    aid, out = sys.argv[1], sys.argv[2]; T0 = time.time()
    f = SafeCurl(f'https://api.dandiarchive.org/api/assets/{aid}/download/'); h = h5py.File(f, 'r')
    E = h['acquisition/ElectricalSeriesEEG']; d = E['data']; ts = E['timestamps']; n = d.shape[0]; c = d.chunks[0]
    assert d.chunks[1] == 1 and d.dtype == np.int16
    pl = d.id.get_create_plist(); filt = [pl.get_filter(i)[0] for i in range(pl.get_nfilters())]
    assert set(filt) <= {1, 2}, filt                    # deflate, optional shuffle
    t0, t1 = float(ts[0]), float(ts[n - 1]); dt = (t1 - t0) / (n - 1); conv = float(d.attrs['conversion']) * 1e6
    reg = E['electrodes'][:]; el = h['general/extracellular_ephys/electrodes']
    cols = np.flatnonzero(el['is_data_valid'][:][reg].astype(bool)); ids = el['id'][:][reg][cols]
    dec = lambda a: [x.decode() if isinstance(x, bytes) else x for x in a]
    ep = {t: (a, b) for t, a, b in zip(dec(h['intervals/epochs/tags'][:]), h['intervals/epochs/start_time'][:], h['intervals/epochs/stop_time'][:])}
    a0, a1 = ep['isoflurane_anesthesia']; st = h['intervals/trials/start_time'][:]
    R = h['processing/behavior/BehavioralTimeSeries/running_speed']; rs = R['data'][:]
    rs_t0, rs_rate = float(R['starting_time'][()]), float(R['starting_time'].attrs['rate'])
    info = {}
    d.id.chunk_iter(lambda ci: info.__setitem__((ci.chunk_offset[0] // c, ci.chunk_offset[1]), (ci.byte_offset, ci.size, ci.filter_mask)))
    plan = stretches(st, t0, t1, a0, a1); arrays, meta, nbytes, terr = {}, [], 0, 0.0
    for k, (state, lo, hi, elo, ehi) in enumerate(plan):
        i0, i1 = int(np.ceil((lo - t0) / dt)), int(np.floor((hi - t0) / dt)) + 1
        for i in (i0, i1 - 1):                          # timing check against the stored timestamps
            terr = max(terr, abs(float(ts[i]) - (t0 + i * dt)))
        rows = range(i0 // c, (i1 - 1) // c + 1); need = sorted((info[(r, ch)][0], info[(r, ch)][1], r, ch, info[(r, ch)][2]) for r in rows for ch in cols)
        groups, g = [], [need[0]]                        # merge byte-adjacent chunks into one request
        for x in need[1:]:
            if x[0] - (g[-1][0] + g[-1][1]) < 65536 and x[0] + x[1] - g[0][0] < 32 * 2 ** 20: g.append(x)
            else: groups.append(g); g = [x]
        groups.append(g)
        def fetch(g):
            a = g[0][0]; buf = curl_range(f.url, a, g[-1][0] + g[-1][1] - 1); res = []
            for off, size, r, ch, fm in g:
                raw = zlib.decompress(buf[off - a:off - a + size])
                v = np.frombuffer(raw, np.uint8).reshape(2, -1).T.reshape(-1).view(np.int16) if 2 in filt and not fm & 2 else np.frombuffer(raw, np.int16)
                res.append((r, ch, v))
            return res, len(buf)
        X = np.zeros(((rows[-1] + 1 - rows[0]) * c, len(cols)), np.int16); col = {ch: j for j, ch in enumerate(cols)}
        with ThreadPoolExecutor(8) as ex:
            for res, nb in ex.map(fetch, groups):
                nbytes += nb
                for r, ch, v in res: X[(r - rows[0]) * c:(r - rows[0]) * c + len(v), col[ch]] = v[:c]
        X = X[i0 - rows[0] * c:i1 - rows[0] * c]
        fl = flat_runs(X, NOMINAL_FS)                    # any good channel; reduced per channel below is not needed (atlas uses any)
        y = continuous(X.astype(np.float64) * conv, NOMINAL_FS, MAINS); m = len(y)
        flm = np.pad(fl, (0, max(0, m * 25 - len(fl))))[:m * 25].reshape(m, 25).any(1)
        tt = t0 + i0 * dt + np.arange(m) / 100.0
        arrays[f'X{k}'] = y.astype(np.float32); arrays[f'F{k}'] = flm
        arrays[f'S{k}'] = np.interp(tt, rs_t0 + np.arange(len(rs)) / rs_rate, rs).astype(np.float32)
        meta.append(dict(state=state, read=[lo, hi], elig=[float(elo), float(ehi)], t_first=float(tt[0]), n100=m,
                         rms_uv=float(np.median(y.std(0))), nonfinite=int((~np.isfinite(X)).sum())))
        print(f'  {state} {lo:.0f}-{hi:.0f}s elig {elo:.0f}-{ehi:.0f} M={len(cols)} {nbytes/1e6:.0f} MB so far', flush=True)
    meta = dict(asset=aid, path=h.filename, fs=1 / dt, t0=t0, t1=t1, cols=cols.tolist(), ids=ids.tolist(), anesthesia=[a0, a1],
                stretches=meta, bytes=nbytes, timing_err_s=terr, filters=filt, rs_rate=rs_rate, seconds=time.time() - T0)
    np.savez(out, meta=json.dumps(meta), **arrays)
    print(f'done {out}: {len(plan)} stretches, {nbytes/1e6:.0f} MB transferred, timing err {terr*1e3:.2f} ms, {time.time()-T0:.0f} s', flush=True)

