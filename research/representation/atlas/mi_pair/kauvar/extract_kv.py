"""Stream the pre-puff segment (0–590 s, or to 5 s before the first DC event if earlier) of one DANDI 001347 sEEG session:
contiguous float64 LFP via curl ranges; bipolar (adjacent contacts on one shaft, OUT OF BRAIN excluded); atlas continuous chain
(notch 60, HP 0.5, LP 40, to 100 Hz). Rules: kauvar-prereg.md.
    python extract_kv.py ASSET_ID OUT.npz"""
import sys, json, re, time
import numpy as np, h5py
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, '/Users/wedi0306/Code/mts-spi-study-cluster/.claude/worktrees/real-world-apps-9b53a1')
from research.representation.atlas.prepare import continuous
from extract_claar import SafeCurl, curl_range, flat_any
dec = lambda a: [x.decode() if isinstance(x, bytes) else str(x) for x in a]

def read_rows(url, off, ncol, r0, r1, itemsize=8, step=8 * 2 ** 20):
    a, b = off + r0 * ncol * itemsize, off + r1 * ncol * itemsize - 1; parts = list(range(a, b + 1, step))
    with ThreadPoolExecutor(8) as ex: bufs = list(ex.map(lambda p: curl_range(url, p, min(p + step, b + 1) - 1), parts))
    return np.frombuffer(b''.join(bufs), '<f8').reshape(r1 - r0, ncol), b - a + 1

if __name__ == '__main__':
    aid, out = sys.argv[1], sys.argv[2]; T0 = time.time(); url = f'https://api.dandiarchive.org/api/assets/{aid}/download/'
    cf = SafeCurl(url); url = cf.url; h = h5py.File(cf, 'r'); v = h['processing/ecephys/LFP/v_trace']; d = v['data']
    assert d.chunks is None and d.compression is None and d.dtype == np.dtype('<f8')
    fs = float(v['starting_time'].attrs['rate']); n, M = d.shape; off = d.id.get_offset(); conv = float(d.attrs['conversion']) * 1e6
    el = h['general/extracellular_ephys/electrodes']; idx = v['electrodes'][:]
    lab, grp, loc = (np.array(dec(el[k][:]))[idx] for k in ('label', 'group_name', 'location'))
    D = h['acquisition/dc_trace/data']; dn = dec(h['acquisition/dc_channels/channel_name'][:]); fsd = float(h['acquisition/dc_trace/starting_time'].attrs['rate'])
    dc = D[:int(630 * fsd)]; td = np.arange(len(dc)) / fsd; first = {}
    cand = [j for j, nm in enumerate(dn) if re.match(r'DC\d\d$|CCEP', nm) and dc[td < 590, j].std() > 0]
    ratio = {j: dc[td >= 600, j].std() / dc[td < 590, j].std() for j in cand}; trig = max(ratio, key=ratio.get)   # puff trigger: most active after 600 s
    x = dc[:, trig]; base = np.median(x[td < 590]); amp = np.percentile(np.abs(x[td >= 600] - base), 99.9)
    k = np.flatnonzero(np.abs(x - base) > 0.5 * amp); first = dict(channel=dn[trig], ratio=round(float(ratio[trig]), 1), first_event_s=round(float(td[k[0]]), 2) if len(k) else None)
    end = 590.0
    if first['first_event_s'] is not None and first['first_event_s'] < 595: end = first['first_event_s'] - 5
    X, nb = read_rows(url, off, M, 0, int(end * fs)); X = X * conv
    num = lambda s: int(re.search(r'(\d+)$', s).group(1)) if re.search(r'(\d+)$', s) else None
    pairs = []
    for i in range(M):
        for j in range(M):
            if grp[i] == grp[j] and num(lab[i]) is not None and num(lab[j]) == num(lab[i]) + 1 and 'OUT OF BRAIN' not in (loc[i], loc[j]):
                pairs.append((i, j))
    B = np.column_stack([X[:, j] - X[:, i] for i, j in pairs]); fl = flat_any(B, int(fs))
    y = continuous(B, fs, 60.0); m = len(y); r = int(round(fs / 100)); flm = np.pad(fl, (0, max(0, m * r - len(fl))))[:m * r].reshape(m, r).any(1)
    meta = dict(asset=aid, path=h.filename, fs=fs, n=n, M=M, end_s=end, first_dc_event_s=first, bytes=nb, pairs=[f'{lab[j]}-{lab[i]}' for i, j in pairs],
                pair_loc=[f'{loc[j]}|{loc[i]}' for i, j in pairs], nonfinite=int((~np.isfinite(X)).sum()), seconds=time.time() - T0)
    np.savez(out, X=y.astype(np.float32), F=flm, meta=json.dumps(meta))
    print(f'done {out}: {len(pairs)} bipolar, end {end:.1f} s, first DC events {first}, {nb/1e6:.0f} MB, {time.time()-T0:.0f} s', flush=True)
