"""Pre-run pyspi cost estimate for the MTS benchmark panel (no pyspi is run).

Base: the HPC.md sizing table for a full benchmarked_p90 table (workers, GiB, hours per recording), charged as
2 SU x max(workers, GiB/3.96) x hours, interpolated bilinearly in log(ordered edges) and log(T). The table was
measured with the fd0de71 pyspi; the pinned 676a49d / r10 runs measured 1.2-3x slower at M<=17 (calibration
printed below), so estimates are multiplied by CAL=2. Expect +/-50%. Below the table (M<16 or T<256) the
estimate is scaled by edges and T^1.3 and floored; those cells are dominated by per-SPI overhead and are cheap
either way. Above the table (M>256) the interpolation is clipped and must not be used.

Usage: python -m research.representation.mts_benchmark.estimate_cost
"""
import numpy as np

MS, TS = [16, 32, 64, 128, 256], [256, 1024, 2048, 4096]
TABLE = {16: [(1, 4, .05), (1, 4, .07), (1, 5, .1), (2, 6, .2)], 32: [(1, 4, .07), (1, 4, .2), (2, 5, .3), (2, 7, .9)],
         64: [(1, 4, .1), (2, 6, .3), (4, 9, .6), (8, 24, 1.1)], 128: [(2, 6, .3), (4, 11, .7), (8, 23, 1.3), (16, 60, 2.7)],
         256: [(4, 11, .5), (12, 34, 1.0), (24, 77, 2.0), (32, 174, 7.1)]}  # '<0.1 h' cells entered as .05-.07
CAL = 2.0
# measured single-core r10 recordings: (name, M, T, seconds), from the discovery and character-transfer execution records
MEASURED = [('CHB-MIT', 16, 2000, 813), ('NeuroTycho', 16, 2000, 558), ('HMC', 8, 3840, 847), ('GRABMyo', 14, 2048, 858),
            ('LANL', 4, 4096, 231), ('fMRI', 17, 947, 145)]


def _charge(workers, gib, hours):
    return 2 * max(workers, gib / 3.96) * hours


LOGSU = np.log([[_charge(*c) for c in TABLE[m]] for m in MS])


def su_per_recording(M, T, cal=CAL):
    if M > MS[-1]:
        raise ValueError('outside the sizing table; measure instead')
    E, lE, lT = M * (M - 1), np.log([m * (m - 1) for m in MS]), np.log(TS)
    x, y = np.clip(np.log(E), lE[0], lE[-1]), np.clip(np.log(T), lT[0], lT[-1])
    i = min(np.searchsorted(lE, x, side='right') - 1, len(MS) - 2)
    j = min(np.searchsorted(lT, y, side='right') - 1, len(TS) - 2)
    fx, fy = (x - lE[i]) / (lE[i + 1] - lE[i]), (y - lT[j]) / (lT[j + 1] - lT[j])
    v = ((1 - fx) * (1 - fy) * LOGSU[i, j] + fx * (1 - fy) * LOGSU[i + 1, j]
         + (1 - fx) * fy * LOGSU[i, j + 1] + fx * fy * LOGSU[i + 1, j + 1])
    s = np.exp(v) * min(1.0, E / 240) * min(1.0, T / 256) ** 1.3
    return cal * max(s, 0.02)


# (dataset, M, T as computed, records under the full-split tier, records under the subsampled tier)
PANEL = [('ImaginedOpenCloseFist', 64, 640, 4725, 1600), ('ImaginedFeetHands', 64, 640, 4770, 1600),
         ('ShortIntervalTask (fixed 32-ch subset, 250 Hz)', 32, 1375, 2780, 1000), ('AutomotiveRoadTrials', 8, 300, 230, 230),
         ('EmoPain', 30, 200, 1323, 1323), ('SelfRegulationSCP1', 6, 896, 561, 561), ('HandMovementDirection', 10, 400, 234, 234),
         ('ABIDE-PCP MSDL', 39, 200, 871, 871), ('TUAB (19 ch, 20 s at 100 Hz)', 19, 2000, 2993, 2993),
         ('NeuroTycho re-cut (16 ch, 10 s at 100 Hz)', 16, 1000, 480, 480), ('BasicMotions (positioning)', 6, 100, 80, 80)]
ALTERNATIVES = [('ShortIntervalTask all 63 ch', 63, 1375, 2780), ('MotorImagery at 500 Hz', 64, 1500, 378),
                ('MotorImagery native 1 kHz', 64, 3000, 378), ('PEMS-SF 64-sensor subset', 64, 144, 440)]


def main():
    print('calibration (measured r10 SU vs table estimate at CAL=1):')
    for name, M, T, sec in MEASURED:
        print(f'  {name:11s} M{M:<3d} T{T:<5d} measured {2 * sec / 3600:.3f}  table {su_per_recording(M, T, 1):.3f}')
    full = sub = 0.0
    print(f'\n{"dataset":48s} {"SU/rec":>7s} {"full":>8s} {"subsampled":>10s}')
    for name, M, T, nf, ns in PANEL:
        s = su_per_recording(M, T); full += s * nf; sub += s * ns
        print(f'{name:48s} {s:7.2f} {s * nf:8.0f} {s * ns:10.0f}')
    print(f'{"pyspi total":48s} {"":7s} {full:8.0f} {sub:10.0f}')
    print('\nalternatives not in the panel:')
    for name, M, T, n in ALTERNATIVES:
        s = su_per_recording(M, T)
        print(f'  {name:46s} {s:7.2f} {s * n:8.0f}')


if __name__ == '__main__':
    main()
