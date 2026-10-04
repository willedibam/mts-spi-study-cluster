"""Match signed and absolute covariance within nested observation-size cells."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from threadpoolctl import threadpool_limits
from scripts.build_pearson_size_control import DATA as PARENT_DATA, LABELS, candidate_signs
from scripts.build_pearson_strength_match import normalize
from scripts.spi_baseline_exploration import ROOT, sha

RUN = 'pearson-size-matched-261004'
DATA = ROOT/'data/representation'/RUN
OUT = ROOT/'results/representation'/RUN
ANGLES = np.linspace(0, 2*np.pi, 33)


def absolute(x):
    c = np.corrcoef(x)
    return float(np.abs(c[~np.eye(len(c), dtype=bool)]).mean())


def correlation(c):
    sd = np.sqrt(np.diag(c))
    return c/sd[:, None]/sd[None, :]


def components(bank, instance, index, m, t):
    label = LABELS[index]
    start = (32-m)//2
    cut = lambda x: x[start:start+m, :t]
    if index < 4:
        signal = normalize(cut(bank[f'{label}-parent-{instance}']))
        noises = [np.random.default_rng(np.random.SeedSequence([261011, instance, index, j])).normal(size=(32, 1000)) for j in [1, 2]]
        return dict(signal=signal, first=cut(noises[0]), second=cut(noises[1]), kind='attenuation')
    noise_index = index-4
    rng = np.random.default_rng(np.random.SeedSequence([261010, instance, noise_index+4]))
    draw = rng.normal if noise_index == 0 else rng.standard_cauchy
    epsilon, common = draw(size=(32, 1000)), draw(size=(1, 1000))
    loadings = rng.permutation(np.linspace(.5, 1.5, 32))[:, None]
    np.testing.assert_array_equal(epsilon, bank[f'{LABELS[index+2]}-parent-{instance}'])
    second_rng = np.random.default_rng(np.random.SeedSequence([261011, instance, index, 2]))
    second = second_rng.normal(size=(32, 1000)) if noise_index == 0 else second_rng.standard_cauchy(size=(32, 1000))
    return dict(signal=cut(loadings*common), first=cut(epsilon), second=cut(second), kind='shared_input')


def absolute_profile(c, theta, target):
    x = c['signal']
    e = np.cos(theta)*c['first'] + np.sin(theta)*c['second']
    x, e = x-x.mean(1, keepdims=True), e-e.mean(1, keepdims=True)
    t = x.shape[1]
    xx, ee = x@x.T/t, e@e.T/t
    xe = (x@e.T+e@x.T)/t
    mask = ~np.eye(len(x), dtype=bool)
    if c['kind'] == 'attenuation':
        def value(s):
            return np.abs(correlation(xx+s*xe+s*s*ee)[mask]).mean()-target
        if value(0) <= 0 or absolute(e) >= target:
            raise ValueError('Absolute target outside attenuation bracket')
        high = 1.
        while value(high) > 0:
            high *= 2
            if high > 1e6:
                raise ValueError('No finite attenuation bracket')
        parameter = brentq(value, 0, high, xtol=1e-13)
        y = x+parameter*e
    else:
        def value(g):
            cov = g*g*xx+(1-g)**2*ee+g*(1-g)*xe
            return np.abs(correlation(cov)[mask]).mean()-target
        if value(0) >= 0:
            raise ValueError('Independent innovations exceed absolute target')
        parameter = brentq(value, 0, 1, xtol=1e-13)
        y = parameter*x+(1-parameter)*e
    y = normalize(y)
    return y, float(parameter), y@y.T/t


def joint_match(c, target_absolute, target_signed, instance):
    m = len(c['signal'])
    signs = candidate_signs(m, instance)
    cache = {}
    def profile(theta):
        if theta not in cache:
            cache[theta] = absolute_profile(c, theta, target_absolute)
        return cache[theta]
    def means(theta):
        cov = profile(theta)[2]
        return (np.einsum('bi,ij,bj->b', signs, cov, signs, optimize=True)-m)/(m*(m-1))-target_signed
    left, fleft = ANGLES[0], means(ANGLES[0])
    for right in ANGLES[1:]:
        fright = means(right)
        crossing = np.flatnonzero(fleft*fright <= 0)
        if len(crossing):
            fraction = np.abs(fleft[crossing])/(np.abs(fleft[crossing])+np.abs(fright[crossing])+1e-300)
            index = int(crossing[np.argmin(fraction)])
            sign = signs[index]
            def value(theta):
                return (sign@profile(theta)[2]@sign-m)/(m*(m-1))-target_signed
            theta = brentq(value, left, right, xtol=1e-12)
            x, parameter, _ = profile(theta)
            return sign[:, None]*x, dict(noise_angle=float(theta), parameter=parameter,
                                         parameter_kind=c['kind'], polarity=sign.astype(int).tolist(), profile_evaluations=len(cache))
        left, fleft = right, fright
    raise ValueError(f'No signed-mean bracket M={m}, T={c["signal"].shape[1]}, instance={instance}')


def build():
    if (DATA/'manifest.json').exists():
        raise FileExistsError('Immutable bank already exists')
    DATA.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    parent_manifest = json.loads((PARENT_DATA/'manifest.json').read_text())
    assert sha(PARENT_DATA/'parents.npz') == parent_manifest['parents_sha256']
    with np.load(PARENT_DATA/'parents.npz') as archive:
        bank = {k: archive[k] for k in archive.files}
    rows, arrays, targets = [], {}, []
    for instance in range(5):
        for m in [8, 16, 32]:
            for t in [100, 500, 1000]:
                parts = [components(bank, instance, i, m, t) for i in range(6)]
                floor = max(absolute(np.cos(theta)*p['first']+np.sin(theta)*p['second']) for p in parts for theta in ANGLES)
                ceiling = min(absolute(p['signal']) for p in parts[:4])
                if ceiling-floor <= .004:
                    raise ValueError(f'No common absolute range M={m},T={t},I={instance}: {floor}, {ceiling}')
                target_a = (floor+ceiling)/2
                # An equicorrelated reference with this sign imbalance realizes
                # b/a exactly; this avoids impossible arbitrary signed targets.
                sign_sum = 2*round(np.sqrt(2*m)/2)
                ratio = (sign_sum**2-m)/(m*(m-1))
                target_b = ratio*target_a
                targets.append(dict(instance=instance, M=m, T=t, noise_floor=floor, signal_ceiling=ceiling,
                                    target_absolute=target_a, target_signed=target_b, reference_sign_sum=sign_sum))
                for index, label in enumerate(LABELS):
                    if index < 6:
                        x, setting = joint_match(parts[index], target_a, target_b, instance)
                    else:
                        start = (32-m)//2
                        x = normalize(bank[f'{label}-parent-{instance}'][start:start+m, :t])
                        setting = dict(parameter_kind='independent_control', polarity=[1]*m)
                    cov = x@x.T/t
                    mask = ~np.eye(m, dtype=bool)
                    b, a = float(cov[mask].mean()), float(np.abs(cov[mask]).mean())
                    if index < 6:
                        assert abs(b-target_b) < 1e-9 and abs(a-target_a) < 1e-9
                    np.testing.assert_allclose(x.var(1), 1, atol=1e-12)
                    np.testing.assert_allclose(x.mean(1), 0, atol=1e-12)
                    name = f'{label}-M{m}-T{t}-I{instance}'
                    arrays[name] = x
                    rows.append(dict(row_id=name, label=label, instance=instance, parent=f'{label}-parent-{instance}',
                        M=m, T=t, cell=f'M{m}-T{t}', corpus_index=len(rows), panel='matched' if index < 6 else 'independent_control',
                        mean_covariance=b, mean_abs_Pearson=a, target_signed=target_b if index < 6 else None,
                        target_absolute=target_a if index < 6 else None, **setting))
                print(f'Calibrated M{m}/T{t}/I{instance}', flush=True)
    metadata = dict(__dataset_names__=np.array([r['row_id'] for r in rows]),
                    __labels_json__=np.array([json.dumps([r['label']]) for r in rows]),
                    __shapes__=np.array([[r['M'], r['T']] for r in rows]), __axis_order__=np.array(['process', 'observation']))
    np.savez_compressed(DATA/'observations.npz', **arrays, **metadata)
    manifest = dict(rows=rows, archive_sha256=sha(DATA/'observations.npz'), script_sha256=sha(Path(__file__)),
        parent_archive_sha256=parent_manifest['parents_sha256'], parent_generator_sha256=parent_manifest['script_sha256'],
        qualification='Five independent parents/class, nine nested views. Signed and absolute targets identical across classes within each size/parent block; pooled class strength distributions identical. Targets depend on size. Data-conditioned observation control, not physical coupling equality.')
    (DATA/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    pd.DataFrame(rows).drop(columns='polarity').to_csv(OUT/'calibration.csv', index=False)
    pd.DataFrame(targets).to_csv(OUT/'targets.csv', index=False)
    print('Saved', len(rows), 'records', manifest['archive_sha256'])


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        build()
