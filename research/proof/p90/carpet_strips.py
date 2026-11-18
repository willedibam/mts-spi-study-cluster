"""Carpet strips: tile a fixed rectangle with MTS carpets whose height encodes M.

A strip holds a whole number of channels of height. Carpet heights come from one unit ``u`` per 8
channels: an M-channel carpet is ``M/8*u - gap`` tall, so two M carpets plus a gap stack to exactly
one 2M carpet and every column fills the strip. Channel height is then nearly constant across M
(the gaps take the slack). Width is a time window, free per carpet: it is solved so the columns
fill the strip width exactly, within aspect bounds.

Layout strings (top to bottom within a column, columns left to right)::

    "32 | 16, 8+8 | 8, 8, 16"

``|`` separates columns, ``,`` stacks levels, ``+`` puts equal carpets side by side in a level.
``/`` stacks full-width bands, each tiled by its own columns (for tall strips, where a single band
is dominated by 32-channel carpets).
"""
from itertools import combinations, product

import numpy as np


def parse(band):
    """One band: columns of levels of carpet channel counts."""
    return [[[int(m) for m in level.split('+')] for level in col.split(',')] for col in band.split('|')]


def format_layout(columns):
    return ' | '.join(', '.join('+'.join(str(m) for m in level) for level in col) for col in columns)


def bands(spec):
    return [b.strip() for b in spec.split('/')]


def channels(band):
    """Channels of height of one band."""
    return sum(level[0] for level in parse(band)[0])


def mirror(spec):
    return ' / '.join(format_layout(parse(b)[::-1]) for b in bands(spec))


def brick(m, u, gap):
    """Height of an M-channel carpet."""
    return m/8*u - gap


def _column_range(col, u, gap, aspect):
    """(low, preferred, high) width of a column; every level shares it."""
    lo, pref, hi = [], [], []
    for level in col:
        k, h = len(level), brick(level[0], u, gap)
        lo.append(k*aspect[0]*h + (k-1)*gap); pref.append(k*aspect[2]*h + (k-1)*gap)
        hi.append(k*aspect[1]*h + (k-1)*gap)
    return max(lo), float(np.mean(pref)), min(hi)


def solve_widths(columns, width, u, gap, aspect):
    """Column widths filling ``width`` exactly, closest to the preferred widths (water-filling).
    ``aspect`` = (low, high, target) width/height per carpet. None if infeasible."""
    lo, pref, hi = np.array([_column_range(c, u, gap, aspect) for c in columns]).T
    total = width - (len(columns)-1)*gap
    if (lo > hi).any() or lo.sum() > total or hi.sum() < total:
        return None
    a, b = -width, width
    for _ in range(60):
        lam = (a+b)/2
        a, b = (lam, b) if np.clip(pref+lam, lo, hi).sum() < total else (a, lam)
    return np.clip(pref+(a+b)/2, lo, hi)


def place(spec, width, u, gap, aspect):
    """Carpet rectangles (x, y from the strip's top-left, w, h; inches) and M, in reading order."""
    out, y0 = [], 0.
    for band in bands(spec):
        columns = parse(band)
        widths = solve_widths(columns, width, u, gap, aspect)
        if widths is None:
            raise ValueError(f'Layout {band!r} does not fit a {width:.2f} in strip within aspect {aspect[:2]}.')
        x = 0.
        for col, cw in zip(columns, widths):
            y = y0
            for level in col:
                h = brick(level[0], u, gap); w = (cw - (len(level)-1)*gap)/len(level)
                out.extend({'x': x + j*(w+gap), 'y': y, 'w': w, 'h': h, 'M': m} for j, m in enumerate(level))
                y += h + gap
            x += cw + gap
        y0 += brick(channels(band), u, gap) + gap
    return out


def _stacks(capacity, sizes=(8, 16, 32), max_levels=4, max_splits=1):
    """Column stacks filling ``capacity``; up to ``max_splits`` levels below 32 channels split
    into two side-by-side carpets."""
    def compositions(rest):
        if rest == 0:
            yield []
        for m in sizes:
            if m <= rest:
                for tail in compositions(rest - m):
                    yield [m] + tail
    stacks = []
    for comp in compositions(capacity):
        if len(comp) > max_levels:
            continue
        small = [i for i, m in enumerate(comp) if m < 32]
        for k in range(max_splits+1):
            for split in combinations(small, k):
                stacks.append([[m, m] if i in split else [m] for i, m in enumerate(comp)])
    return stacks


def _band_cost(band, rects):
    """Neighbouring columns that repeat, and horizontal cuts they share (both read as a grid)."""
    columns = parse(band)
    cuts = [set(np.cumsum([lv[0] for lv in c])[:-1]) for c in columns]
    return .3*sum(a == b for a, b in zip(columns, columns[1:])) + .1*sum(len(a & b) for a, b in zip(cuts, cuts[1:]))


def score(spec, width, u, gap, aspect, n_target, require_all_m=True):
    """Lower is better: carpet aspect error, distance from the target count, unequal counts of
    M = 8, 16 and 32, and the grid penalties.
    Infinite if a band does not fit, or (``require_all_m``) if M = 8, 16 and 32 are not all shown."""
    try:
        rects = place(spec, width, u, gap, aspect)
    except ValueError:
        return np.inf
    if require_all_m and len({r['M'] for r in rects}) < 3:
        return np.inf
    err = np.mean([np.log(r['w']/r['h']/aspect[2])**2 for r in rects])
    counts = np.array([sum(r['M'] == m for r in rects) for m in (8, 16, 32)])
    balance = np.abs(counts - len(rects)/3).sum()  # Each M about equally often.
    return (err + .08*abs(len(rects) - n_target) + .1*balance
            + sum(_band_cost(b, rects) for b in bands(spec)))


def _band_layouts(cap, width, u, gap, aspect, n_items, max_columns, max_splits, max_levels=4, keep=300):
    """Feasible bands: column multisets ranked by aspect error and count (order-free), then every
    distinct ordering of the ``keep`` best."""
    from itertools import combinations_with_replacement, permutations
    stacks = [c for c in _stacks(cap, max_levels=max_levels, max_splits=max_splits)
              if (lambda r: r[0] <= r[2])(_column_range(c, u, gap, aspect))]  # Each column must fit alone.
    n_target = (n_items[0] + n_items[1])/2
    sets = []
    for k in range(1, max_columns+1):
        for combo in combinations_with_replacement(range(len(stacks)), k):
            cols = [stacks[i] for i in combo]
            n = sum(len(level) for col in cols for level in col)
            if n_items[0] <= n <= n_items[1]:
                s = score(format_layout(cols), width, u, gap, aspect, n_target, require_all_m=False)
                if np.isfinite(s):
                    sets.append((s, combo))
    sets.sort()
    for _, combo in sets[:keep]:
        for order in set(permutations(combo)):
            yield format_layout([stacks[i] for i in order])


def rank_layouts(capacities, width, u, gap, aspect, n_items=(5, 8), max_columns=4, max_splits=1, max_levels=4,
                 top=20, band_top=40):
    """Feasible layouts for bands of ``capacities`` channels (top to bottom), ranked by ``score``.
    Mirror images count once. A multi-band strip combines the ``band_top`` best of each band."""
    capacities = [capacities] if np.isscalar(capacities) else list(capacities)
    n_target = (n_items[0] + n_items[1])/2
    if len(capacities) == 1:
        candidates = _band_layouts(capacities[0], width, u, gap, aspect, n_items, max_columns, max_splits, max_levels)
    else:
        per_band = []
        for cap in capacities:
            share = cap/sum(capacities)  # Item count split in proportion to band height.
            limits = (max(1, int(np.floor(n_items[0]*share))), int(np.ceil(n_items[1]*share)))
            ranked = sorted((score(b, width, u, gap, aspect, n_target*share, require_all_m=False), b)
                            for b in _band_layouts(cap, width, u, gap, aspect, limits, max_columns, max_splits, max_levels))
            per_band.append([b for _, b in ranked[:band_top]])
        candidates = (' / '.join(c) for c in product(*per_band))
    ranked, seen = [], set()
    for spec in candidates:
        key = min(spec, mirror(spec))
        if key in seen:
            continue
        seen.add(key)
        if not n_items[0] <= len(place(spec, width, u, gap, aspect)) <= n_items[1]:
            continue
        s = score(spec, width, u, gap, aspect, n_target)
        if np.isfinite(s):
            ranked.append((s, spec))
    ranked.sort()
    return ranked[:top]
