/* Offline Gaussian KDE overlays. Scott bandwidth × 1.2; approximate 80/50%
 * probability-mass contours on a uniform grid. No clustering happens here.
 * Embedded into the gallery at render time; no network or JS dependencies.
 */
function assignedGroups(coordinates, rows, labels) {
  const groups = {};
  rows.forEach((row, i) => {
    const label = labels[row];
    if (label >= 0) (groups[label] ??= []).push(coordinates[i]);
  });
  return groups;
}

function densityContours(points, gridSize = 48) {
  const n = points.length;
  if (n < 3) return [];
  const mean = [0, 0];
  points.forEach(p => { mean[0] += p[0] / n; mean[1] += p[1] / n; });
  let xx = 0, xy = 0, yy = 0;
  points.forEach(p => {
    const dx = p[0] - mean[0], dy = p[1] - mean[1];
    xx += dx * dx / (n - 1); xy += dx * dy / (n - 1); yy += dy * dy / (n - 1);
  });
  const bandwidth2 = (1.2 * Math.pow(n, -1 / 6)) ** 2;
  xx *= bandwidth2; xy *= bandwidth2; yy *= bandwidth2;
  const determinant = xx * yy - xy * xy;
  // A 2D KDE is undefined for coincident/collinear samples; do not invent blobs.
  if (!Number.isFinite(determinant) || determinant <= 0 || determinant <= xx * yy * 1e-12) return [];
  const min = [Math.min(...points.map(p => p[0])) - 3 * Math.sqrt(xx),
               Math.min(...points.map(p => p[1])) - 3 * Math.sqrt(yy)];
  const max = [Math.max(...points.map(p => p[0])) + 3 * Math.sqrt(xx),
               Math.max(...points.map(p => p[1])) + 3 * Math.sqrt(yy)];
  const step = [(max[0] - min[0]) / (gridSize - 1), (max[1] - min[1]) / (gridSize - 1)];
  const density = new Float64Array(gridSize * gridSize);
  for (let j = 0; j < gridSize; j++) for (let i = 0; i < gridSize; i++) {
    const x = min[0] + i * step[0], y = min[1] + j * step[1];
    let value = 0;
    for (const p of points) {
      const dx = x - p[0], dy = y - p[1];
      value += Math.exp(-.5 * (yy * dx * dx - 2 * xy * dx * dy + xx * dy * dy) / determinant);
    }
    density[j * gridSize + i] = value;
  }
  // Uniform cell area and Gaussian normalizer cancel when ranking grid mass.
  const sorted = Array.from(density).sort((a, b) => b - a), total = sorted.reduce((a, b) => a + b, 0);
  if (!(total > 0)) return [];
  return [.8, .5].map(mass => {
    let sum = 0, threshold = sorted[sorted.length - 1];
    for (const value of sorted) { sum += value; if (sum >= mass * total) { threshold = value; break; } }
    return {mass, path: contourPath(density, gridSize, min, step, threshold * (1 + 1e-12))};
  }).filter(level => level.path);
}

function contourPath(values, n, min, step, threshold) {
  // Marching squares with interpolated crossings and shared grid-edge IDs.
  const nodes = new Map(), adjacent = new Map();
  function edge(horizontal, x, y) {
    const key = `${horizontal ? 'h' : 'v'}:${x}:${y}`;
    if (!nodes.has(key)) {
      const a = values[y * n + x], b = values[(y + (horizontal ? 0 : 1)) * n + x + (horizontal ? 1 : 0)];
      const t = Math.max(0, Math.min(1, (threshold - a) / (b - a)));
      nodes.set(key, [min[0] + (x + (horizontal ? t : 0)) * step[0], min[1] + (y + (horizontal ? 0 : t)) * step[1]]);
    }
    return key;
  }
  const pairs = {1:[[3,0]], 2:[[0,1]], 3:[[3,1]], 4:[[1,2]], 6:[[0,2]],
    7:[[3,2]], 8:[[2,3]], 9:[[2,0]], 11:[[1,2]], 12:[[1,3]], 13:[[0,1]], 14:[[3,0]]};
  for (let y = 0; y < n - 1; y++) for (let x = 0; x < n - 1; x++) {
    const v = [values[y*n+x], values[y*n+x+1], values[(y+1)*n+x+1], values[(y+1)*n+x]];
    const mask = v.reduce((a, b, i) => a | (b >= threshold ? 1 << i : 0), 0);
    if (mask === 0 || mask === 15) continue;
    let segments = pairs[mask];
    if (mask === 5 || mask === 10) {
      const centerHigh = v.reduce((a,b) => a+b, 0) / 4 >= threshold;
      segments = (mask === 5) === centerHigh ? [[0,1],[2,3]] : [[3,0],[1,2]];
    }
    // Only ask for crossing edges: non-crossings can have identical densities.
    const getEdge = i => i === 0 ? edge(true,x,y) : i === 1 ? edge(false,x+1,y) : i === 2 ? edge(true,x,y+1) : edge(false,x,y);
    for (const [a,b] of segments) {
      const first = getEdge(a), second = getEdge(b);
      if (!adjacent.has(first)) adjacent.set(first, []);
      if (!adjacent.has(second)) adjacent.set(second, []);
      adjacent.get(first).push(second); adjacent.get(second).push(first);
    }
  }
  const visited = new Set(), paths = [];
  for (const start of adjacent.keys()) {
    if (visited.has(start)) continue;
    const loop = [];
    let current = start, previous = null, closed = false;
    while (!visited.has(current)) {
      visited.add(current); loop.push(nodes.get(current));
      const next = (adjacent.get(current) || []).find(key => key !== previous);
      if (next === undefined) break;
      if (next === start) { closed = true; break; }
      previous = current; current = next;
    }
    if (closed && loop.length >= 3) paths.push('M' + loop.map(p => p.map(v => v.toFixed(3)).join(',')).join('L') + 'Z');
  }
  return paths.join('');
}

if (typeof module !== 'undefined' && module.exports) module.exports = {assignedGroups, densityContours, contourPath};
