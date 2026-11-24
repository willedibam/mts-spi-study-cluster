"""Brain meshes and species silhouettes for fig:mi (d), from public downloads, into <figwin>/assets.npz.

    python fig_assets.py <download dir> <figwin dir>

Expected in <download dir> (fetched 23 November 2026; none is committed):
  pial_left.gii.gz   fsaverage5 left pial surface (MNI305), nilearn repository
                     https://raw.githubusercontent.com/nilearn/nilearn/main/nilearn/datasets/data/fsaverage5/pial_left.gii.gz
  997.obj            Allen Mouse CCF 2017 whole-brain ("root") mesh, Allen Institute (cite the CCFv3, Wang et al. 2020)
                     http://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/annotation/ccf_2017/structure_meshes/997.obj
  NMT_v2.0_sym/      NMT v2.0 symmetric macaque template (cite Jung et al. 2021); only its surfaces/lh.gray_surface.rsl.gii, from
                     https://afni.nimh.nih.gov/pub/dist/atlases/macaque/nmt/NMT_v2.0_sym.tgz
  NMT_v2.0_sym/NMT_v2.0_sym_05mm/CHARM_in_NMT_v2.0_sym_05mm.nii.gz
                     the CHARM hierarchical macaque atlas (cite Jung et al. 2021) from the same tarball, for the region maps
  top.nrrd           Allen CCF 2017 isocortex areas seen from above, one hemisphere, 10 µm (ccf_streamlines assets; cite
                     Wang et al. 2020): https://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/
                     cortical_coordinates/ccf_2017/ccf_streamlines_assets/master_updated/top.nrrd
  ggseg-0.1/         the MIT-licensed ggseg 0.1 sdist from PyPI, unpacked (its Desikan-Killiany 2D outlines; cite Mowinckel
                     & Vidal-Piñeiro 2020); read as data, not installed
  macaque.svg, mouse.svg, human.svg
                     PhyloPic silhouettes, all CC0 1.0: Macaca mulatta by Matthew Schmitz (4e9c5666-79cc-4766-a7d3-d547514b0d77),
                     Mus musculus by Daniel Jaron (92989e35-4e68-4a2d-b3a2-191ba9da671a), Homo sapiens by T. Michael Keesey
                     (f33e111a-7257-4ad0-a2c1-663663784d3f); https://images.phylopic.org/images/<id>/vector.svg

Stores per species `<name>_v` (vertices, float32) and `<name>_f` (faces, int32) in the source's own space, and the
silhouettes as matplotlib path arrays `<name>_sv` (vertices) and `<name>_sc` (codes). Region maps (fig:mi (d) "map"
icons): `macaque_lab` (CHARM level-1 and level-2 label per vertex, majority-smoothed over the mesh), `mouse_map` (the top
view's area labels, 20 µm, one hemisphere, midline at the last column), and the left lateral DK outlines as
`human_map_v`, `human_map_c` (path arrays, SVG units, y down), `human_map_s` (subpath starts) and `human_map_o`/`_oc`
(the hemisphere's outline).
"""
import base64
import gzip
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET
import zlib

import numpy as np
from matplotlib.path import Path as MPath


def gifti(path):
    """Vertices and faces of a GIfTI surface (base64, optionally zlib-compressed), without nibabel."""
    raw = Path(path).read_bytes()
    root = ET.fromstring(gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw)
    out = {}
    for da in root.iter("DataArray"):
        kind = da.get("Intent")
        dt = {"NIFTI_TYPE_FLOAT32": "<f4", "NIFTI_TYPE_INT32": "<i4"}[da.get("DataType")]
        b = base64.b64decode(da.find("Data").text)
        if da.get("Encoding") == "GZipBase64Binary":
            b = zlib.decompress(b)
        dims = [int(da.get(f"Dim{i}")) for i in range(int(da.get("Dimensionality")))]
        a = np.frombuffer(b, dtype=np.dtype(dt).newbyteorder(">" if da.get("Endian") == "BigEndian" else "<"))
        out[kind] = a.reshape(dims, order="F" if da.get("ArrayIndexingOrder") == "ColumnMajorOrder" else "C")
    return out["NIFTI_INTENT_POINTSET"].astype(np.float32), out["NIFTI_INTENT_TRIANGLE"].astype(np.int32)


def obj(path):
    v, f = [], []
    for line in open(path):
        if line.startswith("v "):
            v.append(line.split()[1:4])
        elif line.startswith("f "):
            f.append([int(t.split("/")[0]) - 1 for t in line.split()[1:4]])
    return np.array(v, np.float32), np.array(f, np.int32)


def potrace_svg(path):
    """A potrace SVG's path data (M/m, L/l, C/c, Z) as matplotlib path arrays, y up, in the SVG's own units."""
    s = Path(path).read_text()
    d = " ".join(re.findall(r'\sd="([^"]*)"', s))
    toks = re.findall(r"[MmLlCcZz]|-?\d*\.?\d+(?:e-?\d+)?", d)
    V, C = [], []
    cur = start = np.zeros(2); i = 0; cmd = None
    while i < len(toks):
        if re.fullmatch(r"[A-Za-z]", toks[i]):
            cmd = toks[i]; i += 1
            if cmd in "Zz":
                V.append(start); C.append(MPath.CLOSEPOLY); cur = start
                continue
        n = {"M": 2, "m": 2, "L": 2, "l": 2, "C": 6, "c": 6}[cmd]
        p = np.array(toks[i:i + n], float).reshape(-1, 2); i += n
        if cmd.islower():
            p = p + cur
        if cmd in "Mm":
            V.append(p[0]); C.append(MPath.MOVETO); cur = start = p[0]
            cmd = "l" if cmd == "m" else "L"                   # implicit lineto after moveto
        elif cmd in "Ll":
            V.append(p[0]); C.append(MPath.LINETO); cur = p[0]
        else:
            V.extend(p); C.extend([MPath.CURVE4] * 3); cur = p[2]
    return np.array(V, np.float32) * .1, np.array(C, np.uint8)         # potrace's scale(0.1, -0.1), y kept up


def nifti(path):
    """A gzipped little-endian NIfTI-1 volume and its sform affine (no scaling; enough for CHARM)."""
    b = gzip.decompress(Path(path).read_bytes())
    dim = np.frombuffer(b[40:56], "<i2"); dt = {2: "u1", 4: "<i2", 8: "<i4", 16: "<f4", 512: "<u2"}[int(np.frombuffer(b[70:72], "<i2")[0])]
    off = int(np.frombuffer(b[108:112], "<f4")[0]); shape = tuple(int(x) for x in dim[1:dim[0] + 1])
    a = np.frombuffer(b[off:off + int(np.prod(shape)) * np.dtype(dt).itemsize], dt).reshape(shape, order="F")
    return a, np.vstack([np.frombuffer(b[280:328], "<f4").reshape(3, 4), [0, 0, 0, 1]])


def charm_labels(path, v, f, levels=(0, 1), smooth=4):
    """CHARM labels at the surface's vertices: each vertex takes its voxel's label, or the nearest labelled voxel's,
    then `smooth` rounds of a majority vote over its mesh neighbours (removes speckle in deep sulci)."""
    from scipy.ndimage import distance_transform_edt
    from scipy.sparse import coo_matrix
    a, A = nifti(path); a = a.reshape(a.shape[:3] + (-1,))
    ijk = np.rint((np.c_[v, np.ones(len(v))] @ np.linalg.inv(A).T)[:, :3]).astype(int)
    e = np.r_[f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]
    adj = coo_matrix((np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(len(v),) * 2).tocsr()
    out = []
    for lv in levels:
        L = a[..., lv]; idx = distance_transform_edt(L == 0, return_distances=False, return_indices=True)
        lab = L[tuple(idx[:, ijk[:, 0], ijk[:, 1], ijk[:, 2]])]
        u, k = np.unique(lab, return_inverse=True)
        for _ in range(smooth):
            oh = coo_matrix((np.ones(len(k)), (np.arange(len(k)), k)), shape=(len(k), len(u))).tocsr()
            k = np.asarray((adj @ oh + oh).argmax(1)).ravel()
        out.append(u[k])
    return np.array(out, np.int16)


def nrrd2d(path):
    """A gzip-encoded 2-D NRRD (uint16), rows first."""
    raw = Path(path).read_bytes(); i = raw.index(b"\n\n") + 2; hdr = raw[:i].decode()
    assert "encoding: gzip" in hdr and "type: uint16" in hdr and "endian: little" in hdr
    sizes = [int(x) for x in re.search(r"sizes: (.*)", hdr).group(1).split()]
    return np.frombuffer(gzip.decompress(raw[i:]), "<u2").reshape(sizes[::-1])


def svg_abs(s):
    """ggseg's SVG path strings (absolute M, L, Q, C, Z; comma pairs) as matplotlib path arrays."""
    cm = {"M": MPath.MOVETO, "L": MPath.LINETO, "Q": MPath.CURVE3, "C": MPath.CURVE4, "Z": MPath.CLOSEPOLY}
    V, C = [], []; t = re.split("([A-Za-z])", s)[1:]; start = np.zeros((1, 2))
    for c, vals in zip(t[::2], t[1::2]):
        p = np.array([e.split(",") for e in vals.split()], float).reshape(-1, 2) if c.upper() != "Z" else start
        if c.upper() == "M":
            start = p[:1]
        V.append(p); C += [cm[c.upper()]] * len(p)
    return np.concatenate(V), np.array(C, np.uint8)


def ggseg_dk_lateral_left(root):
    """The left lateral view of ggseg's Desikan-Killiany 2D atlas: every left region's subpaths that lie inside the
    lateral view's outline box (the same files also hold the medial view)."""
    dk = Path(root) / "ggseg/data/dk"; ov, oc = svg_abs((dk / "lateral_left").read_text())
    e = MPath(ov, oc).get_extents(); lo, hi = np.array([e.x0, e.y0]), np.array([e.x1, e.y1])
    V, C, S = [], [], []; n = 0
    for fn in sorted(dk.glob("*_left")):
        if fn.name in ("lateral_left", "medial_left"):
            continue
        v, c = svg_abs(fn.read_text()); st = np.flatnonzero(c == MPath.MOVETO).tolist() + [len(c)]
        for a, b in zip(st[:-1], st[1:]):
            m = v[a:b][c[a:b] != MPath.CLOSEPOLY].mean(0)
            if (lo <= m).all() and (m <= hi).all():
                V.append(v[a:b]); C.append(c[a:b]); S.append(n); n += b - a
    return np.concatenate(V).astype(np.float32), np.concatenate(C), np.array(S), ov.astype(np.float32), oc


def macaque(nmt):
    """The NMT v2.0 symmetric template's left gray-matter (pial) surface."""
    return gifti(Path(nmt) / "NMT_v2.0_sym_surfaces/lh.gray_surface.rsl.gii")


if __name__ == "__main__":
    dl, figwin = Path(sys.argv[1]), Path(sys.argv[2])
    out = {}
    for name, (v, f) in (("human", gifti(dl / "pial_left.gii.gz")), ("mouse", obj(dl / "997.obj")),
                         ("macaque", macaque(dl / "NMT_v2.0_sym"))):
        out[name + "_v"], out[name + "_f"] = v, f
        print(name, v.shape, f.shape, v.min(0).round(1), v.max(0).round(1))
    for name in ("macaque", "mouse", "human"):
        out[name + "_sv"], out[name + "_sc"] = potrace_svg(dl / f"{name}.svg")
    out["macaque_lab"] = charm_labels(dl / "NMT_v2.0_sym/NMT_v2.0_sym_05mm/CHARM_in_NMT_v2.0_sym_05mm.nii.gz",
                                      out["macaque_v"], out["macaque_f"])
    out["mouse_map"] = nrrd2d(dl / "top.nrrd")[::2, ::2][:, :285]          # 20 µm; columns up to the midline (570 at 10 µm)
    (out["human_map_v"], out["human_map_c"], out["human_map_s"], out["human_map_o"],
     out["human_map_oc"]) = ggseg_dk_lateral_left(dl / "ggseg-0.1")
    np.savez_compressed(figwin / "assets.npz", **out)
