"""Brain meshes and species silhouettes for fig:mi (d), from public downloads, into <figwin>/assets.npz.

    python fig_assets.py <download dir> <figwin dir>

Expected in <download dir> (fetched 23 November 2026; none is committed):
  pial_left.gii.gz   fsaverage5 left pial surface (MNI305), nilearn repository
                     https://raw.githubusercontent.com/nilearn/nilearn/main/nilearn/datasets/data/fsaverage5/pial_left.gii.gz
  997.obj            Allen Mouse CCF 2017 whole-brain ("root") mesh, Allen Institute (cite the CCFv3, Wang et al. 2020)
                     http://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/annotation/ccf_2017/structure_meshes/997.obj
  NMT_v2.0_sym/      NMT v2.0 symmetric macaque template (cite Jung et al. 2021); only its surfaces/lh.gray_surface.rsl.gii, from
                     https://afni.nimh.nih.gov/pub/dist/atlases/macaque/nmt/NMT_v2.0_sym.tgz
  macaque.svg, mouse.svg, human.svg
                     PhyloPic silhouettes, all CC0 1.0: Macaca mulatta by Matthew Schmitz (4e9c5666-79cc-4766-a7d3-d547514b0d77),
                     Mus musculus by Daniel Jaron (92989e35-4e68-4a2d-b3a2-191ba9da671a), Homo sapiens by T. Michael Keesey
                     (f33e111a-7257-4ad0-a2c1-663663784d3f); https://images.phylopic.org/images/<id>/vector.svg

Stores per species `<name>_v` (vertices, float32) and `<name>_f` (faces, int32) in the source's own space, and the
silhouettes as matplotlib path arrays `<name>_sv` (vertices) and `<name>_sc` (codes).
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
    np.savez_compressed(figwin / "assets.npz", **out)
