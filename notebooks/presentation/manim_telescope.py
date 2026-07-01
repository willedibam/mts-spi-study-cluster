"""Manim telescope: SPI-SPI methods pipeline as a self-similar zoom, L0 -> L4.

One dataset throughout (M10 var-phi-0.95 instance, full 297-SPI library + in the feature matrix).
Built as a *persistent board*: the camera roams to each stage, finished stages stay parked, and
each stage's output flows into the next via TransformFromCopy (source stays, a copy morphs forward).
A final pull-back reveals the whole evolution. Storyboard: manim_telescope_storyboard.md.

Render (from project root):
    .venv/bin/manim -ql notebooks/presentation/manim_telescope.py Telescope
Per-level (built centered, no roam/continuity):
    .venv/bin/manim -ql notebooks/presentation/manim_telescope.py L0   # L1 L2 L3 L4
"""
from pathlib import Path

import numpy as np
import matplotlib
import seaborn as sns
from scipy.stats import zscore, rankdata
from statsmodels.nonparametric.smoothers_lowess import lowess
from manim import *

# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[2]
M10_DIR = ROOT / "data/embeddings/proof_benchmarked90_260603/var-phi-0.95_cpl-0.4/M10_T500_I9"
FEAT_ROW = 749
TRACE = (4, 7)

ICEFIRE = sns.color_palette("icefire", as_cmap=True)
GRAYCM = matplotlib.colormaps["gray"]
VIRIDIS = matplotlib.colormaps["viridis"]
SPI_COL = {"r": ManimColor("#4C72B0"), "rho": ManimColor("#DD8452"), "MI": ManimColor("#55A868")}
SPI_TEX = {"r": "r", "rho": r"\rho", "MI": r"\mathrm{MI}_r"}
REG = ManimColor("#E14BD2")
BOUNDS = {"r": (-1.0, 1.0), "rho": (-1.0, 1.0), "MI": (0.0, None)}

TT = TexTemplate()
TT.add_to_preamble(r"\usepackage{mathdots}")     # for \iddots


def linfoot(mi):
    return float(np.sqrt(np.clip(1 - np.exp(-2 * max(float(mi), 0.0)), 0.0, 1.0)))


def heat_rgb(mat, cmap, vmin, vmax):
    norm = matplotlib.colors.Normalize(vmin, vmax)
    return (cmap(norm(np.asarray(mat, float)))[..., :3] * 255).astype(np.uint8)


def mcolor(v, cmap, vmin, vmax):
    norm = matplotlib.colors.Normalize(vmin, vmax)
    return rgb_to_color(cmap(norm(v))[:3])


def offdiag_lower(mat):
    n = mat.shape[0]
    sym = 0.5 * (mat + mat.T)
    vals, posmap, k = [], {}, 0
    for i in range(n):
        for j in range(i):
            posmap[(i, j)] = k
            vals.append(sym[i, j])
            k += 1
    return np.array(vals), posmap


def bounds(key, mat):
    lo, hi = BOUNDS[key]
    return (lo, float(np.nanmax(mat)) if hi is None else hi)


def load_m10():
    ts = np.load(M10_DIR / "timeseries.npy").astype(float)
    npz = np.load(M10_DIR / "spi_mpis.npz")
    names = list(npz.files)
    mpis = {"r": np.asarray(npz["cov_EmpiricalCovariance"], float),
            "rho": np.asarray(npz["spearmanr"], float),
            "MI": np.asarray(npz["mi_kraskov_NN-4"], float)}
    used = {"cov_EmpiricalCovariance", "spearmanr", "mi_kraskov_NN-4"}
    extra = [np.asarray(npz[n], float) for n in names if n not in used][:2]
    last = np.asarray(npz[names[-1]], float)
    return ts, mpis, len(names), extra, last


# --------------------------------------------------------------------------- #
class Telescope(MovingCameraScene):
    def construct(self):
        self.load()
        S0 = LEFT * 18.5 + UP * 0.4
        S1 = LEFT * 8.5 + UP * 0.4
        S2 = RIGHT * 3.0 + UP * 0.5
        S3 = RIGHT * 14.0 + UP * 0.4
        S4 = RIGHT * 25.5 + UP * 0.3
        # keep ~2 stages in frame so each output is seen morphing into the next input
        self.camera.frame.move_to(S0 + RIGHT * 1.8).set(width=15)

        d0 = self.build_l0(S0)
        self.frame_pair(S0, S1); d1 = self.build_l1(S1, d0)
        self.frame_pair(S1, S2); d2 = self.build_l2(S2, d1["bars"])
        self.frame_pair(S2, S3); d3 = self.build_l3(S3, d2["fronts"])
        self.frame_pair(S3, S4); self.build_l4(S4, d3["f_mob"])

        stages = [d0["group"], d1["group"], d2["group"], d3["group"], self.l4_group]
        board = Group(*stages)
        self.play(self.camera.frame.animate.move_to(board.get_center()).set(width=board.width * 1.06),
                  run_time=2.4)
        arrows = VGroup(*[Arrow(a.get_right(), b.get_left(), buff=0.3, stroke_width=3, color=GRAY_B,
                                max_tip_length_to_length_ratio=0.02) for a, b in zip(stages, stages[1:])])
        self.play(LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.25), run_time=2)
        self.wait(1.2)

    def load(self):
        self.ts, self.mpis, self.K, self.stack_extra, self.last_mpi = load_m10()
        self.M = self.mpis["r"].shape[0]
        i, j = TRACE
        self.vals = {"r": float(self.mpis["r"][i, j]), "rho": float(self.mpis["rho"][i, j]),
                     "MI": linfoot(self.mpis["MI"][i, j])}

    def cam(self, center, width):
        self.play(self.camera.frame.animate.move_to(center).set(width=width), run_time=1.1)

    def frame_pair(self, a, b, pad=9.5):
        """Frame two consecutive stages together so the output->input morph between them is visible."""
        center = (a + b) / 2 + UP * 0.2
        self.play(self.camera.frame.animate.move_to(center).set(width=abs(b[0] - a[0]) + pad), run_time=1.2)

    # -- L0: MTS heatmap with a time-axis break ---------------------------- #
    def build_l0(self, at):
        Z = zscore(self.ts, axis=0).T
        C, (W, H) = 70, (3.9, 3.0)
        img_l = self._img(heat_rgb(Z[:, :C], ICEFIRE, -2, 2), W, H)
        img_r = self._img(heat_rgb(Z[:, -C:], ICEFIRE, -2, 2), W, H)
        img_l.move_to(at + LEFT * 2.25)
        img_r.next_to(img_l, RIGHT, buff=0.5)
        grp = Group(img_l, img_r)

        gx = (img_l.get_right()[0] + img_r.get_left()[0]) / 2
        def brk(y):
            return VGroup(Line([gx - 0.11, y - 0.13, 0], [gx - 0.01, y + 0.13, 0]),
                          Line([gx + 0.01, y - 0.13, 0], [gx + 0.11, y + 0.13, 0])).set_stroke(WHITE, 3)
        breaks = VGroup(brk(img_l.get_top()[1]), brk(img_l.get_bottom()[1]))
        M_lab = MathTex("M").next_to(grp, LEFT, buff=0.3)
        T_lab = MathTex("T").next_to(grp, DOWN, buff=0.35)

        rh, top = H / self.M, img_r.get_top()[1]
        braces = VGroup()
        anchors = {}
        for ch in TRACE:
            yc = top - (ch + 0.5) * rh
            edge = Line([img_r.get_right()[0], yc - rh / 2, 0], [img_r.get_right()[0], yc + rh / 2, 0])
            br = Brace(edge, RIGHT, buff=0.05)
            lab = MathTex(rf"X_t^{{({ch})}}").scale(0.7).next_to(br, RIGHT, buff=0.1).set_color(WHITE)
            braces.add(br, lab)
            anchors[ch] = lab.get_right()

        title = Tex("MTS", font_size=34).next_to(grp, UP, buff=0.3)
        self.play(FadeIn(grp), FadeIn(breaks), Write(M_lab), Write(T_lab), Write(title))
        self.play(*[GrowFromCenter(b) for b in braces if isinstance(b, Brace)],
                  *[Write(b) for b in braces if isinstance(b, MathTex)])
        self.wait(0.3)
        return {"group": Group(grp, breaks, M_lab, T_lab, braces, title), "anchor": anchors}

    # -- L1: three SPI phase panels + Linfoot bars ------------------------- #
    def build_l1(self, at, d0):
        i, j = TRACE
        N = 180
        idx = np.linspace(0, self.ts.shape[0] - 1, N).astype(int)
        Xi, Xj = zscore(self.ts[:, i])[idx], zscore(self.ts[:, j])[idx]
        panels = VGroup(self._panel("pearson", Xi, Xj), self._panel("spearman", rankdata(Xi), rankdata(Xj)),
                        self._panel("mi", Xi, Xj)).arrange(RIGHT, buff=0.5).move_to(at + UP * 1.4)
        bars = self._bars().next_to(panels, DOWN, buff=0.7)
        title = Tex(rf"extract pair $(X^{{({i})}}, X^{{({j})}})$", font_size=30).next_to(panels, UP, buff=0.3)

        conn = None
        if d0 and "anchor" in d0:
            src = d0["anchor"][i]
            conn = Arrow(src + RIGHT * 0.1, panels.get_left() + LEFT * 0.15, buff=0.2,
                         stroke_width=3, color=GRAY_B, max_tip_length_to_length_ratio=0.06)
        if conn is not None:
            self.play(GrowArrow(conn), Write(title))
        else:
            self.play(Write(title))
        self.play(LaggedStart(*[Create(p[0]) for p in panels], lag_ratio=0.3))
        self.play(LaggedStart(*[FadeIn(p[1], scale=0.6) for p in panels], lag_ratio=0.3, run_time=1.5))
        self.play(*[Create(p[2]) for p in panels], *[Write(p[3]) for p in panels])
        self.play(LaggedStart(*[GrowFromEdge(b[0], DOWN) for b in bars["bars"]], lag_ratio=0.2),
                  LaggedStart(*[Write(b[1]) for b in bars["bars"]], lag_ratio=0.2),
                  LaggedStart(*[Write(b[2]) for b in bars["bars"]], lag_ratio=0.2),
                  Create(bars["base"]), Write(bars["ylab"]))
        self.wait(0.3)
        grp = Group(panels, bars, title, *( [conn] if conn is not None else [] ))
        return {"group": grp, "bars": {k: bars["bars"][n][0] for n, k in enumerate(("r", "rho", "MI"))}}

    def _panel(self, kind, xs, ys):
        raw = kind != "spearman"
        lo, hi = (-3, 3) if raw else (0, len(xs) + 1)
        ax = Axes(x_range=[lo, hi], y_range=[lo, hi], x_length=2.7, y_length=2.7,
                  tips=False, axis_config={"include_ticks": False})
        dots = VGroup(*[Dot(ax.c2p(x, y), radius=0.028, color=GRAY_C, fill_opacity=0.5)
                        for x, y in zip(xs, ys)])
        if kind == "mi":
            sm = lowess(ys, xs, frac=0.5)
            fit = VMobject(color=REG, stroke_width=4).set_points_smoothly([ax.c2p(x, y) for x, y in sm])
            title = MathTex(r"\mathrm{MI}\ \text{(Kraskov)}").scale(0.55)
        else:
            m, b = np.polyfit(xs, ys, 1)
            xr = [lo + 0.3, hi - 0.3]
            fit = Line(ax.c2p(xr[0], m * xr[0] + b), ax.c2p(xr[1], m * xr[1] + b), color=REG, stroke_width=4)
            title = MathTex(r"\text{Pearson}" if kind == "pearson" else r"\text{Spearman (ranks)}").scale(0.55)
        title.next_to(ax, UP, buff=0.15)
        return VGroup(ax, dots, fit, title)

    def _bars(self):
        base_h = 2.0
        row = VGroup()
        for key in ("r", "rho", "MI"):
            v = self.vals[key]
            rect = Rectangle(width=0.55, height=max(base_h * v, 1e-3),
                             fill_color=SPI_COL[key], fill_opacity=1, stroke_width=0)
            val = MathTex(f"{v:.2f}").scale(0.55).next_to(rect, UP, buff=0.06)
            klab = MathTex(SPI_TEX[key]).scale(0.65).next_to(rect, DOWN, buff=0.12)
            row.add(VGroup(rect, val, klab))
        for b in row:
            b[0].align_to(row[0][0], DOWN)
        row.arrange(RIGHT, buff=0.5, aligned_edge=DOWN)
        base = Line(row.get_corner(DL) + LEFT * 0.15, row.get_corner(DR) + RIGHT * 0.15, stroke_width=1.5, color=GRAY_B)
        ylab = Tex("SPI value (MI: Linfoot)").scale(0.5).rotate(PI / 2).next_to(row, LEFT, buff=0.2)
        return VDict({"bars": row, "base": VGroup(base), "ylab": VGroup(ylab)})

    # -- L2: 45-degree stack of K MPIs ------------------------------------- #
    def build_l2(self, at, bar_src=None):
        self.stack_base = at + LEFT * 2.6 + DOWN * 1.85     # centers the 45-deg stack on `at`
        u = RIGHT * 0.85 + UP * 0.62
        fronts = {"r": self._mpi_grid("r"), "rho": self._mpi_grid("rho"), "MI": self._mpi_grid("MI")}
        for p, key in enumerate(("r", "rho", "MI")):
            fronts[key].move_to(self.stack_base).shift(u * p).set_z_index(30 - p)
        cards = Group(*[self._mpi_card(m, 3 + d, u) for d, m in enumerate(self.stack_extra)])
        ell = MathTex(r"\iddots", tex_template=TT).scale(1.5).move_to(self.stack_base + u * 5).set_z_index(5)
        back = self._mpi_card(self.last_mpi, 6, u)

        kbrace = BraceBetweenPoints(fronts["r"].get_corner(UL), back.get_corner(UL),
                                    direction=normalize(np.array([-1.0, 1.0, 0.0])))
        klab = kbrace.get_tex(rf"K = {self.K}\ \text{{SPIs}}").scale(0.7)
        mbrace = Brace(fronts["r"][0], DOWN, buff=0.1)          # bracket below the MPIs
        mlab = mbrace.get_tex("M").scale(0.8)

        self.play(FadeIn(back), *[FadeIn(c) for c in cards], FadeIn(ell))
        self.play(LaggedStart(*[FadeIn(fronts[k], shift=u * 0.3) for k in ("MI", "rho", "r")], lag_ratio=0.25))
        self.play(GrowFromCenter(kbrace), Write(klab), GrowFromCenter(mbrace), Write(mlab))
        # continuity: L1 bars -> the (i,j) cell of each front MPI
        ti, tj = TRACE
        if bar_src is not None:
            self.play(*[TransformFromCopy(bar_src[k], fronts[k][0][ti * self.M + tj].copy())
                        for k in ("r", "rho", "MI")], run_time=1.4)
        self.wait(0.3)
        grp = Group(*cards, ell, back, *fronts.values(), kbrace, klab, mbrace, mlab)
        return {"group": grp, "fronts": fronts}

    def _mpi_grid(self, key, cell=0.30):
        mat = self.mpis[key]
        vmin, vmax = bounds(key, mat)
        n = self.M
        cells = VGroup()
        for i in range(n):
            for j in range(n):
                sq = Square(side_length=cell, stroke_width=0)
                if i == j:
                    sq.set_fill(BLACK, 0).set_stroke(GRAY_D, 0.5)
                else:
                    sq.set_fill(mcolor(mat[i, j], GRAYCM, vmin, vmax), 1)
                cells.add(sq)
        cells.arrange_in_grid(rows=n, cols=n, buff=0)
        ti, tj = TRACE
        border = cells[ti * n + tj].copy().set_fill(opacity=0).set_stroke(SPI_COL[key], 4)
        title = MathTex(SPI_TEX[key]).scale(0.8).next_to(cells, UP, buff=0.12).set_color(SPI_COL[key])
        return VGroup(cells, border, title)

    def _mpi_card(self, mat, p, u, cell=0.30):
        img = self._img(heat_rgb(mat, GRAYCM, np.nanmin(mat), np.nanmax(mat)), cell * self.M, cell * self.M)
        img.move_to(self.stack_base).shift(u * p).set_z_index(20 - p)
        img.set_opacity(max(0.25, 0.7 - 0.12 * p))
        return img

    # -- L3: unravel off-diagonals -> barcode -> SPI-SPI scatter ----------- #
    def build_l3(self, at, fronts=None):
        vec_r, posmap = offdiag_lower(self.mpis["r"])
        vec_rho, _ = offdiag_lower(self.mpis["rho"])
        tk = posmap[(max(TRACE), min(TRACE))]

        bar_r = self._barcode(vec_r, "r", tk)
        bar_rho = self._barcode(vec_rho, "rho", tk)
        VGroup(bar_r, bar_rho).arrange(DOWN, buff=0.5, aligned_edge=LEFT).move_to(at + UP * 2.4)

        ax = Axes(x_range=[-1, 1, 0.5], y_range=[-1, 1, 0.5], x_length=4, y_length=4,
                  tips=False, axis_config={"include_ticks": False}).move_to(at + DOWN * 1.6 + LEFT * 1.6)
        xlab = MathTex("r").scale(0.9).next_to(ax.x_axis, RIGHT, buff=0.15).set_color(SPI_COL["r"])
        ylab = MathTex(r"\rho").scale(0.9).next_to(ax.y_axis, UP, buff=0.15).set_color(SPI_COL["rho"])
        dots = VGroup(*[Dot(ax.c2p(xv, yv), radius=0.05, color=SPI_COL["r"] if k == tk else GRAY_C,
                            fill_opacity=1 if k == tk else 0.5) for k, (xv, yv) in enumerate(zip(vec_r, vec_rho))])
        m, b = np.polyfit(vec_r, vec_rho, 1)
        reg = ax.plot(lambda x: m * x + b, x_range=[-0.9, 0.98], color=REG, stroke_width=4)
        f = float(np.corrcoef(vec_r, vec_rho)[0, 1])
        flab = MathTex(rf"f_{{ij}} = {f:.2f}").scale(0.75).next_to(ax, RIGHT, buff=0.6).shift(UP * 0.6)
        note = Tex("each point $=$ a channel-pair").scale(0.5).next_to(flab, DOWN, buff=0.4, aligned_edge=LEFT)

        # continuity: front-MPI off-diagonals -> barcodes -> scatter
        if fronts is not None:
            src_r = self._offdiag_cells(fronts["r"][0])
            src_rho = self._offdiag_cells(fronts["rho"][0])
            self.play(TransformFromCopy(src_r, bar_r[0]), TransformFromCopy(src_rho, bar_rho[0]), run_time=1.6)
            self.play(FadeIn(bar_r[1]), FadeIn(bar_r[2]), FadeIn(bar_rho[1]), FadeIn(bar_rho[2]))
        else:
            self.play(FadeIn(bar_r), FadeIn(bar_rho))
        self.play(Create(ax), Write(xlab), Write(ylab))
        self.play(TransformFromCopy(bar_r[0], dots), TransformFromCopy(bar_rho[0], dots), run_time=1.5)
        self.play(Create(reg), Write(flab), FadeIn(note))
        self.wait(0.3)
        grp = VGroup(bar_r, bar_rho, ax, xlab, ylab, dots, reg, flab, note)
        return {"group": grp, "f_mob": flab}

    def _offdiag_cells(self, grid_cells):
        n = self.M
        return VGroup(*[grid_cells[i * n + j] for i in range(n) for j in range(i)])

    def _barcode(self, vec, key, trace_k, cell=0.20):
        vmin, vmax = bounds(key, self.mpis[key])
        cells = VGroup(*[Square(side_length=cell, stroke_width=0).set_fill(mcolor(v, GRAYCM, vmin, vmax), 1)
                         for v in vec]).arrange(RIGHT, buff=0)
        border = cells[trace_k].copy().set_fill(opacity=0).set_stroke(SPI_COL[key], 3)
        lab = MathTex(SPI_TEX[key]).scale(0.7).next_to(cells, LEFT, buff=0.25).set_color(SPI_COL[key])
        return VGroup(cells, border, lab)

    # -- L4: feature matrix, exact cell ------------------------------------ #
    def build_l4(self, at, f_src=None):
        d = np.load(ROOT / "notebooks/presentation/_l4_cache.npz", allow_pickle=True)
        disp = np.nan_to_num(d["disp"])
        trow, tcols = int(d["trow"]), [int(x) for x in d["tcols"]]
        nrow, ncol = int(d["nrow"]), int(d["ncol"])
        vlo, vhi = np.percentile(disp, [2, 98])
        img = self._img(heat_rgb(disp, VIRIDIS, vlo, vhi), 9, 4.4).move_to(at)

        br_rows = Brace(img, LEFT); rows_lab = br_rows.get_tex(rf"{nrow}\ \text{{MTS}}").scale(0.7)
        br_cols = Brace(img, DOWN); cols_lab = br_cols.get_tex(rf"{ncol:,}\ \text{{features}}").scale(0.7)

        dn, dm = disp.shape[1], disp.shape[0]
        W, H = img.width, img.height
        cx = img.get_left()[0] + (tcols[0] + 0.5) / dn * W
        cy = img.get_top()[1] - (trow + 0.5) / dm * H
        cell = Square(0.16, stroke_color=SPI_COL["r"], stroke_width=3, fill_opacity=0).move_to([cx, cy, 0])
        cap = Tex(r"one $f_{ij}$: one of ${\sim}40{,}000$ SPI-pair features (297 SPIs)", font_size=26).next_to(img, UP, buff=0.35)

        self.play(FadeIn(img, scale=0.4), Write(cap))
        self.play(GrowFromCenter(br_rows), Write(rows_lab), GrowFromCenter(br_cols), Write(cols_lab))
        if f_src is not None:
            self.play(TransformFromCopy(f_src, cell), Flash(cell, color=SPI_COL["r"], flash_radius=0.35))
        else:
            self.play(Create(cell))
        self.wait(0.3)
        self.l4_group = Group(img, cap, br_rows, rows_lab, br_cols, cols_lab, cell)

    def _img(self, rgb, w, h):
        im = ImageMobject(rgb)
        im.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
        return im.stretch_to_fit_width(w).stretch_to_fit_height(h)


# --------------------------------------------------------------------------- #
class L0(Telescope):
    def construct(self):
        self.load(); self.build_l0(ORIGIN)


class L1(Telescope):
    def construct(self):
        self.load(); self.build_l1(ORIGIN, None)


class L2(Telescope):
    def construct(self):
        self.load(); self.build_l2(ORIGIN, None)


class L3(Telescope):
    def construct(self):
        self.load(); self.build_l3(ORIGIN, None)


class L4(Telescope):
    def construct(self):
        self.load(); self.build_l4(ORIGIN, None)
