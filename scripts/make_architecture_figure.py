"""Fig. 1 of the paper (UCR-MFG architecture): paper/figures/architecture.png.

  python scripts/make_architecture_figure.py      # -> paper/figures/architecture.png
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans", "font.size": 8})
OUT = "paper/figures/architecture"
COL = {"data": "#ececec", "base": "#e3e9fb", "unc": "#fde7cf", "graph": "#e0f4e0", "out": "#fbe3e3"}
W, H = 2.7, 1.5  # box size in TikZ units (slightly taller for 3-line labels)


def box(ax, x, y, text, kind, w=W, h=H, **kw):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.08",
                                fc=COL[kind], ec="black", lw=0.8, **kw))
    ax.text(x, y, text, ha="center", va="center", fontsize=6.0, linespacing=1.3)
    return dict(x=x, y=y, w=w, h=h, l=x - w / 2, r=x + w / 2, t=y + h / 2, b=y - h / 2)


def arrow(ax, pts, color="black", ls="-"):
    for (x0, y0), (x1, y1) in zip(pts[:-2], pts[1:-1]):
        ax.plot([x0, x1], [y0, y1], color=color, lw=0.9, ls=ls, solid_capstyle="butt")
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=9, color=color, lw=0.9,
                                 linestyle=ls, shrinkA=0, shrinkB=0))


def label(ax, x, y, text, ha="center", va="center", size=5.8):
    ax.text(x, y, text, ha=ha, va=va, fontsize=size, bbox=dict(fc="white", ec="none", pad=0.6))


def tag(ax, x, y, text, ha="center", va="bottom"):
    ax.text(x, y, text, ha=ha, va=va, fontsize=5.2, color="#595959")


def main():
    fig, ax = plt.subplots(figsize=(10.5, 4.1))
    ax.set_xlim(-1.5, 22.1); ax.set_ylim(-5.5, 4.1); ax.set_aspect("equal"); ax.axis("off")
    ORANGE, RED = "#cc6a00", "#b03030"

    img = box(ax, 0, 0, "Input image\nresize, pad,\nflip, jitter", "data")
    bb = box(ax, 3.9, 0, "ResNet-50\n+ FPN", "base")
    pyr = box(ax, 7.8, 0, "Pyramid $F_l$\n$l=3,4,5$\nstrides 8, 16, 32", "base")
    gb = box(ax, 11.7, 0, "Multi-scale\nfuzzy graph\nlocal, kNN,\ncross-scale", "graph")
    rt = box(ax, 15.6, 0, "C2: UC router\ntyped top-$K$, budget\ngrows with $U$", "graph")
    mp = box(ax, 19.5, 0, "C3: benefit-aware\nMPNN\nattention, gate, GRU", "graph")
    uh = box(ax, 11.7, 2.9, "C1: pre-graph uncertainty head\nMC dropout, $T=10$, $p=0.2$", "unc", w=4.2)
    fu = box(ax, 19.5, -3.0, "C4: fusion gate\n$Z=F+(1-\\lambda)\\odot H$", "graph")
    hd = box(ax, 15.6, -3.0, "FCOS head\ncls, ltrb,\ncenterness,\nlog-variance", "base")
    out = box(ax, 11.7, -3.0, "Detections\n+ box std $\\sigma$\n+ $U_a$, $U_e$", "out")
    ax.add_patch(FancyBboxPatch((-1.2, -3.0 - 0.85), 7.3, 1.7, boxstyle="round,pad=0,rounding_size=0.08",
                                fc="#fffbe8", ec="black", lw=0.8, ls="--"))
    ax.text(-0.95, -3.0, "Training objective (stage-aware)\n"
            "$\\mathcal{L}_{det}+\\lambda_v\\mathcal{L}_{var}+\\lambda_u\\mathcal{L}_u+\\lambda_g\\mathcal{L}_{graph}$\n"
            "$+\\lambda_i\\mathcal{L}_{inst}+\\lambda_b\\mathcal{L}_{ben}+\\lambda_c\\mathcal{L}_{cons}$",
            ha="left", va="center", fontsize=6.0, linespacing=1.5)

    tag(ax, bb["x"], bb["t"] + 0.05, "stage 1+")
    tag(ax, gb["x"], gb["t"] + 0.05, "stage 2+")
    tag(ax, rt["l"], rt["t"] + 0.05, "stage 3+", ha="left")
    tag(ax, mp["l"], mp["t"] + 0.05, "stage 3+", ha="left")
    tag(ax, fu["r"], fu["b"] - 0.05, "stage 4", ha="right", va="top")
    tag(ax, uh["x"], uh["t"] + 0.05, "stage 1+ (MC passes from stage 2)")

    arrow(ax, [(img["r"], 0), (bb["l"], 0)])
    arrow(ax, [(bb["r"], 0), (pyr["l"], 0)])
    arrow(ax, [(pyr["r"], 0), (gb["l"], 0)]); label(ax, (pyr["r"] + gb["l"]) / 2, 0.28, "$h_i$")
    arrow(ax, [(gb["r"], 0), (rt["l"], 0)]); label(ax, (gb["r"] + rt["l"]) / 2, 0.28, "$\\mathcal{E}, k_{ij}$")
    arrow(ax, [(rt["r"], 0), (mp["l"], 0)]); label(ax, (rt["r"] + mp["l"]) / 2, 0.28, "$m_{ij}, s_{ij}$")
    arrow(ax, [(mp["x"], mp["b"]), (fu["x"], fu["t"])]); label(ax, mp["x"] + 0.2, -1.5, "$H_l$", ha="left")
    arrow(ax, [(fu["l"], -3.0), (hd["r"], -3.0)]); label(ax, (fu["l"] + hd["r"]) / 2, -2.72, "$Z_l$")
    arrow(ax, [(hd["l"], -3.0), (out["r"], -3.0)])
    arrow(ax, [(pyr["x"], pyr["t"]), (pyr["x"], uh["y"]), (uh["l"], uh["y"])])
    label(ax, pyr["x"] - 0.15, 1.7, "sg$(F_l)$", ha="right")
    arrow(ax, [(pyr["x"], pyr["b"]), (pyr["x"], -4.85), (fu["x"] - 0.6, -4.85), (fu["x"] - 0.6, fu["b"])])
    label(ax, 13.6, -5.0, "$F_l$ (skip connection)", va="top")
    arrow(ax, [(uh["r"], uh["y"] - 0.2), (rt["x"] + 0.5, uh["y"] - 0.2), (rt["x"] + 0.5, rt["t"])], ORANGE, "--")
    label(ax, rt["x"] + 0.62, 1.7, "$U_a, U_e$", ha="left")
    arrow(ax, [(uh["r"], uh["y"]), (mp["x"] + 0.5, uh["y"]), (mp["x"] + 0.5, mp["t"])], ORANGE, "--")
    arrow(ax, [(uh["r"], uh["y"] + 0.2), (21.7, uh["y"] + 0.2), (21.7, -3.0), (fu["r"], -3.0)], ORANGE, "--")
    arrow(ax, [(hd["x"], hd["t"]), (rt["x"], rt["b"])], RED, "-.")
    label(ax, hd["x"] + 0.15, -1.5, "benefit\n$b_{ij}=-\\partial\\mathcal{L}_{det}/\\partial m_{ij}$", ha="left")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT + ".png", bbox_inches="tight", pad_inches=0.02, dpi=400)
    print(f"saved {OUT}.png")


if __name__ == "__main__":
    main()
