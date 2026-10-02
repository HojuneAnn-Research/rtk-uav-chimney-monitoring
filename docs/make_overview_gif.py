"""Animated overview of the pipeline (docs/overview.gif) from the real point clouds of
Surveys 1 and 4 of the study, seen from the side (X: easting, Z: height).

    python docs/make_overview_gif.py --data data
"""
import argparse
import io
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pipeline as P                                     # noqa: E402

ORIGIN = np.array([461521.484497, 5619304.977997, 65.2285])
LATER, EXAG = 4, 20                                       # survey shown; exaggeration of the offset
BLUE, ORANGE, GREEN, GREY = "#0072B2", "#E69F00", "#009E73", "#B0B0B0"


def rot_interp(R, t):
    """Rotation by the fraction t of the angle of R (axis-angle)."""
    ang = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if ang < 1e-12:
        return np.eye(3)
    k = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(ang))
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    a = t * ang
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default=str(pathlib.Path(__file__).with_name("overview.gif")))
    a = ap.parse_args()
    data = pathlib.Path(a.data)
    x1 = P.read_cloud(data / "survey_1.npz") - ORIGIN
    xk = P.read_cloud(data / f"survey_{LATER}.npz") - ORIGIN
    ref = P.Reference(x1)
    ep = P.Epoch(ref, xk)
    G = ep.coregistration()
    d, T = ep.displacement(G)
    off = P.survey_offset_cm(G, ref.foot)

    rng = np.random.default_rng(0)
    pick = lambda x, n: x[rng.choice(len(x), min(n, len(x)), replace=False)]
    f = ref.foot
    side = lambda x: (x[:, 0] - f[0], x[:, 2] - f[2])     # side view relative to the foot
    r1 = pick(x1, 30000)
    rk = pick(xk, 30000)
    stable = np.hypot(rk[:, 0] - f[0], rk[:, 1] - f[1]) >= ref.p["r_min"]
    rk_al = P.apply(G, rk)                                # later cloud after co-registration
    src = pick(ref.src, 9000)
    top0, top1 = ref.p_top, P.apply(T, ref.p_top[None])[0]

    frames = []
    ZOOM = (-0.30, 0.22, 1.36, 1.68)                       # x0, x1, z0, z1 of the inset around the top

    def draw(ax, layers, arrow, ms):
        for x, c, s, al in layers:
            u, v = side(x)
            ax.scatter(u, v, s=s, c=c, alpha=al, linewidths=0, rasterized=True)
        if arrow is not None:
            (u0, v0), (u1, v1) = [(p[0] - f[0], p[2] - f[2]) for p in arrow]
            ax.annotate("", (u1, v1), (u0, v0), arrowprops=dict(arrowstyle="-|>", color="black", lw=1.8))
            ax.plot([u0], [v0], "o", ms=ms, mfc="white", mec="black", mew=1.2)
            ax.plot([u1], [v1], "o", ms=ms, color="black")

    def frame(caption, layers, arrow=None, note=None, zoom=False):
        fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=100)
        fig.subplots_adjust(left=0.09, right=0.98, bottom=0.11, top=0.84)
        draw(ax, layers, arrow, 5)
        ax.set_xlim(-1.6, 1.6)
        ax.set_ylim(-0.15, 1.85)
        ax.set_aspect("equal")
        ax.set_xlabel("Horizontal position, X (m)")
        ax.set_ylabel("Height, Z (m)")
        ax.grid(ls=":", lw=0.5, c="0.7")
        if zoom:
            x0, x1, z0, z1 = ZOOM
            top = [(x[(x[:, 2] - f[2]) > z0 - 0.05], c, s * 2.5, al) for x, c, s, al in layers]
            axi = ax.inset_axes([0.62, 0.28, 0.37, 0.55])
            draw(axi, top, arrow, 7)
            axi.set_xlim(x0, x1)
            axi.set_ylim(z0, z1)
            axi.set_aspect("equal")
            axi.set_xticks([])
            axi.set_yticks([])
            axi.set_title("top, enlarged", fontsize=9, pad=3)
            ax.indicate_inset_zoom(axi, edgecolor="0.25", lw=0.8)
        fig.text(0.02, 0.955, caption[0], fontsize=12.5, weight="bold", va="top")
        fig.text(0.02, 0.895, caption[1], fontsize=10.5, va="top")
        if note:
            ax.text(0.01, 0.97, note, transform=ax.transAxes, ha="left", va="top", fontsize=9.5,
                    bbox=dict(fc="white", ec="0.4", lw=0.6))
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    def hold(img, n):
        frames.extend([img] * n)

    # 1. reference
    hold(frame(("1  Survey 1, the reference",
                "Point cloud from an RTK drone survey, without ground control points"),
               [(r1, BLUE, 1.6, 0.55)], note="blue: Survey 1"), 18)
    # 2. later survey
    hold(frame(("2  Survey 4, after the model was tilted",
                "Both clouds are in real-world coordinates from the RTK positions of the camera"),
               [(r1, BLUE, 1.6, 0.35), (rk, ORANGE, 1.6, 0.55)],
               note="blue: Survey 1 (reference)\norange: Survey 4 (model tilted)", zoom=True), 24)
    # 3. offset of the survey, exaggerated, then removed on the stable area
    cap3 = ("3  Remove the small offset between the surveys",
            f"Survey 4 is offset by {np.linalg.norm(off):.1f} cm as a whole. Matching the surroundings "
            f"(green) removes it.")
    for s in list(np.linspace(1, EXAG, 6)) + [EXAG] * 8 + list(np.linspace(EXAG, 0, 10)) + [0] * 12:
        pos = rk_al + s * (rk - rk_al)
        frames.append(frame(cap3, [(r1, BLUE, 1.6, 0.25), (pos[~stable], GREY, 1.6, 0.5),
                                   (pos[stable], GREEN, 1.6, 0.6)],
                            note=(f"offset shown {EXAG} times larger" if s > 1.01 else
                                  "green: stable surroundings, more than\n0.8 m from the chimney")))
    # 4. rigid ICP of the chimney
    cap4 = ("4  Fit the reference chimney onto the new one",
            "Rigid registration (ICP): the shape is kept, only position and tilt change")
    R = T[:3, :3]
    c0 = src.mean(0)                                      # rotate about the chimney centroid
    c1 = P.apply(T, c0[None])[0]
    for s in [0] * 6 + list(np.linspace(0, 1, 14)) + [1] * 8:
        moved = (src - c0) @ rot_interp(R, s).T + c0 + s * (c1 - c0)
        frames.append(frame(cap4, [(rk_al, ORANGE, 1.6, 0.4), (moved, BLUE, 1.6, 0.75)],
                            note="blue: reference chimney\norange: Survey 4", zoom=True))
    # 5. displacement of the top
    hold(frame(("5  Read the movement of the top",
                f"Top moved {d[0]:+.2f} cm horizontally and {d[2]:+.2f} cm vertically "
                f"(ruler: -10.5 and +2.1 cm)"),
               [(rk_al, ORANGE, 1.6, 0.35), (P.apply(T, src), BLUE, 1.6, 0.6)], arrow=(top0, top1),
               note="white dot: top before\nblack dot: top after", zoom=True), 36)

    q, cache = [], {}
    for im in frames:                                     # one palette per distinct frame
        if id(im) not in cache:
            cache[id(im)] = im.quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
        q.append(cache[id(im)])
    q[0].save(a.out, save_all=True, append_images=q[1:], duration=110, loop=0, optimize=True)
    print(f"{a.out}: {len(q)} frames, offset {off.round(2)} cm, displacement {d.round(2)} cm")


if __name__ == "__main__":
    main()
