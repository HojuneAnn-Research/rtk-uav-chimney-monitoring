"""Reproduce the displacement results of the paper from the eight survey point clouds
and check them against the values printed in the paper.

    python reproduce_paper.py --data data [--out results]

Input: data/survey_1.npz ... survey_8.npz (from extract_surveys.py). Survey 1 is the
reference of every survey. The script prints Table 5 of the paper (marker-free rows
computed here, marker-based rows from the manual picks), the georeferencing offsets,
the ground-plane variant and the sensitivity to the stable area, then a list of checks.
It exits with status 1 if any computed value differs from the paper.
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np

import pipeline as P

SURVEYS = range(2, 9)
ORIGIN = np.array([461521.484497, 5619304.977997, 65.2285])   # local origin of the study (UTM 30N, m)

# Ground truth (Table 2): ruler readings at the top of the model, cm
ACTUAL = {2: (-3.5, 1.3), 3: (-7.0, 1.5), 4: (-10.5, 2.1), 5: (0.0, -2.5),
          6: (-3.5, -1.2), 7: (-7.0, -1.0), 8: (-10.5, -0.5)}
ALPHA_DEG = {2: 1.250, 3: 2.495, 4: 3.734, 5: 0.0, 6: 1.250, 7: 2.495, 8: 3.734}   # Table 3
D_HINGE, H_MODEL, MARKER_DEPTH = 0.375, 1.6, 0.078      # m (Eq. (1); markers below the top-centre point)

# Marker-based estimates (Table 5): mean of P1-P4, each picked ten times in CloudCompare
MARKER_BASED = {2: (-3.85, 0.82), 3: (-7.32, 0.75), 4: (-10.87, 1.87), 5: (-0.60, -2.10),
                6: (-4.05, -0.75), 7: (-6.72, -1.63), 8: (-10.77, -1.55)}
MARKER_CENTROID = ORIGIN + np.array([0.004, -0.017, 0.715])   # centroid of P1-P4 in Survey 1 (UTM, m)

# Values printed in the paper (cm)
PAPER_ROWS = {
    "rtk_only": {2: (-3.92, 1.12), 3: (-7.81, 0.63), 4: (-11.45, 1.91), 5: (-0.75, -1.70),
                 6: (-4.32, -0.75), 7: (-7.23, -1.75), 8: (-11.46, -1.26)},
    "coregistered": {2: (-3.32, 0.75), 3: (-7.02, 1.21), 4: (-10.53, 1.77), 5: (-0.42, -2.51),
                     6: (-3.79, -1.78), 7: (-7.15, -1.33), 8: (-10.56, -0.62)},
}


def eq1(alpha_deg, h):
    """Eq. (1): |dX| and dZ (cm) of a point at height h above the hinged base."""
    a = np.radians(alpha_deg)
    return (h * np.sin(a) + D_HINGE * (1 - np.cos(a))) * 100, (D_HINGE * np.sin(a) - h * (1 - np.cos(a))) * 100


def errors(est, act):
    e = np.array([[abs(est[k][0] - act[k][0]), abs(est[k][1] - act[k][1])] for k in SURVEYS])
    return dict(mean_x=e[:, 0].mean(), max_x=e[:, 0].max(), mean_z=e[:, 1].mean(), max_z=e[:, 1].max()), e


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--origin", default=",".join(f"{v}" for v in ORIGIN),
                    help="local origin 'x,y,z' (default: the origin of the study)")
    a = ap.parse_args()
    origin = np.array([float(v) for v in a.origin.split(",")])
    data, out = pathlib.Path(a.data), pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    load = lambda k: P.read_cloud(data / f"survey_{k}.npz") - origin

    # --- main run: RTK georeferencing only and co-registered (r_min = 0.8 m)
    t0 = time.time()
    ref = P.Reference(load(1))
    ep, rows, G, T = {}, {"rtk_only": {}, "coregistered": {}}, {}, {}
    for k in SURVEYS:
        ep[k] = P.Epoch(ref, load(k))
        rows["rtk_only"][k], _ = ep[k].displacement()
        G[k] = ep[k].coregistration()
        rows["coregistered"][k], T[k] = ep[k].displacement(G[k])
    t_main = time.time() - t0

    # --- marker-based method at the marker centroid, with and without the same co-registration
    act_marker = {k: (-round(eq1(ALPHA_DEG[k], H_MODEL - MARKER_DEPTH)[0], 2) + 0.0, ACTUAL[k][1]) for k in SURVEYS}
    pm = MARKER_CENTROID - origin
    shift = {k: (P.apply(G[k], pm[None])[0] - pm) * 100 for k in SURVEYS}     # correction at the markers
    mb_co = {k: (MARKER_BASED[k][0] + shift[k][0], MARKER_BASED[k][1] + shift[k][2]) for k in SURVEYS}

    # --- offsets of the surveys at the foot of the axis (minus the correction)
    off = np.array([P.survey_offset_cm(G[k], ref.foot) for k in SURVEYS])

    # --- ground-plane variant specified first
    ground = {k: ep[k].displacement(ep[k].ground_plane_correction())[0] for k in SURVEYS}

    # --- sensitivity to the inner radius of the stable area
    sens, dz_survey = {}, 0.0
    for r_min in (1.0, 1.3, 1.6, 2.0):
        est = {k: ep[k].displacement(ep[k].coregistration(r_min))[0] for k in SURVEYS}
        sens[r_min] = errors({k: (v[0], v[2]) for k, v in est.items()}, ACTUAL)
        dz_survey = max(dz_survey, max(abs(est[k][2] - rows["coregistered"][k][2]) for k in SURVEYS))

    # --- Table 5
    xz = lambda d: {k: (v[0], v[2]) for k, v in d.items()}
    blocks = [("Marker-based method (at the marker centroid)", MARKER_BASED, act_marker),
              ("Marker-free method, RTK georeferencing only", xz(rows["rtk_only"]), ACTUAL),
              ("Marker-free method, co-registered on the stable area", xz(rows["coregistered"]), ACTUAL)]
    stats = {}
    print("Table 5 (cm): estimated, actual and absolute error of dX and dZ")
    for name, est, act in blocks:
        print(f"  {name}")
        for k in SURVEYS:
            x, z = round(est[k][0], 2) + 0.0, round(est[k][1], 2) + 0.0
            print(f"    Survey {k}: {x:7.2f} {act[k][0]:7.2f} {abs(x - act[k][0]):5.2f} | "
                  f"{z:6.2f} {act[k][1]:5.1f} {abs(z - act[k][1]):5.2f}")
        stats[name], _ = errors(est, act)
        s = stats[name]
        print(f"    mean {s['mean_x']:.2f} max {s['max_x']:.2f} | mean {s['mean_z']:.2f} max {s['max_z']:.2f}")
    s_mbco, _ = errors(mb_co, act_marker)
    s_ground, _ = errors(xz(ground), ACTUAL)
    print(f"Marker-based + co-registration: X mean {s_mbco['mean_x']:.2f} max {s_mbco['max_x']:.2f} | "
          f"Z mean {s_mbco['mean_z']:.2f} max {s_mbco['max_z']:.2f}")
    print(f"Ground-plane correction:        X mean {s_ground['mean_x']:.2f} max {s_ground['max_x']:.2f} | "
          f"Z mean {s_ground['mean_z']:.2f} max {s_ground['max_z']:.2f}")
    hor = np.hypot(off[:, 0], off[:, 1])
    print(f"Survey offsets at the foot: horizontal {hor.min():.2f} to {hor.max():.2f} cm, vertical "
          f"{off[:, 2].min():+.2f} to {off[:, 2].max():+.2f} cm, 3D up to {np.linalg.norm(off, axis=1).max():.2f} cm")
    dy = np.array([rows["coregistered"][k][1] for k in SURVEYS])
    print(f"dY after co-registration: {dy.min():+.2f} to {dy.max():+.2f} cm")
    for r_min, (s, _) in sens.items():
        print(f"Stable area beyond {r_min} m: X mean {s['mean_x']:.2f} max {s['max_x']:.2f} | "
              f"Z mean {s['mean_z']:.2f} max {s['max_z']:.2f}")
    print(f"Largest change of a vertical estimate over these stable areas: {dz_survey:.3f} cm")
    print(f"Main run (8 surveys, both variants): {t_main:.0f} s")

    # --- checks against the paper
    c = stats["Marker-free method, co-registered on the stable area"]
    r = stats["Marker-free method, RTK georeferencing only"]
    m = stats["Marker-based method (at the marker centroid)"]
    sx = [s["mean_x"] for s, _ in sens.values()]
    mx = [s["max_x"] for s, _ in sens.values()]
    dz = max(abs(s[q] - c[q]) for s, _ in sens.values() for q in ("mean_z", "max_z"))
    checks = [(f"{v} S{k} {comp}", PAPER_ROWS[v][k][i], round(rows[v][k][j], 2) + 0.0)
              for v in PAPER_ROWS for k in SURVEYS for i, (j, comp) in enumerate(((0, "dX"), (2, "dZ")))]
    checks += [("co-registered X mean/max", (0.16, 0.42), (round(c["mean_x"], 2), round(c["max_x"], 2))),
               ("co-registered Z mean/max", (0.32, 0.58), (round(c["mean_z"], 2), round(c["max_z"], 2))),
               ("RTK only X mean/max", (0.71, 0.96), (round(r["mean_x"], 2), round(r["max_x"], 2))),
               ("RTK only Z mean/max", (0.57, 0.87), (round(r["mean_z"], 2), round(r["max_z"], 2))),
               ("marker-based X mean/max", (0.60, 0.88), (round(m["mean_x"], 2), round(m["max_x"], 2))),
               ("marker-based Z mean/max", (0.57, 1.05), (round(m["mean_z"], 2), round(m["max_z"], 2))),
               ("marker-based + co-registration X mean/max", (0.12, 0.27),
                (round(s_mbco["mean_x"], 2), round(s_mbco["max_x"], 2))),
               ("marker-based + co-registration Z max", 0.84, round(s_mbco["max_z"], 2)),
               ("ground-plane X mean", 0.73, round(s_ground["mean_x"], 2)),
               ("ground-plane Z mean/max", (0.30, 0.58), (round(s_ground["mean_z"], 2), round(s_ground["max_z"], 2))),
               ("offsets horizontal min/max", (0.08, 1.07), (round(hor.min(), 2), round(hor.max(), 2))),
               ("offsets vertical min/max", (-0.64, 1.03), (round(off[:, 2].min(), 2), round(off[:, 2].max(), 2))),
               ("offsets 3D max", 1.25, round(np.linalg.norm(off, axis=1).max(), 2)),
               ("dY range", (-1.01, 0.48), (round(dy.min(), 2), round(dy.max(), 2))),
               ("sensitivity X mean range", (0.14, 0.22), (round(min(sx), 2), round(max(sx), 2))),
               ("sensitivity X max range", (0.43, 0.53), (round(min(mx), 2), round(max(mx), 2))),
               ("sensitivity change of Z mean/max <= 0.01", True, bool(round(dz, 2) <= 0.01))]
    fails = [(n, p, v) for n, p, v in checks if p != v]
    print(f"\nChecks against the paper: {len(checks) - len(fails)} of {len(checks)} agree")
    for n, p, v in fails:
        print(f"  DIFFERS: {n}: paper {p}, computed {v}")

    json.dump(dict(origin=origin.tolist(), p_top=(ref.p_top + origin).tolist(),
                   foot=(ref.foot + origin).tolist(), radius_cm=ref.cyl[2] * 100,
                   rtk_only_cm={k: v.tolist() for k, v in rows["rtk_only"].items()},
                   coregistered_cm={k: v.tolist() for k, v in rows["coregistered"].items()},
                   ground_plane_cm={k: v.tolist() for k, v in ground.items()},
                   offset_at_foot_cm={k: o.tolist() for k, o in zip(SURVEYS, off)},
                   marker_based_coregistered_cm=mb_co,
                   statistics={**stats, "Marker-based + co-registration": s_mbco, "Ground-plane correction": s_ground,
                               **{f"Stable area beyond {r} m": s for r, (s, _) in sens.items()}},
                   G_local={k: g.tolist() for k, g in G.items()}, T_local={k: t.tolist() for k, t in T.items()},
                   checks_failed=[n for n, _, _ in fails]),
              open(out / "paper_results.json", "w"), indent=1, default=float)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
