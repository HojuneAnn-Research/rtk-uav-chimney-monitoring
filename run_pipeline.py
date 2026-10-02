"""Displacement of the top of a slender structure between a reference point cloud and
one or more later point clouds, with and without co-registration on the stable area.

    python run_pipeline.py REFERENCE LATER [LATER ...] [--r-min 0.8] [--out results.json]

All clouds have to be georeferenced in the same real-world frame (for example by RTK
direct georeferencing, without ground control points), in metres with Z up, and
cropped to the same region of interest around the structure.
"""
import argparse
import json
import time

import numpy as np

import pipeline as P


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reference")
    ap.add_argument("later", nargs="+")
    ap.add_argument("--r-min", type=float, default=P.PARAMS["r_min"],
                    help="inner radius of the stable area in m (default %(default)s)")
    ap.add_argument("--origin", default="auto",
                    help="local origin 'x,y,z' subtracted before processing; 'auto' uses the "
                         "rounded centroid of the reference cloud")
    ap.add_argument("--out", default="results.json")
    a = ap.parse_args()

    t0 = time.time()
    x_ref = P.read_cloud(a.reference)
    origin = (np.round(x_ref.mean(0), 0) if a.origin == "auto"
              else np.array([float(v) for v in a.origin.split(",")]))
    ref = P.Reference(x_ref - origin)
    print(f"reference: {a.reference} ({len(x_ref)} points), structure radius "
          f"{ref.cyl[2] * 100:.1f} cm, top-centre {P.height(ref.p_top[None], ref.plane)[0]:.3f} m "
          f"above the ground")

    out = dict(reference=a.reference, origin=origin.tolist(), r_min=a.r_min,
               p_top=(ref.p_top + origin).tolist(), epochs=[])
    print(f"{'epoch':28s} {'RTK only: dX dY dZ (cm)':>26s}   {'co-registered: dX dY dZ (cm)':>30s}   "
          f"{'offset at foot (cm)':>22s}")
    for path in a.later:
        ep = P.Epoch(ref, P.read_cloud(path) - origin)
        d_rtk, _ = ep.displacement()
        G = ep.coregistration(a.r_min)
        d_co, T = ep.displacement(G)
        off = P.survey_offset_cm(G, ref.foot)
        print(f"{path[-28:]:28s} {d_rtk[0]:8.2f} {d_rtk[1]:7.2f} {d_rtk[2]:7.2f}      "
              f"{d_co[0]:8.2f} {d_co[1]:7.2f} {d_co[2]:7.2f}      {off[0]:6.2f} {off[1]:6.2f} {off[2]:6.2f}")
        out["epochs"].append(dict(file=path, rtk_only_cm=d_rtk.tolist(), coregistered_cm=d_co.tolist(),
                                  offset_at_foot_cm=off.tolist(), G=G.tolist(), T=T.tolist()))
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"{len(a.later)} epochs in {time.time() - t0:.0f} s; results in {a.out}")


if __name__ == "__main__":
    main()
