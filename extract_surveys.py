"""Extract the eight survey point clouds of the study from the CloudCompare file
Chimney.bin and save them as survey_1.npz ... survey_8.npz (UTM zone 30N, metres).

    python extract_surveys.py Chimney.bin --out data
"""
import argparse
import pathlib
import re

import numpy as np

import ccbin

# Pix4D project name -> survey number in the paper (intact: no settlement, caseN: tilt level N)
SURVEYS = {"Chim5_intact": 1, "Chim5_intact_case1": 2, "Chim5_intact_case2": 3, "Chim5_intact_case3": 4,
           "Chim5_settlement": 5, "Chim5_settlement_case1": 6, "Chim5_settlement_case2": 7,
           "Chim5_settlement_case3": 8}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bin")
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    _, root = ccbin.read(a.bin)
    for g in root["children"]:
        c = g["children"][0]
        k = SURVEYS[re.match(r"(Chim5_[a-z_0-9]+?)_group1", c["name"]).group(1)]
        assert c["scale"] == 1.0
        xyz = c["xyz"].astype(np.float64) - c["shift"]      # CloudCompare: global = local - shift
        np.savez(out / f"survey_{k}.npz", xyz=xyz, rgb=c["rgba"][:, :3])
        print(f"survey {k}: {len(xyz):7d} points, E {xyz[:, 0].min():.2f}-{xyz[:, 0].max():.2f}, "
              f"N {xyz[:, 1].min():.2f}-{xyz[:, 1].max():.2f}, Z {xyz[:, 2].min():.2f}-{xyz[:, 2].max():.2f}")


if __name__ == "__main__":
    main()
