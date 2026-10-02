"""Marker-free displacement of a slender structure between GCP-free RTK-UAV point clouds.

The four steps follow Section 2.4 of the paper:

  (i)   detection of the structure: ground plane of the reference epoch by RANSAC,
        ground plane of each later epoch by least squares, DBSCAN on the points above
        the ground, cylinder fit, radial filter and top-centre point p_top;
  (ii)  co-registration of each later epoch to the reference on the stable area, that
        is, all points farther than r_min horizontally from the foot of the axis
        (6-DoF point-to-plane ICP, starting from the identity);
  (iii) registration of the structure by point-to-plane ICP, starting from the
        translation between the centroids;
  (iv)  displacement of the top-centre point, d = T(p_top) - p_top.

Without step (ii), the point clouds are compared in the frames given by their own RTK
georeferencing ("RTK georeferencing only" in the paper).

Coordinates are in metres, with X = easting, Y = northing and Z = up. The clouds are
shifted to a local origin before processing, which keeps the coordinates small.
"""
import pathlib

import numpy as np
import open3d as o3d

REG = o3d.pipelines.registration

PARAMS = dict(
    ransac_threshold=0.01,         # m, inlier threshold of the reference ground plane
    ransac_iterations=2000,
    seed=1,                        # Open3D random seed for RANSAC
    h_detect=0.30,                 # m, points above this height are clustered; reference structure cut
    h_target=0.15,                 # m, lower cut of the structure in a later epoch
    dbscan_eps=0.05,               # m
    dbscan_min_points=20,
    radial_tolerance=0.03,         # m, structure points within this distance of the cylinder
    top_percentile=99.5,           # axial percentile of the top-centre point
    ground_annulus=(0.8, 2.0),     # m, ground points used to refit the plane of a later epoch
    ground_band=0.03,              # m, ground points within this distance of the reference plane
    r_min=0.8,                     # m, inner radius of the stable area
    icp_distances=(0.05, 0.02, 0.01),  # m, maximum correspondence distance per stage
    icp_iterations=50,             # per stage
    knn_normals=20,
)


# ----------------------------------------------------------------------------- input

def read_cloud(path):
    """XYZ (N x 3, float64, real-world coordinates) from .npz (key 'xyz'), a text file
    (.txt, .csv, .xyz, .asc: X, Y, Z in the first three columns; lines starting with
    '//' or '#' are skipped) or any format Open3D reads (.ply, .pcd)."""
    path = pathlib.Path(path)
    suf = path.suffix.lower()
    if suf == ".npz":
        return np.load(path)["xyz"].astype(np.float64)
    if suf in (".txt", ".csv", ".xyz", ".asc"):
        with open(path, encoding="utf-8", errors="replace") as f:
            rows = [ln for ln in f if ln.strip() and not ln.lstrip().startswith(("//", "#"))]
        delim = "," if "," in rows[0] else None
        return np.loadtxt(rows, delimiter=delim, usecols=(0, 1, 2), dtype=np.float64)
    return np.asarray(o3d.io.read_point_cloud(str(path)).points, dtype=np.float64)


def pcd(x):
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.ascontiguousarray(x)))


def apply(T, x):
    return x @ T[:3, :3].T + T[:3, 3]


# ----------------------------------------------------------------- step (i): detection

def plane_from_nd(n, d):
    s = np.sign(n[2]) or 1.0
    n = np.asarray(n, float) * s
    d = float(d) * s
    L = np.linalg.norm(n)
    return n / L, d / L


def ransac_ground(x, p=PARAMS):
    o3d.utility.random.seed(p["seed"])
    (a, b, c, d), _ = pcd(x).segment_plane(p["ransac_threshold"], 3, p["ransac_iterations"])
    return plane_from_nd([a, b, c], d)


def height(x, plane):
    n, d = plane
    return x @ n + d


def detect_structure(x, plane, p=PARAMS):
    """Indices of the DBSCAN cluster with the largest vertical extent above h_detect."""
    h = height(x, plane)
    idx = np.where(h > p["h_detect"])[0]
    lab = np.asarray(pcd(x[idx]).cluster_dbscan(p["dbscan_eps"], p["dbscan_min_points"]))
    best, ext = None, -1.0
    for L in set(lab) - {-1}:
        m = lab == L
        e = np.ptp(h[idx[m]])
        if e > ext:
            best, ext = idx[m], e
    if best is None:
        raise RuntimeError("no structure found above the ground plane")
    return best


def circle_fit(u, v):
    """Algebraic circle fit refined by Gauss-Newton on the geometric distance."""
    A = np.c_[u, v, np.ones_like(u)]
    D, E, F = np.linalg.lstsq(A, -(u ** 2 + v ** 2), rcond=None)[0]
    uc, vc = -D / 2, -E / 2
    R = np.sqrt(uc ** 2 + vc ** 2 - F)
    for _ in range(10):
        du, dv = u - uc, v - vc
        rho = np.hypot(du, dv)
        J = np.c_[-du / rho, -dv / rho, -np.ones_like(u)]
        step = np.linalg.lstsq(J, -(rho - R), rcond=None)[0]
        uc, vc, R = uc + step[0], vc + step[1], R + step[2]
        if np.abs(step).max() < 1e-7:
            break
    return uc, vc, R


def _basis(a):
    e1 = np.cross(a, [0, 1, 0])
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(a, e1)


def fit_cylinder(pts, n_slices=10, n_iter=3):
    """Point c on the axis, upward unit direction a and radius R (median over slices).
    The axis passes through the centres of circles fitted to n_slices slices."""
    m = pts.mean(0)
    a = np.linalg.svd(pts - m, full_matrices=False)[2][0]
    a = a if a[2] > 0 else -a
    c = m
    for _ in range(n_iter):
        e1, e2 = _basis(a)
        s = (pts - c) @ a
        edges = np.linspace(np.percentile(s, 2), np.percentile(s, 98), n_slices + 1)
        cen, rad = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            q = pts[(s >= lo) & (s < hi)]
            if len(q) < 50:
                continue
            uc, vc, R = circle_fit((q - c) @ e1, (q - c) @ e2)
            cen.append(c + uc * e1 + vc * e2 + (lo + hi) / 2 * a)
            rad.append(R)
        cen = np.array(cen)
        c = cen.mean(0)
        a = np.linalg.svd(cen - c, full_matrices=False)[2][0]
        a = a if a[2] > 0 else -a
    return c, a, float(np.median(rad))


def structure_points(x, plane, hmin, cyl, p=PARAMS):
    """Indices of the points above hmin within radial_tolerance of the cylinder surface."""
    c, a, R = cyl
    cand = np.where(height(x, plane) >= hmin)[0]
    v = x[cand] - c
    rho = np.linalg.norm(v - np.outer(v @ a, a), axis=1)
    return cand[np.abs(rho - R) < p["radial_tolerance"]]


def top_centre(pts, cyl, p=PARAMS):
    c, a, _ = cyl
    return c + np.percentile((pts - c) @ a, p["top_percentile"]) * a


def axis_foot(cyl, plane):
    c, a, _ = cyl
    n, d = plane
    return c - (c @ n + d) / (a @ n) * a


def fit_ground(x, ref_plane, foot, p=PARAMS):
    """Least-squares plane z = p0 x + p1 y + p2 of the ground annulus around the foot,
    after removal of residuals larger than three standard deviations."""
    lo, hi = p["ground_annulus"]
    r = np.hypot(x[:, 0] - foot[0], x[:, 1] - foot[1])
    q = x[(r >= lo) & (r <= hi) & (np.abs(height(x, ref_plane)) < p["ground_band"])]
    keep = np.ones(len(q), bool)
    for _ in range(3):
        A = np.c_[q[keep, 0], q[keep, 1], np.ones(keep.sum())]
        coef = np.linalg.lstsq(A, q[keep, 2], rcond=None)[0]
        res = q[:, 2] - np.c_[q[:, 0], q[:, 1], np.ones(len(q))] @ coef
        keep = np.abs(res) < 3 * res[keep].std()
    return plane_from_nd(np.array([-coef[0], -coef[1], 1.0]), -coef[2])


# --------------------------------------------------------------------- registration

def icp(source, target_pcd, init, p=PARAMS):
    """Point-to-plane ICP with the staged correspondence distances (target needs normals)."""
    s = source if isinstance(source, o3d.geometry.PointCloud) else pcd(source)
    T = init.copy()
    crit = REG.ICPConvergenceCriteria(1e-10, 1e-10, p["icp_iterations"])
    for dmax in p["icp_distances"]:
        T = REG.registration_icp(s, target_pcd, dmax, T,
                                 REG.TransformationEstimationPointToPlane(), crit).transformation
    return T


def with_normals(x, p=PARAMS):
    t = pcd(x)
    t.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(p["knn_normals"]))
    return t


# --------------------------------------------------------------------------- epochs

class Reference:
    """Quantities of the reference epoch used by every later epoch."""

    def __init__(self, x, p=PARAMS):
        self.x, self.p = x, p
        self.plane = ransac_ground(x, p)
        self.cyl = fit_cylinder(x[detect_structure(x, self.plane, p)])
        self.src = x[structure_points(x, self.plane, p["h_detect"], self.cyl, p)]
        self.p_top = top_centre(self.src, self.cyl, p)
        self.foot = axis_foot(self.cyl, self.plane)
        self.r = np.hypot(x[:, 0] - self.foot[0], x[:, 1] - self.foot[1])
        self._stable = {}

    def stable_target(self, r_min):
        if r_min not in self._stable:
            self._stable[r_min] = with_normals(self.x[self.r >= r_min], self.p)
        return self._stable[r_min]


class Epoch:
    """Structure of a later epoch, detected on its own ground plane."""

    def __init__(self, ref, x):
        p = ref.p
        self.ref, self.x = ref, x
        self.plane = fit_ground(x, ref.plane, ref.foot, p)
        cyl = fit_cylinder(x[detect_structure(x, self.plane, p)])
        self.tgt_idx = structure_points(x, self.plane, p["h_target"], cyl, p)
        h = height(x[self.tgt_idx], self.plane)
        self.band_idx = self.tgt_idx[h >= p["h_detect"]]     # same height band as the reference
        self.r = np.hypot(x[:, 0] - ref.foot[0], x[:, 1] - ref.foot[1])

    def coregistration(self, r_min=None):
        """Step (ii): 4x4 transformation of this epoch onto the reference stable area."""
        r_min = self.ref.p["r_min"] if r_min is None else r_min
        return icp(self.x[self.r >= r_min], self.ref.stable_target(r_min), np.eye(4), self.ref.p)

    def ground_plane_correction(self):
        """Variant specified first in the paper: rigid correction that maps the ground
        plane of this epoch onto that of the reference (vertical shift at the axis foot
        and the two tilts); it leaves X, Y and the rotation about Z unchanged."""
        ref = self.ref
        n1, d1 = fit_ground(ref.x, ref.plane, ref.foot, ref.p)
        nk, dk = self.plane
        ez = np.array([0, 0, 1.0])
        f1 = ref.foot - (ref.foot @ n1 + d1) / n1[2] * ez
        fk = f1 - (f1 @ nk + dk) / nk[2] * ez
        v = np.cross(nk, n1)
        s, c = np.linalg.norm(v), float(nk @ n1)
        K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        Rg = np.eye(3) + K + K @ K * ((1 - c) / s ** 2) if s > 1e-15 else np.eye(3)
        G = np.eye(4)
        G[:3, :3] = Rg
        G[:3, 3] = f1 - Rg @ fk
        return G

    def displacement(self, G=None):
        """Steps (iii) and (iv) after the correction G (identity: RTK georeferencing only).
        Returns the displacement of p_top in cm and the transformation T."""
        ref = self.ref
        G = np.eye(4) if G is None else G
        tgt = with_normals(apply(G, self.x[self.tgt_idx]), ref.p)
        init = np.eye(4)
        init[:3, 3] = apply(G, self.x[self.band_idx]).mean(0) - ref.src.mean(0)
        T = icp(ref.src, tgt, init, ref.p)
        return (apply(T, ref.p_top[None])[0] - ref.p_top) * 100, T


def survey_offset_cm(G, point):
    """Offset of a later epoch relative to the reference at a point: minus the
    correction that the co-registration G applies there (cm)."""
    return -(apply(G, point[None])[0] - point) * 100
