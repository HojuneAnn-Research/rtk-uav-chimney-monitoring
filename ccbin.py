"""Minimal reader for CloudCompare BIN v2 files (data version 52) holding groups,
point clouds with RGBA colours and 2D labels. Written for the files of this study;
it stops with an error on any structure it does not understand."""
import struct, sys
import numpy as np

class R:
    def __init__(self, b, trace=False):
        self.b, self.p, self.trace = b, 0, trace
    def take(self, n):
        s = self.b[self.p:self.p + n]; self.p += n; return s
    def u8(self):  return self.take(1)[0]
    def b1(self):
        v = self.u8(); assert v in (0, 1), f"bool expected at {self.p-1}, got {v}"; return bool(v)
    def u32(self): return struct.unpack("<I", self.take(4))[0]
    def i32(self): return struct.unpack("<i", self.take(4))[0]
    def u64(self): return struct.unpack("<Q", self.take(8))[0]
    def f64(self): return struct.unpack("<d", self.take(8))[0]
    def qstr(self):
        n = struct.unpack(">I", self.take(4))[0]
        return None if n == 0xFFFFFFFF else self.take(n).decode("utf-16-be")
    def qvariant(self):
        t = struct.unpack(">I", self.take(4))[0]; isnull = self.u8()
        if t == 1:  return bool(self.u8())
        if t == 2:  return struct.unpack(">i", self.take(4))[0]
        if t == 3:  return struct.unpack(">I", self.take(4))[0]
        if t == 4:  return struct.unpack(">q", self.take(8))[0]
        if t == 5:  return struct.unpack(">Q", self.take(8))[0]
        if t == 6:  return struct.unpack(">d", self.take(8))[0]
        if t == 10: return self.qstr()
        raise ValueError(f"QVariant type {t} at {self.p}")

def gl_matrix(r):
    return np.frombuffer(r.take(64), dtype="<f4").reshape(4, 4).T.copy()   # column-major -> row-major

def hobject_me(r, o):
    o["visible"] = r.b1(); o["lockedVis"] = r.b1(); o["colorsShown"] = r.b1()
    o["normalsShown"] = r.b1(); o["sfShown"] = r.b1()
    if r.b1(): o["tempColor"] = r.take(3)
    if r.b1(): o["glTrans"] = gl_matrix(r)
    o["showName"] = r.b1()

def generic_array(r, dtype):
    comp = r.u8(); n = r.u32()
    a = np.frombuffer(r.take(n * comp * np.dtype(dtype).itemsize), dtype=dtype).reshape(n, comp)
    return a

def parse_object(r, depth=0):
    start = r.p
    o = {"class": r.u64(), "uid": r.u32(), "name": r.qstr(), "flags": r.u32()}
    nmeta = r.u32(); o["meta"] = {r.qstr(): r.qvariant() for _ in range(nmeta)}
    c = o["class"]
    if r.trace:
        print("  " * depth + f"@{start} class=0x{c:x} uid={o['uid']} name={o['name']!r} meta={len(o['meta'])}")
    if c == 0x1:                       # group (HIERARCHY_OBJECT)
        hobject_me(r, o)
    elif c == 0x101:                   # point cloud
        hobject_me(r, o)
        o["shift"] = np.array([r.f64(), r.f64(), r.f64()]); o["scale"] = r.f64()
        if r.b1(): o["visibility"] = generic_array(r, np.uint8)
        o["pointSize"] = r.u8()
        o["xyz"] = generic_array(r, "<f4")
        if r.b1(): o["rgba"] = parse_object(r, depth + 1)["data"]
        if r.b1(): o["normals"] = parse_object(r, depth + 1)["data"]
        nsf = r.u32(); assert nsf == 0, f"{nsf} scalar fields at {r.p}: not supported"
        o["sfScaleShown"] = r.b1()
        o["sfIndex"] = r.i32()
        ngrid = r.u32(); assert ngrid == 0, f"{ngrid} grids at {r.p}: not supported"
        assert not r.b1(), "waveforms not supported"
    elif c & 0x10 and "colors" in (o["name"] or ""):   # RGBA colour array (ARRAY bit 0x10)
        o["data"] = generic_array(r, np.uint8)
    elif c & 0x20:                     # 2D label (LABEL bit 0x20): handled by parse_label
        parse_label(r, o)
    else:
        raise ValueError(f"unknown class 0x{c:x} ({o['name']!r}) at {start}")
    nchild = r.u32()
    o["children"] = [parse_object(r, depth + 1) for _ in range(nchild)]
    o["selection"] = r.u32()
    o["transHistory"] = gl_matrix(r)
    return o

def parse_label(r, o):
    hobject_me(r, o)
    n = r.u32(); o["picks"] = []
    for _ in range(n):
        idx, cloud, mesh = r.u32(), r.u32(), r.u32()
        uv = (r.f64(), r.f64()); centre = r.b1()
        o["picks"].append({"index": idx, "cloud": cloud, "mesh": mesh, "uv": uv, "centre": centre})
    o["screenPos"] = struct.unpack("<2f", r.take(8))
    o["fullBody"] = r.b1(); o["in2D"] = r.b1(); o["legend"] = r.b1()

def read(path, trace=False):
    b = open(path, "rb").read()
    r = R(b, trace)
    assert r.take(3) == b"CCB"; flags = r.u8() - 48; ver = r.u32()
    root = parse_object(r)
    assert r.p == len(b), f"stopped at {r.p} of {len(b)}"
    return ver, root

if __name__ == "__main__":
    ver, root = read(sys.argv[1], trace=True)
    print("version", ver)
