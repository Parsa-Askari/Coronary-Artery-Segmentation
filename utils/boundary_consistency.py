
import numpy as np
import cv2
from pathlib import Path


def _boundary_point(gt, a, b, min_contact=3):
    A, B = (gt == a), (gt == b)
    if not (A.any() and B.any()):
        return None
    c = cv2.dilate(A.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & B
    if c.sum() < min_contact:
        return None
    y, x = np.nonzero(c)
    return np.array([y.mean(), x.mean()])


def scan(label_dir, pairs, class_map, radius=40, min_images=10):
    name2id = {v: k for k, v in class_map.items()}
    acc = {p: dict(pos=[], ncomp=[]) for p in pairs}
    for f in sorted(Path(label_dir).glob("*.npy")):
        gt = np.load(f)
        H, W = gt.shape
        for nm in pairs:
            a, b = name2id.get(nm[0]), name2id.get(nm[1])
            if a is None or b is None:
                continue
            pt = _boundary_point(gt, a, b)
            if pt is None:
                continue
            acc[nm]["pos"].append([pt[0] / H, pt[1] / W])

            yy, xx = np.ogrid[:H, :W]
            near = ((yy - pt[0]) ** 2 + (xx - pt[1]) ** 2) <= radius ** 2
            others = set(np.unique(gt[near])) - {0, a, b}
            acc[nm]["ncomp"].append(len(others))
    out = []
    for nm, d in acc.items():
        if len(d["pos"]) < min_images:
            continue
        P = np.array(d["pos"])
        # mean distance from the centroid = spatial spread in image coords
        spread = float(np.linalg.norm(P - P.mean(0), axis=1).mean())
        out.append(dict(pair=f"{nm[0]}|{nm[1]}", n=len(P), spread=spread,
                        n_competing=float(np.mean(d["ncomp"]))))
    return sorted(out, key=lambda r: r["spread"])


def report(rows):
    print(f"{'pair':>10} {'n':>5} {'pos spread':>11} {'competing classes':>18}")
    for r in rows:
        print(f"{r['pair']:>10} {r['n']:5d} {r['spread']:11.3f} {r['n_competing']:18.2f}")
    print("\npos spread   = mean distance from the mean boundary location, "
          "in normalised image units")
    print("competing    = other segment classes within `radius` px of the boundary")