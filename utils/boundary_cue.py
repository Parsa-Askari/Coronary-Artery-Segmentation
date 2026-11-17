"""
Measure a FIXED list of anatomically consecutive segment pairs, and say explicitly
why any requested pair produced no measurement.
"""
import numpy as np
import cv2
from pathlib import Path
from scipy import ndimage
from skimage.morphology import skeletonize

# full 25-segment map (left + right); adjust to your on-disk numbering

FULL_MAP = {1:'1',2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',10:'9a',
            11:'10',12:'10a',13:'11',14:'12',15:'12a',16:'13',17:'14',18:'14a',
            19:'15',20:'16',21:'16a',22:'16b',23:'16c',24:'12b',25:'14b'}

PAIRS = [('1','2'), ('2','3'), ('3','4'), ('3','16'),
         ('5','6'), ('5','11'), ('5','12'),
         ('6','7'), ('7','8'), ('11','13')]

def _branch_points(skel):
    nb = ndimage.convolve(skel.astype(int), np.ones((3,3), int), mode="constant") * skel
    return (nb >= 4) & skel


def scan_pairs(label_dir,image_dir, pairs=PAIRS, class_map=FULL_MAP, img_loader=None,
               band=4, n_control=25, min_contact=3, seed=0):
    rng = np.random.default_rng(seed)
    name2id = {v: k for k, v in class_map.items()}
    stats = {p: dict(rows=[], both_present=0, touched=0) for p in pairs}

    for f in sorted(Path(label_dir).glob("*.npy")):
        gt = np.load(f)

        img = cv2.imread(Path(image_dir)/ f"{Path(f).stem}.png")
        vessel = gt > 0
        skel = skeletonize(vessel)
        if skel.sum() < 10:
            continue
        dt = cv2.distanceTransform(vessel.astype(np.uint8), cv2.DIST_L2, 5)
        bp = _branch_points(skel)
        sy, sx = np.nonzero(skel); cent = np.stack([sy, sx], 1).astype(float)
        k3 = np.ones((3,3), np.uint8)

        def _step(p):
            d = np.linalg.norm(cent - p, axis=1)
            near = cent[(d > band) & (d < 3*band)]
            if len(near) < 6: return None
            v = near - p
            ax = v[np.argmax(np.linalg.norm(v,axis=1))]
            ax = ax/(np.linalg.norm(ax)+1e-9)
            pr = v @ ax
            s1, s2 = near[pr>0].astype(int), near[pr<=0].astype(int)
            if len(s1)<3 or len(s2)<3: return None
            r = abs(dt[s1[:,0],s1[:,1]].mean() - dt[s2[:,0],s2[:,1]].mean())
            i = np.nan if img is None else abs(
                float(img[s1[:,0],s1[:,1]].mean()) - float(img[s2[:,0],s2[:,1]].mean()))
            return r, i

        for pr_names in pairs:
            a, b = (name2id.get(pr_names[0]), name2id.get(pr_names[1]))
            if a is None or b is None:
                continue
            A, B = (gt == a), (gt == b)
            if not (A.any() and B.any()):
                continue
            stats[pr_names]["both_present"] += 1
            contact = cv2.dilate(A.astype(np.uint8), k3).astype(bool) & B
            if contact.sum() < min_contact:
                continue
            stats[pr_names]["touched"] += 1
            by, bx = np.nonzero(contact); pt = np.array([by.mean(), bx.mean()])
            py, px = np.nonzero(bp)
            d_br = float(np.min(np.hypot(py-pt[0], px-pt[1]))) if len(py) else np.nan
            at = _step(pt)
            if at is None: continue
            ctrl = [_step(cent[j]) for j in
                    rng.choice(len(cent), min(n_control, len(cent)), replace=False)]
            ctrl = np.array([c for c in ctrl if c is not None])
            if not len(ctrl): continue
            stats[pr_names]["rows"].append(
                (d_br, at[0], float(ctrl[:,0].mean()), at[1], float(np.nanmean(ctrl[:,1]))))
    return stats


def report(stats, cue_thresh=1.2):
    """
    Ranked by radius ratio -- the informative axis.

    d_branch is reported but NOT used to judge: the branch that defines a segment
    boundary is frequently unannotated in ARCADE, so distance to a mask-skeleton
    branch point is dominated by which vessels happen to be labelled, not by where
    the anatomical takeoff is.
    """
    print(f"{'pair':>10} {'n':>4} {'imgs both':>10} {'touched':>8} "
          f"{'d_branch':>9} {'r_step':>7} {'r_ctrl':>7} {'ratio':>7} "
          f"{'i_step':>7} {'i_ctrl':>7} {'i_ratio':>8}")
    scored, missing = [], []
    for p, s in stats.items():
        nm = f"{p[0]}|{p[1]}"
        if not s["rows"]:
            why = ("never co-occur" if s["both_present"] == 0
                   else f"co-occur in {s['both_present']} but never touch")
            missing.append((nm, s, why))
            continue
        A = np.array(s["rows"])
        d, r, rc, i, ic = np.nanmean(A, axis=0)
        scored.append((r/(rc+1e-9), nm, len(A), s, d, r, rc, i, ic))

    for ratio, nm, n, s, d, r, rc, i, ic in sorted(scored):
        iratio = i/(ic+1e-9)
        tag = "  <- no visual cue" if ratio < cue_thresh else ""
        print(f"{nm:>10} {n:4d} {s['both_present']:10d} {s['touched']:8d} "
              f"{d:9.1f} {r:7.2f} {rc:7.2f} {ratio:7.2f} "
              f"{i:7.2f} {ic:7.2f} {iratio:8.2f}{tag}")
    for nm, s, why in missing:
        print(f"{nm:>10} {0:4d} {s['both_present']:10d} {s['touched']:8d}   -- {why}")


def cue_vs_confusion(stats, conf_rate, cue_thresh=1.2):
    """
    Correlate the measured visual cue against how often the model actually confuses
    the pair.

    conf_rate : {"6|7": 0.13, ...}  symmetric confusion rate read off the validation
                confusion matrix, e.g. (P(a->b) + P(b->a)) / 2.
    """
    from scipy.stats import spearmanr
    xs, ys, names = [], [], []
    for p, s in stats.items():
        nm = f"{p[0]}|{p[1]}"
        if not s["rows"] or nm not in conf_rate:
            continue
        A = np.array(s["rows"])
        _, r, rc, _, _ = np.nanmean(A, axis=0)
        xs.append(r/(rc+1e-9)); ys.append(conf_rate[nm]); names.append(nm)
    if len(xs) < 3:
        print("not enough pairs"); return None
    rho, pv = spearmanr(xs, ys)
    print(f"{'pair':>10} {'cue ratio':>10} {'confusion':>10}")
    for nm, x, y in sorted(zip(names, xs, ys), key=lambda t: t[1]):
        print(f"{nm:>10} {x:10.2f} {y:10.3f}")
    print(f"\nSpearman rho = {rho:+.3f}  (p = {pv:.4f}), n = {len(xs)}")
    print("negative rho -> weaker visual cue goes with MORE confusion")
    return dict(rho=rho, p=pv, ratios=xs, confusion=ys, names=names)

if __name__ == "__main__" :
    stats = scan_pairs(
        "/home/parsa/Masters/Coronary Artry/Datasets/Arcade/codes/dataset/syntax/labels/train",
        "/home/parsa/Masters/Coronary Artry/Datasets/Arcade/codes/dataset/syntax/images/train"
    )
    print(report(stats=stats))
