
import argparse
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon
from tqdm import tqdm

RING_PX, GUARD_PX, BG_PX = 10, 2, 25


def load_masks(lab_path):
    lab = np.load(lab_path)
    if lab.ndim == 3:                                  # (H,W,C) one-hot
        masks = {c: lab[:, :, c].astype(bool) for c in range(lab.shape[2])}
    else:                                              # (H,W) class ids
        masks = {int(c): lab == c for c in np.unique(lab) if c != 0}
    return {c: m for c, m in masks.items() if m.sum() >= 50}


def _ring(mask, all_vessels):
    """Band around `mask`, minus a guard zone and every other vessel."""
    outer = cv2.dilate(mask.astype(np.uint8), np.ones((RING_PX * 2 + 1,) * 2, np.uint8))
    guard = cv2.dilate(all_vessels.astype(np.uint8), np.ones((GUARD_PX * 2 + 1,) * 2, np.uint8))
    return (outer > 0) & (guard == 0)


def cnr_auc(img, mask, all_vessels):
    ring = _ring(mask, all_vessels)
    v, b = img[mask].astype(float), img[ring].astype(float)
    if len(v) < 20 or len(b) < 20 or b.std() < 1e-6:
        return np.nan, np.nan
    cnr = abs(v.mean() - b.mean()) / b.std()
    u = mannwhitneyu(v, b, alternative="two-sided").statistic
    auc = max(u, len(v) * len(b) - u) / (len(v) * len(b))   # direction-agnostic
    return cnr, auc


def bg_noise(img, all_vessels):
    far = cv2.dilate(all_vessels.astype(np.uint8), np.ones((BG_PX * 2 + 1,) * 2, np.uint8)) == 0
    if far.sum() < 500:
        return np.nan
    hp = img.astype(float) - cv2.medianBlur(img, 5).astype(float)   # kill the slow field
    return hp[far].std()


def score_image(img, masks):
    allv = np.zeros(img.shape, bool)
    for m in masks.values():
        allv |= m
    rows = []
    for c, m in masks.items():
        cnr, auc = cnr_auc(img, m, allv)
        rows.append(dict(cls=c, cnr=cnr, auc=auc, zero=(img[m] == 0).mean()))
    return rows, bg_noise(img, allv)


# --------------------------------------------------------------------------- #
def _init_worker():
    cv2.setNumThreads(0)      # stop OpenCV threading inside each process


def _score_one(task):
    """One image, every directory. Top-level so it pickles."""
    lp, img_paths = task                       # img_paths: {name: Path}
    masks = load_masks(lp)
    if not masks:
        return None
    out = {}
    for name, ip in img_paths.items():
        img = cv2.imread(str(ip), cv2.IMREAD_GRAYSCALE)   # collapses your 3x-stacked gray
        if img is None:
            return None
        out[name] = score_image(img, masks)
    return out


def run(dirs, labels_dir, split, per_class, workers=None):
    names = list(dirs)
    stems = sorted(p.stem for p in (dirs[names[0]] / split).glob("*.png"))

    tasks = []
    for s in stems:
        lp = labels_dir / split / f"{s}.npy"
        ips = {n: dirs[n] / split / f"{s}.png" for n in names}
        if lp.exists() and all(p.exists() for p in ips.values()):
            tasks.append((lp, ips))
    if not tasks:
        raise SystemExit("no image/label pairs found - check --a/--b/--labels/--split")

    workers = workers or min(os.cpu_count() or 1, len(tasks))
    per_img = {n: [] for n in names}
    per_cls = {n: {} for n in names}

    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as ex:
        for res in tqdm(ex.map(_score_one, tasks, chunksize=4),
                        total=len(tasks), desc=f"{split} x{workers}"):
            if res is None:
                continue
            for n in names:
                rows, noise = res[n]
                per_img[n].append(dict(
                    cnr=np.nanmean([r["cnr"] for r in rows]),
                    auc=np.nanmean([r["auc"] for r in rows]),
                    zero=np.mean([r["zero"] for r in rows]),
                    bg=noise))
                for r in rows:
                    per_cls[n].setdefault(r["cls"], []).append(r)

    print(f"\n{len(per_img[names[0]])} images, split={split}\n")
    print(f"{'set':<10}{'CNR':>8}{'AUC':>8}{'zero%':>8}{'bg_noise':>10}")
    for n in names:
        d = per_img[n]
        print(f"{n:<10}{np.nanmean([x['cnr'] for x in d]):>8.3f}"
              f"{np.nanmean([x['auc'] for x in d]):>8.3f}"
              f"{100 * np.mean([x['zero'] for x in d]):>8.2f}"
              f"{np.nanmean([x['bg'] for x in d]):>10.2f}")

    if len(names) == 2:
        a, b = names
        print(f"\npaired {b} - {a}  (Wilcoxon signed-rank, n={len(per_img[a])})")
        for k, better in [("cnr", "higher"), ("auc", "higher"),
                          ("zero", "lower"), ("bg", "lower")]:
            x = np.array([d[k] for d in per_img[a]])
            y = np.array([d[k] for d in per_img[b]])
            ok = ~(np.isnan(x) | np.isnan(y))
            d = y[ok] - x[ok]
            p = wilcoxon(d).pvalue if np.any(d) else 1.0
            print(f"  {k:<5} delta={d.mean():+8.4f}  p={p:<10.2e} ({better} is better)")

    if per_class:
        print("\nper-class (zero% / AUC)")
        for c in sorted(per_cls[names[0]]):
            line = f"  {str(c):<5}n={len(per_cls[names[0]][c]):<5}"
            for n in names:
                r = per_cls[n][c]
                line += (f"{n}: {100 * np.mean([x['zero'] for x in r]):5.1f}% "
                         f"{np.nanmean([x['auc'] for x in r]):.3f}   ")
            print(line)


if __name__ == "__main__":
    mp.freeze_support()
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="image dir (contains train/val/test)")
    ap.add_argument("--b", default=None, help="second image dir to compare against")
    ap.add_argument("--labels", required=True, help="label dir (contains train/val/test)")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--per-class", action="store_true")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    dirs = {"a": Path(args.a)}
    if args.b:
        dirs["b"] = Path(args.b)
    run(dirs, Path(args.labels), args.split, args.per_class, args.workers)