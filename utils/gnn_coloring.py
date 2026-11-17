import numpy as np, cv2
from pathlib import Path
from scipy.spatial import cKDTree
from tqdm import tqdm

def assign_labels_to_mask(binary_fg, node_coords, node_labels):
    """Give every foreground pixel the label of its nearest node."""
    out = np.zeros(binary_fg.shape, np.int64)
    ys, xs = np.where(binary_fg)
    if len(ys) == 0 or len(node_coords) == 0:
        return out
    _, idx = cKDTree(node_coords).query(np.stack([ys, xs], 1))
    out[ys, xs] = node_labels[idx]
    return out


def compute_gnn_ceiling(gnn_root, pred_root, gt_root, class_map,
                        image_shape=512, parts=("train", "val", "test"),
                        coord_cols=(0, 1), bg_label=0, verbose=True):
    gnn_root, pred_root, gt_root = Path(gnn_root), Path(pred_root), Path(gt_root)
    fg_classes = [c for c in class_map if c != bg_label]
    results = {}

    for part in parts:
        lab_dir, feat_dir = gnn_root / part / "labels", gnn_root / part / "features"
        inter   = {c: 0 for c in fg_classes}     # |pred ∩ gt|
        psum    = {c: 0 for c in fg_classes}     # |pred|
        gsum    = {c: 0 for c in fg_classes}     # |gt|
        covered = {c: 0 for c in fg_classes}     # gt px falling inside the binary mask
        per_img = {c: [] for c in fg_classes}
        n_used = n_skip = 0

        for lp in tqdm(sorted(lab_dir.glob("*.npy"))):
            stem = lp.stem
            fp, gp = feat_dir / f"{stem}.npy", gt_root / part / f"{stem}.npy"
            pp = pred_root / part / f"{stem}.png"
            if not (fp.exists() and gp.exists() and pp.exists()):
                n_skip += 1; continue

            node_labels = np.load(lp).astype(np.int64).reshape(-1)
            feats  = np.load(fp)
            coords = feats[:, list(coord_cols)].astype(np.float64) * image_shape

            png = cv2.imread(str(pp), cv2.IMREAD_GRAYSCALE)
            if png is None:
                n_skip += 1; continue
            fg = png > 127
            gt = np.load(gp).astype(np.int64)

            pred = assign_labels_to_mask(fg, coords, node_labels)
            n_used += 1

            for c in fg_classes:
                p, g = (pred == c), (gt == c)
                i, pc, gc = int((p & g).sum()), int(p.sum()), int(g.sum())
                inter[c] += i; psum[c] += pc; gsum[c] += gc
                covered[c] += int((g & fg).sum())
                if gc or pc:
                    per_img[c].append(2 * i / (pc + gc))

        res = {}
        for c in fg_classes:
            res[class_map[c]] = {
                "dice_global": (2*inter[c]/(psum[c]+gsum[c])) if (psum[c]+gsum[c]) else float("nan"),
                "dice_per_image_mean": float(np.mean(per_img[c])) if per_img[c] else float("nan"),
                "gt_px": gsum[c], "pred_px": psum[c],
                "binary_recall": (covered[c]/gsum[c]) if gsum[c] else float("nan"),
            }
        results[part] = res

        if verbose:
            print(f"\n===== {part}  ({n_used} images, {n_skip} skipped) =====")
            print(f"{'class':<8s} {'dice(glob)':>11s} {'dice(img)':>10s} "
                  f"{'binaryRecall':>13s} {'gt_px':>9s} {'pred_px':>9s}")
            for name, r in res.items():
                print(f"{name:<8s} {r['dice_global']:>11.4f} {r['dice_per_image_mean']:>10.4f} "
                      f"{r['binary_recall']:>13.4f} {r['gt_px']:>9d} {r['pred_px']:>9d}")
            ds = [r["dice_global"] for r in res.values() if not np.isnan(r["dice_global"])]
            print(f"{'MEAN':<8s} {np.mean(ds):>11.4f}")
    return results