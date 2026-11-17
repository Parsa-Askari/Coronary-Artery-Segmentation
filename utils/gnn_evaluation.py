import numpy as np
from .gnn_helpers import propagate_labels

CLASS_MAP = {1:'1',2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',10:'9a',
             11:'10',12:'10a',13:'11',14:'12',15:'12a',16:'13',17:'14',18:'14a',
             19:'15',20:'16',21:'16a',22:'16b',23:'16c',24:'12b',25:'14b'}
_BASE_NAME_TO_GLOBAL = {v: k for k, v in CLASS_MAP.items()}


def build_name_maps(family_labels, class_map=CLASS_MAP):
    """
    Handle collapsed class names like '9a+10' or '1+2+3' in family_labels.

    Every member of a collapsed group is given the SAME global id (the smallest id
    among its members), so pred and gt land on one class. Overlapping groups are
    merged transitively via union-find, e.g. '9a+10' and '10+10a' -> one class.

    Returns (name_to_global, global_to_display).
    """
    base = {v: k for k, v in class_map.items()}
    parent = {n: n for n in base}                       # union-find over atomic names

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:                                    # keep the smaller-id root
            lo, hi = (ra, rb) if base[ra] <= base[rb] else (rb, ra)
            parent[hi] = lo

    for fam in family_labels.values():
        for name in fam.values():
            if "+" not in str(name):
                continue
            parts = str(name).split("+")
            for p in parts:
                if p not in base:
                    raise KeyError(f"collapsed name {name!r} contains unknown segment {p!r}")
            for p in parts[1:]:
                union(parts[0], p)

    groups = {}
    for n in base:
        groups.setdefault(find(n), []).append(n)

    name_to_global, global_to_display = {}, {}
    for root, members in groups.items():
        gid = base[root]
        members_sorted = sorted(members, key=lambda m: base[m])
        disp = "+".join(members_sorted) if len(members) > 1 else members_sorted[0]
        for m in members:
            name_to_global[m] = gid
        global_to_display[gid] = disp

    # collapsed names themselves resolve to their group id too
    for fam in family_labels.values():
        for name in fam.values():
            if "+" in str(name):
                name_to_global[str(name)] = name_to_global[str(name).split("+")[0]]
    return name_to_global, global_to_display


def _merge_pred(fam_dict, cnn_for_img, img_shape, family_labels, name_to_global,
                own_classes=None, ignore_bg=False):
    h, w = img_shape[:2]
    merged = np.zeros((h, w), np.int64)
    prio = np.zeros((h, w), np.int8)
    conflicts = 0

    for fam, payload in fam_dict.items():
        binm = cnn_for_img.get(fam)
        if binm is None:
            continue
        labels = np.asarray(payload[0])
        coords = np.asarray(payload[1])[:, :2]
        if ignore_bg:
            keep = labels != 0
            if keep.any():
                labels, coords = labels[keep], coords[keep]

        fam_mask = propagate_labels(binm, coords, labels, img_shape)
        for ci, seg in family_labels[fam].items():
            if ci == 0:
                continue
            gid = name_to_global[str(seg)]
            is_own = True if own_classes is None else (ci in own_classes.get(fam, []))
            p = 2 if is_own else 1
            sel = (fam_mask == ci)
            conflicts += int((sel & (prio > 0) & (merged != gid)).sum())
            take = sel & (p >= prio)
            merged[take] = gid
            prio[take] = p
    return merged, conflicts


def _merge_gt(gt_fams, family_labels, name_to_global):
    merged = None
    for fam, gt in gt_fams.items():
        gt = np.asarray(gt)
        if merged is None:
            merged = np.zeros_like(gt, np.int64)
        for ci, seg in family_labels[fam].items():
            if ci == 0:
                continue
            merged[gt == ci] = name_to_global[str(seg)]
    return merged


def evaluate_merged(gnn_preds, cnn_vessels, gt_vessels, family_labels,
                    img_shape=(512, 512), own_classes=None, ignore_bg=False,
                    class_map=CLASS_MAP, eps=1e-8, verbose=True):
    """Handles collapsed class names ('9a+10', '1+2+3') in family_labels."""
    name_to_global, global_to_display = build_name_maps(family_labels, class_map)

    inter, psum, gsum = {}, {}, {}
    total_conf = 0
    for name in set(gt_vessels) | set(gnn_preds):
        pred, conf = _merge_pred(gnn_preds.get(name, {}), cnn_vessels.get(name, {}),
                                 img_shape, family_labels, name_to_global,
                                 own_classes, ignore_bg)
        total_conf += conf
        gt = _merge_gt(gt_vessels.get(name, {}), family_labels, name_to_global)
        if gt is None:
            gt = np.zeros(img_shape[:2], np.int64)

        for gid in set(np.unique(pred)) | set(np.unique(gt)):
            if gid == 0:
                continue
            p, g = (pred == gid), (gt == gid)
            inter[gid] = inter.get(gid, 0.0) + float((p & g).sum())
            psum[gid]  = psum.get(gid, 0.0) + float(p.sum())
            gsum[gid]  = gsum.get(gid, 0.0) + float(g.sum())

    per_class = {global_to_display[g]: 2.0 * i / (psum[g] + gsum[g] + eps)
                 for g, i in sorted(inter.items())}
    macro = float(np.mean(list(per_class.values()))) if per_class else 0.0
    if verbose:
        if total_conf:
            print(f"{total_conf} px where two families disagreed")
        if ignore_bg:
            print("ignore_bg=True: background nodes dropped; every fg pixel gets a class")
    return macro, per_class