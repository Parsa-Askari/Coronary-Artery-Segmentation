import numpy as np
import cv2
from .gnn_helpers import propagate_labels

CLASS_MAP = {1:'1',2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',10:'9a',
             11:'10',12:'10a',13:'11',14:'12',15:'12a',16:'13',17:'14',18:'14a',
             19:'15',20:'16',21:'16a',22:'16b',23:'16c',24:'12b',25:'14b'}
NAME_TO_GLOBAL = {v: k for k, v in CLASS_MAP.items()}

_LUT = np.zeros((256, 3), np.uint8)
for _i, _gid in enumerate(sorted(NAME_TO_GLOBAL.values())):
    _h = int(179 * _i / max(1, len(NAME_TO_GLOBAL) - 1))
    _LUT[_gid] = cv2.cvtColor(np.uint8([[[_h, 210, 245]]]), cv2.COLOR_HSV2BGR)[0, 0]


def _merge(fam_dict, cnn_for_img, img_shape, family_labels, from_gt=False):
    h, w = img_shape[:2]
    merged = np.zeros((h, w), np.int64)
    claimed = np.zeros((h, w), bool)
    conflicts = 0
    for fam, payload in fam_dict.items():
        if from_gt:
            fam_mask = np.asarray(payload)
        else:
            binm = cnn_for_img.get(fam)
            if binm is None:
                continue
            fam_mask = propagate_labels(binm, np.asarray(payload[1])[:, :2],
                                        np.asarray(payload[0]), img_shape)
        for ci, seg_name in family_labels[fam].items():
            if ci == 0:
                continue
            sel = (fam_mask == ci)
            conflicts += int((sel & claimed).sum())
            merged[sel] = NAME_TO_GLOBAL[seg_name]
            claimed |= sel
    return merged, conflicts


def _annotate(rgb, title, legend_items, legend):
    cv2.putText(rgb, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    if legend:
        y = 44
        for gid, n in legend_items:
            cv2.rectangle(rgb, (8, y - 9), (22, y + 1),
                          tuple(int(v) for v in _LUT[gid]), -1)
            cv2.putText(rgb, f"{CLASS_MAP[gid]}  ({n})", (28, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (235, 235, 235), 1, cv2.LINE_AA)
            y += 17
    return rgb


def _colourise(mask, title, legend):
    rgb = np.zeros((*mask.shape, 3), np.uint8)
    items = []
    for gid in np.unique(mask):
        if gid:
            rgb[mask == gid] = _LUT[gid]
            items.append((int(gid), int((mask == gid).sum())))
    return _annotate(rgb, title, items, legend)


def _draw_binary(cnn_for_img, img_shape, title="CNN binary mask (all families)"):
    """Union of the per-family CNN binary masks -- what the GNN had to work with."""
    h, w = img_shape[:2]
    rgb = np.zeros((h, w, 3), np.uint8)
    union = np.zeros((h, w), bool)
    for binm in cnn_for_img.values():
        union |= (np.asarray(binm) > 0)
    rgb[union] = (235, 235, 235)
    cv2.putText(rgb, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(rgb, f"fg px ({int(union.sum())})", (8, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
    return rgb


def _draw_nodes(fam_dict, img_shape, family_labels, radius=3):
    h, w = img_shape[:2]
    canvas = np.zeros((h, w, 3), np.uint8)
    counts = {}
    for fam, payload in fam_dict.items():
        labels = np.asarray(payload[0]).astype(int)
        pts = np.asarray(payload[1], float)[:, :2].copy()
        pts[:, 0] *= h
        pts[:, 1] *= w
        for (r, c), ci in zip(pts, labels):
            if ci == 0 or ci not in family_labels[fam]:
                continue
            gid = NAME_TO_GLOBAL[family_labels[fam][ci]]
            cv2.circle(canvas, (int(round(c)), int(round(r))), radius,
                       tuple(int(v) for v in _LUT[gid]), -1)
            counts[gid] = counts.get(gid, 0) + 1
    return canvas, counts


def draw_gnn_masks(gnn_preds, cnn_vessels,family_labels,
                   gt_vessels=None, img_shape=(512, 512),
                   legend=True, node_radius=3,
                   report_conflicts=True):
    """
    One 2x2 BGR image per img_name:

        [ CNN binary mask   ][ GNN node labels ]
        [ propagated pred   ][ ground truth    ]

    Returns {img_name: HxWx3 uint8 BGR} -- pass straight to cv2.imwrite.
    """
    out = {}
    h, w = img_shape[:2]
    for name, fam_dict in gnn_preds.items():
        cnn_for_img = cnn_vessels.get(name, {})

        tl = _draw_binary(cnn_for_img, img_shape)
        nodes_rgb, counts = _draw_nodes(fam_dict, img_shape, family_labels, node_radius)
        tr = _annotate(nodes_rgb, "GNN node labels", sorted(counts.items()), legend)

        pred, conf = _merge(fam_dict, cnn_for_img, img_shape, family_labels)
        if report_conflicts and conf:
            print(f"{name}: {conf} px claimed by >1 family")
        bl = _colourise(pred, "propagated prediction", legend)

        if gt_vessels is not None and name in gt_vessels:
            gt_merged, _ = _merge(gt_vessels[name], {}, img_shape, family_labels,
                                  from_gt=True)
            br = _colourise(gt_merged, "ground truth", legend)
        else:
            br = np.zeros((h, w, 3), np.uint8)
            cv2.putText(br, "no ground truth", (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (120, 120, 120), 1, cv2.LINE_AA)

        vsep = np.full((h, 6, 3), 70, np.uint8)
        top = np.hstack([tl, vsep, tr])
        bot = np.hstack([bl, vsep, br])
        hsep = np.full((6, top.shape[1], 3), 70, np.uint8)
        out[name] = np.vstack([top, hsep, bot])
    return out