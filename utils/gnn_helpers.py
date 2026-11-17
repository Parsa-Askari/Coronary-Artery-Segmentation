from scipy.ndimage import distance_transform_edt
import numpy as np
from skimage.morphology import skeletonize
from scipy.ndimage import convolve
import networkx as nx
from scipy.spatial import cKDTree
from scipy.ndimage import label as cc_label
from pathlib import Path
from scipy import ndimage
import cv2
import os
from tqdm import tqdm
import re
import torch
from collections import deque
###IE###
###SS###
_AXES = {
    "left":        lambda r, c:  c,
    "right":       lambda r, c: -c,
    "top":         lambda r, c:  r,
    "bottom":      lambda r, c: -r,
    "topleft":     lambda r, c:  r + c,
    "topright":    lambda r, c:  r - c,
    "bottomleft":  lambda r, c:  c - r,
    "bottomright": lambda r, c: -(r + c),
}

def find_branch_depth(coords, edge_index, root_idx, N,
                      min_branch_nodes=5, max_depth=6):
    depth = np.zeros(N, np.int64)
    if root_idx is None or N == 0 or edge_index.shape[1] == 0:
        return depth.astype(np.float32) / max_depth, depth

    G = nx.Graph(); G.add_nodes_from(range(N)); G.add_edges_from(edge_index.T.tolist())
    if root_idx not in G:
        return depth.astype(np.float32) / max_depth, depth

    parent = {root_idx: None}; order = [root_idx]; dq = deque([root_idx])
    while dq:
        u = dq.popleft()
        for v in G.neighbors(u):
            if v not in parent:
                parent[v] = u; order.append(v); dq.append(v)

    children = {u: [] for u in parent}
    for v, p in parent.items():
        if p is not None: children[p].append(v)

    subtree = {u: 1 for u in parent}
    for u in reversed(order):
        p = parent[u]
        if p is not None: subtree[p] += subtree[u]

    for u in order:
        kids = children[u]
        real = [v for v in kids if subtree[v] >= min_branch_nodes]   # spur filter
        is_bp = len(real) >= 2
        for v in kids:
            depth[v] = depth[u] + (1 if is_bp and v in real else 0)

    depth = np.minimum(depth, max_depth)
    return depth.astype(np.float32) / max_depth, depth

def fit_feature_stats(train_feature_dir, save_path=None):
    train_feature_dir = Path(train_feature_dir)
    n = 0
    s = ss = None
    for f in train_feature_dir.glob("*.npy"):
        x = np.load(f).astype(np.float64)
        if s is None:
            s = np.zeros(x.shape[1]); ss = np.zeros(x.shape[1])
        s  += x.sum(0)
        ss += (x ** 2).sum(0)
        n  += x.shape[0]
    mean = s / n
    std  = np.sqrt(np.clip(ss / n - mean ** 2, 1e-12, None))
    mean, std = mean.astype(np.float32), std.astype(np.float32)
    if save_path:
        np.savez(save_path, mean=mean, std=std)
    return  mean, std

def sample_at_nodes(field, coords):
    """field: (H,W). coords: (N,2) row,col. -> (N,) values at node pixels."""
    rr = np.clip(coords[:, 0].round().astype(int), 0, field.shape[0] - 1)
    cc = np.clip(coords[:, 1].round().astype(int), 0, field.shape[1] - 1)
    return field[rr, cc].astype(np.float32)

def fix_16_family(mask,index_16):
    mask[mask<index_16] = 0
    if(5 in mask):
        mask[mask!=0]-=4
        return mask
    else:
        mask[mask!=0]-= 19
        return mask

def find_radius(binary_mask, coords, norm=20.0):
    dt = distance_transform_edt(binary_mask > 0)          # (H,W): radius field
    r = sample_at_nodes(dt, coords)
    return (r / norm).astype(np.float32)

def find_degree(edge_index, N, norm=4.0):
    if edge_index.shape[1] == 0:
        return np.zeros(N, dtype=np.float32)
    deg = np.bincount(edge_index[0], minlength=N).astype(np.float32)
    return deg / norm  

def find_root(coords, class1_mask):
    if coords.shape[0] == 0 or not class1_mask.any():
        return None
    sk1 = skeletonize(class1_mask > 0)
    k = np.array([[1,1,1],[1,0,1],[1,1,1]])
    nbr = convolve(sk1.astype(int), k, mode="constant")
    ep = np.argwhere((sk1) & (nbr == 1))    
    if len(ep) == 0:
        ep = np.argwhere(sk1)        
    ostium = ep[np.argmin(ep[:, 0])]           
    d = np.sqrt(((coords - ostium) ** 2).sum(1))
    return int(np.argmin(d))

# def find_left_root(coords, gt_mask, branch_idx, root_idx, axis="left"):
#     rows = coords[:, 0].astype(int)
#     cols = coords[:, 1].astype(int)
#     node_cls = gt_mask[rows, cols]

#     on_branch = np.where(node_cls == branch_idx)[0]
#     if on_branch.size == 0:
#         return None

#     if axis == "left":
#         proximal = cols
#     elif axis == "top":
#         proximal = rows
#     elif axis == "topleft":
#         proximal = rows + cols 
#     else:
#         raise ValueError(f"unknown axis {axis!r}")

#     for cand in root_idx[branch_idx]:
#         if cand == branch_idx:
#             return int(on_branch[np.argmin(proximal[on_branch])])

#         parent = (gt_mask == cand)
#         if parent.any():
#             dist = ndimage.distance_transform_edt(~parent)
#             k = np.argmin(dist[rows[on_branch], cols[on_branch]])
#             return int(on_branch[k])

#     return int(on_branch[np.argmin(proximal[on_branch])])



def find_left_root(coords, pred_mask=None, axis="left", edge_index=None, N=None):

    coords = np.asarray(coords)
    if coords.size == 0:
        return None
    if axis not in _AXES:
        raise ValueError(f"unknown axis {axis!r}; expected one of {sorted(_AXES)}")

    rows = coords[:, 0].astype(float)
    cols = coords[:, 1].astype(float)
    score = _AXES[axis](rows, cols)

    cand = np.arange(len(coords))
    if edge_index is not None and np.asarray(edge_index).size:
        ei = np.asarray(edge_index)
        n = len(coords) if N is None else N
        deg = np.bincount(ei[0], minlength=n)
        ends = np.where(deg == 1)[0]
        if ends.size:
            cand = ends

    return int(cand[np.argmin(score[cand])])

def find_right_root(coords, class_mask,c=3):

    if(c==3):
        if coords.shape[0] == 0:
            return None
        if not class_mask.any():
            return None
        d = distance_transform_edt(~class_mask)
        rr = np.clip(coords[:, 0].round().astype(int), 0, d.shape[0] - 1)
        cc = np.clip(coords[:, 1].round().astype(int), 0, d.shape[1] - 1)
        return int(np.argmin(d[rr, cc]))
    else:
        if coords.shape[0] == 0:
            return None
        if not class_mask.any():
            return None
        d = distance_transform_edt(class_mask)
        rr = np.clip(coords[:, 0].round().astype(int), 0, d.shape[0] - 1)
        cc = np.clip(coords[:, 1].round().astype(int), 0, d.shape[1] - 1)
        return int(np.argmin(d[rr, cc]))
    
def find_dist_from_root(root_idx,edge_index,coords,image_shape,N):
    if root_idx is None:
        geo_n     = np.full(N, -1.0, np.float32)
        euc_n     = np.zeros(N, np.float32)
        man_n     = np.zeros(N, np.float32)
        reachable = np.zeros(N, np.float32)
    else:

        G = nx.Graph()
        G.add_nodes_from(range(N))
        G.add_edges_from(edge_index.T.tolist())
        hops = nx.single_source_shortest_path_length(G, root_idx)
        geo = np.full(N, -1.0, np.float32)                
        for n, d in hops.items():
            geo[n] = d
        reachable = (geo >= 0).astype(np.float32)
        gmax = geo[geo >= 0].max() if (geo >= 0).any() else 1.0
        geo_n = np.where(geo >= 0, geo / max(1.0, gmax), -1.0).astype(np.float32)

        diff = coords - coords[root_idx]
        euc_n = (np.sqrt((diff ** 2).sum(1)) / (np.sqrt(2) * image_shape)).astype(np.float32)
        man_n = (np.abs(diff).sum(1) / (2 * image_shape)).astype(np.float32)
        
    return geo_n , euc_n , man_n , reachable
    
# def set_labels(coords,gt_mask):
#     rr = np.clip(coords[:,0].round().astype(int), 0, gt_mask.shape[0]-1)
#     cc = np.clip(coords[:,1].round().astype(int), 0, gt_mask.shape[1]-1)
#     node_labels = gt_mask[rr, cc].reshape(-1)
#     return node_labels

def set_labels(coords, gt_mask, max_dist=6.0, k=9, vote=True):
    """
    Label graph nodes from a GT mask, robust to CNN/GT misalignment.

    Direct lookup fails when a node sits 1-2 px off the GT vessel (the skeleton comes
    from the CNN mask, the labels from GT). Instead:
      1. take the k nearest LABELLED pixels within `max_dist`
      2. distance-weighted majority vote among them
      3. nothing within max_dist -> label 0 (node really is on a false-positive vessel)
    """
    coords = np.asarray(coords, float)
    ys, xs = np.nonzero(gt_mask)
    if len(ys) == 0:
        return np.zeros(len(coords), np.int64)

    pts = np.stack([ys, xs], 1).astype(float)
    lab = np.asarray(gt_mask)[ys, xs].astype(np.int64)

    kk = min(k, len(pts))
    d, idx = cKDTree(pts).query(coords, k=kk)
    if kk == 1:
        d, idx = d[:, None], idx[:, None]

    out = np.zeros(len(coords), np.int64)
    for i in range(len(coords)):
        m = d[i] <= max_dist
        if not m.any():
            continue                           
        if not vote:
            out[i] = lab[idx[i][m][0]]
            continue
        w = 1.0 / (d[i][m] + 1e-6)
        scores = {}
        for c, wt in zip(lab[idx[i][m]], w):
            scores[c] = scores.get(c, 0.0) + wt
        out[i] = max(scores, key=scores.get)
    return out

def load_feature_stats(path):
    d = np.load(path)
    return list(d["cols"]), d["mean"], d["std"]

def apply_feature_stats(features, mean, std):
    out = features.copy()
    out = (out - mean) / (std + 1e-6)
    return out



def left_mapping_function(old_class_map,new_class_map):
    if(len(new_class_map)!=17):
        collapse = {
            # 1 = LAD territory
            '6':   '1',
            '7':   '1',
            '8':   '1',
            '9':   '1',
            '9a':  '1',
            '10':  '1',
            '10a': '1',

            # 2 = LCx territory
            '11':  '2',
            '12':  '2',
            '12a': '2',
            '12b': '2',
            '13':  '2',
            '14':  '2',
            '14a': '2',
            '14b': '2',
            '15':  '2',

            # 3 = left main
            '5':   '3',
        }
      
        final_label = {idx: collapse.get(lbl, lbl) for idx, lbl in old_class_map.items()}

        surviving = sorted(set(final_label.values()), key=int)
        new_class_map = {i + 1: lbl for i, lbl in enumerate(surviving)}
        label_to_new = {lbl: i for i, lbl in new_class_map.items()}

        remap = {0: 0}
        for old_idx, lbl in final_label.items():
            remap[old_idx] = label_to_new[lbl]

        lut = np.zeros(max(remap) + 1, dtype=np.uint8)
        for old_idx, new_idx in remap.items():
            lut[old_idx] = new_idx
    else:
        label_to_new = {label: idx for idx, label in new_class_map.items()}
        parent = lambda lbl: re.match(r'\d+', lbl).group()
        remap = {0: 0}
        for old_idx, label in old_class_map.items():
            remap[old_idx] = label_to_new[parent(label)]

        lut = np.zeros(max(remap) + 1, dtype=np.uint8)
        for old_idx, new_idx in remap.items():
            lut[old_idx] = new_idx
            
    return lut


def left_gnn_mapping_function(old_class_map, new_class_map, family,verbose=False):
    if(family == "1"):
        collapse = {
            "bg":0,
            "6":1,
            "7":2,
            "8":3,
            "9":4,
            "9a":5,
            "10":5
        }
    elif(family == "2"):
        collapse = {
            "bg":0,
            "11":1,
            "12":2,
            "12a":3,
            "12b":3,
            "13":4,
            "14":5,
            "14a":5,
            "14b":5,
            "15":5,
                        
        }
    final_label = {idx: collapse.get(lbl, 0) for idx, lbl in old_class_map.items()}

    surviving = sorted({v for v in final_label.values() if v != 0}, key=int)
    new_class_map = {i + 1: lbl for i, lbl in enumerate(surviving)}
    label_to_new = {lbl: i for i, lbl in new_class_map.items()}
    label_to_new[0] = 0                                   # <- the fix

    remap = {0: 0}
    for old_idx, lbl in final_label.items():
        remap[old_idx] = label_to_new[lbl]

    lut = np.zeros(max(remap) + 1, dtype=np.uint8)
    for old_idx, new_idx in remap.items():
        lut[old_idx] = new_idx
    print(lut,new_class_map)
    return lut

def load_set(names,base_gnn_path,part,skip_c = -1):
    gnn_preds = {}
    cnn_vessels = {}
    gt_vessels = {}
    
    for name in tqdm(names):
        base_feature_path = base_gnn_path / part / name/ "features"
        base_label_path = base_gnn_path / part / name / "labels"
        base_pred_path = base_gnn_path / part / name / "pred_masks"
        base_true_path = base_gnn_path / part / name / "true_masks"

        classes = os.listdir(base_pred_path)
        classes = [Path(c).stem for c in classes]
        gnn_preds[name] = {}
        cnn_vessels[name] = {}
        gt_vessels[name] = {}
        
        for c in classes:
            c = int(c)
            pred = (cv2.imread(base_pred_path / f"{c}.png",cv2.IMREAD_GRAYSCALE) /255).astype(int)
            gt = np.load(base_true_path / f"{c}.npy")
            labels = np.load(base_label_path / f"{c}.npy")
            features = np.load(base_feature_path / f"{c}.npy")

            cnn_vessels[name][c] = pred
            
            gnn_preds[name][c] = [labels,features]
            if(c!=skip_c):
                gt_vessels[name][c] = gt

    return gnn_preds , cnn_vessels , gt_vessels

def propagate_labels(binary_mask, coords_norm, node_labels, img_shape=None):
    h, w = binary_mask.shape[:2] if img_shape is None else img_shape[:2]
    out = np.zeros((h, w), np.int64)

    ys, xs = np.nonzero(binary_mask > 0)        # every foreground pixel
    if len(ys) == 0 or len(coords_norm) == 0:
        return out

    pts = np.asarray(coords_norm, float)[:, :2].copy()
    pts[:, 0] *= h                              # de-normalise (coords are 0-1)
    pts[:, 1] *= w

    nn = cKDTree(pts).query(np.stack([ys, xs], 1))[1]   # nearest node per pixel
    out[ys, xs] = np.asarray(node_labels, np.int64)[nn] # inherit its label
    return out


def bridge_components(coords, edge_index, max_gap=12.0):
    """Connect nearest endpoints of separate components if within max_gap px."""
    import networkx as nx
    G = nx.Graph(); G.add_nodes_from(range(len(coords)))
    G.add_edges_from(zip(edge_index[0], edge_index[1]))
    comps = [list(c) for c in nx.connected_components(G)]
    if len(comps) < 2:
        return edge_index
    new = []
    main = max(comps, key=len)
    tree = cKDTree(coords[main])
    for c in comps:
        if c is main: continue
        d, j = tree.query(coords[c])
        k = int(np.argmin(d))
        if d[k] <= max_gap:
            new.append((c[k], main[j[k]]))
    if not new:
        return edge_index
    e = np.array(new).T
    return np.concatenate([edge_index, e, e[::-1]], axis=1)


def node_pixel_weights(binary_mask, coords_norm, img_shape=None, power=1.0,
                       clip_pct=(1, 99), normalise=True):
    h, w = binary_mask.shape[:2] if img_shape is None else img_shape[:2]
    coords = np.asarray(coords_norm, float)[:, :2]
    n = len(coords)
    if n == 0:
        return np.zeros(0, np.float32)
 
    ys, xs = np.nonzero(np.asarray(binary_mask) > 0)
    if len(ys) == 0:
        return np.ones(n, np.float32)
 
    pts = coords.copy()
    pts[:, 0] *= h
    pts[:, 1] *= w
    nn = cKDTree(pts).query(np.stack([ys, xs], 1))[1]
    counts = np.bincount(nn, minlength=n).astype(np.float32)
 
    counts = np.maximum(counts, 1.0)                 # a node owning 0 px still trains
    wgt = counts ** power
    if clip_pct is not None:
        lo, hi = np.percentile(wgt, clip_pct)
        wgt = np.clip(wgt, lo, hi)
    if normalise:
        wgt = wgt / wgt.mean()
    return wgt.astype(np.float32)



if __name__ =="__main__" : 
    old_class_map = {
        0:"bg",1:"6",2:"7",3:"8",4:"9",5:"9a",6:"10"
    }
    class_map = {
        0:"bg",1:"6",2:"7",3:"8",4:"9",4:"9a+10"
    }
    left_gnn_mapping_function(old_class_map,class_map,family="1")

