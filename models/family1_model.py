import numpy as np
import torch
import networkx as nx
###IE###
from .gnn_model import VesselGraphSAGE
from utils.helpers import load_model
###SS###
# class ids in the COLLAPSED prediction (family 1)
BG, C6, C7, C8, C9, C9A10 = 0, 1, 2, 3, 4, 5
# ids in the EXPANDED output
OUT_9A, OUT_10 = 5, 6
LONE_THRESHOLD = 0.55          # a single component below this -> 9a, above -> 10


def _parent_arc(nodes_p, coords, G, prox_pts):
    """
    Arc position 0..1 of every parent node, measured from the end nearest segment 6.
    Falls back to Euclidean ordering if the parent subgraph is fragmented.
    """
    if len(nodes_p) == 0:
        return None
    sub = G.subgraph(nodes_p)
    comp = max(nx.connected_components(sub), key=len) if sub.number_of_nodes() else set()
    comp = list(comp)
    if len(comp) < 2:
        return None

    # proximal end = parent node closest to segment 6
    if len(prox_pts):
        d = np.linalg.norm(coords[comp][:, None, :] - prox_pts[None, :, :], axis=2).min(1)
        src = comp[int(np.argmin(d))]
    else:
        src = comp[0]

    try:
        dist = nx.single_source_dijkstra_path_length(sub.subgraph(comp), src, weight="w")
    except Exception:
        dist = {}
    if len(dist) < 2:
        ref = coords[src]
        dist = {n: float(np.linalg.norm(coords[n] - ref)) for n in comp}

    mx = max(dist.values()) or 1.0
    return {n: v / mx for n, v in dist.items()}


def split_merged_nodes(pred, coords, edge_index, batch=None,
                       lone_threshold=LONE_THRESHOLD):
    """
    pred        : (N,) collapsed class ids, 5 == '9a+10'
    coords      : (N,2) node coords (normalised is fine, only ratios are used)
    edge_index  : (2,E)
    Returns (N,) expanded ids where 5 -> 9a and 6 -> 10.
    """
    pred = np.asarray(pred).reshape(-1).copy()
    coords = np.asarray(coords, float)
    ei = np.asarray(edge_index)
    out = pred.copy()
    if not (pred == C9A10).any():
        return out

    G = nx.Graph()
    G.add_nodes_from(range(len(pred)))
    for u, v in zip(ei[0], ei[1]):
        G.add_edge(int(u), int(v), w=float(np.linalg.norm(coords[u] - coords[v])))

    batch = np.zeros(len(pred), np.int64) if batch is None else np.asarray(batch).reshape(-1)
    for b in np.unique(batch):
        m = batch == b
        idx = np.nonzero(m)[0]
        arc = _parent_arc(idx[pred[idx] == C7], coords, G,
                          coords[idx[pred[idx] == C6]])
        merged = idx[pred[idx] == C9A10]
        if len(merged) == 0:
            continue
        if arc is None:                       # no usable segment 7 -> leave as 9a
            out[merged] = OUT_9A
            continue

        pnodes = np.array(list(arc.keys()))
        pvals = np.array([arc[n] for n in pnodes])

        comps = [np.array(sorted(c)) for c in
                 nx.connected_components(G.subgraph(merged.tolist()))]
        scored = []
        for c in comps:
            d = np.linalg.norm(coords[c][:, None, :] - coords[pnodes][None, :, :], axis=2)
            scored.append((float(pvals[int(np.unravel_index(np.argmin(d), d.shape)[1])]), c))
        scored.sort(key=lambda t: t[0])       # proximal first

        if len(scored) == 1:
            f, c = scored[0]
            out[c] = OUT_9A if f < lone_threshold else OUT_10
        else:
            out[scored[0][1]] = OUT_9A
            for _, c in scored[1:]:
                out[c] = OUT_10
    return out


class LADFULL(torch.nn.Module):
    def __init__(self, args, cord_idx=(0, 1)):
        super().__init__()
        #./outputs/2026-08-16_19-08-38 [full_gt_tests_branch[1]_side[left]]
        self.gnn_model = load_model(
            model_class=VesselGraphSAGE,
            path="./outputs/2026-08-17_00-37-14 [full_gt_tests_branch[1]_side[left]]",
            device="cuda",
            model_inputs={},
            model_name="model_1",
            task_type = "gnn"
        )
        self.cord_idx = list(cord_idx)
        self.lone_threshold = LONE_THRESHOLD

    @torch.no_grad()
    def forward(self, x, edge_index, batch=None):
        logits = self.gnn_model(x, edge_index, batch)     # (N, 6) collapsed
        pred = logits.argmax(1)

        coords = x[:, self.cord_idx].detach().cpu().numpy()
        expanded = split_merged_nodes(
            pred.detach().cpu().numpy(), coords,
            edge_index.detach().cpu().numpy(),
            None if batch is None else batch.detach().cpu().numpy(),
            self.lone_threshold,
        )
        return torch.as_tensor(expanded, device=pred.device, dtype=pred.dtype)
    