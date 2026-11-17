import numpy as np
import torch
import networkx as nx
###IE###
from .gnn_model import VesselGraphSAGE
from utils.helpers import load_model
###SS###
PASSTHROUGH = {0: 0, 1: 1, 2: 2, 4: 4}          # '13' stays at 4

SPLITS = [
    # 12a+12b -> 12a(3), 12b(8)   proximal to distal along segment 11
    dict(merged=3, parent=1, outputs=[3, 8],           thresholds=[0.50]),
    # 14 group -> 14(5), 14a(6), 14b(9), 15(7)   proximal to distal along segment 13
    dict(merged=5, parent=4, outputs=[5, 6, 9, 7],     thresholds=[0.25, 0.50, 0.75]),
]


def _parent_arc(pnodes, coords, geo, G):
    """
    Arc position 0..1 for every parent node, oriented so 0 = proximal.

    Orientation comes from geo_root: the parent end with the smaller geodesic distance
    to the graph root is the proximal one. That works for every segment without needing
    a separate 'what is upstream of this' rule.
    """
    if len(pnodes) < 2:
        return None
    sub = G.subgraph(pnodes.tolist())
    comp = list(max(nx.connected_components(sub), key=len)) if sub.number_of_nodes() else []
    if len(comp) < 2:
        return None

    g = geo[comp]
    valid = g >= 0
    src = comp[int(np.argmin(np.where(valid, g, np.inf)))] if valid.any() else comp[0]

    try:
        dist = nx.single_source_dijkstra_path_length(sub.subgraph(comp), src, weight="w")
    except Exception:
        dist = {}
    if len(dist) < 2:
        ref = coords[src]
        dist = {n: float(np.linalg.norm(coords[n] - ref)) for n in comp}
    mx = max(dist.values()) or 1.0
    return {n: v / mx for n, v in dist.items()}


def split_multi(pred, coords, geo, edge_index, batch=None,
                splits=SPLITS, passthrough=PASSTHROUGH):
    """
    pred  : (N,) collapsed ids
    coords: (N,2)     geo: (N,) geo_root feature (-1 where unreachable)
    Returns (N,) expanded ids.
    """
    pred = np.asarray(pred).reshape(-1)
    coords = np.asarray(coords, float)
    geo = np.asarray(geo, float).reshape(-1)
    ei = np.asarray(edge_index)

    out = np.zeros_like(pred)
    for c_in, c_out in passthrough.items():
        out[pred == c_in] = c_out

    G = nx.Graph(); G.add_nodes_from(range(len(pred)))
    for u, v in zip(ei[0], ei[1]):
        G.add_edge(int(u), int(v), w=float(np.linalg.norm(coords[u] - coords[v])))

    batch = np.zeros(len(pred), np.int64) if batch is None else np.asarray(batch).reshape(-1)
    for b in np.unique(batch):
        idx = np.nonzero(batch == b)[0]
        for sp in splits:
            merged = idx[pred[idx] == sp["merged"]]
            if len(merged) == 0:
                continue
            outs = sp["outputs"]
            arc = _parent_arc(idx[pred[idx] == sp["parent"]], coords, geo, G)
            if arc is None:                       # no usable parent -> most proximal label
                out[merged] = outs[0]
                continue

            pn = np.array(list(arc.keys()))
            pv = np.array([arc[n] for n in pn])
            comps = [np.array(sorted(c)) for c in
                     nx.connected_components(G.subgraph(merged.tolist()))]
            scored = []
            for c in comps:
                d = np.linalg.norm(coords[c][:, None, :] - coords[pn][None, :, :], axis=2)
                j = int(np.unravel_index(np.argmin(d), d.shape)[1])
                scored.append((float(pv[j]), c))
            scored.sort(key=lambda t: t[0])

            if len(scored) == len(outs):
                for o, (_, c) in zip(outs, scored):   # exact count -> pure ordering
                    out[c] = o
            else:                                     # otherwise bin by attach fraction
                th = sp["thresholds"]
                for f, c in scored:
                    out[c] = outs[int(np.searchsorted(th, f))]
    return out


class LCXFULL(torch.nn.Module):
    def __init__(self, args,cord_idx=(0, 1), geo_idx=2, path=None):
        super().__init__()
        self.gnn_model = load_model(
            model_class=VesselGraphSAGE, 
            path="./outputs/2026-08-18_11-05-38 [full_gt_tests_branch[2]_side[left]]",
            device="cuda", model_inputs={},
            model_name="model_2",
            task_type = "gnn"
        )
        self.cord_idx = list(cord_idx)
        self.geo_idx = geo_idx

    @torch.no_grad()
    def forward(self, x, edge_index, batch=None):
        logits = self.gnn_model(x, edge_index, batch)          # (N,6) collapsed
        pred = logits.argmax(1)
        expanded = split_multi(
            pred.detach().cpu().numpy(),
            x[:, self.cord_idx].detach().cpu().numpy(),
            x[:, self.geo_idx].detach().cpu().numpy(),
            edge_index.detach().cpu().numpy(),
            None if batch is None else batch.detach().cpu().numpy(),
        )
        return torch.as_tensor(expanded, device=pred.device, dtype=pred.dtype)