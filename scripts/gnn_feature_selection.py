import numpy as np
import torch
from sklearn.metrics import f1_score
from tqdm.auto import tqdm

@torch.no_grad()
def _collect_preds(model, loader, device):
    model.eval()
    preds, labels = [], []
    for data, _, name in loader:                     # <-- unpack your 3-tuple
        logits = my_forward(model, data, device)     # forward_fn handles squeeze/.to
        preds.append(logits.argmax(1).cpu())
        labels.append(data.y.squeeze(0).cpu())
    return torch.cat(preds).numpy(), torch.cat(labels).numpy()


def _macro_f1(y_true, y_pred, include_bg, bg_label=0):
    classes = np.unique(y_true)
    if not include_bg:
        classes = classes[classes != bg_label]
    if len(classes) == 0:
        return 0.0
    return f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)

def my_forward(model, data, device):
    features = data.x.squeeze(0).to(device)
    edge_index = data.edge_index.squeeze(0).to(device)
    batch = data.batch.squeeze(0).to(device)
    out = model(features, edge_index, batch=batch)
    return out[0] if isinstance(out, (tuple, list)) else out 

@torch.no_grad()
def permutation_importance(
    model,
    loader,
    feature_names,
    device="cuda",
    include_bg=False,
    bg_label=0,
    n_repeats=5,
    seed=0,
    group=None,                              # e.g. [2,3,4] to shuffle geo,euc,man together
):

    g = torch.Generator().manual_seed(seed)

    base_pred, base_lab = _collect_preds(model, loader, device)
    baseline = _macro_f1(base_lab, base_pred, include_bg, bg_label)

    results = {"baseline_f1": baseline, "features": {}}

    def shuffled_f1(cols):
        preds, labels = [], []
        for data, _, name in loader:  
            data = data.clone()
            x = data.x.squeeze(0)
            perm = torch.randperm(x.shape[0], generator=g)
            for c in cols:
                x[:, c] = x[:, c][perm]
            data.x = x.unsqueeze(0) if data.x.ndim == 3 else x 
            logits = my_forward(model, data, device)
            preds.append(logits.argmax(1).cpu())
            labels.append(data.y.squeeze(0).cpu())
        return _macro_f1(torch.cat(labels).numpy(),
                         torch.cat(preds).numpy(), include_bg, bg_label)

    # per-feature
    for fi in range(len(feature_names)):
        drops = np.array([baseline - shuffled_f1([fi]) for _ in range(n_repeats)])
        results["features"][feature_names[fi]] = {
            "importance_mean": float(drops.mean()),
            "importance_std": float(drops.std()),
            "f1_when_shuffled": float(baseline - drops.mean()),
        }

    # optional grouped importance for correlated features
    if group is not None:
        drops = np.array([baseline - shuffled_f1(group) for _ in range(n_repeats)])
        gname = "GROUP[" + ",".join(feature_names[i] for i in group) + "]"
        results["features"][gname] = {
            "importance_mean": float(drops.mean()),
            "importance_std": float(drops.std()),
            "f1_when_shuffled": float(baseline - drops.mean()),
        }

    return results


def print_report(results):
    print(f"baseline macro-F1: {results['baseline_f1']:.4f}\n")
    print(f"{'feature':<24s} {'importance':>11s} {'std':>7s} {'f1_shuffled':>12s}")
    ranked = sorted(results["features"].items(),
                    key=lambda kv: kv[1]["importance_mean"], reverse=True)
    for name, r in ranked:
        print(f"{name:<24s} {r['importance_mean']:>+11.4f} {r['importance_std']:>7.4f} "
              f"{r['f1_when_shuffled']:>12.4f}")
    print("\nhigh importance = model relies on it. ~0 = unused OR redundant with "
          "another feature. negative = noise.")