
from dataclasses import dataclass
import numpy as np
from scipy.ndimage import label, binary_dilation

S8 = np.ones((3, 3), int)            # 8-connectivity

# ---- label indices (EDIT THESE to match your tensor's class values) --------
BG = 0
C1, C2, C3, C4, C16 = 1, 2, 3, 4, 5   # 5 == merged-16 in the 6-class model
PROXIMAL = (C1, C2, C3)               # each is a single component
DAUGHTERS = (C4, C16)                 # must attach to C3, each single component


@dataclass
class RuleConfig:
    single_proximal: bool = True      # rule 1
    attach_daughters: bool = True     # rule 2
    single_daughters: bool = True     # rule 3
    superior_is_16: bool = True       # rule 4
    verbose: bool = False             # log every change (use on a few images)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _components(binary):
    lab, n = label(binary, structure=S8)
    return [lab == i for i in range(1, n + 1)]


def _touches(a, b):
    return bool((binary_dilation(a, S8) & b).any())


def _majority_neighbour(out, comp):
    """Label of the most common foreground pixel bordering comp (excluding comp)."""
    ring = binary_dilation(comp, S8) & ~comp & (out > 0)
    vals = out[ring]
    if vals.size == 0:
        return BG
    return int(np.bincount(vals).argmax())


def _centroid_row(binary):
    ys, _ = np.where(binary)
    return ys.mean() if ys.size else np.inf


# ---------------------------------------------------------------------------
# the rules
# ---------------------------------------------------------------------------
def apply_rules(pred, cfg=RuleConfig()):
    """pred: (H,W) int ndarray. Returns a cleaned copy. Pure, no side effects."""
    out = pred.copy()
    log = []

    # RULE 1 -- each proximal class is its single largest component.
    if cfg.single_proximal:
        for c in PROXIMAL:
            comps = _components(out == c)
            if len(comps) > 1:
                comps.sort(key=lambda m: m.sum(), reverse=True)
                for extra in comps[1:]:
                    out[extra] = BG                 # drop; rule 2 may re-adopt
                    log.append(f"R1: dropped extra component of class {c} ({extra.sum()} px)")

    # RULE 2 -- 4 and 16 must attach to 3. Any component that doesn't ->
    #           relabel to its majority foreground neighbour (usually the vessel
    #           it was carved out of). This is the image-12 fix.
    if cfg.attach_daughters:
        three = out == C3
        for c in DAUGHTERS:
            for comp in _components(out == c):
                if not _touches(comp, three):
                    new = _majority_neighbour(out, comp)
                    out[comp] = new
                    log.append(f"R2: class {c} component not touching 3 -> relabelled {new} ({comp.sum()} px)")

    # RULE 3 -- 4 and 16 each a single component (Betti-0 <= 1). image-47 fix.
    if cfg.single_daughters:
        for c in DAUGHTERS:
            comps = _components(out == c)
            if len(comps) > 1:
                comps.sort(key=lambda m: m.sum(), reverse=True)
                for extra in comps[1:]:
                    out[extra] = BG
                    log.append(f"R3: dropped duplicate class {c} ({extra.sum()} px)")

    # RULE 4 -- if both daughters exist, superior(smaller row)=16, inferior=4.
    #           Swaps them if the model got the vertical order backwards.
    if cfg.superior_is_16:
        m4, m16 = out == C4, out == C16
        if m4.any() and m16.any():
            if _centroid_row(m16) > _centroid_row(m4):   # 16 is BELOW 4 -> wrong
                out[m4], out[m16] = C16, C4
                log.append("R4: swapped 4<->16 (16 was inferior to 4)")

    if cfg.verbose and log:
        print("\n".join(log))
    return out


def apply_rules_batch(pred, cfg=RuleConfig()):
    """pred: (B,H,W) torch tensor OR ndarray. Returns same type/device."""
    is_torch = hasattr(pred, "cpu")
    arr = pred.cpu().numpy() if is_torch else np.asarray(pred)
    out = arr.copy()
    for b in range(out.shape[0]):
        out[b] = apply_rules(out[b], cfg)
    if is_torch:
        import torch
        return torch.from_numpy(out).to(pred.device)
    return out


# ---------------------------------------------------------------------------
# VALIDATION -- run this on your TRAIN GT before trusting any rule.
# A correct rule almost never changes a correct ground-truth mask. If a rule
# rewrites GT often, it is WRONG for your data (e.g. a real config where 16 has
# two branches) and you must disable it.
# ---------------------------------------------------------------------------
def validate_rules(gt_masks, cfg=RuleConfig()):
    """gt_masks: iterable of (H,W) int GT arrays in the SAME label space."""
    n, changed = 0, 0
    changed_px_total = 0
    per_rule = {k: 0 for k in ("R1", "R2", "R3", "R4")}
    for gt in gt_masks:
        gt = np.asarray(gt)
        out = apply_rules(gt, cfg)
        n += 1
        diff = int((out != gt).sum())
        if diff:
            changed += 1
            changed_px_total += diff
    print(f"images checked            : {n}")
    print(f"GT masks the rules ALTERED: {changed}  ({100*changed/max(1,n):.1f}%)")
    print(f"total pixels changed      : {changed_px_total}")
    print()
    if changed / max(1, n) > 0.05:
        print("WARNING: rules change >5% of GT masks. At least one rule contradicts")
        print("your ground truth. Toggle rules off one at a time to find which, and")
        print("disable it. Do NOT apply rules that rewrite correct anatomy.")
    else:
        print("OK: rules rarely touch GT -> safe to apply at eval.")


if __name__ == "__main__":
    # image-12 (spurious 16 island in the gold) + image-47 (duplicate 4)
    p = np.zeros((40, 40), np.int64)
    p[10:20, 5] = C3
    p[20:22, 5:35] = C4
    p[20:22, 18:24] = C16          # spurious island, not touching 3
    p[30:32, 25:35] = C4           # duplicate gold, disconnected
    out = apply_rules(p, RuleConfig(verbose=True))
    print("class16 px:", (p == C16).sum(), "->", (out == C16).sum())
    print("class4 comps:", len(_components(p == C4)), "->", len(_components(out == C4)))