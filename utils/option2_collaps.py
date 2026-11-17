import numpy as np
import torch

BASE_CLASS_MAP = {1:'5', 2:'12', 3:'6', 4:'9', 5:'7', 6:'9a+10', 7:'8+10a',
                  8:'11', 9:'12a+12b', 10:'13', 11:'14+14a+14b+15'}
N_BASE = len(BASE_CLASS_MAP) + 1                      # 12

STAGE1_MAP = {1:'5', 2:'6', 3:'7', 4:'8', 5:'11', 6:'13'}          # mains
STAGE2_MAP = {1:'12', 2:'9', 3:'9a+10', 4:'12a+12b',
              5:'14+14a+14b+15'}                                    # children, no 10a


def build_lut(stage_map, base_map=BASE_CLASS_MAP):
    """stage class idx -> baseline idx, matched by segment NAME (order-independent)."""
    name_to_base = {}
    for b, nm in base_map.items():
        for part in nm.split('+'):
            name_to_base[part] = b
    lut = np.zeros(max(stage_map) + 1, np.int64)
    for si, nm in stage_map.items():
        tgt = {name_to_base[p] for p in nm.split('+') if p in name_to_base}
        if len(tgt) != 1:
            raise ValueError(f"stage class {si} ({nm}) -> {tgt}; expected exactly one")
        lut[si] = tgt.pop()
    return lut


LUT1 = build_lut(STAGE1_MAP)
LUT2 = build_lut(STAGE2_MAP)


def combine(logits1, logits2, mode="confidence", return_probs=True,
            labels_from_probs=False, lut1=None, lut2=None):
    """
    logits1 : (B, 1+len(STAGE1_MAP), H, W)   stage-1 mains
    logits2 : (B, 1+len(STAGE2_MAP), H, W)   stage-2 children
    returns (labels (B,H,W) int64, probs (B,N_BASE,H,W) float)
    """
    l1 = LUT1 if lut1 is None else lut1
    l2 = LUT2 if lut2 is None else lut2

    if logits1.shape[1] != len(l1):
        raise ValueError(f"logits1 has {logits1.shape[1]} channels but lut1 covers "
                         f"{len(l1)} (bg + {len(l1)-1} classes)")
    if logits2.shape[1] != len(l2):
        raise ValueError(f"logits2 has {logits2.shape[1]} channels but lut2 covers "
                         f"{len(l2)} (bg + {len(l2)-1} classes)")

    p1, p2 = logits1.softmax(1), logits2.softmax(1)
    c1, c2 = p1.argmax(1), p2.argmax(1)
    m1, m2 = c1 > 0, c2 > 0
    dev = p1.device

    b1 = torch.as_tensor(l1, device=dev)[c1]
    b2 = torch.as_tensor(l2, device=dev)[c2]

    out = torch.zeros_like(b1)
    out[m2] = b2[m2]
    out[m1] = b1[m1]
    if mode == "confidence":
        both = m1 & m2
        if both.any():
            take2 = both & (p2.max(1).values > p1.max(1).values)
            out[take2] = b2[take2]
    if not return_probs:
        return out, None

    B, _, H, W = p1.shape
    probs = torch.zeros(B, N_BASE, H, W, device=dev, dtype=p1.dtype)

    w2 = torch.ones(B, 1, H, W, device=dev, dtype=p1.dtype)
    if mode == "stage1":
        w2 = w2 * (~m1).unsqueeze(1).to(p1.dtype)

    # maximum() is safe whether or not the two LUTs share a baseline channel
    probs[:, l1[1:]] = torch.maximum(probs[:, l1[1:]], p1[:, 1:])
    probs[:, l2[1:]] = torch.maximum(probs[:, l2[1:]], p2[:, 1:] * w2)

    probs[:, 0] = torch.sqrt(p1[:, 0] * p2[:, 0])
    probs = probs / probs.sum(1, keepdim=True).clamp(min=1e-8)
    if labels_from_probs:
        out = probs.argmax(1)
    return out, probs