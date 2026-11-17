
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import json
import warnings

import numpy as np
import torch

LabelOrRegion = Union[int, Tuple[int, ...]]

__all__ = [
    "evaluate_model",
    "compute_metrics_from_arrays",
    "compute_tp_fp_fn_tn",
    "region_or_label_to_mask",
    "aggregate_results",
    "save_summary_json",
    "print_summary",
    "default_predict_fn",
    "regions_logits_to_segmentation",
]


# --------------------------------------------------------------------------------------
# 1. Primitives (1:1 with nnunetv2/evaluation/evaluate_predictions.py)
# --------------------------------------------------------------------------------------

def region_or_label_to_mask(segmentation: np.ndarray, region_or_label: LabelOrRegion) -> np.ndarray:
    """int -> (seg == label). tuple -> union of the labels in the tuple."""
    if np.isscalar(region_or_label):
        return segmentation == region_or_label
    mask = np.zeros_like(segmentation, dtype=bool)
    for r in region_or_label:
        mask[segmentation == r] = True
    return mask


def compute_tp_fp_fn_tn(mask_ref: np.ndarray, mask_pred: np.ndarray,
                        ignore_mask: Optional[np.ndarray] = None):
    if ignore_mask is None:
        use_mask = np.ones_like(mask_ref, dtype=bool)
    else:
        use_mask = ~ignore_mask
    tp = np.sum((mask_ref & mask_pred) & use_mask)
    fp = np.sum(((~mask_ref) & mask_pred) & use_mask)
    fn = np.sum((mask_ref & (~mask_pred)) & use_mask)
    tn = np.sum(((~mask_ref) & (~mask_pred)) & use_mask)
    return tp, fp, fn, tn


def compute_metrics_from_arrays(seg_ref: np.ndarray,
                                seg_pred: np.ndarray,
                                labels_or_regions: Sequence[LabelOrRegion],
                                ignore_label: Optional[int] = None) -> Dict[str, Any]:
    """
    nnU-Net's compute_metrics(), but on in-memory label maps instead of files.
    seg_ref and seg_pred must be integer LABEL MAPS of identical shape.
    """
    assert seg_ref.shape == seg_pred.shape, \
        f"shape mismatch: ref {seg_ref.shape} vs pred {seg_pred.shape}"

    ignore_mask = (seg_ref == ignore_label) if ignore_label is not None else None

    metrics: Dict[LabelOrRegion, Dict[str, Any]] = {}
    for r in labels_or_regions:
        mask_ref = region_or_label_to_mask(seg_ref, r)
        mask_pred = region_or_label_to_mask(seg_pred, r)
        tp, fp, fn, tn = compute_tp_fp_fn_tn(mask_ref, mask_pred, ignore_mask)

        m: Dict[str, Any] = {}
        if tp + fp + fn == 0:
            m["Dice"] = np.nan
            m["IoU"] = np.nan
        else:
            m["Dice"] = 2 * tp / (2 * tp + fp + fn)
            m["IoU"] = tp / (tp + fp + fn)
        m["FP"] = fp
        m["TP"] = tp
        m["FN"] = fn
        m["TN"] = tn
        m["n_pred"] = fp + tp
        m["n_ref"] = fn + tp
        metrics[r] = m
    return {"metrics": metrics}


def aggregate_results(results: List[dict],
                      labels_or_regions: Sequence[LabelOrRegion]) -> Tuple[dict, dict]:
    """nnU-Net's aggregation block from compute_metrics_on_folder()."""
    metric_list = list(results[0]["metrics"][labels_or_regions[0]].keys())

    # mean metric per class  (nanmean over cases)
    means: Dict[LabelOrRegion, Dict[str, float]] = {}
    for r in labels_or_regions:
        means[r] = {}
        for m in metric_list:
            vals = [i["metrics"][r][m] for i in results]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN slice
                means[r][m] = np.nanmean(vals)

    # foreground mean  (plain mean over per-class means, background key skipped)
    foreground_mean: Dict[str, float] = {}
    for m in metric_list:
        values = [means[k][m] for k in means.keys() if not (k == 0 or k == "0")]
        foreground_mean[m] = np.mean(values)

    return means, foreground_mean


# --------------------------------------------------------------------------------------
# 2. Logits -> label map
# --------------------------------------------------------------------------------------

def default_predict_fn(logits: torch.Tensor) -> torch.Tensor:
    """
    (B, C, H, W) logits -> (B, H, W) integer label map.
      C == 1 -> binary, sigmoid > 0.5
      C >  1 -> softmax/argmax over the channel dim (channel index == label id)
    Override via the predict_fn argument if your head differs.
    """
    if logits.shape[1] == 1:
        return (torch.sigmoid(logits[:, 0]) > 0.5).long()
    return logits.argmax(dim=1).long()


def regions_logits_to_segmentation(regions: Sequence[LabelOrRegion]) -> Callable:
    """
    predict_fn factory for REGION-BASED heads (independent sigmoid per region).
    Mirrors nnU-Net's LabelManager.convert_probabilities_to_segmentation: regions are
    painted in order, each writing its LAST label id, so later regions overwrite earlier
    ones. Channel i of the head must correspond to regions[i].

        predict_fn = regions_logits_to_segmentation([(1, 2, 3), (2, 3), (3,)])
    """
    def _fn(logits: torch.Tensor) -> torch.Tensor:
        assert logits.shape[1] == len(regions), \
            f"head has {logits.shape[1]} channels but {len(regions)} regions were given"
        probs = torch.sigmoid(logits)
        seg = torch.zeros(probs.shape[0], *probs.shape[2:], dtype=torch.long, device=probs.device)
        for i, c in enumerate(regions):
            label = c if np.isscalar(c) else c[-1]
            seg[probs[:, i] > 0.5] = int(label)
        return seg
    return _fn


# --------------------------------------------------------------------------------------
# 3. Main entry point
# --------------------------------------------------------------------------------------

@torch.no_grad()
def evaluate_model(model: torch.nn.Module,
                   dataloader,
                   labels_or_regions: Sequence[LabelOrRegion],
                   device: Union[str, torch.device] = "cuda",
                   ignore_label: Optional[int] = None,
                   predict_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                   image_key: str = "imgs",
                   mask_key: str = "mask",
                   case_id_key: Optional[str] = None,
                   output_file: Optional[str] = None,
                   autocast: bool = False,
                   verbose: bool = True) -> Dict[str, Any]:
    """
    Run nnU-Net-v2-style evaluation of `model` over `dataloader`.

    Args
    ----
    model             : torch module. Called as model(img) with img of shape (B, 1, H, W).
                        A list/tuple return (deep supervision) is handled: element [0] is used.
    dataloader        : yields dicts; data[image_key] -> (B, 1, H, W), data[mask_key] -> (B, H, W).
                        B is expected to be 1 but any B is handled (each item scored separately).
    labels_or_regions : FOREGROUND labels only, e.g. [1, 2, 3]. Do NOT include 0 — nnU-Net
                        never evaluates background. Region-based: [(1, 2, 3), (2, 3), (3,)].
    device            : 'cuda', 'cpu', torch.device(...).
    ignore_label      : voxels where GT == ignore_label are excluded from all counts.
    predict_fn        : logits (B, C, H, W) -> label map (B, H, W). Defaults to
                        default_predict_fn. Use regions_logits_to_segmentation(...) for regions.
    image_key/mask_key: dict keys in the batch.
    case_id_key       : optional dict key holding a case name/id. Falls back to 'case_0000'.
    output_file       : optional path; writes a summary.json in nnU-Net's format.
    autocast          : run the forward pass under torch.autocast (mixed precision).
    verbose           : print the summary table at the end.

    Returns
    -------
    dict with keys: metric_per_case, mean, foreground_mean, global_dice
    """
    assert len(labels_or_regions) > 0, "labels_or_regions is empty"
    if any((k == 0 or k == (0,)) for k in labels_or_regions):
        warnings.warn("label 0 (background) was passed in. nnU-Net evaluates foreground only; "
                      "it will be scored in 'mean' but skipped in 'foreground_mean'.")
    if predict_fn is None:
        predict_fn = default_predict_fn

    device = torch.device(device)
    was_training = model.training
    model.eval()
    model.to(device)

    results: List[dict] = []
    # accumulators for the aggregated ("global"/pseudo) Dice
    g_tp = {r: 0 for r in labels_or_regions}
    g_fp = {r: 0 for r in labels_or_regions}
    g_fn = {r: 0 for r in labels_or_regions}

    counter = 0
    for batch_idx, data in enumerate(dataloader):
        img = data[image_key]
        gt = data[mask_key]

        assert img.ndim == 4, f"expected img (B, 1, H, W), got {tuple(img.shape)}"
        if gt.ndim == 4 and gt.shape[1] == 1:
            gt = gt[:, 0]                      # (B, 1, H, W) -> (B, H, W), just in case
        assert gt.ndim == 3, f"expected mask (B, H, W), got {tuple(gt.shape)}"

        img = img.to(device, non_blocking=True).float()

        if autocast and device.type == "cuda":
            with torch.autocast("cuda", enabled=True):
                logits = model(img)[0]
        else:
            logits = model(img)[0]

        if isinstance(logits, (list, tuple)):   # deep supervision -> highest resolution
            logits = logits[0]
        assert logits.ndim == 4, f"expected logits (B, C, H, W), got {tuple(logits.shape)}"

        seg_pred = predict_fn(logits.float()).cpu().numpy().astype(np.int64)
        seg_ref = gt.cpu().numpy().astype(np.int64)

        for b in range(seg_ref.shape[0]):
            if case_id_key is not None and case_id_key in data:
                cid = data[case_id_key][b]
                cid = cid.item() if torch.is_tensor(cid) else cid
                case_id = str(cid)
            else:
                case_id = f"case_{counter:04d}"

            res = compute_metrics_from_arrays(seg_ref[b], seg_pred[b],
                                              labels_or_regions, ignore_label)
            res["case_id"] = case_id
            results.append(res)

            for r in labels_or_regions:
                g_tp[r] += int(res["metrics"][r]["TP"])
                g_fp[r] += int(res["metrics"][r]["FP"])
                g_fn[r] += int(res["metrics"][r]["FN"])
            counter += 1

    if was_training:
        model.train()

    if len(results) == 0:
        raise RuntimeError("dataloader yielded no batches")

    means, foreground_mean = aggregate_results(results, labels_or_regions)

    # aggregated / "global" Dice: pool counts over the whole set, then one Dice per class.
    # This is what nnU-Net prints as "Pseudo dice" during training. It is NOT summary.json.
    global_dice: Dict[Any, float] = {}
    for r in labels_or_regions:
        denom = 2 * g_tp[r] + g_fp[r] + g_fn[r]
        global_dice[r] = (2 * g_tp[r] / denom) if denom > 0 else float("nan")
    fg_keys = [k for k in global_dice if not (k == 0 or k == "0")]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        global_dice["foreground_mean"] = float(np.nanmean([global_dice[k] for k in fg_keys]))

    result = {
        "metric_per_case": results,
        "mean": means,
        "foreground_mean": foreground_mean,
        "global_dice": global_dice,
    }

    if output_file is not None:
        save_summary_json(result, output_file)
    if verbose:
        print_summary(result, labels_or_regions)
    return result


# --------------------------------------------------------------------------------------
# 4. Output helpers
# --------------------------------------------------------------------------------------

def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(i) for i in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if np.isnan(v) else v          # json has no NaN; nnU-Net writes NaN, we write null
    if isinstance(obj, np.ndarray):
        return _to_jsonable(obj.tolist())
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def save_summary_json(result: Dict[str, Any], output_file: str) -> None:
    """Write summary.json with stringified keys, like nnU-Net does."""
    assert output_file.endswith(".json"), "output_file should end with .json"
    with open(output_file, "w") as f:
        json.dump(_to_jsonable(result), f, sort_keys=True, indent=4)


def print_summary(result: Dict[str, Any],
                  labels_or_regions: Sequence[LabelOrRegion],
                  label_names: Optional[Dict[Any, str]] = None) -> None:
    n = len(result["metric_per_case"])
    print(f"\nnnU-Net-style evaluation over {n} case(s)")
    print(f"{'class':<16}{'Dice':>10}{'IoU':>10}{'n_valid':>10}{'globalDice':>12}")
    print("-" * 58)
    for r in labels_or_regions:
        name = (label_names or {}).get(r, str(r))
        dices = [c["metrics"][r]["Dice"] for c in result["metric_per_case"]]
        n_valid = int(np.sum(~np.isnan(dices)))
        print(f"{name:<16}{result['mean'][r]['Dice']:>10.4f}{result['mean'][r]['IoU']:>10.4f}"
              f"{n_valid:>10}{result['global_dice'][r]:>12.4f}")
    print("-" * 58)
    print(f"{'foreground_mean':<16}{result['foreground_mean']['Dice']:>10.4f}"
          f"{result['foreground_mean']['IoU']:>10.4f}{'':>10}"
          f"{result['global_dice']['foreground_mean']:>12.4f}")
    print("\nnote: 'Dice' column = mean of per-case Dice (nnU-Net summary.json).")
    print("      'n_valid' = cases where the class was present in GT or prediction (NaN cases excluded).")
    print("      'globalDice' = Dice of pooled TP/FP/FN (nnU-Net's training-time 'pseudo dice').\n")