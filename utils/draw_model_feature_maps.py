import torch
import os
from pathlib import Path
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import gc
from concurrent.futures import ProcessPoolExecutor
import numpy as np

def _norm(x):
    x = x.astype(np.float32)
    x = x - x.min()
    x = x / (x.max() + 1e-8)
    return x

def _resize_bilinear(arr, out_hw):
    t = torch.from_numpy(arr)[None, None].float()
    t = F.interpolate(t, size=out_hw, mode="bilinear", align_corners=False)
    return t[0, 0].numpy()

def _save_kwargs_for(img_format, img_quality, dpi):
    """Build savefig kwargs for a given output format.

    WebP is the default: lossy but visually indistinguishable at q>=90, and
    5-9x smaller than PNG for gradient-heavy feature maps. JPEG is offered for
    maximum compatibility, PNG for a lossless fallback.
    """
    fmt = img_format.lower().lstrip(".")
    kwargs = dict(dpi=dpi, bbox_inches="tight")
    if fmt == "webp":
        # method=6 is the slowest/smallest; drop to 4 for faster encoding.
        kwargs["pil_kwargs"] = {"quality": img_quality, "method": 6}
    elif fmt in ("jpg", "jpeg"):
        kwargs["pil_kwargs"] = {"quality": img_quality, "optimize": True,
                                "progressive": True}
        # JPEG has no alpha; flatten onto white so nothing goes transparent.
        kwargs["facecolor"] = "white"
    elif fmt == "png":
        pass  # lossless, no quality setting
    else:
        raise ValueError(f"Unsupported img_format: {img_format!r}")
    return fmt, kwargs

def _render_one(payload):
    (img_name, real_img, mask_img, pred_mask_img,
     heatmaps, layer_names, fg_prob, save_path,
     img_format, img_quality, dpi) = payload

    num_plots = 4 + len(layer_names)
    cols = 5
    rows = (num_plots + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(25, rows * 5))
    axes = axes.flatten()

    axes[0].imshow(real_img, cmap="gray")
    axes[0].set_title("Real image")
    axes[0].axis("off")
    
    axes[1].imshow(mask_img, cmap="gray")
    axes[1].set_title("GT mask")
    axes[1].axis("off")
    
    axes[2].imshow(pred_mask_img, cmap="gray")
    axes[2].set_title("Pred mask")
    axes[2].axis("off")

    for ax_idx, (layer_name, hm) in enumerate(zip(layer_names, heatmaps), start=3):
        axes[ax_idx].imshow(hm, cmap="turbo")
        axes[ax_idx].set_title(layer_name)
        axes[ax_idx].axis("off")

    ax_idx = len(layer_names) + 3
    axes[ax_idx].imshow(fg_prob, cmap="turbo")
    axes[ax_idx].set_title("fg probs")
    axes[ax_idx].axis("off")

    for j in range(ax_idx + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    _, save_kwargs = _save_kwargs_for(img_format, img_quality, dpi)
    fig.savefig(save_path, **save_kwargs)
    plt.close(fig)
    return save_path

def _prep_real(x, b):
    x = x.detach().cpu()
    if x.ndim == 4: x = x[b]
    if x.ndim == 3:
        x = x[0] if x.shape[0] == 1 else x[0]
    return _norm(x.float().numpy())

def _prep_mask(mask, b):
    mask = mask.detach().cpu()
    if mask.ndim == 4: mask = mask[b, 0]
    elif mask.ndim == 3: mask = mask[b]
    return mask.numpy()

def _prep_pred(pred_logits, b, threshold=0.5):
    p = pred_logits.detach().cpu()
    if p.ndim == 4:
        if p.shape[1] == 1:
            prob = torch.sigmoid(p[b, 0])
            m = (prob > threshold).float()
        else:
            m = torch.argmax(p[b], dim=0).float()
    elif p.ndim == 3:
        m = p[b].float()
    else:
        raise ValueError(f"Unexpected pred_logits shape: {p.shape}")
    return m.numpy()

def _prep_heatmap(act, b, out_hw):
    a = act[b].numpy()
    a = _resize_bilinear(a, out_hw)
    return _norm(a)

def make_light_hook(layer_name, activation_maps, ACT_MODE):
    def hook(module, inp, out):
        if isinstance(out, (list, tuple)):
            out = out[0]
        with torch.no_grad():
            if ACT_MODE == "mean":
                act = out.detach().mean(dim=1)
            elif ACT_MODE == "max":
                act = out.detach().max(dim=1).values
            else:
                raise ValueError("ACT_MODE must be 'mean' or 'max'")
            activation_maps[layer_name] = act.float().cpu()
    return hook

def draw_feature_maps(
        layers_to_print,
        layer_names,
        data_loaders,
        args,
        model_name,
        best_model,
        ACT_MODE="mean",
        device="cuda",
        num_workers=None,
        max_inflight=None,
        out_folder="model_analysis",
        # --- output size controls ---------------------------------------
        # webp q90 @ dpi 120 -> ~0.9 MB/image (was ~6 MB PNG), no visible loss.
        # Lower dpi to 100 and/or quality to 85 for further shrink.
        img_format="webp",   # "webp" | "jpeg" | "png"
        img_quality=90,      # 1-100 (ignored for png); 90 is a safe high setting
        dpi=120,             # figure resolution; 120 keeps original size
        ):
    hooks = []
    activation_maps = {}
    for layer_name, layer in layers_to_print.items():
        hooks.append(layer.register_forward_hook(
            make_light_hook(layer_name, activation_maps, ACT_MODE)))

    ext = img_format.lower().lstrip(".")
    if ext == "jpg":
        ext = "jpeg"

    base_ds_path = Path(args["paths"]["base_path"])
    save_root = base_ds_path / out_folder / model_name / "feature_maps"
    os.makedirs(save_root, exist_ok=True)

    if num_workers is None:
        num_workers = max(1, (os.cpu_count() or 2) - 1)
    if max_inflight is None:
        max_inflight = num_workers * 3

    futures = []

    def _drain(executor, force=False):
        limit = 0 if force else max_inflight
        while len(futures) > limit:
            done = futures.pop(0)
            done.result()

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        with torch.no_grad():
            for part, dl in data_loaders.items():
                part_path = save_root / part
                os.makedirs(part_path, exist_ok=True)

                for batch_idx, batch in enumerate(tqdm(dl, desc=part)):
                    new_image_not_normal = batch["new_image_not_normal"]
                    img = batch["new_image"]
                    mask = batch["mask"]
                    names = batch["name"]

                    activation_maps.clear()
                    img = img.to(device, non_blocking=True)

                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        pred_logits,_ = best_model(img)

                    if isinstance(pred_logits, (list, tuple)):
                        fg_src = pred_logits[0]
                        pred_logits = pred_logits[0]
                    else:
                        fg_src = pred_logits

                    fg_prob_batch = torch.softmax(fg_src, dim=1)[:, 1].detach().cpu().numpy()

                    B = img.shape[0]
                    H, W = img.shape[-2:]

                    acts_cpu = {ln: activation_maps[ln] for ln in layer_names}

                    for b in range(B):
                        img_name = names[b]
                        if isinstance(img_name, (tuple, list)):
                            img_name = img_name[0]
                        img_name = Path(str(img_name)).stem

                        heatmaps = [
                            _prep_heatmap(acts_cpu[ln], b, (H, W))
                            for ln in layer_names
                        ]

                        payload = (
                            img_name,
                            _prep_real(new_image_not_normal, b),
                            _prep_mask(mask, b),
                            _prep_pred(pred_logits, b, 0.5),
                            heatmaps,
                            list(layer_names),
                            fg_prob_batch[b],
                            str(part_path / f"{img_name}.{ext}"),
                            img_format,
                            img_quality,
                            dpi,
                        )
                        futures.append(executor.submit(_render_one, payload))

                    _drain(executor)

                    del img, pred_logits, fg_src
                    activation_maps.clear()
                    if batch_idx % 10 == 0:
                        gc.collect()
                        torch.cuda.empty_cache()

            _drain(executor, force=True)

    for h in hooks:
        h.remove()
    gc.collect()
    torch.cuda.empty_cache()