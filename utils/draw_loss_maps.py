import torch
import os
from pathlib import Path
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from tqdm.auto import tqdm
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch.nn as nn
import cv2
from .helpers import soft_skeletonize

def _norm(x):
    # Optimize: Convert to float32 without copying if it already is, then do operations in-place
    x = np.asarray(x, dtype=np.float32)
    min_val = x.min()
    x -= min_val
    max_val = x.max()
    x /= (max_val + 1e-8)
    return x

def _save_kwargs_for(img_format, img_quality, dpi):
    """Build savefig kwargs for a given output format.

    WebP is the default: lossy but visually indistinguishable at q>=90, and
    5-9x smaller than PNG for these gradient-heavy loss maps. It also keeps the
    sharp blue vessel overlays crisp (JPEG would blur those edges). JPEG is
    offered for maximum compatibility, PNG for a lossless fallback.
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

class CLDiceLossSpatial(nn.Module):
    def __init__(self, eps=1e-5, k=40):
        super(CLDiceLossSpatial, self).__init__()
        self.k = k
        self.eps = eps
        self.sum_dims = (0, 1) 

    def forward(self, pred_binary_mask, binary_gt_mask, gt_skel):
        pred_skel = soft_skeletonize(pred_binary_mask, k=self.k)
        
        t_prec = (pred_skel * binary_gt_mask).sum(dim=self.sum_dims) / (pred_skel.sum(dim=self.sum_dims) + self.eps)
        t_rec = (gt_skel * pred_binary_mask).sum(dim=self.sum_dims) / (gt_skel.sum(dim=self.sum_dims) + self.eps)
        
        cldice = 2 * ((t_prec * t_rec) / (t_prec + t_rec + self.eps))
        cldice_loss = 1 - cldice
        
        return cldice_loss

class CustomCELossSpatial(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.pos_weight = args["pos_weight"]
        self.bg_weight = args["bg_weight"]
        self.small_vessel_weights = args["small_vessel_weights"]

    def forward(self, gt_masks, pred_masks, small_vmasks=None):
        # Optimize: Use full_like for cleaner allocation
        w = torch.full_like(
            gt_masks, 
            fill_value=self.bg_weight, 
            dtype=pred_masks.dtype
        )
        w[gt_masks != 0] = self.pos_weight

        if small_vmasks is not None:
            w[small_vmasks == 1] = self.small_vessel_weights

        ce = F.cross_entropy(pred_masks, gt_masks, reduction="none")
        
        weighted_ce = ce * w
        reduce_dims = tuple(range(weighted_ce.ndim - 2))
        
        loss_map = weighted_ce.sum(dim=reduce_dims) / w.sum(dim=reduce_dims).clamp_min(1e-6)
        
        return loss_map

class TverskyLossSpatial(nn.Module):
    def __init__(self, args, eps=1e-6):
        super().__init__()
        self.eps = eps

        self.sum_dims = (0,) 
        self.alpha = args["alpha"]
        self.beta = args["beta"]
        self.gamma = args["gamma"]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if args["weights"] is not None:
            print("tverskey w is set : ", args["weights"])
            self.register_buffer("t_alpha", torch.tensor(args["weights"], dtype=torch.float32, device=device))
        else:
            self.t_alpha = None

    def forward(self, pred_probs, gt):
        gt = gt.float()

        tp = (gt * pred_probs).sum(dim=self.sum_dims)
        fp = ((1 - gt) * pred_probs).sum(dim=self.sum_dims)
        fn = (gt * (1 - pred_probs)).sum(dim=self.sum_dims)

        ti = (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)
        loss_c = (1 - ti).clamp_min(0) ** self.gamma

        if self.t_alpha is None:
            return loss_c.mean(dim=0) 
        
        w = self.t_alpha.view(-1, 1, 1)
        spatial_loss_map = (w * loss_c).sum(dim=0) / (w.sum() + self.eps)
        
        return spatial_loss_map
    
def _draw_one(inps):
    (gt_rgb_img, pred_rgb_img, pred_probs,
     skel_mask, gt_mask, pred_classes, 
     loss_maps, save_path,
     img_format, img_quality, dpi) = inps
    
    num_plots = 4 + len(loss_maps)
    cols = 3
    rows = (num_plots + cols - 1) // cols
    
    # Optimize: Bypass pyplot state machine to prevent severe memory leaks in workers
    fig = Figure(figsize=(cols*10, rows*10))
    canvas = FigureCanvas(fig)
    axes = fig.subplots(rows, cols).flatten()

    gt_rgb_img[gt_mask != 0] = [0, 0, 255]
    pred_rgb_img[pred_classes != 0] = [0, 0, 255]

    axes[0].imshow(gt_rgb_img)
    axes[0].set_title("Real image")
    axes[0].axis("off")

    axes[1].imshow(pred_rgb_img)
    axes[1].set_title("Pred image")
    axes[1].axis("off")

    axes[2].imshow(skel_mask)
    axes[2].set_title("mask skel")
    axes[2].axis("off")

    axes[3].imshow(pred_probs, cmap="turbo")
    axes[3].set_title("pred probs")
    axes[3].axis("off")

    for i, (loss_name, loss_map) in enumerate(loss_maps.items(), start=4):
        axes[i].imshow(loss_map, cmap="turbo")
        axes[i].set_title(loss_name)
        axes[i].axis("off")

    fig.tight_layout()
    _, save_kwargs = _save_kwargs_for(img_format, img_quality, dpi)
    fig.savefig(save_path, **save_kwargs)
    fig.clf() # Fully clear the figure context
    return save_path

def draw_loss_plots(
        model,
        dataloaders,
        model_name,
        args,
        num_workers=None,
        max_inflight=None,
        out_folder="model_analysis",
        # --- output size controls ---------------------------------------
        # webp q90 @ dpi 120 -> ~5-9x smaller than PNG, no visible loss.
        # Lower dpi to 100 and/or quality to 85 for further shrink.
        img_format="webp",   # "webp" | "jpeg" | "png"
        img_quality=90,      # 1-100 (ignored for png); 90 is a safe high setting
        dpi=120):            # figure resolution; 120 keeps original size
    
    base_ds_path = Path(args["paths"]["base_path"])
    save_root = base_ds_path / out_folder / model_name / "loss_maps"
    os.makedirs(save_root, exist_ok=True)

    ext = img_format.lower().lstrip(".")
    if ext == "jpg":
        ext = "jpeg"

    if num_workers is None:
        num_workers = max(1, (os.cpu_count() or 2) - 1)
    if max_inflight is None:
        max_inflight = num_workers * 3

    eps = 1e-5

    cldice_loss_fn = CLDiceLossSpatial(k=args["cl_dice_conf"]["k"], eps=eps)
    ce_loss_fn = CustomCELossSpatial(args=args["ce_loss_conf"])
    dice_loss_fn = TverskyLossSpatial(eps=eps, args=args["tversky_conf"])

    softmax = nn.Softmax(dim=1)
    class_count = args["class_count"]
    
    # Optimize: Use a set to track futures for backpressure
    active_futures = set() 
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        with torch.no_grad():
            for part, dl in dataloaders.items():
                part_path = save_root / part
                os.makedirs(part_path, exist_ok=True)
                
                for batch_idx, batch in enumerate(tqdm(dl, desc=part)):
                    new_image_not_normal = batch["new_image_not_normal"]
                    img = batch["new_image"].to("cuda", non_blocking=True)
                    gt_masks = batch["mask"].to("cuda", non_blocking=True)
                    small_vmasks = batch["small_vmask"].to("cuda", non_blocking=True)
                    skel_masks = batch["skel_mask"].to("cuda", non_blocking=True)
                    name = batch["name"][0]
                    save_path = part_path / f"{name}.{ext}"

                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        pred_masks,_ = model(img)
                        pred_masks = pred_masks[0]
                        pred_classes = torch.argmax(pred_masks, dim=1)

                    pred_classes_np = pred_classes.cpu().numpy()
                    gt_masks_np = gt_masks.cpu().numpy()
                    
                    is_background = (gt_masks_np[0] == 0) & (pred_classes_np[0] == 0)
                    
                    onehot_gt_masks = F.one_hot(gt_masks, num_classes=class_count)
                    onehot_gt_masks = onehot_gt_masks.permute(0, 3, 1, 2).float()

                    pred_probs = softmax(pred_masks)

                    mask_ce_loss_map = ce_loss_fn(
                        pred_masks=pred_masks,
                        gt_masks=gt_masks,
                        small_vmasks=small_vmasks
                    ).cpu().numpy()
                    
                    fg_pred_probs = pred_probs[:,1:]
                    fg_onehot_gt_masks = onehot_gt_masks[:,1:]
                    dice_loss_map = dice_loss_fn(
                        pred_probs=fg_pred_probs,
                        gt=fg_onehot_gt_masks
                    ).cpu().numpy()

                    cl_dice_loss_map = cldice_loss_fn(
                        pred_binary_mask=pred_probs[:,1:],
                        binary_gt_mask=gt_masks.unsqueeze(1).float(),
                        gt_skel=skel_masks.unsqueeze(1).float()
                    ).cpu().numpy()

                    mask_ce_loss_map = _norm(mask_ce_loss_map)
                    dice_loss_map = _norm(dice_loss_map)
                    cl_dice_loss_map = _norm(cl_dice_loss_map)
                   
                    cl_dice_loss_map[is_background] = 0.0

                    loss_maps = {
                        "cldice loss": cl_dice_loss_map,
                        "dice_loss": dice_loss_map,
                        "ce loss": mask_ce_loss_map
                    }
                    
                    raw_img = new_image_not_normal.cpu().numpy()[0,0]
                    gt_rgb_img = cv2.cvtColor(raw_img, cv2.COLOR_GRAY2RGB)
                    pred_rgb_img = gt_rgb_img.copy() # Optimize: Avoid redundant OpenCV conversion
                    
                    pred_probs_np = pred_probs[0,1].cpu().numpy()

                    payload = (
                        gt_rgb_img,
                        pred_rgb_img,
                        pred_probs_np,
                        skel_masks.cpu().numpy()[0],
                        gt_masks_np[0],
                        pred_classes_np[0],
                        loss_maps,
                        save_path,
                        img_format,
                        img_quality,
                        dpi,
                    )
                    
                    future = executor.submit(_draw_one, payload)
                    active_futures.add(future)

                    # --- BACKPRESSURE IMPLEMENTATION ---
                    # Blocks the main thread from queuing more items if we hit max_inflight.
                    if len(active_futures) >= max_inflight:
                        done, active_futures = concurrent.futures.wait(
                            active_futures, return_when=concurrent.futures.FIRST_COMPLETED
                        )
                        # Optional: Raise exceptions if any occurred in the worker
                        for f in done:
                            f.result() 

                    # Optimize: Explicitly delete large tensors immediately to free VRAM/RAM
                    del batch, img, gt_masks, small_vmasks, skel_masks
                    del pred_masks, pred_classes, pred_probs, onehot_gt_masks
                    del payload, loss_maps
                    
            # Drain any remaining futures at the end of the run
            for f in concurrent.futures.as_completed(active_futures):
                f.result()