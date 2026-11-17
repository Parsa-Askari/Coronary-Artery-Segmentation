import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import albumentations as A
import cv2
from tqdm import tqdm
from models.swin_upernet_cascade import SwinUperNetCascade
from models.smp_models import DeepLabV3Plus
from utils.helpers import load_model,read_images,make_result_dataloaders
from utils.gnn_dataset_generation_args import args
from utils.right_rules import apply_rules ,RuleConfig
from utils.graph_extraction import (
    generate_robust_mask,
    create_gnn_dataset_parallel_limited,
    
)
class EnsembledModels(torch.nn.Module):
    def __init__(self, paths):
        super().__init__()
        self.models = torch.nn.ModuleList([load_model(model_class= c, path=p,device="cuda",model_inputs={}) for p,c in paths])
        for m in self.models:
            m.eval()

    @torch.no_grad()
    def forward(self, img, side=None):
        probs = torch.stack([
            F.softmax(m(img, side)[0].float(), dim=1) for m in self.models
        ]).mean(0)
        # return LOG-probs: argmax is unchanged, and a downstream softmax is a
        # no-op that recovers the averaged probs. Drop-in for your eval code.
        return [torch.log(probs.clamp_min(1e-8))]
    
import numpy as np
from scipy.ndimage import label
 
S8 = np.ones((3, 3), int)
 
 
def _ncomp(binary):
    return label(binary, structure=S8)[1]
 
 
def bridge_class(out, c, max_gap):
    """Connect same-class fragments of class `c` in-place. Returns #px added."""
    m = out == c
    lab, n = label(m, structure=S8)
    if n < 2:
        return 0
 
    comps = [np.argwhere(lab == i) for i in range(1, n + 1)]
    filled = 0
    for i in range(n):
        for j in range(i + 1, n):
            A, B = comps[i], comps[j]
            # closest pair of pixels between the two components
            d = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1))
            ai, bj = np.unravel_index(d.argmin(), d.shape)
            gap = d[ai, bj]
            if gap > max_gap + 1:            # too far -> a real separate vessel
                continue
            p0, p1 = A[ai], B[bj]
            steps = int(np.ceil(gap)) + 1
            ys = np.linspace(p0[0], p1[0], steps).round().astype(int)
            xs = np.linspace(p0[1], p1[1], steps).round().astype(int)
            for y, x in zip(ys, xs):
                if out[y, x] == 0:           # ONLY paint background
                    out[y, x] = c
                    filled += 1
    return filled
 
 
def bridge(pred, classes=(1, 2, 3, 4, 5), max_gap=4):
    """pred: (H,W) int ndarray -> bridged copy."""
    out = pred.copy()
    for c in classes:
        bridge_class(out, c, max_gap)
    return out
 
 
def bridge_batch(pred, classes=(1, 2, 3, 4, 5), max_gap=4):
    """pred: (B,H,W) torch tensor or ndarray -> same type/device."""
    is_torch = hasattr(pred, "cpu")
    arr = pred.cpu().numpy() if is_torch else np.asarray(pred)
    out = arr.copy()
    for b in range(out.shape[0]):
        out[b] = bridge(out[b], classes, max_gap)
    if is_torch:
        import torch
        return torch.from_numpy(out).to(pred.device)
    return out
 

def validate_bridge(gt_masks, classes=(1, 2, 3, 4, 5), max_gap=4):
    n = added_total = touched = 0
    for gt in gt_masks:
        gt = np.asarray(gt)
        out = bridge(gt, classes, max_gap)
        add = int((out != gt).sum())
        n += 1
        added_total += add
        touched += add > 0
    print(f"GT masks checked        : {n}")
    print(f"masks changed by bridge : {touched} ({100*touched/max(1,n):.1f}%)")
    print(f"total px added to GT     : {added_total}  (mean {added_total/max(1,n):.1f}/img)")
    if added_total / max(1, n) > 20:
        print("WARNING: bridging adds many px to correct GT -> max_gap too large,")
        print("         you are joining vessels that are genuinely separate. Lower it.")
    else:
        print("OK: bridging barely touches GT -> safe. Sweep max_gap on predictions.")

if __name__ == "__main__":

    model_paths = [
        ["./outputs/2026-07-16_17-29-02 [multi-right-6class-50epo-aux_branchpoint-nobias-cbam-upsample]",SwinUperNetCascade],
        ["./outputs/2026-07-17_00-28-41 [multi-right-6class-50epo-aux_branchpoint-nobias-cbam-upsample]",SwinUperNetCascade],
        ["./outputs/2026-07-17_01-04-15 [multi-right-6class-50epo-aux_branchpoint-nobias-cbam-upsample]",SwinUperNetCascade],
        # ["./outputs/2026-07-17_10-41-50 [multi-right-6class-50epo-aux_branchpoint-nobias-cbam-upsample]",SwinUperNetCascade],
        ["./outputs/2026-07-19_14-27-22 [model[DeepLabV3Plus]-multi-right-6class-50epo-aux_branchpoint-nobias-cbam-upsample]",DeepLabV3Plus]
    ]
    
    normalizer = A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        max_pixel_value=255.0,
    )

    transform = A.Compose([],additional_targets={'v_mask': 'mask',"expert_mask":"mask","topology_mask":"mask"})

    train_images = read_images(
        paths = args["paths"],
        part = "train",
        chosen_labels=None,
        hard_classes = None,
        bootstrap = args["bootstrap"],
        training_mode = args["training_mode"],
        binary_mode = args["binary_training_mode"],
        side_mode = args["side_mode"],
        in_c = args["in_c"],
        cut_borders = args["cut_borders"],
        class_count = args["class_count"]

    )
    valid_images = read_images(
        paths = args["paths"],
        part = "val",
        chosen_labels=None,
        bootstrap = args["bootstrap"],
        training_mode = args["training_mode"],
        binary_mode = args["binary_training_mode"],
        side_mode = args["side_mode"],
        in_c = args["in_c"],
        cut_borders = args["cut_borders"],
        class_count = args["class_count"]
    )
    test_images = read_images(
        paths = args["paths"],
        part = "test",
        chosen_labels=None,
        bootstrap = args["bootstrap"],
        training_mode = args["training_mode"],
        binary_mode = args["binary_training_mode"],
        side_mode = args["side_mode"],
        in_c = args["in_c"],
        cut_borders = args["cut_borders"],
        class_count = args["class_count"]
    )


    re_train_loader , re_valid_loader , re_test_loader  = make_result_dataloaders(
        train_images=train_images,
        valid_images=valid_images,
        test_images=test_images,
        normalizer=normalizer,      
        transforms=transform
    )
    # dataloaders = {
    #     "train":re_train_loader,
    #     # "val":re_valid_loader,
    #     # "test":re_test_loader
    # }
    model = EnsembledModels(model_paths).cuda()
    model.eval()
    """
    baseline:
    tensor(0.8016) tensor(0.8027) tensor(0.8013)
    single_proximal:
    tensor(0.7994) tensor(0.8026) tensor(0.7974)
    attach_daughters:
    tensor(0.7908) tensor(0.8115) tensor(0.7741)
    single_daughters:
    tensor(0.7956) tensor(0.8080) tensor(0.7852)
    superior_is_16:
    tensor(0.8016) tensor(0.8027) tensor(0.8013)
    
    """
    total_TP = torch.zeros(args["class_count"])
    total_FP = torch.zeros(args["class_count"])
    total_FN = torch.zeros(args["class_count"])
    with torch.no_grad():
        for data in tqdm(re_valid_loader):
            
            imgs = data["new_image"].cuda()
            masks = data["mask"].cuda()
            with torch.autocast(device_type="cuda",dtype=torch.float16):
                pred_logits = model(imgs)[0]
                pred_mask = torch.argmax(pred_logits,dim=1)[0]
            
            # new_pred_mask = apply_rules(pred_mask.cpu().numpy(),
            #     RuleConfig(
            #         single_proximal = False,
            #         attach_daughters = False,
            #         single_daughters = False,
            #         superior_is_16 = True
            #     ))
            new_pred_mask = bridge(pred_mask.cpu().numpy(),max_gap=30)
            new_pred_mask = torch.from_numpy(new_pred_mask)

            onehot_preds = F.one_hot(new_pred_mask,num_classes=args["class_count"])
            onehot_preds = onehot_preds.permute(2, 0, 1).float().cuda()

            onehot_gt = F.one_hot(masks[0].long(),num_classes=args["class_count"])
            onehot_gt = onehot_gt.permute(2, 0, 1).float()

            TP = ((onehot_gt*onehot_preds).sum(dim=(1,2))).cpu()
            FP = (((1-onehot_gt)*onehot_preds).sum(dim=(1,2))).cpu()
            FN = ((onehot_gt*(1-onehot_preds)).sum(dim=(1,2))).cpu()

            total_TP += TP
            total_FP += FP
            total_FN += FN

        dice_score = ((2 * total_TP + 1e-8) / (2 * total_TP + total_FP + total_FN + 1e-8))[1:]
        precision = (total_TP /(total_FP + total_TP + 1e-8))[1:]
        recall = (total_TP /(total_FN + total_TP + 1e-8))[1:]
        
        print(dice_score.mean(),precision.mean(),recall.mean())


