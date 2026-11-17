import torch 
from tqdm.notebook import tqdm
import random
import copy
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from pathlib import Path
import os
import cv2
import matplotlib
import numpy as np
###IE###
from utils.helpers import TP_TN_FP_FN ,resize_dsv
###SS###
import torch
import torch.nn.functional as F
import random


def _grads_for(loss_items, leaf, store_device, out):
    """One backward per loss onto `leaf`; frees the graph on the last term."""
    n = len(loss_items)
    for i, (name, L) in enumerate(loss_items):
        g = torch.autograd.grad(
            L, leaf,
            retain_graph=(i < n - 1),   # <-- free the graph on the final term
            allow_unused=True,
        )[0]
        if g is not None:
            out[name] = g.flatten().to(store_device)   # offload, then drop the GPU copy
        del g


def _measure_grads(logits, recompute_loss_dict_fn, branch_preds=None,
                   recompute_branch_loss_dict_fn=None,
                   sample=None, store_device="cpu"):
    if sample is not None:                    # diagnose on a slice of the batch
        logits = logits[:sample]
        if branch_preds is not None:
            branch_preds = branch_preds[:sample]

    grads = {}

    leaf = logits.detach().float().requires_grad_(True)
    loss_dict = recompute_loss_dict_fn(leaf)
    items = [(n, L) for n, L in loss_dict.items()
             if torch.is_tensor(L) and L.requires_grad]
    _grads_for(items, leaf, store_device, grads)
    del loss_dict, items, leaf                # recomputed graph is gone here

    if branch_preds is not None:
        b_leaf = branch_preds.detach().float().requires_grad_(True)
        branch_loss_dict = recompute_branch_loss_dict_fn(b_leaf)
        items = [(f"branch_{n}", L) for n, L in branch_loss_dict.items()
                 if torch.is_tensor(L) and L.requires_grad]
        _grads_for(items, b_leaf, store_device, grads)
        del branch_loss_dict, items, b_leaf

    with torch.no_grad():
        raw_norms = {n: g.norm().item() for n, g in grads.items()}
        names = list(grads)
        cosines = {}
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                cosines[f"{names[i]}|{names[j]}"] = F.cosine_similarity(
                    grads[names[i]], grads[names[j]], dim=0
                ).item()

    del grads
    return raw_norms, cosines


def train_fn(
        model,
        imgs,
        gt_masks,
        loss_weights,
        optimizer,
        loss_fn,
        scaler,
        args,
        ep =None,
        small_vmasks=None,
        skel_masks=None,
        topology_masks = None,
        sides=None,
        measure_prob=0.01,   # gate the whole diagnostic
        ):

    optimizer.zero_grad()
    loss_dict = {}
    do_measure = random.random() < measure_prob

    # ---------------- real forward (unchanged, autocast/fp16) ----------------
    with torch.autocast(device_type=args["device"], dtype=torch.float16):
        
        pred_masks,aux_loss = model(imgs,gt_mask=gt_masks,ep=ep)
        
        loss, loss_dict = loss_fn(
            pred_masks=pred_masks[0],
            gt_masks=gt_masks,
            skel_masks=skel_masks,
            ignore_mask=small_vmasks
        )
        
        loss = loss + (0.5*aux_loss)
        

        total_dsv_loss = 0
        dsv_targets = []  # stash targets so we can recompute in fp32 later
        for i, w in enumerate(loss_weights):
            dsv_pred_masks = pred_masks[i + 1]
            size = dsv_pred_masks.shape[-2:]
            dsv_gt_masks = resize_dsv(gt_masks, size, "nearest")
            # if size[0] == 512:
            #     dsv_skel_masks = skel_masks
            # else:
            #     dsv_skel_masks = None
            dsv_skel_masks = resize_dsv(skel_masks, size, "nearest")
            dsv_small_vmasks = resize_dsv(small_vmasks, size, "nearest")

            dsv_loss, _ = loss_fn(
                pred_masks=dsv_pred_masks,
                gt_masks=dsv_gt_masks,
                skel_masks=dsv_skel_masks,
                ignore_mask=dsv_small_vmasks,
            )
            total_dsv_loss += (w * dsv_loss)
            dsv_targets.append((i, w, dsv_pred_masks,
                                dsv_gt_masks, dsv_skel_masks, dsv_small_vmasks))

        loss = loss + (0.7 * total_dsv_loss)

    # ---------------- diagnostic (fp32, isolated, gated) ----------------
    if do_measure:
        # main head
        raw, cos = _measure_grads(
            pred_masks[0],
            lambda leaf: loss_fn(
                pred_masks=leaf,
                gt_masks=gt_masks,
                skel_masks=skel_masks,
                ignore_mask=small_vmasks,
                # aux_loss=aux_loss,
            )[1],
        )
        print("++++++++ main loss grad norms ++++++++")
        print(raw)
        print("cosines:", cos)

        print("++++++++ dsv grad norms ++++++++")
        for (i, w, dsv_pred, dsv_gt, dsv_skel, dsv_sv) in dsv_targets:
            raw, cos = _measure_grads(
                dsv_pred,
                lambda leaf, gt=dsv_gt, sk=dsv_skel, sv=dsv_sv: loss_fn(
                    pred_masks=leaf,
                    gt_masks=gt,
                    # skel_masks=sk,
                    ignore_mask=sv,
                )[1],
            )
            # also show weighted contribution: this head is scaled by 0.7 * w
            weighted = {n: v * (0.7 * w) for n, v in raw.items()}
            print(f"[dsv {i} | w={w}] raw={raw}")
            print(f"[dsv {i} | w={w}] weighted={weighted}  cosines={cos}")

    # ---------------- real backward / step (unchanged) ----------------
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

    if do_measure:
        print("--- Total Norm ---")
        print(total_norm)

    scaler.step(optimizer)
    scaler.update()

    loss = loss.detach().cpu().item() # branch_loss_dict
    for loss_name in loss_dict:
        loss_dict[loss_name] = loss_dict[loss_name].detach().cpu().item()

    # for loss_name in branch_loss_dict:
    #     loss_dict[f"branch_{loss_name}"] = branch_loss_dict[loss_name].detach().cpu().item()

    loss_dict["total loss"] = loss

    return loss_dict, pred_masks[0].detach()
    
def trainer(args,recorder,model,optimizer,loss_fn,train_loader,valid_loader,
            lr_sch=None,loss_weights=[1],choice_score = "dice",test_loader=None):
    device = args["device"]
    epochs = args["epochs"]
    class_count = args["class_count"]
    full_report_cycle = args["full_report_cycle"]
    training_mode = args["training_mode"]
    use_ignore_mask = args["use_ignore_mask"]

    scaler = torch.amp.GradScaler(device = device) 
    best_val_dice = float("-inf")
    
    best_model = copy.deepcopy(model)
    for ep in tqdm(range(epochs)):
        total_TP =  torch.zeros(class_count)
        total_FP = torch.zeros(class_count)
        total_FN = torch.zeros(class_count)
        # if(ep == 50):
        #     print("start moe training")
        #     model.train_moe = True
        model.train()
        class_wise_report = False
        for data in tqdm(train_loader):
            """
                img : (B,C,H,W)
                gt_binary_masks : (B,H,W)
                expert_mask : (B,H,W)
                gt_multi_masks : (B,H,W)
                v_mask : (B,H,W)
            """
            imgs = data["imgs"].to(device)
            sides = data["side"].int().to(device).reshape(-1)
            gt_masks = data["mask"].to(device)
            small_vmasks = data["small_vmask"].to(device)
            skel_masks = data["skel_mask"].to(device)
            # topology_masks = data["topology_mask"].to(device)
            if(use_ignore_mask == False):
                small_vmasks = None
            loss_dict , mask_preds = train_fn(
                model = model,
                imgs = imgs,
                gt_masks = gt_masks,
                small_vmasks = small_vmasks,
                sides = sides,
                optimizer = optimizer,
                loss_fn = loss_fn,
                scaler = scaler,
                args = args,
                loss_weights = loss_weights,
                skel_masks = skel_masks,
                ep = ep,
                measure_prob = 0.0
            )

            
            TP , _ , FP , FN = TP_TN_FP_FN(mask_preds,gt_masks,process_preds=True)
            total_TP += TP
            total_FP += FP
            total_FN += FN
                
            recorder.add_losses("train",loss_dict)
            
            
        current_lr = [group['lr'] for group in optimizer.param_groups][0]
        print(f"current lr : {current_lr:.4}")
        if(lr_sch is not None):
            lr_sch.step()
        
       
        dice_score = (2 * total_TP + 1e-8) / (2 * total_TP + total_FP + total_FN + 1e-8)
        precision = total_TP /(total_FP + total_TP + 1e-8) 
        recall = total_TP /(total_FN + total_TP + 1e-8) 
        
        recorder.add_metrics(
            dice_score.tolist(),
            precision.tolist(),
            recall.tolist(),
            part = "train"
        )
        
        recorder.print_loss_report("train",ep)
        recorder.print_metrics_report("train",ep,class_wise=False)
        print("<=>"*20)
        
        if((ep+1)%full_report_cycle==0):
            class_wise_report=True
            
        val_dice,val_per,val_rec = evaluation(
            recorder=recorder,
            model=model,
            loss_fn=loss_fn,
            valid_loader=valid_loader,
            class_wise_report=class_wise_report,
            class_count = class_count,
            epoch=ep,
            device=device,
            training_mode = training_mode,
            part="valid",
            use_ignore_mask = use_ignore_mask
        )
        if(test_loader is not None):
            tes_dice,tes_per,tes_rec = evaluation(
                recorder=recorder,
                model=model,
                loss_fn=loss_fn,
                valid_loader=test_loader,
                class_wise_report=class_wise_report,
                class_count = class_count,
                epoch=ep,
                device=device,
                training_mode = training_mode,
                part="test",
                use_ignore_mask = use_ignore_mask
            )
        if(choice_score=="dice"):
            score = val_dice
        else:
            score = (val_rec*args["socre_params"]["r"])+(val_per*args["socre_params"]["p"])
        if(score>best_val_dice):
            print(f"New Best! : dice = {val_dice} , recall = {val_rec} , precision = {val_per}")
            best_model = copy.deepcopy(model)
            best_val_dice = score
            best_recall = val_rec
            best_ep = ep + 1
    print(f"best result at epoch {best_ep} with dice {best_val_dice} and recall : {best_recall}")
    return best_model



@torch.no_grad()
def evaluation(recorder,model,loss_fn,valid_loader,class_count,training_mode,use_ignore_mask,
               class_wise_report=False,epoch=None,device="cuda",part="valid"):
    model.eval()
    total_TP = total_TP = torch.zeros(class_count)
    total_FP = torch.zeros(class_count)
    total_FN = torch.zeros(class_count)

    if(part=="valid"):
        vis_names = ["63","36","69","129","30","26","10","93","182","33","22","187","49","1","43"]
        base_vis_path = Path("./val_vis")
        for name in vis_names:
            os.makedirs(base_vis_path / name,exist_ok=True)
        cmap = matplotlib.colormaps['tab20']  # good for up to 20 classes
        COLORS = (np.array([cmap(i / max(class_count-1, 1))[:3] for i in range(class_count)]) * 255).astype(np.uint8)
        COLORS = COLORS[:, ::-1]
    for data in valid_loader:
        """
            img : (B,C,H,W)
            gt_binary_masks : (B,H,W)
            gt_multi_masks : (B,H,W)
            v_masks : (B,H,W)
        """

        imgs = data["imgs"].to(device)
        sides = data["side"].to(device).int().reshape(-1)
        gt_masks = data["mask"].to(device)
        small_vmasks = data["small_vmask"].to(device)
        skel_masks = data["skel_mask"].to(device)
        name = data["name"][0]
        # topology_masks = data["topology_mask"].to(device)
        if(use_ignore_mask == False):
            small_vmasks = None
        with torch.autocast(device_type=device,dtype=torch.float16):
           
            pred_masks,aux_loss = model(imgs,gt_mask=gt_masks,ep=epoch)#,topology_masks.unsqueeze(1).float()
            pred_masks = pred_masks[0]
            
            loss , loss_dict = loss_fn(
                pred_masks = pred_masks,
                gt_masks = gt_masks,
                skel_masks = skel_masks,
                ignore_mask = small_vmasks
            )

            # branch_loss, branch_loss_dict = branch_loss_fn(
            #     pred_masks=branch_preds,
            #     gt_masks=topology_masks,
            #     # skel_masks=skel_masks,
            #     small_vmasks=small_vmasks
            # )
            # side_loss = model._side_loss(sides)
            # loss = loss + (0.05*side_loss)
            # loss_dict["side_loss"] = side_loss
            loss = loss + (0.5*aux_loss)
            loss = loss.detach().cpu().item()
            

            for loss_name in loss_dict:
                loss_dict[loss_name] = loss_dict[loss_name].detach().cpu().item()
            # for loss_name in branch_loss_dict:
            #     loss_dict[f"branch_{loss_name}"] = branch_loss_dict[loss_name].detach().cpu().item()
            
            loss_dict["total loss"] = loss
            
        
        TP , _ , FP , FN = TP_TN_FP_FN(pred_masks,gt_masks,process_preds=True)
        total_TP += TP
        total_FP += FP
        total_FN += FN
        
        recorder.add_losses(part,loss_dict)

        if(part == "valid" and name in vis_names):
            img = cv2.imread(f"./dataset/syntax/images/val/{name}.png",cv2.IMREAD_COLOR)
            pred_class = torch.argmax(pred_masks,dim=1).cpu().numpy()[0]
            color_mask = COLORS[pred_class]
            
            overlay = cv2.addWeighted(img, 0.6, color_mask, 0.4, 0)
            cv2.imwrite(base_vis_path / name / f"{epoch}.png",overlay)

    dice_score = (2 * total_TP + 1e-8) / (2 * total_TP + total_FP + total_FN + 1e-8)
    precision = total_TP /(total_FP + total_TP + 1e-8) 
    recall = total_TP /(total_FN + total_TP + 1e-8) 

    recorder.add_metrics(
        dice_score.tolist(),
        precision.tolist(),
        recall.tolist(),
        part = part
    )
    recorder.print_loss_report(part,epoch)
    recorder.print_metrics_report(part,epoch,class_wise=class_wise_report)
    print("-"*60)
    return dice_score[1:].mean().item() , precision[1:].mean().item() , recall[1:].mean().item()