import torch 
from tqdm.notebook import tqdm
import random
import copy
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
###IE###
from utils.helpers import TP_TN_FP_FN 
###SS###
def train_fn(model,imgs,gt_masks,loss_weights,optimizer,loss_fn,scaler,args):
    optimizer.zero_grad()
    loss_dict={}

    with torch.autocast(device_type=args["device"],dtype=torch.float16):
        pred_masks = model(imgs)
        loss , loss_dict = loss_fn(
            pred_masks = pred_masks[0],
            gt_masks = gt_masks
        )

        for i,w in enumerate(loss_weights[1:]):
            
            dsv_pred_masks = pred_masks[i+1]
            dsv_gt_masks =  F.interpolate(
                input=gt_masks.unsqueeze(1).float(),
                size=dsv_pred_masks.shape[-2:],
                mode="nearest"
            ).squeeze(1).long()

            dsv_loss , dsv_loss_dict = loss_fn(
                pred_masks = dsv_pred_masks,
                gt_masks = dsv_gt_masks
            )
            loss += (w*dsv_loss)
            for loss_type,l in dsv_loss_dict.items():
                loss_dict[loss_type] += (w*l)

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

    if random.random() < 0.001:
        with torch.no_grad():
            print("--- Total Norm ---")
            print(total_norm)
            # print("\n--- Gradient norms ---")
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         grad_norm = param.grad.data.norm().item()
            #         print(f"{name:30s}: {grad_norm:.6f}")
            print("----------------------\n")
    scaler.step(optimizer)
    scaler.update()
    
    loss = loss.detach().cpu().item()
    for loss_name in loss_dict:
        loss_dict[loss_name] = loss_dict[loss_name].detach().cpu().item()
    
    loss_dict["total loss"] = loss
    return loss_dict , pred_masks[0].detach()
    
def trainer(args,recorder,model,optimizer,loss_fn,train_loader,valid_loader,
            lr_sch=None,loss_weights=[1]):
    device = args["device"]
    epcohs = args["epcohs"]
    class_count = args["class_count"]
    full_report_cycle = args["full_report_cycle"]
    training_mode = args["training_mode"]

    scaler = torch.amp.GradScaler(device = device) 
    best_val_dice = float("-inf")
    
    best_model = copy.deepcopy(model)
    for ep in tqdm(range(epcohs)):
        total_TP =  torch.zeros(class_count)
        total_FP = torch.zeros(class_count)
        total_FN = torch.zeros(class_count)
        
        model.train()
        class_wise_report = False
        for imgs, gt_binary_masks, gt_multi_masks in tqdm(train_loader):
            """
            img : (B,C,H,W)
            gt_binary_masks : (B,H,W)
            gt_multi_masks : (B,H,W)
            """
            imgs = imgs.to(device)
            if(training_mode=="binary"):
                gt_masks = gt_binary_masks.to(device)
            else:
                gt_masks = gt_multi_masks.to(device)


            loss_dict , mask_preds = train_fn(
                model = model,
                imgs = imgs,
                gt_masks = gt_masks,
                optimizer = optimizer,
                loss_fn = loss_fn,
                scaler = scaler,
                args = args,
                loss_weights = loss_weights
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
            
        val_dice = evaluation(
            recorder=recorder,
            model=model,
            loss_fn=loss_fn,
            valid_loader=valid_loader,
            class_wise_report=class_wise_report,
            class_count = class_count,
            epoch=ep,
            device=device,
            training_mode = training_mode
        )
        
        if(val_dice>best_val_dice):
            print(f"New Best! : dice = {val_dice}")
            best_model = copy.deepcopy(model)
            best_val_dice = val_dice
            best_ep = ep + 1
    print(f"best result at epoch {best_ep} with dice {best_val_dice}")
    return best_model



@torch.no_grad()
def evaluation(recorder,model,loss_fn,valid_loader,class_count,training_mode,
               class_wise_report=False,epoch=None,device="cuda"):
    model.eval()
    total_TP = total_TP = torch.zeros(class_count)
    total_FP = torch.zeros(class_count)
    total_FN = torch.zeros(class_count)
    
    for imgs, gt_binary_masks, gt_multi_masks in valid_loader:
        """
        img : (B,C,H,W)
        gt_binary_masks : (B,H,W)
        gt_multi_masks : (B,H,W)
        """
        imgs = imgs.to(device)
        if(training_mode=="binary"):
            gt_masks = gt_binary_masks.to(device)
        else:
            gt_masks = gt_multi_masks.to(device)
        
        with torch.autocast(device_type=device,dtype=torch.float16):
            pred_masks = model(imgs)[0]
            loss , loss_dict = loss_fn(
                pred_masks = pred_masks,
                gt_masks = gt_masks
            )
            
            loss = loss.detach().cpu().item()
            for loss_name in loss_dict:
                loss_dict[loss_name] = loss_dict[loss_name].detach().cpu().item()
        
            loss_dict["total loss"] = loss
        
        
        TP , _ , FP , FN = TP_TN_FP_FN(pred_masks,gt_masks,process_preds=True)
        total_TP += TP
        total_FP += FP
        total_FN += FN
        
        recorder.add_losses("valid",loss_dict)
        
    dice_score = (2 * total_TP + 1e-8) / (2 * total_TP + total_FP + total_FN + 1e-8)
    precision = total_TP /(total_FP + total_TP + 1e-8) 
    recall = total_TP /(total_FN + total_TP + 1e-8) 

    recorder.add_metrics(
        dice_score.tolist(),
        precision.tolist(),
        recall.tolist(),
        part = "valid"
    )
    recorder.print_loss_report("valid",epoch)
    recorder.print_metrics_report("valid",epoch,class_wise=class_wise_report)
    print("-"*60)
    return dice_score[1:].mean().item()