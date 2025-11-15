import torch 
from tqdm.notebook import tqdm
import random
###IE###
from utils.helpers import TP_TN_FP_FN
###SS###
def train_fn(model,img,gt_masks,optimizer,loss_fn,scaler,args,device,loss_weights=[1]):
    optimizer.zero_grad()
    loss_dict={}
    with torch.autocast(device_type=args["device"],dtype=torch.float16):
        pred_masks =  model(img)
        loss = 0
        for i,pred_mask in enumerate(pred_masks) : 

            gt_mask = gt_masks[i].to(device)
            loss_weight = loss_weights[i]
            layer_loss , layer_loss_dict = loss_fn(pred_mask , gt_mask)
            if(i==0):
                loss_dict = layer_loss_dict
                pred_mask_last = pred_mask.detach()
                gt_mask_last = gt_mask
            else:
                loss_dict = {key : loss_dict[key]+ loss_weight*layer_loss_dict[key] for key in layer_loss_dict}

            loss += loss_weight*layer_loss

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

    if random.random() < 0.01:
        with torch.no_grad():
            print("--- Total Norm ---")
            print(total_norm)
            # print("\n--- Gradient norms ---")
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         grad_norm = param.grad.data.norm().item()
            #         print(f"{name:30s}: {grad_norm:.6f}")
            # print("----------------------\n")
    scaler.step(optimizer)
    scaler.update()
    
    loss = loss.detach().cpu().item()
    for loss_name in loss_dict:
        loss_dict[loss_name] = loss_dict[loss_name].detach().cpu().item()
    
    loss_dict["total loss"] = loss
    return loss_dict , pred_mask_last , gt_mask_last
    
def trainer(args,recorder,model,optimizer,loss_fn,train_loader,valid_loader,loss_weights=[1]):
    device = args["device"]
    epcohs = args["epcohs"]
    class_count = args["class_count"]
    full_report_cycle = args["full_report_cycle"]
    scaler = torch.amp.GradScaler(device = device) 
    for ep in tqdm(range(epcohs)):
        total_TP =  torch.zeros(class_count)
        total_FP = torch.zeros(class_count)
        total_FN = torch.zeros(class_count)
        
        model.train()
        class_wise_report = False
        for img , gt_masks  in tqdm(train_loader) : 
            # gt_mask = gt_mask.long()
            img = img.to(device)
            # gt_mask = gt_mask.to(device)

            loss_dict , pred_mask , gt_mask = train_fn(
                model = model,
                img = img,
                gt_masks = gt_masks,
                optimizer = optimizer,
                loss_fn = loss_fn,
                scaler = scaler,
                args = args,
                device = device,
                loss_weights = loss_weights
            )
            
            TP , _ , FP , FN = TP_TN_FP_FN(pred_mask,gt_mask,process_preds=True)
            total_TP += TP
            total_FP += FP
            total_FN += FN
            
            recorder.add_losses("train",loss_dict)
            
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
            
        evaluation(
            recorder=recorder,
            model=model,
            loss_fn=loss_fn,
            valid_loader=valid_loader,
            class_wise_report=class_wise_report,
            class_count = class_count,
            epoch=ep,
            device=device)
            
@torch.no_grad()
def evaluation(recorder,model,loss_fn,valid_loader,class_count,class_wise_report=False,epoch=None,device="cuda"):
    model.eval()
    total_TP = total_TP = torch.zeros(class_count)
    total_FP = torch.zeros(class_count)
    total_FN = torch.zeros(class_count)
    
    for img , gt_mask  in valid_loader:
        img = img.to(device)
        gt_mask = gt_mask[0].to(device)
        
        with torch.autocast(device_type=device,dtype=torch.float16):
            pred_mask  = model(img)[0]
            loss , loss_dict = loss_fn(pred_mask , gt_mask)

            loss = loss.detach().cpu().item()
            for loss_name in loss_dict:
                loss_dict[loss_name] = loss_dict[loss_name].detach().cpu().item()
        
            loss_dict["total loss"] = loss
        
        TP , _ , FP , FN = TP_TN_FP_FN(pred_mask,gt_mask,process_preds=True)
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