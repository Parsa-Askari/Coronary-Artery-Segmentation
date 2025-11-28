import torch 
from tqdm.notebook import tqdm
import random
import copy
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
###IE###
from utils.helpers import TP_TN_FP_FN
###SS###
def model_sanity_check(model,loss,total_norm):
    if random.random() < 0.05:
        with torch.no_grad():
            print("--- Total Norm ---")
            print(loss,total_norm)
            # print("\n--- Gradient norms ---")
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         grad_norm = param.grad.data.norm().item()
            #         print(f"{name:30s}: {grad_norm:.6f}")
            # print("----------------------\n")

    
def train_fn(model,img,ground_truths,optimizer,loss_fn,scaler,args,device,loss_weights=[1],use_amp=False):
    optimizer.zero_grad()
    loss_dict={}

    if(use_amp):
        with torch.autocast(device_type=args["device"],dtype=torch.float16,enabled=use_amp):
            preds =  model(img)
            loss , loss_dict = loss_fn(preds , ground_truths)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
        
        model_sanity_check(model,loss,total_norm)
        
        scaler.step(optimizer)
        scaler.update()
    
    else:

        with torch.autocast(device_type=args["device"],dtype=torch.float16,enabled=use_amp):
            preds =  model(img)
            loss , loss_dict = loss_fn(preds , ground_truths)

        loss.backward()
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

        model_sanity_check(model,loss,total_norm)

        optimizer.step()

    loss = loss.detach().cpu().item()
    for loss_name in loss_dict:
        loss_dict[loss_name] = loss_dict[loss_name].detach().cpu().item()
    
    loss_dict["total loss"] = loss
    pred_mask = preds[-1].detach()
    return loss_dict , pred_mask 
    
def trainer(args,recorder,model,optimizer,loss_fn,train_loader,valid_loader,lr_sch=None,loss_weights=[1]):
    device = args["device"]
    epcohs = args["epcohs"]
    class_count = args["class_count"]
    full_report_cycle = args["full_report_cycle"]
    use_amp = args["use_amp"]
    just_binary_trining = args["just_binary_trining"]
    scaler = torch.amp.GradScaler(device = device,init_scale=2**8)
    best_val_dice = float("-inf")
    best_model = copy.deepcopy(model)
    best_ep = 0
    for ep in tqdm(range(epcohs)):
        total_TP =  torch.zeros(class_count)
        total_FP = torch.zeros(class_count)
        total_FN = torch.zeros(class_count)

        model.train()
        class_wise_report = False
        labels_hist = [[],[]]
        for img,gt_side_label,gt_binary_mask,gt_abs_mask,gt_mask in tqdm(train_loader) : 
            # gt_mask = gt_mask.long()
            img = img.to(device)
            if(just_binary_trining):
                gt_binary_mask = gt_binary_mask.to(device)
                ground_truths = [
                    gt_binary_mask
                ]
            else:
                gt_mask = gt_mask.to(device)
                gt_side_label = gt_side_label.view(-1,1,1,1).to(device)
                gt_abs_mask = gt_abs_mask.to(device)
                ground_truths = [
                    gt_side_label,
                    gt_abs_mask,gt_mask
                ]

            loss_dict , pred_mask = train_fn(
                model = model,
                img = img,
                ground_truths = ground_truths,
                optimizer = optimizer,
                loss_fn = loss_fn,
                scaler = scaler,
                args = args,
                device = device,
                loss_weights = loss_weights,
                use_amp = use_amp
            )
            if(just_binary_trining):
                ground_truth_mask = gt_binary_mask.squeeze(1)
            else:
                ground_truth_mask = gt_mask
            # labels_hist[0]+=[(F.sigmoid(pred_labels.detach().cpu().reshape(-1))>=0.5).tolist()]
            # labels_hist[1]+=[gt_side_label.cpu().reshape(-1).tolist()]
            # TP , _ , FP , FN = TP_TN_FP_FN(
            #     pred_mask,
            #     gt_mask,
            #     process_preds=True
            # )
            TP , _ , FP , FN = TP_TN_FP_FN(
                pred_mask,
                ground_truth_mask,
                process_preds=True
            )
            total_TP += TP
            total_FP += FP
            total_FN += FN
            
            recorder.add_losses("train",loss_dict)
            

        current_lr = [group['lr'] for group in optimizer.param_groups][0]
        print(f"current lr : {current_lr:.4}")
        # print("train acc",accuracy_score(labels_hist[1],labels_hist[0]))
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
            use_amp = use_amp,
            just_binary_trining = just_binary_trining
        )
        if(val_dice>best_val_dice):
            print(f"New Best! : dice = {val_dice}")
            best_model = copy.deepcopy(model)
            best_val_dice = val_dice
            best_ep = ep + 1
    print(f"best result at epoch {best_ep} with dice {best_val_dice}")
    return best_model
@torch.no_grad()
def evaluation(recorder,model,loss_fn,valid_loader,class_count,
               class_wise_report=False,epoch=None,device="cuda",
               use_amp=True,just_binary_trining=False):
    model.eval()
    total_TP = torch.zeros(class_count)
    total_FP = torch.zeros(class_count)
    total_FN = torch.zeros(class_count)

    labels_hist = [[],[]]
    for img,gt_side_label,gt_binary_mask,gt_abs_mask,gt_mask in valid_loader:
        img = img.to(device)
        if(just_binary_trining):
            gt_binary_mask = gt_binary_mask.to(device)
            ground_truths = [
                gt_binary_mask
            ]
        else:
            gt_mask = gt_mask.to(device)
            gt_side_label = gt_side_label.view(-1,1,1,1).to(device)
            gt_abs_mask = gt_abs_mask.to(device)
            ground_truths = [
                gt_side_label,
                gt_abs_mask,gt_mask
            ]
        with torch.autocast(device_type=device,dtype=torch.float16,enabled=use_amp):
            preds  = model(img)
            loss , loss_dict = loss_fn(preds , ground_truths)

            loss = loss.detach().cpu().item()
            for loss_name in loss_dict:
                loss_dict[loss_name] = loss_dict[loss_name].detach().cpu().item()
        
            loss_dict["total loss"] = loss
        if(just_binary_trining):
            ground_truth_mask = gt_binary_mask.squeeze(1)
        else:
            ground_truth_mask = gt_mask

        pred_mask = preds[-1]
        # labels_hist[0]+=[(F.sigmoid(pred_labels.detach().cpu().reshape(-1))>=0.5).tolist()]
        # labels_hist[1]+=[gt_side_label.cpu().reshape(-1).tolist()]
        
        # TP , _ , FP , FN = TP_TN_FP_FN(pred_mask,gt_mask,process_preds=True)

        TP , _ , FP , FN  = TP_TN_FP_FN(
            pred_mask,
            ground_truth_mask,
            process_preds=True
        )
        total_TP += TP
        total_FP += FP
        total_FN += FN
        

        recorder.add_losses("valid",loss_dict)

    # print("test acc",accuracy_score(labels_hist[1],labels_hist[0]))
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