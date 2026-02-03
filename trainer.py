import torch 
from tqdm.notebook import tqdm
import random
import copy
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
###IE###
from utils.helpers import TP_TN_FP_FN , process_targets , no_teacher_forcing_pipeline
###SS###
def train_fn(model,contexts,targets,optimizer,loss_fn,scaler,args):
    optimizer.zero_grad()
    loss_dict={}
    with torch.autocast(device_type=args["device"],dtype=torch.float16):
        pred_targets =  model(contexts)
         
        loss , loss_dict = loss_fn(pred_targets , targets)
        

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
            # print("----------------------\n")
    scaler.step(optimizer)
    scaler.update()
    
    loss = loss.detach().cpu().item()
    for loss_name in loss_dict:
        loss_dict[loss_name] = loss_dict[loss_name].detach().cpu().item()
    
    loss_dict["total loss"] = loss
    return loss_dict , pred_targets.detach(), targets.detach()
    
def trainer(args,recorder,model,optimizer,loss_fn,train_loader,valid_loader,lr_sch=None,loss_weights=[1]):
    device = args["device"]
    epcohs = args["epcohs"]
    class_count = args["class_count"]
    full_report_cycle = args["full_report_cycle"]
    context_shape = (args["in_c"],)+ args["image_shape"]
    target_shape = args["image_shape"]
    scaler = torch.amp.GradScaler(device = device) 
    best_val_dice = float("-inf")
    
    best_model = copy.deepcopy(model)
    for ep in tqdm(range(epcohs)):
        total_TP =  torch.zeros(class_count)
        total_FP = torch.zeros(class_count)
        total_FN = torch.zeros(class_count)
        
        model.train()
        class_wise_report = False
        for contexts , targets  in tqdm(train_loader) : 
            # gt_mask = gt_mask.long()
            t_max = contexts.shape[1]
            b_size = contexts.shape[0]

            contexts = contexts.view(-1,*context_shape)
            targets = targets.view(-1,*target_shape)
            contexts = contexts.to(device)
            targets = targets.to(device)
         
            loss_dict , pred_targets , targets = train_fn(
                model = model,
                contexts = contexts,
                targets = targets,
                optimizer = optimizer,
                loss_fn = loss_fn,
                scaler = scaler,
                args = args,

            )
            pred_targets , targets = process_targets(
                pred_targets=pred_targets,
                targets= targets,
                b_size=b_size,
                t_max=t_max,
                target_shape=target_shape,
                class_count=class_count
            )

            TP , _ , FP , FN = TP_TN_FP_FN(pred_targets,targets,process_preds=False)
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
            context_shape = context_shape,
            target_shape = target_shape,
            epoch=ep,
            device=device)
        
        if(val_dice>best_val_dice):
            print(f"New Best! : dice = {val_dice}")
            best_model = copy.deepcopy(model)
            best_val_dice = val_dice
            best_ep = ep + 1
    print(f"best result at epoch {best_ep} with dice {best_val_dice}")
    return best_model



@torch.no_grad()
def evaluation(recorder,model,loss_fn,valid_loader,class_count,context_shape,target_shape,class_wise_report=False,epoch=None,device="cuda"):
    model.eval()
    total_TP = total_TP = torch.zeros(class_count)
    total_FP = torch.zeros(class_count)
    total_FN = torch.zeros(class_count)
    
    for contexts , targets  in valid_loader:

        with torch.autocast(device_type=device,dtype=torch.float16):
            pred_targets , targets ,b_size , t_max= no_teacher_forcing_pipeline(
                model = model,
                contexts = contexts,
                targets = targets,
                class_count = class_count,
                device = device
            )

            pred_targets = pred_targets.reshape(-1,class_count,*target_shape)
            targets = targets.reshape(-1,*target_shape)
            loss , loss_dict = loss_fn(pred_targets , targets)

            loss = loss.detach().cpu().item()
            for loss_name in loss_dict:
                loss_dict[loss_name] = loss_dict[loss_name].detach().cpu().item()
        
            loss_dict["total loss"] = loss
        
        pred_targets , targets = process_targets(
            pred_targets=pred_targets,
            targets= targets,
            b_size=b_size,
            t_max=t_max,
            target_shape=target_shape,
            class_count=class_count
        )
        TP , _ , FP , FN = TP_TN_FP_FN(pred_targets,targets,process_preds=False)
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