import torch 
from tqdm.notebook import tqdm
import random
import copy
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
###IE###
from utils.helpers import TP_TN_FP_FN
###SS###
def send_to_device(array,device):
    if (isinstance(array,list)):
        for i in range(len(array)):
            array[i] = array[i].to(device)
    else:
        array = [array.to(device)]
    return array

def model_sanity_check(model,loss,total_norm):
    if random.random() <0.01:
        with torch.no_grad():
            print("--- Total Norm ---")
            # with open("./grad_log.txt","a") as f:
            #     f.write(f"{loss.detach().item()},{total_norm.detach().item()}\n")
            print(loss,total_norm)

            if(random.random() < 0.05):
                print("\n--- Gradient norms ---")
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.data.norm().item()
                        data = torch.norm(param).item()
                        print(f"{name:30s}: {grad_norm:.6f} - {data:.6f}")
                        # c+=f"{name:30s}: {grad_norm:.6f} - {data:.6f}\n"
            # with open("./grad_log.txt","a") as f:
            #     f.write(c)
            #     f.write("-----------------\n")
            print("----------------------\n")

def calculate_loss_loop(preds,ground_truths,loss_weights,loss_fn):
    loss = 0
    for i,pred in enumerate(preds):
        ground_truth = ground_truths[i]
        loss_weight = loss_weights[i]
        layer_loss , layer_loss_dict = loss_fn(pred , ground_truth)
        
        loss += loss_weight*layer_loss
        if(i==0):
            loss_dict = {key:value for key,value in layer_loss_dict.items()}
        else:
            loss_dict = {key : loss_dict[key] + (loss_weight*layer_loss_dict[key]) for key in layer_loss_dict}
    return loss_dict ,loss

def train_fn(model,img,ground_truths,optimizer,loss_fn,scaler,args,device,loss_weights=[1],use_amp=False):
    optimizer.zero_grad()
    loss_dict={}

    if(use_amp):
        with torch.autocast(device_type=args["device"],dtype=torch.float16,enabled=use_amp):
            preds =  model(img)
            loss_dict ,loss = calculate_loss_loop(
                preds=preds,
                ground_truths=ground_truths,
                loss_weights=loss_weights,
                loss_fn=loss_fn
            )
                
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
        
        model_sanity_check(model,loss,total_norm)
        
        scaler.step(optimizer)
        scaler.update()
    
    else:
        with torch.autocast(device_type=args["device"],dtype=torch.float16,enabled=use_amp):
            preds =  model(img)
            loss_dict ,loss = calculate_loss_loop(
                preds=preds,
                ground_truths=ground_truths,
                loss_weights=loss_weights,
                loss_fn=loss_fn
            )

        loss.backward()
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

        model_sanity_check(model,loss,total_norm)

        optimizer.step()

    loss = loss.detach().cpu().item()
    for loss_name in loss_dict:
        loss_dict[loss_name] = loss_dict[loss_name].detach().cpu().item()
    
    loss_dict["total loss"] = loss
    pred_mask = preds[0].detach()
    return loss_dict , pred_mask 



def trainer(args,recorder,model,optimizer,loss_fn,train_loader,valid_loader,
            lr_sch=None,loss_weights=[1],binary_encoder=None):
    device = args["device"]
    epcohs = args["epcohs"]
    class_count = args["class_count"]
    full_report_cycle = args["full_report_cycle"]
    use_amp = args["use_amp"]
    just_binary_trining = args["just_binary_trining"]
    binary_type = args["binary_type"]
    scaler = torch.amp.GradScaler(device = device)
    best_val_dice = float("-inf")
    
    best_model = copy.deepcopy(model)
    best_ep = 0
    for ep in tqdm(range(epcohs)):
        total_TP =  torch.zeros(class_count)
        total_FP = torch.zeros(class_count)
        total_FN = torch.zeros(class_count)

        model.train()
        class_wise_report = False
        for img,gt_masks in tqdm(train_loader) : 
            # gt_mask = gt_mask.long()
            img = img.to(device)
            gt_masks = send_to_device(gt_masks,device)
            loss_dict , pred_mask = train_fn(
                model = model,
                img = img,
                ground_truths = gt_masks,
                optimizer = optimizer,
                loss_fn = loss_fn,
                scaler = scaler,
                args = args,
                device = device,
                loss_weights = loss_weights,
                use_amp = use_amp
            )
            
            ground_truth_mask = gt_masks[0].squeeze(1)
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
            just_binary_trining = just_binary_trining,
            binary_type=binary_type,
            loss_weights = loss_weights
        )
        if(val_dice>best_val_dice):
            print(f"New Best! : dice = {val_dice}")
            best_model = copy.deepcopy(model)
            best_val_dice = val_dice
            best_ep = ep + 1
    print(f"best result at epoch {best_ep} with dice {best_val_dice}")
    return best_model
@torch.no_grad()
def evaluation(recorder,model,loss_fn,valid_loader,class_count,binary_type,
               class_wise_report=False,epoch=None,device="cuda",
               use_amp=True,just_binary_trining=False,loss_weights=[1]):
    model.eval()
    total_TP = torch.zeros(class_count)
    total_FP = torch.zeros(class_count)
    total_FN = torch.zeros(class_count)

    labels_hist = [[],[]]
    
    for img,gt_masks in valid_loader:
        img = img.to(device)

        with torch.autocast(device_type=device,dtype=torch.float16,enabled=use_amp):
            preds  = model(img)
            gt_masks = send_to_device(gt_masks,device)
            loss_dict ,loss = calculate_loss_loop(
                preds=preds,
                ground_truths=gt_masks,
                loss_weights=loss_weights,
                loss_fn=loss_fn
            )

            loss = loss.detach().cpu().item()
            for loss_name in loss_dict:
                loss_dict[loss_name] = loss_dict[loss_name].detach().cpu().item()
        
            loss_dict["total loss"] = loss
        
        ground_truth_mask = gt_masks[0].squeeze(1)
        pred_mask = preds[0]
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