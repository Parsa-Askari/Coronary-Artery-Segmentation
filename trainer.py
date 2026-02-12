import torch 
from tqdm.notebook import tqdm
import random
import copy
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
###IE###
from utils.helpers import TP_TN_FP_FN , process_masks , no_teacher_forcing_pipeline
###SS###
def train_fn(model,imgs,masks,m_labels,m_takens,current_labels,optimizer,
             loss_fn,scaler,args,device):
    optimizer.zero_grad()
    loss_dict={}
    b_size=imgs.shape[0]
    seq_len = imgs.shape[1]
    mini_batch_size = args["mini_batch_size"]
    itter_count = (seq_len//mini_batch_size) + 1

    imgs = imgs.to(device)
    loss_dict = {}
    loss = 0
    full_mask_preds = []
    for i in range(itter_count):
        mini_imgs = imgs[:,i*mini_batch_size:(i+1)*mini_batch_size]
        mini_m_labels = m_labels[:,i*mini_batch_size:(i+1)*mini_batch_size]
        mini_m_takens = m_takens[:,i*mini_batch_size:(i+1)*mini_batch_size]
        mini_current_labels = current_labels[:,i*mini_batch_size:(i+1)*mini_batch_size]

        mini_imgs = mini_imgs.to(device)
        mini_m_labels = mini_m_labels.to(device)
        mini_m_takens = mini_m_takens.to(device)
        mini_current_labels = mini_current_labels.to(device)

        B,T,C,H,W = mini_imgs.shape
        class_counts = mini_current_labels.shape[-1]
        
        with torch.autocast(device_type=args["device"],dtype=torch.float16):
        
            pred_m_labels , pred_current_labels , _ =  model(
                img = mini_imgs[:,:-1].reshape(-1,C,H,W),
                m_taken = mini_m_takens[:,:-1].float().reshape(-1,1,H,W),
                v_seen = torch.cumsum(mini_current_labels[:,:-1],dim=1).float().reshape(-1,class_counts)
            )

            mini_loss , mini_loss_dict = loss_fn(
                pred_m_labels = pred_m_labels,
                gt_m_labels = mini_m_labels[:,1:].reshape(-1,H,W),
                pred_current_labels = pred_current_labels,
                gt_current_labels = mini_current_labels[:,1:].reshape(-1,class_counts)
            )
        print(pred_m_labels.shape)
        full_mask_preds.append(pred_m_labels.reshape(B,T,class_counts,H,W).detach())

        loss += mini_loss
        for key , l in mini_loss_dict.items():
            if(key not in loss_dict):
                loss_dict[key]=0
            loss_dict[key]+=l


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
    for l in full_mask_preds:
        print(l.shape)
    full_mask_preds = torch.cat(full_mask_preds,dim=1)
    loss = loss.detach().cpu().item()
    for loss_name in loss_dict:
        loss_dict[loss_name] = loss_dict[loss_name].detach().cpu().item()
    
    loss_dict["total loss"] = loss
    return loss_dict , full_mask_preds
    
def trainer(args,recorder,model,optimizer,loss_fn,train_loader,valid_loader,
            lr_sch=None,loss_weights=[1]):
    device = args["device"]
    epcohs = args["epcohs"]
    class_count = args["class_count"]
    full_report_cycle = args["full_report_cycle"]
    target_shape = args["image_shape"]
    report_class_wise = args["report_class_wise"]
    scaler = torch.amp.GradScaler(device = device) 
    best_val_dice = float("-inf")
    
    best_model = copy.deepcopy(model)
    for ep in tqdm(range(epcohs)):
        total_TP =  torch.zeros(class_count)
        total_FP = torch.zeros(class_count)
        total_FN = torch.zeros(class_count)
        
        model.train()
        class_wise_report = False
        for imgs , masks , m_labels , m_takens , current_labels in tqdm(train_loader) : 
            """
            img : (B,seq_len,C,H,W)
            masks : (B,1,H,W)
            m_labels : (B, seq_len, H, W)
            m_takens : (B, seq_len, 1, H, W)
            current_labels : (B, seq_len, 26)
            """
            seq_len = masks.shape[1]
            b_size = imgs.shape[0]

            loss_dict , mask_preds = train_fn(
                model = model,
                imgs = imgs,
                masks = masks,
                m_labels = m_labels,
                m_takens = m_takens,
                current_labels = current_labels,
                optimizer = optimizer,
                loss_fn = loss_fn,
                scaler = scaler,
                args = args,
                device = device

            )
            masks = masks.to(device)

            mask_preds  = process_masks(
                pred_targets=mask_preds,
                b_size=b_size,
                t_max=seq_len,
                target_shape=target_shape,
                class_count=class_count
            )

            TP , _ , FP , FN = TP_TN_FP_FN(mask_preds,masks,process_preds=False)
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
        
        recorder.print_loss_report("train",ep,class_wise=report_class_wise)
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
            target_shape = target_shape,
            report_class_wise = report_class_wise,
            epoch=ep,
            device=device
            
        )
        
        if(val_dice>best_val_dice):
            print(f"New Best! : dice = {val_dice}")
            best_model = copy.deepcopy(model)
            best_val_dice = val_dice
            best_ep = ep + 1
    print(f"best result at epoch {best_ep} with dice {best_val_dice}")
    return best_model



@torch.no_grad()
def evaluation(recorder,model,loss_fn,valid_loader,class_count,report_class_wise,
               target_shape,class_wise_report=False,epoch=None,device="cuda"):
    model.eval()
    total_TP = total_TP = torch.zeros(class_count)
    total_FP = torch.zeros(class_count)
    total_FN = torch.zeros(class_count)
    
    for imgs , masks , stop_labels , full_masks in valid_loader:
        """
        img : (B,C,H,W)
        targets : (B,seq_len,H,W)
        stop_labels : (B,seq_len)
        """
        seq_len = masks.shape[1]
        b_size = imgs.shape[0]

        masks = masks.reshape(-1,*target_shape)

        imgs = imgs.to(device)
        masks = masks.to(device)
        stop_labels = stop_labels.to(device)
        
        with torch.autocast(device_type=device,dtype=torch.float16):
            pred_stop_labels,pred_masks =  model(imgs,seq_len)


            loss , loss_dict , class_wise_loss = loss_fn(
                masks = masks,
                pred_masks = pred_masks,
                stop_labels = stop_labels,
                pred_stop_labels = pred_stop_labels,

            )
            masks = masks.cpu()
            full_masks = full_masks.to(device)
            
            loss = loss.detach().cpu().item()
            for loss_name in loss_dict:
                loss_dict[loss_name] = loss_dict[loss_name].detach().cpu().item()
        
            loss_dict["total loss"] = loss
        
        pred_masks  = process_masks(
            pred_targets=pred_masks,
            b_size=b_size,
            t_max=seq_len,
            target_shape=target_shape,
            class_count=class_count
        )
        TP , _ , FP , FN = TP_TN_FP_FN(pred_masks,full_masks,process_preds=False)
        total_TP += TP
        total_FP += FP
        total_FN += FN
        
        recorder.add_losses("valid",loss_dict,class_wise_loss)
        
    dice_score = (2 * total_TP + 1e-8) / (2 * total_TP + total_FP + total_FN + 1e-8)
    precision = total_TP /(total_FP + total_TP + 1e-8) 
    recall = total_TP /(total_FN + total_TP + 1e-8) 

    recorder.add_metrics(
        dice_score.tolist(),
        precision.tolist(),
        recall.tolist(),
        part = "valid"
    )
    recorder.print_loss_report("valid",epoch,class_wise=report_class_wise)
    recorder.print_metrics_report("valid",epoch,class_wise=class_wise_report)
    print("-"*60)
    return dice_score[1:].mean().item()