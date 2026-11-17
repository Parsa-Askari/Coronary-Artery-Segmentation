import torch
import copy
from tqdm.auto import tqdm
import torch.nn.functional as F
import random
import copy
###IE###
###SS###
def train_fn(model,loss_fn,optimizer,scaler,imgs,
             features,edge_index,labels,batch,training_mode,device,node_weights):
    optimizer.zero_grad()
    with torch.autocast(device_type=device,dtype=torch.float16):
        if(training_mode=="side"):
            y_pred = model(imgs)
        else:
            y_pred = model(features,edge_index,batch=batch)
        
        
        loss , loss_dict = loss_fn(
            y_pred,
            labels,
            node_weights = node_weights
        )
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

    # if random.random() < 0.1:
    #     with torch.no_grad():
    #         print("--- Total Norm ---")
    #         print(total_norm)
    #             # print("\n--- Gradient norms ---")
    #             # for name, param in model.named_parameters():
    #             #     if param.grad is not None:
    #             #         grad_norm = param.grad.data.norm().item()
    #             #         print(f"{name:30s}: {grad_norm:.6f}")
    #             # print("----------------------")

    scaler.step(optimizer)
    scaler.update()

    loss = loss.detach().cpu().item()
    for loss_name in loss_dict:
        loss_dict[loss_name] = loss_dict[loss_name].detach().cpu().item()
    
    y_pred =  F.softmax(y_pred,dim=1)
    return loss_dict , y_pred.detach().cpu().numpy()

def trainer(args,recorder,model,optimizer,loss_fn,
            train_loader,valid_loader,
            lr_sch=None,loss_weights=[1]):
    
    device = args["device"]
    epochs = args["epochs"]
    training_mode = args["training_mode"]

    scaler = torch.amp.GradScaler(device = device) 
    best_val_f1 = float("-inf")
    best_model = copy.deepcopy(model)
    for ep in tqdm(range(epochs)):
        model.train()
        class_wise_report = False
        for data,labels,name in train_loader:
            """
                features : (1,n,f)
                edge_index : (1,2,e)
                labels : (1,n)
                imgs : (B,C,H,W)
                side : (B)
            """
            if(training_mode=="side"):
                labels = labels.to(device)
                imgs = data.to(device)
                features = None
                edge_index = None
                batch = None
            else:
                
                features = data.x.to(device)
                edge_index = data.edge_index.to(device)
                labels = data.y.to(device)
                node_weights = data.node_weights.to(device)
                batch = data.batch.to(device)
                imgs = None
                if(data.x.dim == 3):
                    features = features.squeeze(0)
                    edge_index = edge_index.squeeze(0)
                    labels = labels.squeeze(0)
                    batch = batch.squeeze(0)
            loss_dict , y_pred= train_fn(
                model = model,
                loss_fn = loss_fn,
                optimizer = optimizer,
                scaler = scaler,
                imgs = imgs,
                features = features,
                edge_index = edge_index,
                labels = labels,
                batch = batch,
                device = device,
                training_mode = training_mode,
                node_weights = node_weights
            )
            labels = labels.cpu().numpy().reshape(-1)
            recorder.save_loss_history(loss_dict,"train")
            recorder.save_pred_history(labels,y_pred,"train")
        
        recorder = evaluation(
            recorder=recorder,
            model=model,
            loss_fn=loss_fn,
            valid_loader=valid_loader,
            epoch=ep,
            training_mode = training_mode,
            device=device
        )
        
        val_f1 = recorder.generate_full_report(ep)
        print("++++++++++++++++++++++++++++++++++++++++++++")
        if(best_val_f1<val_f1):
            print(f"New Best! : F1 = {val_f1}")
            best_val_f1 = val_f1
            best_model = copy.deepcopy(model)
            best_ep = ep + 1
    print(f"best result at epoch {best_ep} with f1 {best_val_f1}")
    return best_model

@torch.no_grad()
def evaluation(recorder,model,loss_fn,valid_loader,training_mode,epoch,device="cuda"):
    model.eval()
    for data,labels,name in valid_loader:
        if(training_mode=="side"):
            labels = labels.to(device)
            imgs = data.to(device)
            features = None
            edge_index = None
            batch = None
        else:
            features = data.x.to(device)
            edge_index = data.edge_index.to(device)
            labels = data.y.to(device)
            batch = data.batch.to(device)
            node_weights = data.node_weights.to(device)
            imgs = None
            if(data.x.dim == 3):
                features = features.squeeze(0)
                edge_index = edge_index.squeeze(0)
                labels = labels.squeeze(0)
                batch = batch.squeeze(0)

        with torch.autocast(device_type=device,dtype=torch.float16):
            if(training_mode=="side"):
                y_pred = model(imgs)
            else:
                y_pred = model(features,edge_index,batch=batch)
            loss , loss_dict = loss_fn(
                y_pred,
                labels,
                node_weights = node_weights
            )
        for loss_name in loss_dict:
            loss_dict[loss_name] = loss_dict[loss_name].cpu().item()

        y_pred =  F.softmax(y_pred,dim=1)
        y_pred = y_pred.cpu().numpy()

        labels = labels.cpu().numpy().reshape(-1)
        recorder.save_loss_history(loss_dict,"val")
        recorder.save_pred_history(labels,y_pred,"val")

    return recorder    
    