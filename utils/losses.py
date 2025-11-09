import torch.nn as nn
import torch.nn.functional as F
import torch
###IE###
class UnetLoss(nn.Module):
    def __init__(self,args,eps = 1e-8):
        super(UnetLoss,self).__init__()
        self.class_count = args["class_count"]
        self.loss_type = args["loss_type"]
        self.bce_fn = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)
        self.eps = eps
        self.alpha = args["alpha"]
        self.beta = args["beta"]
        self.gamma = args["gamma"]
        self.sum_dims = (2,3)
        if(self.loss_type=="dice loss"):
            print("loss is set to dice")
            self.loss_fn = DiceLoss(self.eps,self.sum_dims)
        elif(self.loss_type=="tversky loss"):
            print("loss is set to tversky")
            self.loss_fn = TverskyLoss(self.eps,self.sum_dims,self.alpha,self.beta,self.gamma)

    def forward(self,pred_mask , gt_mask):
        
        # Cross Entropy Loss
        ce_loss = self.bce_fn(pred_mask, gt_mask)
        # Dice Loss
        onehot_mask = F.one_hot(gt_mask, num_classes=self.class_count)
        onehot_mask = onehot_mask.permute(0, 3, 1, 2).float()  

        prob = self.softmax(pred_mask)

        forground_prob = prob[:,1:]
        forground_onehot_mask = onehot_mask[:,1:]
        present_class = forground_onehot_mask.sum(dim=self.sum_dims)>0
        
        second_loss = self.loss_fn(
            pred_probs = forground_prob,
            gt = forground_onehot_mask,
            present_class=present_class
        )
        total_loss = ce_loss + second_loss

        loss_dict = {
            "CE loss" : ce_loss,
            self.loss_type : second_loss
        }
        return total_loss , loss_dict

class DiceLoss(nn.Module):
    def __init__(self,eps,sum_dims):
        super(DiceLoss,self).__init__()
        self.eps = eps
        self.sum_dims = sum_dims
    def forward(self,pred_probs,gt,present_class=None):
        dice_numerator = (gt * pred_probs).sum(dim=self.sum_dims)
        dice_denominator = gt.sum(dim=self.sum_dims) + pred_probs.sum(dim=self.sum_dims)
        per_class_dice_loss = (2*dice_numerator)/(dice_denominator + self.eps)
        if(present_class is None):
            dice_loss = -per_class_dice_loss.mean()
        else:
            dice_loss = -per_class_dice_loss[present_class].mean()
        return dice_loss

class TverskyLoss(nn.Module):
    def __init__(self,eps,sum_dims,alpha=0.3,beta=0.7,gamma=1.33):
        super(TverskyLoss,self).__init__()
        self.eps = eps
        self.sum_dims = sum_dims
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
    def forward(self,pred_probs,gt,present_class=None):
        tp = (gt * pred_probs).sum(dim=self.sum_dims)
        fp = ((1-gt)*pred_probs).sum(dim=self.sum_dims)
        fn = ((1-pred_probs)*gt).sum(dim=self.sum_dims)
        t_index = (tp + self.eps) / (tp + self.alpha*fp + self.beta*fn + self.eps) 
        if(present_class is None):
            t_index = t_index.mean()
        else:
            t_index = t_index[present_class].mean()
        return (1 - t_index)**self.gamma 

    
    
