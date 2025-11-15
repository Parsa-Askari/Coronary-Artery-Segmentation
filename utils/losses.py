import torch.nn as nn
import torch.nn.functional as F
import torch
###IE###
from .helpers import soft_skeletonize
###SS###
class UnetLoss(nn.Module):
    def __init__(self,args,eps = 1e-8):
        super(UnetLoss,self).__init__()
        self.class_count = args["class_count"]
        self.loss_type = args["loss_type"]
        self.alpha = args["alpha"]
        self.beta = args["beta"]
        self.t_gamma = args["t_gamma"]
        self.f_gamma = args["f_gamma"]
        self.k = args["k"]
        self.loss_coefs = args["loss_coefs"]
        # self.focal_fn = FocalCrossEntropy(
        #     f_gamma=self.f_gamma,
        #     eps=eps,
        #     f_alpha=args["f_alpha"],
        #     f_loss_scale = args["f_loss_scale"]
        # )
        if(args["f_alpha"] is not None):
            w = torch.tensor(args["f_alpha"],dtype=torch.float32,device="cuda")
            self.ce_fn = nn.CrossEntropyLoss(weight=w)
        else :
            self.ce_fn = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)
        self.eps = eps
        self.sum_dims = (0,2,3)
        if(self.loss_type=="dice loss"):
            print("loss is set to dice")
            self.loss_fn = DiceLoss(self.eps,self.sum_dims)
        elif(self.loss_type=="tversky loss"):
            print("loss is set to tversky")
            self.loss_fn = TverskyLoss(self.eps,self.sum_dims,self.alpha,self.beta,self.t_gamma)
        self.cldice_fn = CLDiceLoss(sum_dims=self.sum_dims,eps=self.eps,k=self.k) 
    def forward(self,pred_mask , gt_mask ):
        
        onehot_mask = F.one_hot(gt_mask, num_classes=self.class_count)
        onehot_mask = onehot_mask.permute(0, 3, 1, 2).float()  

        prob = self.softmax(pred_mask)

        # Cross Entropy Loss
        ce_loss = self.ce_fn(pred_mask,gt_mask)
        # Dice/Tversky Loss
        forground_prob = prob[:,1:]
        forground_onehot_mask = onehot_mask[:,1:]
        # present_class = forground_onehot_mask.sum(dim=self.sum_dims)>0
        
        second_loss = self.loss_fn(
            pred_probs = forground_prob,
            gt = forground_onehot_mask
        )

        ce_loss = self.loss_coefs["CE"]*ce_loss
        second_loss = self.loss_coefs["Second"]*second_loss

        total_loss = second_loss + ce_loss

        loss_dict = {
            "CE loss" : ce_loss,
            self.loss_type : second_loss
        }
        return total_loss , loss_dict

class FocalCrossEntropy(nn.Module):
    def __init__(self,f_gamma,eps,f_loss_scale=1,f_alpha=None):
        super(FocalCrossEntropy,self).__init__()
        self.f_gamma = f_gamma
        if(f_alpha is not None):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.f_alpha = torch.tensor(f_alpha).to(device)
        else:
            self.f_alpha = f_alpha
        self.eps = eps
        self.f_loss_scale = f_loss_scale
    def forward(self,prob,onehot_mask):
        # prob : (B,C,H,W)
        # onehot_mask : (B,C,H,W)
        # gt_mask = (B,H,W)

        p = (prob*onehot_mask).sum(dim=1) # (B,H,W)
        pt = torch.clamp(p,self.eps,1-self.eps)
        focal_weights = (1-pt)**self.f_gamma
        focal_loss = focal_weights*(torch.log(pt))
        if(self.f_alpha is not None):
            alpha_b = self.f_alpha.view(1, -1, 1, 1).type_as(prob)
            class_w = (alpha_b*onehot_mask).sum(dim=1)
        else :
            class_w = 1.0
        return -self.f_loss_scale*(class_w*focal_loss).mean()

class CLDiceLoss(nn.Module):
    def __init__(self,eps,sum_dims,k=40):
        super(CLDiceLoss,self).__init__()
        self.k=k
        self.eps = eps
        self.sum_dims = (1,2)

    def forward(self,pred_binary_mask , gt_mask,gt_skel):

        binary_pred = (pred_binary_mask>=0.5).type_as(pred_binary_mask)
        binary_gt = (gt_mask!=0).type_as(gt_mask)

        pred_skel = soft_skeletonize(binary_pred,k=self.k)

        t_prec = (pred_skel*binary_gt + self.eps).sum(dim=self.sum_dims)/(pred_skel.sum(dim=self.sum_dims) +self.eps)
        t_rec = (gt_skel*binary_pred + self.eps).sum(dim=self.sum_dims)/(gt_skel.sum(dim=self.sum_dims) +self.eps)
        
        cldice = 2*((t_prec*t_rec)/(t_prec+t_rec))
        cldice_loss = 1 - cldice.mean()
        return cldice_loss
class DiceLoss(nn.Module):
    def __init__(self,eps,sum_dims):
        super(DiceLoss,self).__init__()
        self.eps = eps
        self.sum_dims = sum_dims
    def forward(self,pred_probs,gt):
        tp = (gt * pred_probs).sum(dim=self.sum_dims)
        fp = ((1-gt)*pred_probs).sum(dim=self.sum_dims)
        fn = ((1-pred_probs)*gt).sum(dim=self.sum_dims)
        per_class_dice_score = (2*tp +self.eps)/(2*tp + fp + fn + self.eps)
        # if(present_class is None):
        #     dice_loss = -per_class_dice_score.mean()
        # else:
        #     dice_loss = -per_class_dice_score[present_class].mean()
        dice_loss = -per_class_dice_score.mean()
        return dice_loss

class TverskyLoss(nn.Module):
    def __init__(self,eps,sum_dims,alpha=0.3,beta=0.7,gamma=1.33):
        super(TverskyLoss,self).__init__()
        self.eps = eps
        self.sum_dims = sum_dims
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
    def forward(self,pred_probs,gt):
        tp = (gt * pred_probs).sum(dim=self.sum_dims)
        fp = ((1-gt)*pred_probs).sum(dim=self.sum_dims)
        fn = ((1-pred_probs)*gt).sum(dim=self.sum_dims)
        t_index = (tp + self.eps) / (tp + self.alpha*fp + self.beta*fn + self.eps) 

        t_index = t_index.mean()
        
        return (1 - t_index)**self.gamma 

    
    
