import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.ops.focal_loss import sigmoid_focal_loss
from monai.losses import FocalLoss
###IE###
from .helpers import soft_skeletonize
###SS###
class UnetLoss(nn.Module):
    def __init__(self,args,eps = 1e-8):
        super(UnetLoss,self).__init__()
        self.class_count = args["class_count"]
        self.loss_type = args["loss_type"]
        self.mask_ce_weights = args["mask_ce_weights"]
        self.eps = eps
        self.args = args
        self.sum_dims = (0,2,3)
        
        # self.focal_fn = FocalCrossEntropy(
        #     f_gamma=self.f_gamma,
        #     eps=eps,
        #     args=args["focal_ce_conf"]
        # )
        self.softmax = nn.Softmax(dim=1)
        self.current_label_loss_fn = nn.CrossEntropyLoss()
        if(args["mask_ce_weights"] is not None):
            w = torch.tensor(
                args["mask_ce_weights"],
                dtype=torch.float32,
                device="cuda"
            )
            self.ce_fn = nn.CrossEntropyLoss(weight=w)
            
        else :
            self.ce_fn = nn.CrossEntropyLoss()

        if(self.loss_type=="dice"):
            print("loss is set to dice")
            self.loss_fn = DiceLoss(self.eps,self.sum_dims)
        elif(self.loss_type=="tversky"):
            print("loss is set to tversky")
            self.loss_fn = TverskyLoss(
                eps = self.eps,
                sum_dims= self.sum_dims,
                args = args["tversky_conf"]
            )
    def forward(self,pred_m_labels,gt_m_labels,pred_current_labels,gt_current_labels):
        """
        pred_m_labels : (BxT,class_counts,H,W)
        gt_m_labels : (BxT,H,W)
        pred_current_labels : (BxT,class_counts)
        gt_current_labels : (BxT,class_counts)
        """

        onehot_gt_masks = F.one_hot(gt_m_labels, num_classes=self.class_count)
        onehotgt__masks = onehot_gt_masks.permute(0, 3, 1, 2).float()

        gt_current_labels = torch.argmax(gt_current_labels,dim=1)

        prob = self.softmax(pred_m_labels)
        
        ### Mask Losses
        # Cross Entropy Loss 
        mask_ce_loss = self.ce_fn(pred_m_labels,gt_m_labels)
        # Dice/Tversky Loss
        forground_probs = prob[:,1:]
        forground_onehot_masks = onehotgt__masks[:,1:]
        second_loss = self.loss_fn(
            pred_probs = forground_probs,
            gt = forground_onehot_masks
        )
        ## Current Label Losses
        current_label_loss = self.current_label_loss_fn(
            pred_current_labels,
            gt_current_labels
        )

        total_loss = (
            self.args["mask_ce_conf"]["imp_coef"]*mask_ce_loss 
            + self.args[f"{self.loss_type}_conf"]["imp_coef"]*second_loss 
            + self.args["label_ce_conf"]["imp_coef"]*current_label_loss
        )

        loss_dict = {
            "mask CE-Loss" : mask_ce_loss,
            "label CE-Loss": current_label_loss,
            self.loss_type : second_loss
        }
        return total_loss , loss_dict

class FocalCrossEntropy(nn.Module):
    def __init__(self,args,eps,mask_ce_weights=None):
        super(FocalCrossEntropy,self).__init__()
        self.eps = eps
        self.f_loss_scale = args["loss_scale"]
        self.f_gamma = args["gamma"]
        if(mask_ce_weights is not None):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.mask_ce_weights = torch.tensor(mask_ce_weights).to(device)
        else:
            self.mask_ce_weights = mask_ce_weights

    def forward(self,prob,onehot_mask):
        # prob : (B,C,H,W)
        # onehot_mask : (B,C,H,W)
        # gt_mask = (B,H,W)

        p = (prob*onehot_mask).sum(dim=1) # (B,H,W)
        pt = torch.clamp(p,self.eps,1-self.eps)
        focal_weights = (1-pt)**self.f_gamma
        focal_loss = focal_weights*(torch.log(pt))
        if(self.mask_ce_weights is not None):
            alpha_b = self.mask_ce_weights.view(1, -1, 1, 1).type_as(prob)
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
    def __init__(self, args,eps=1e-6, sum_dims=(0,2,3) ):
        super().__init__()
        self.eps = eps
        self.sum_dims = sum_dims
        self.alpha = args["alpha"]
        self.beta = args["beta"]
        self.gamma = args["gamma"]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if args["weights"] is not None:
            self.register_buffer("t_alpha", torch.tensor(args["weights"], dtype=torch.float32,device=device))
        else:
            self.t_alpha = None

    def forward(self, pred_probs, gt,batch_size=None,seq_len =None):
        
        if(batch_size is not None):
            pred_probs = pred_probs.reshape((batch_size,seq_len,*pred_probs.shape[1:]))
            gt = gt.reshape((batch_size,seq_len,*gt.shape[1:]))

        gt = gt.float()
        tp = (gt * pred_probs).sum(dim=self.sum_dims)
        fp = ((1 - gt) * pred_probs).sum(dim=self.sum_dims)
        fn = (gt * (1 - pred_probs)).sum(dim=self.sum_dims)
        ti = (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)
        
        loss_c = (1 - ti).clamp_min(0) ** self.gamma
        if(batch_size is not None):
            
            loss_c = loss_c.reshape(-1)
        if self.t_alpha is None:
            return loss_c.mean() 
        w = self.t_alpha
        return (w * loss_c).sum() / (w.sum() + self.eps) 
    
    
    
