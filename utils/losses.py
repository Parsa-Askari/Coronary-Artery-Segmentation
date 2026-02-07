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
        self.alpha = args["alpha"]
        self.beta = args["beta"]
        self.t_gamma = args["t_gamma"]
        self.mask_ce_weights = args["mask_ce_weights"]
        self.k = args["k"]
        # self.focal_fn = FocalCrossEntropy(
        #     f_gamma=self.f_gamma,
        #     eps=eps,
        #     mask_ce_weights=args["mask_ce_weights"],
        #     f_loss_scale = args["f_loss_scale"]
        # )
        self.stop_ce = nn.BCEWithLogitsLoss()
        if(args["mask_ce_weights"] is not None):
            w = torch.tensor(
                args["mask_ce_weights"],
                dtype=torch.float32,
                device="cuda"
            )
            self.ce_fn = nn.CrossEntropyLoss(weight=w)
        else :
            self.ce_fn = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)
        self.eps = eps
        self.sum_dims = (0,3,4)
        if(self.loss_type=="dice loss"):
            print("loss is set to dice")
            self.loss_fn = DiceLoss(self.eps,self.sum_dims)
        elif(self.loss_type=="tversky loss"):
            print("loss is set to tversky")
            self.loss_fn = TverskyLoss(
                eps = self.eps,
                sum_dims= self.sum_dims,
                alpha = self.alpha,
                beta = self.beta,
                gamma = self.t_gamma,
                t_alpha = args["t_alpha"]
                )
        self.cldice_fn = CLDiceLoss(sum_dims=self.sum_dims,eps=self.eps,k=self.k) 
    def forward(self,masks,pred_masks,stop_labels,pred_stop_labels,
                batch_size=None,seq_len=None):
        """
        masks : (BxT,C,H,W)
        pred_masks : (BxT,class_counts,H,W)
        stop_labels : (BxT)
        pred_stop_labels : (BxT)
        """
        onehot_masks = F.one_hot(masks, num_classes=self.class_count)
        onehot_masks = onehot_masks.permute(0, 3, 1, 2).float()  

        stop_labels = stop_labels.reshape(-1).float()
        pred_stop_labels = pred_stop_labels.reshape(-1)

        prob = self.softmax(pred_masks)
        
        # Cross Entropy Loss
        ce_loss = self.ce_fn(pred_masks,masks)
        # Dice/Tversky Loss
        forground_probs = prob[:,1:]
        forground_onehot_masks = onehot_masks[:,1:]
        # present_class = forground_onehot_mask.sum(dim=self.sum_dims)>0
        
        second_loss , class_wise_loss = self.loss_fn(
            pred_probs = forground_probs,
            gt = forground_onehot_masks,
            batch_size = batch_size,
            seq_len = seq_len
        )
        # stop_loss = self.stop_ce(pred_stop_labels,stop_labels)
        total_loss = ce_loss + second_loss #+ (0.1*stop_loss)

        loss_dict = {
            "CE Loss" : ce_loss,
            # "Stop Loss":stop_loss,
           self.loss_type : second_loss
        }
        return total_loss , loss_dict , class_wise_loss

class FocalCrossEntropy(nn.Module):
    def __init__(self,f_gamma,eps,f_loss_scale=1,mask_ce_weights=None):
        super(FocalCrossEntropy,self).__init__()
        self.f_gamma = f_gamma
        if(mask_ce_weights is not None):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.mask_ce_weights = torch.tensor(mask_ce_weights).to(device)
        else:
            self.mask_ce_weights = mask_ce_weights
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
    def __init__(self, eps=1e-6, sum_dims=(0,2,3), alpha=0.3, beta=0.7, gamma=1.33, t_alpha=None):
        super().__init__()
        self.eps = eps
        self.sum_dims = sum_dims
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        device = "cuda" if torch.cuda.is_available() else "cput"
        if t_alpha is not None:
            self.register_buffer("t_alpha", torch.tensor(t_alpha, dtype=torch.float32,device=device))
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
            return loss_c.mean() , loss_c.detach().cpu().numpy()
        w = self.t_alpha
        return (w * loss_c).sum() / (w.sum() + self.eps) , loss_c.detach().cpu().numpy()
    
    
    
