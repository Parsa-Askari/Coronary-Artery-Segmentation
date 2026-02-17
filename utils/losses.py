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
        self.training_mode = args["training_mode"]
        self.eps = eps
        self.args = args
        self.sum_dims = (0,2,3)
        
        self.softmax = nn.Softmax(dim=1)
        ### masks cross-entropy loss (SHARED)
        if(args["mask_ce_weights"] is not None):
            w = torch.tensor(
                args["mask_ce_weights"],
                dtype=torch.float32,
                device="cuda"
            )
            self.ce_fn = nn.CrossEntropyLoss(weight=w)
                
        else :
            self.ce_fn = nn.CrossEntropyLoss()
        ### masks dice loss (SHARED)
        if(self.loss_type=="dice"):
                print("loss is set to dice")
                self.loss_fn = DiceLoss(self.eps,self.sum_dims)
        elif(self.loss_type=="tversky"):
            print("loss is set to tversky")
            self.loss_fn = TverskyLoss(
                eps = self.eps,
                sum_dims= self.sum_dims,
                args = args["tversky_conf"])
        ### JUST BINARY

        if(self.training_mode=="binary"):
            self.cl_dice_loss_fn = CLDiceLoss(
                eps=eps,
                sum_dims=self.sum_dims,
                k=args["cl_dice_conf"]["k"]
            )

    def forward(self,pred_masks,gt_masks):
        if(self.training_mode=="binary"):
            """
            inputs : pred_masks , gt_masks
                - pred_masks : (B,2,H,W)
                - gt_masks : (B,H,W)

            """
            onehot_gt_masks = F.one_hot(gt_masks, num_classes=self.class_count)
            onehot_gt_masks = onehot_gt_masks.permute(0, 3, 1, 2).float()

            pred_probs = self.softmax(pred_masks)

            ### Cross Entropy Loss
            mask_ce_loss = self.ce_fn(pred_masks,gt_masks)
            ### Mask Dice Losses
            fg_pred_probs = pred_probs[:,1:]
            fg_onehot_gt_masks = onehot_gt_masks[:,1:]
            dice_family_loss = self.loss_fn(
                pred_probs = fg_pred_probs,
                gt = fg_onehot_gt_masks
            )
            ### Mask CL-Dice
            cl_dice_loss = self.cl_dice_loss_fn(
                pred_binary_mask = pred_probs[:,1:],
                binary_gt_mask = gt_masks.unsqueeze(1).float()
            )
            ### Recordings
            total_loss = (
                self.args["mask_ce_conf"]["imp_coef"]*mask_ce_loss 
                + self.args[f"{self.loss_type}_conf"]["imp_coef"]*dice_family_loss 
                + self.args["cl_dice_conf"]["imp_coef"]*cl_dice_loss
            )

            loss_dict = {
                "mask CE-Loss" : mask_ce_loss,
                self.loss_type : dice_family_loss,
                "cl dice loss" : cl_dice_loss
            }
        elif(self.training_mode=="multi-class"):
            pass
        
        
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
        self.sum_dims = (1,2,3)

    def forward(self,pred_binary_mask , binary_gt_mask):

        pred_skel = soft_skeletonize(pred_binary_mask,k=self.k)
        gt_skel = soft_skeletonize(binary_gt_mask,k=self.k)

        t_prec = (pred_skel*binary_gt_mask ).sum(dim=self.sum_dims)/(pred_skel.sum(dim=self.sum_dims) +self.eps)
        t_rec = (gt_skel*pred_binary_mask ).sum(dim=self.sum_dims)/(gt_skel.sum(dim=self.sum_dims) +self.eps)
        
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
    
    
    
