import torch.nn as nn
import torch.nn.functional as F
import torch
###IE###
from .helpers import soft_skeletonize
###SS###
class MainLossFn(nn.Module):
    def __init__(self,args,eps = 1e-8):
        super(MainLossFn,self).__init__()
        class_count = args["class_count"]
        abs_class_count = args["abs_class_count"]
        self.loss_type = args["loss_type"]
        self.alpha = args["alpha"]
        self.beta = args["beta"]
        self.t_gamma = args["t_gamma"]
        self.f_gamma = args["f_gamma"]
        self.k = args["k"]
        self.loss_coefs = args["loss_coefs"]
        self.just_binary_trining = args["just_binary_trining"]
        remove_bg = args["remove_bg"]
        entropy_fn = FocalCrossEntropy(
            f_gamma=self.f_gamma,
            eps=eps,
            f_alpha=args["f_alpha"]
        )
        # if(args["f_alpha"] is not None):
        #     w = torch.tensor(args["f_alpha"],dtype=torch.float32,device="cuda")
        #     self.entropy_fn = nn.CrossEntropyLoss(weight=w)
        # else :
        #     self.entropy_fn = nn.CrossEntropyLoss()
        
        self.eps = eps
        self.sum_dims = (0,2,3)
        if(self.loss_type=="dice loss"):
            print("loss is set to dice")
            multi_loss_fn = DiceLoss(self.eps,self.sum_dims)
        elif(self.loss_type=="tversky loss"):
            print("loss is set to tversky")
            multi_loss_fn = TverskyLoss(self.eps,self.sum_dims,self.alpha,
                                        self.beta,self.t_gamma)
            
        cldice_fn = CLDiceLoss(sum_dims=self.sum_dims,eps=self.eps,k=self.k) 
        
        bce_fn = nn.BCEWithLogitsLoss()
        self.mask_loss_fn = MultiClassLoss(
            class_count=class_count,
            multi_loss_fn=multi_loss_fn,
            entropy_fn=entropy_fn
        )
        
        self.abs_mask_loss_fn = MultiClassLoss(
            class_count=abs_class_count,
            multi_loss_fn=multi_loss_fn,
            entropy_fn=entropy_fn
        )
        
        self.binary_mask_loss_fn =BinaryClassLoss(
            cldice_fn=cldice_fn,
            dice_fn=TverskyLoss(
                self.eps,
                self.sum_dims,
                self.alpha,
                self.beta,
                self.t_gamma
            ),
            remove_bg = remove_bg
            # dice_fn=DiceLoss(self.eps,self.sum_dims)
        )
        
        self.label_loss_fn = SideLoss(
            entropy_fn=bce_fn
        )
        
        
        
    def forward(self,preds , ground_truths ):
        if(self.just_binary_trining):
            gt_binary_mask= ground_truths[0]
            pred_binary_mask = preds[0]
            binary_mask_loss , dice_loss , cldice_loss , bce_loss = self.binary_mask_loss_fn(
                pred_binary_mask = pred_binary_mask,
                gt_binary_mask = gt_binary_mask
            )
            total_loss = (
                binary_mask_loss 
            )
            loss_dict = {
                "binary loss" : binary_mask_loss,
                "bianry cldice loss ": cldice_loss,
                "binary dice loss":dice_loss,
                "binary BCE loss":bce_loss
            }
        else:
            gt_side_label,gt_binary_mask,gt_abs_mask,gt_mask = ground_truths
            pred_side_label ,pred_abs_mask, pred_mask = preds
        
            label_loss = self.label_loss_fn(
                pred_label = pred_side_label,
                gt_label=gt_side_label
            )
            

            abs_mask_loss = self.abs_mask_loss_fn(
                pred_mask = pred_abs_mask,
                gt_mask = gt_abs_mask
            )

            mask_loss = self.mask_loss_fn(
                pred_mask = pred_mask,
                gt_mask = gt_mask
            )
            loss_dict = {
                "label loss" : label_loss,
                f"{self.loss_type}_main" : mask_loss,
                f"{self.loss_type}_abs" : abs_mask_loss,
            }
        
        return total_loss , loss_dict
    
class MultiClassLoss(nn.Module):
    def __init__(self,class_count,multi_loss_fn,entropy_fn):
        super(MultiClassLoss,self).__init__()
        self.class_count = class_count
        self.softmax = nn.Softmax(dim=1)
        self.multi_loss_fn = multi_loss_fn
        self.entropy_fn = entropy_fn
    def forward(self,pred_mask,gt_mask):

        onehot_mask = F.one_hot(gt_mask, num_classes=self.class_count)
        onehot_mask = onehot_mask.permute(0, 3, 1, 2).float()  
        prob = self.softmax(pred_mask)

        forground_prob = prob[:,1:]
        forground_onehot_mask = onehot_mask[:,1:]

        second_loss = self.multi_loss_fn(
            pred_probs = forground_prob,
            gt = forground_onehot_mask
        )

        en_loss = self.entropy_fn(prob,onehot_mask)

        return second_loss + en_loss
    
class SideLoss(nn.Module):
    def __init__(self,entropy_fn):
        super(SideLoss,self).__init__()    
        self.entropy_fn = entropy_fn
    def forward(self,pred_label,gt_label):
        en_loss = self.entropy_fn(
            pred_label.reshape(-1),
            gt_label.reshape(-1).float()
        )
        return en_loss

class BinaryClassLoss(nn.Module):
    def __init__(self,cldice_fn,dice_fn,remove_bg):
        super(BinaryClassLoss,self).__init__()
        self.cldice_fn = cldice_fn
        self.dice_fn = dice_fn
        self.coef =0.4
        self.softmax =nn.Softmax(dim=1)
        self.cross_etropy = nn.CrossEntropyLoss()
        self.remove_bg = remove_bg
    def forward(self,pred_binary_mask,gt_binary_mask):
        bce_loss = self.cross_etropy(
            pred_binary_mask,
            gt_binary_mask.squeeze(1)
        )
        onehot_mask = F.one_hot(gt_binary_mask.squeeze(1), num_classes=2) # B,H,W,2
        onehot_gt_binary_mask = onehot_mask.permute(0, 3, 1, 2).float()  # B,2,H,W
        prob = self.softmax(pred_binary_mask)

        if(self.remove_bg):
            cldice_loss = self.cldice_fn(
                binary_pred = prob[:,1:2,...],
                binary_gt = gt_binary_mask.float()
            )
        else :     
            cldice_loss_fg = self.cldice_fn(
                binary_pred = prob[:,1:2,...],
                binary_gt = onehot_gt_binary_mask[:,1:2,...]
            )
            cldice_loss_bg = self.cldice_fn(
                binary_pred = prob[:,0:1,...],
                binary_gt = onehot_gt_binary_mask[:,0:1,...]
            )
            cldice_loss = 0.5*(cldice_loss_fg+cldice_loss_bg)

        dice_loss = self.dice_fn(
            pred_probs = prob,
            gt = onehot_gt_binary_mask
        )
        loss = (
            bce_loss + 
            (1-self.coef)*dice_loss + 
            (self.coef)*cldice_loss
        )
        return loss, dice_loss , cldice_loss , bce_loss
    
class FocalCrossEntropy(nn.Module):
    def __init__(self,f_gamma,eps,f_alpha=None):
        super(FocalCrossEntropy,self).__init__()
        self.f_gamma = f_gamma
        if(f_alpha is not None):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.f_alpha = torch.tensor(f_alpha).to(device)
        else:
            self.f_alpha = f_alpha
        self.eps = eps
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
        return -(class_w*focal_loss).mean()

class CLDiceLoss(nn.Module):
    def __init__(self,eps,sum_dims,k=40):
        super(CLDiceLoss,self).__init__()
        self.k=k
        self.eps = eps
        self.sum_dims = sum_dims

    def forward(self,binary_pred , binary_gt):
        # print(binary_pred.shape,binary_gt.shape,gt_skel.shape)
        pred_skel = soft_skeletonize(binary_pred,k=self.k)
        gt_skel = soft_skeletonize(binary_gt,k=self.k)
        
        num_prec = (pred_skel * binary_gt).sum(dim=self.sum_dims) + self.eps
        den_prec = pred_skel.sum(dim=self.sum_dims) + self.eps
        t_prec   = num_prec / den_prec

        num_rec = (gt_skel * binary_pred).sum(dim=self.sum_dims) + self.eps
        den_rec = gt_skel.sum(dim=self.sum_dims) + self.eps
        t_rec   = num_rec / den_rec

        cldice = 2 * ( (t_prec * t_rec) / (t_prec + t_rec + self.eps) )
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
        dice_loss = 1 - per_class_dice_score.mean()
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

    
    
