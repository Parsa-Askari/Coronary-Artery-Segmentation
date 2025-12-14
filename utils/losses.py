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
        self.binary_type = args["binary_type"]
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
            clce_fn = CLCELoss(
                k=self.k,
                eps = eps,
                sum_dims = self.sum_dims
            ),
            remove_bg = remove_bg
            # dice_fn=DiceLoss(self.eps,self.sum_dims)
        )
        
        self.label_loss_fn = SideLoss(
            entropy_fn=bce_fn
        )
        
        
        
    def forward(self,pred , ground_truth ):
        if(self.just_binary_trining):
            binary_mask_loss , dice_loss  , bce_loss = self.binary_mask_loss_fn(
                pred_binary_mask = pred,
                gt_binary_mask = ground_truth
            )
            total_loss = (
                binary_mask_loss 
            )
            loss_dict = {
                "binary loss" : binary_mask_loss,
                "binary dice loss":dice_loss,
                "binary BCE loss":bce_loss,
                # "bianry cldice loss ": cldice_loss,
                # "binary clce loss" : clce_loss
            }
        else:
            total_loss , second_loss , en_loss = self.mask_loss_fn(
                pred_mask = pred,
                gt_mask = ground_truth.squeeze(1)
            )
            loss_dict = {
                "dice loss" : second_loss,
                "CE loss":en_loss,
                # "bianry cldice loss ": cldice_loss,
                # "binary clce loss" : clce_loss
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

        return second_loss + en_loss , second_loss , en_loss
    
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
    def __init__(self,cldice_fn,clce_fn,dice_fn,remove_bg):
        super(BinaryClassLoss,self).__init__()
        self.cldice_fn = cldice_fn
        self.dice_fn = dice_fn
        self.coef =0.4
        self.softmax =nn.Softmax(dim=1)
        # self.cross_etropy = nn.CrossEntropyLoss()
        self.cross_etropy = FocalCrossEntropy(
            f_gamma=1.5,
            eps=1e-6,
            f_alpha=[1.0,6.0]
        )
        self.clce_fn = clce_fn
        self.remove_bg = remove_bg
    def forward(self,pred_binary_mask,gt_binary_mask):
        # bce_loss = self.cross_etropy(
        #     pred_binary_mask,
        #     gt_binary_mask.squeeze(1)
        # )

        onehot_mask = F.one_hot(gt_binary_mask.squeeze(1), num_classes=2) # B,H,W,2
        onehot_gt_binary_mask = onehot_mask.permute(0, 3, 1, 2).float()  # B,2,H,W
        prob = self.softmax(pred_binary_mask)


        bce_loss = self.cross_etropy(
            prob = prob,
            onehot_mask = onehot_gt_binary_mask
        )
        # clce_loss = self.clce_fn(
        #     pred_binary_mask = pred_binary_mask,
        #     binary_gt = onehot_gt_binary_mask
        # )
        if(self.remove_bg):
            # cldice_loss = self.cldice_fn(
            #     binary_pred = prob[:,1:2,...],
            #     binary_gt = gt_binary_mask.float()
            # )
            dice_loss = self.dice_fn(
                pred_probs = prob[:,1:2,...],
                gt = gt_binary_mask
            )
        else :     
            # cldice_loss_fg = self.cldice_fn(
            #     binary_pred = prob[:,1:2,...],
            #     binary_gt = onehot_gt_binary_mask[:,1:2,...]
            # )
            # cldice_loss_bg = self.cldice_fn(
            #     binary_pred = prob[:,0:1,...],
            #     binary_gt = onehot_gt_binary_mask[:,0:1,...]
            # )
            # cldice_loss = 0.5*(cldice_loss_fg+cldice_loss_bg)

            dice_loss = self.dice_fn(
                pred_probs = prob,
                gt = onehot_gt_binary_mask
            )
        loss = (
            bce_loss + 
            dice_loss 
            # clce_loss
            
        )
        ##(self.coef)*cldice_loss
        return loss, dice_loss , bce_loss #, clce_loss
    
class CLCELoss(nn.Module):
    def __init__(self, eps, sum_dims, k=40):
        super(CLCELoss, self).__init__()
        self.k = k
        self.eps = eps
        self.sum_dims = sum_dims

    def forward(self, pred_binary_mask, binary_gt):
        gt_idx = binary_gt.argmax(dim=1)

        cross_ent = F.cross_entropy(pred_binary_mask, gt_idx, reduction="none")

        probs = pred_binary_mask.softmax(dim=1)

        pred_skel = soft_skeletonize(probs, k=self.k)
        gt_skel   = soft_skeletonize(binary_gt, k=self.k)

        tprec = (cross_ent * gt_skel[:, 1]).mean(dim=(1,2))
        tsens = (cross_ent * pred_skel[:, 1]).mean(dim=(1,2))
        return (tprec + tsens).sum()
    
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

    
    
