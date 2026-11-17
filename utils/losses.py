import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import torch
# from monai.losses import FocalLoss as GnnFocalLoss
###IE###
from .helpers import soft_skeletonize ,popov_edge_map , improved_vessel_map
###SS###


class UnetLoss(nn.Module):
    def __init__(self,args,eps = 1e-8):
        super(UnetLoss,self).__init__()
        self.class_count = args["class_count"]
        self.loss_type = args["loss_type"]
        self.mask_ce_weights = args["mask_ce_weights"]
        self.training_mode = args["training_mode"]
        self.binary_training_mode = args["binary_training_mode"]
        self.eps = eps
        self.args = args
        self.sum_dims = (0,2,3)
        
        self.softmax = nn.Softmax(dim=1)
        
        if(self.training_mode=="binary" or self.training_mode=="branch"):
            self.build_binary_losses(args,eps)
        elif(self.training_mode=="multi"):
            self.build_binary_losses(args,eps)
        elif(self.training_mode=="cleaning"):
            self.build_binary_losses(args,eps)
        else:
            raise ValueError("training mode is not defined")
        
        ### JUST BINARY

            
    def build_binary_losses(self,args,eps):
        
        if(self.binary_training_mode=="full"):
            ### masks cross-entropy loss 
            self.ce_fn = VesselnessRelaxedCELoss(
                expert_vessel_pos_weight = args["mask_frangi_ce_conf"]["expert_vessel_pos_weight"],
                model_vessel_pos_weight = args["mask_frangi_ce_conf"]["model_vessel_pos_weight"],
                unlabeled_vessel_weight = args["mask_frangi_ce_conf"]["unlabeled_vessel_weight"],
                bg_weight = args["mask_frangi_ce_conf"]["bg_weight"],
                kernel_size = args["mask_frangi_ce_conf"]["kernel_size"],
                outer_boundary_boost = args["mask_frangi_ce_conf"]["outer_boundary_boost"],
                use_expert_boundary_only = args["mask_frangi_ce_conf"]["use_expert_boundary_only"]
            )
            # self.ce_fn = PopovWeightedEdgeBCELoss(
            #     vessel_pos_weight = args["mask_frangi_ce_conf"]["vessel_pos_weight"],
            #     edge_boost = args["mask_frangi_ce_conf"]["unlabeled_vessel_weight"],
            #     bg_weight = args["mask_frangi_ce_conf"]["bg_weight"]
            # )

        else:
            ### masks cross-entropy loss 
            if(args["mask_ce_weights"] is None):
                args["mask_ce_weights"] = [1]*self.class_count
                
            # w = torch.tensor(
            #     args["mask_ce_weights"],
            #     dtype=torch.float32,
            #     device="cuda"
            # )
            # self.ce_fn = nn.CrossEntropyLoss(weight=w)
            if(self.training_mode=="branch"):
                self.ce_fn  = CostumeBranchCELoss(args["ce_loss_conf"])
            elif(self.training_mode=="binary"):
                # self.ce_fn  = CostumeCELoss(args["ce_loss_conf"])
                
                # self.ce_fn = BinaryFocalLoss(args["fce_conf"])
                self.ce_fn = FocalLoss(args=args["fce_conf"])
            else:
                self.ce_fn = FocalLoss(args["fce_conf"])
            ### masks dice loss
        if(self.loss_type=="dice"):
            print("loss is set to dice")
            self.loss_fn = DiceLoss(self.eps,self.sum_dims)
        elif(self.loss_type=="tversky"):
            print("loss is set to tversky")
            if(self.binary_training_mode=="full"):
                self.loss_fn = VesselnessRelaxedTverskyLoss(
                    expert_vessel_pos_weight = args["mask_frangi_dice_conf"]["expert_vessel_pos_weight"],
                    model_vessel_pos_weight = args["mask_frangi_dice_conf"]["model_vessel_pos_weight"],
                    unlabeled_vessel_weight = args["mask_frangi_dice_conf"]["unlabeled_vessel_weight"],
                    bg_weight = args["mask_frangi_dice_conf"]["bg_weight"],
                    alpha = args["mask_frangi_dice_conf"]["alpha"],
                    beta = args["mask_frangi_dice_conf"]["beta"],
                    outer_boundary_boost = args["mask_frangi_dice_conf"]["outer_boundary_boost"],
                    kernel_size = args["mask_frangi_dice_conf"]["kernel_size"],
                    use_expert_boundary_only = args["mask_frangi_dice_conf"]["use_expert_boundary_only"]
                )
                # self.loss_fn = TverskyLoss(
                #     eps = self.eps,
                #     sum_dims= self.sum_dims,
                #     args = args["tversky_conf"]
                # )
            else:
                self.loss_fn = TverskyLoss(
                    eps = self.eps,
                    sum_dims= self.sum_dims,
                    args = args["tversky_conf"]
                )
        elif(self.loss_type =="g_dice"):
            print("loss is set to g_dice")
            self.loss_fn = GeneralizedDiceLoss(
                eps = self.eps,
                sum_dims= self.sum_dims
            )
            
            
        # CL-dice
        # self.cl_dice_loss_fn = CLDiceLoss(
        #     eps=eps,
        #     k=args["cl_dice_conf"]["k"]
        # )
        self.recall_skel_loss_fn = SoftSkeletonRecallLoss(
            apply_nonlin=lambda z: torch.softmax(z, 1), 
            smooth=1e-5
        )

        # PARENT = {2:1, 3:2, 4:3, 5:1, 6:5}
        # self.betti_loss_fn = Betti0Loss(parent=PARENT)
    def forward(
            self,
            pred_masks,
            gt_masks,
            skel_masks=None,
            aux_loss=None,
            ignore_mask=None):
        
        """
        inputs : pred_masks , gt_masks
            - pred_masks : (B,2,H,W)
            - gt_masks : (B,H,W)
            - expert_masks : (B,H,W)
            - v_masks : (B,H,W)

        """
        
        
        onehot_gt_masks = F.one_hot(gt_masks, num_classes=self.class_count)
        onehot_gt_masks = onehot_gt_masks.permute(0, 3, 1, 2).float()

        pred_probs = self.softmax(pred_masks)

        ### Cross Entropy Loss

        mask_ce_loss = self.ce_fn(
            inputs = pred_masks,
            targets = gt_masks.long(),
            ignore_mask = ignore_mask
        )
        
        # mask_ce_loss = self.ce_fn(
        #     inputs = pred_masks, 
        #     targets = gt_masks.long()
        # )
            

        fg_pred_probs = pred_probs[:,1:]
        fg_onehot_gt_masks = onehot_gt_masks[:,1:]
        if(self.loss_type == "tversky"):
            dice_family_loss = self.loss_fn(
                pred_probs = fg_pred_probs,
                gt = fg_onehot_gt_masks,
                ignore_mask = ignore_mask
            )
        elif(self.loss_type == "g_dice"):
            dice_family_loss = self.loss_fn(
                pred_probs = pred_probs,
                gt = onehot_gt_masks,
                ignore_mask = ignore_mask
            )
        ## Mask Dice Losses
        # if(skel_masks is not None):
        #     onehot_gt_skels = F.one_hot(skel_masks, num_classes=self.class_count)
        #     onehot_gt_skels = onehot_gt_skels.permute(0, 3, 1, 2).float()#[:,1:]
        #     cl_dice_loss = self.cl_dice_loss_fn(
        #         pred_probs = pred_probs,
        #         gt_mask = onehot_gt_masks,
        #         gt_skel = onehot_gt_skels
        #     )
        ### Mask CL-Dice
        # recall_skel_loss = self.recall_skel_loss_fn(
        #     x = pred_masks, 
        #     skel = skel_masks
        # )
        if(ignore_mask is not None):
            loss_mask = (~ignore_mask.unsqueeze(1).bool()).float()
        else: 
            loss_mask = None
            
        recall_skel_loss = self.recall_skel_loss_fn(pred_masks, gt_masks,loss_mask=loss_mask)
        ### Recordings
        total_loss = (
            self.args["ce_loss_conf"]["imp_coef"]*mask_ce_loss
            + self.args[f"{self.loss_type}_conf"]["imp_coef"]*dice_family_loss 
            # + recall_skel_loss
            + self.args["recall_skel_coef"]["imp_coef"]*recall_skel_loss
            # + self.args["cl_dice_conf"]["imp_coef"]*cl_dice_loss
        )

        loss_dict = {
            "mask CE-Loss" : self.args["ce_loss_conf"]["imp_coef"] * mask_ce_loss,
            # "cl_dice_loss" : self.args["cl_dice_conf"]["imp_coef"] * cl_dice_loss,
            self.loss_type : self.args[f"{self.loss_type}_conf"]["imp_coef"] * dice_family_loss,
            # "recall skel" : recall_skel_loss
            "recall skel": self.args["recall_skel_coef"]["imp_coef"]*recall_skel_loss
        }

        
        
        
        return total_loss , loss_dict


class SoftSkeletonRecallLoss(nn.Module):
    """
    x    : (B, C, H, W) raw logits
    skel : (B, 1, H, W) or (B, H, W) int label map from make_skeleton_target
    Returns NEGATIVE mean recall -> range [-1, 0], lower is better.
    """

    def __init__(self, apply_nonlin=None, batch_dice: bool = False,
                 do_bg: bool = False, smooth: float = 1e-5):
        super().__init__()
        if do_bg:
            raise RuntimeError("skeleton recall does not work with background")
        self.apply_nonlin = apply_nonlin
        self.batch_dice = batch_dice
        self.smooth = smooth

    def forward(self, x, skel, loss_mask=None):
        shp_x, shp_y = x.shape, skel.shape

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)
        x = x[:, 1:]                                   # drop background

        axes = list(range(2, len(shp_x)))

        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                skel = skel.view((shp_y[0], 1, *shp_y[1:]))
            if all(i == j for i, j in zip(shp_x, shp_y)):
                y_onehot = skel[:, 1:]
            else:
                gt = skel.long()
                y_onehot = torch.zeros(shp_x, device=x.device, dtype=torch.float32)
                y_onehot.scatter_(1, gt, 1)
                y_onehot = y_onehot[:, 1:]
            sum_gt = (y_onehot.sum(axes) if loss_mask is None
                      else (y_onehot * loss_mask).sum(axes))

        inter_rec = ((x * y_onehot).sum(axes) if loss_mask is None
                     else (x * y_onehot * loss_mask).sum(axes))

        if self.batch_dice:
            inter_rec = inter_rec.sum(0)
            sum_gt = sum_gt.sum(0)

        rec = (inter_rec + self.smooth) / torch.clip(sum_gt + self.smooth, 1e-8)
        return -rec.mean()
    
class BinaryFocalLoss(nn.Module):
    # Alpha tuned to empirical 0.0358 fg/bg ratio
    def __init__(self,args):
        super().__init__()
        self.alpha = args["alpha"]
        self.gamma = args["gamma"]

    def forward(self, inputs, targets):
        
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.gamma = args["gamma"]
        self.extra_w = args.get("extra_w", 1)

        alpha = args.get("alpha")
        if alpha is not None and not torch.is_tensor(alpha):
            print("alpha is set for focal cross entropy : ", alpha)
            alpha = torch.tensor(alpha, dtype=torch.float32,device ="cuda")
        self.register_buffer("alpha", alpha)

    def forward(self, inputs, targets, hard_mask=None, ignore_mask=None):
        if targets.ndim == inputs.ndim:
            targets = targets.squeeze(1)
        targets = targets.long()

        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            focal_loss = self.alpha[targets] * focal_loss

        if hard_mask is not None:
            if hard_mask.ndim == focal_loss.ndim + 1:
                hard_mask = hard_mask.squeeze(1)
            w = torch.ones_like(focal_loss)
            w[hard_mask == 1] = self.extra_w
            focal_loss = focal_loss * w

        if ignore_mask is None:
            return focal_loss.mean()

        if ignore_mask.ndim == focal_loss.ndim + 1:
            ignore_mask = ignore_mask.squeeze(1)
        keep = (~ignore_mask.bool()).to(focal_loss.dtype)

        return (focal_loss * keep).sum() / keep.sum().clamp(min=1.0)
    
class CLDiceLoss(nn.Module):
    def __init__(self, eps=1.0, k=10, per_class=True):
        super().__init__()
        self.k = k
        self.smooth = eps
        self.sum_dims = (2, 3) if per_class else (1, 2, 3)

    def forward(self, pred_probs, gt_mask, gt_skel):
        pred_skel = soft_skeletonize(pred_probs, k=self.k)
        s, d = self.smooth, self.sum_dims

        t_prec = ((pred_skel * gt_mask).sum(d) + s) / (pred_skel.sum(d) + s)
        t_rec  = ((gt_skel * pred_probs).sum(d) + s) / (gt_skel.sum(d) + s)

        cldice = 2 * (t_prec * t_rec) / (t_prec + t_rec + self.smooth * 1e-8)
        return 1 - cldice.mean()


    
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
    def __init__(self, args, eps=1e-6, sum_dims=(0, 2, 3)):
        super().__init__()
        self.eps = eps
        self.sum_dims = sum_dims
        self.alpha = args["alpha"]
        self.beta = args["beta"]
        self.gamma = args["gamma"]
        if args["weights"] is not None:
            print("tverskey w is set : ", args["weights"])
            self.register_buffer("t_alpha", torch.tensor(args["weights"], dtype=torch.float32))
        else:
            self.t_alpha = None

    def forward(self, pred_probs, gt, ignore_mask=None):
        gt = gt.float()

        if ignore_mask is not None:
            keep = (~ignore_mask.bool()).to(gt.dtype)
            if keep.ndim == gt.ndim - 1:
                keep = keep.unsqueeze(1)

            gt = gt * keep
            pred_probs = pred_probs * keep

        tp = (gt * pred_probs).sum(dim=self.sum_dims)
        fp = ((1 - gt) * pred_probs).sum(dim=self.sum_dims)
        fn = (gt * (1 - pred_probs)).sum(dim=self.sum_dims)
        ti = (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)
        loss_c = (1 - ti).clamp_min(0) ** self.gamma
        if self.t_alpha is None:
            return loss_c.mean()
        w = self.t_alpha
        return (w * loss_c).sum() / (w.sum() + self.eps)


class GeneralizedDiceLoss(nn.Module):
    def __init__(self, eps=1e-6, sum_dims=(0, 2, 3)):
        super().__init__()
        self.eps = eps
        self.sum_dims = sum_dims

    def forward(self, pred_probs, gt,ignore_mask=None):
        gt = gt.float()
        if ignore_mask is not None:
            keep = (~ignore_mask.bool()).float()
            if keep.ndim == gt.ndim - 1:
                keep = keep.unsqueeze(1)
            pred_probs = pred_probs * keep  
            gt = gt * keep

        true_volume = gt.sum(dim=self.sum_dims)
        weights = 1.0 / (true_volume ** 2 + self.eps)
        present = true_volume > 0
        if present.any():
            weights = torch.where(present, weights, weights[present].max())
        else:
            weights = torch.zeros_like(weights)

        intersection = (pred_probs * gt).sum(dim=self.sum_dims)
        denominator  = (pred_probs + gt).sum(dim=self.sum_dims)

        num = 2.0 * (weights * intersection).sum() + self.eps
        den = (weights * denominator).sum() + self.eps
        return 1.0 - num / den
    
class CostumeCELoss(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.pos_weight = args["pos_weight"]
        self.bg_weight = args["bg_weight"]
        self.small_vessel_weights = args["small_vessel_weights"]
    def forward(self,gt_masks,pred_masks,small_vmasks=None):
        w = torch.full(
            size=gt_masks.shape,
            fill_value=self.bg_weight,
            device=gt_masks.device,
            dtype=pred_masks.dtype,
        )
        w[gt_masks!=0] = self.pos_weight

        if(small_vmasks is not None):
            
            w[small_vmasks==1] = self.small_vessel_weights

        
        ce = F.cross_entropy(pred_masks, gt_masks, reduction="none")
        loss = (ce * w).sum() / w.sum().clamp_min(1e-6)
        return loss
    
class CostumeBranchCELoss(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.pos_weight = args["pos_weight"]
        self.bg_weight = args["bg_weight"]
    def forward(self,gt_masks,pred_masks,small_vmasks=None):
        w = torch.full(
            size=gt_masks.shape,
            fill_value=self.bg_weight,
            device=gt_masks.device,
            dtype=pred_masks.dtype,
        )
        w[gt_masks!=0] = self.pos_weight

        ce = F.cross_entropy(pred_masks, gt_masks, reduction="none")
        loss = (ce * w).sum() / w.sum().clamp_min(1e-6)
        return loss
    
class VesselnessRelaxedCELoss(nn.Module):
    def __init__(
        self,
        expert_vessel_pos_weight: float = 2.0,
        model_vessel_pos_weight : float = 1.0,
        bg_weight: float = 0.3,
        unlabeled_vessel_weight: float = 0.03,
        kernel_size : float = 3,
        outer_boundary_boost : float = 2.0,
        use_expert_boundary_only : bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.expert_vessel_pos_weight = float(expert_vessel_pos_weight)
        self.model_vessel_pos_weight = float(model_vessel_pos_weight)
        self.bg_weight = float(bg_weight)
        self.unlabeled_vessel_weight = float(unlabeled_vessel_weight)
        self.eps = eps
        self.kernel_size = kernel_size
        self.outer_boundary_boost = outer_boundary_boost
        self.use_expert_boundary_only = use_expert_boundary_only
        
    @torch.no_grad()
    def _get_boundary(self,gt):
        """
            gt: (B,H,W)
        """
        z = gt.unsqueeze(1) # (B,1,H,W)
        dilated = F.max_pool2d(
            z,
            kernel_size=self.kernel_size,
            stride = 1,
            padding = self.kernel_size // 2
        )[:,0]
        boundary = (dilated - gt).clamp(min=0.0,max=1.0)

        return boundary
    def forward(self, logits, gt_mask,expert_masks, v_masks):
        """
        logits:  (B, 2, H, W)
        gt_mask: (B, H, W) or (B, 1, H, W), values 0/1
        v_masks: (B, H, W) or (B, 1, H, W), preferably normalized to [0, 1]
        """

        if gt_mask.ndim == 4:
            gt_mask = gt_mask[:, 0]
        if v_masks.ndim == 4:
            v_masks = v_masks[:, 0]

        target = gt_mask.to(logits.device).long()
        if(self.use_expert_boundary_only):
            boundary = self._get_boundary(expert_masks.float())
        else:
            boundary = self._get_boundary(gt_mask.float())
    
  
        bg_w = (
            self.bg_weight * (1.0 - v_masks)
            + self.unlabeled_vessel_weight * v_masks
        )

        pos_w = torch.full_like(v_masks, self.model_vessel_pos_weight)
        pos_w[expert_masks!=0] = self.expert_vessel_pos_weight

        w = torch.where(target == 1, pos_w, bg_w)
        outer_penalty = self.outer_boundary_boost * boundary * (1.0 - v_masks)
        w = w + outer_penalty

        ce = F.cross_entropy(logits, target, reduction="none")

        loss = (ce * w).sum() / w.sum().clamp_min(self.eps)

        return loss
    
class VesselnessRelaxedTverskyLoss(nn.Module):
    def __init__(
        self,
        expert_vessel_pos_weight: float = 2.0,
        model_vessel_pos_weight : float = 1.0,
        bg_weight: float = 0.3,
        unlabeled_vessel_weight: float = 0.03,
        kernel_size : float = 3,
        outer_boundary_boost : float = 2.0,
        use_expert_boundary_only : bool = True,
        alpha: float = 0.3,   # FP penalty
        beta: float = 0.7,    # FN penalty
        eps: float = 1e-6,
    ):
        super().__init__()
        self.expert_vessel_pos_weight = float(expert_vessel_pos_weight)
        self.model_vessel_pos_weight = float(model_vessel_pos_weight)
        self.bg_weight = float(bg_weight)
        self.unlabeled_vessel_weight = float(unlabeled_vessel_weight)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.eps = eps
        self.kernel_size = kernel_size
        self.outer_boundary_boost = outer_boundary_boost
        self.use_expert_boundary_only = use_expert_boundary_only
    @torch.no_grad()
    def _get_boundary(self,gt):
        """
            gt: (B,H,W)
        """
        z = gt.unsqueeze(1) # (B,1,H,W)
        dilated = F.max_pool2d(
            z,
            kernel_size=self.kernel_size,
            stride = 1,
            padding = self.kernel_size // 2
        )[:,0]
        boundary = (dilated - gt).clamp(min=0.0,max=1.0)

        return boundary
    def forward(self, logits, gt_mask,expert_masks, v_masks):
        """
        logits:  (B, 2, H, W)
        gt_mask: (B, H, W) or (B, 1, H, W), values 0/1
        v_masks: (B, H, W) or (B, 1, H, W), preferably normalized to [0, 1]

        """

        if gt_mask.ndim == 4:
            gt_mask = gt_mask[:, 0]
        if v_masks.ndim == 4:
            v_masks = v_masks[:, 0]

        device = logits.device
        gt_mask = gt_mask.float()
        v_masks = v_masks.float()

        if(self.use_expert_boundary_only):
            boundary = self._get_boundary(expert_masks.float())
        else:
            boundary = self._get_boundary(gt_mask.float())
        
        prob = torch.softmax(logits, dim=1)[:, 1]

        bg_w = (
            self.bg_weight * (1.0 - v_masks)
            + self.unlabeled_vessel_weight * v_masks
        )

        pos_w = torch.full_like(v_masks, self.model_vessel_pos_weight)
        pos_w[expert_masks!=0] = self.expert_vessel_pos_weight

        w = torch.where(gt_mask == 1, pos_w, bg_w)
        outer_penalty = self.outer_boundary_boost * boundary * (1.0 - v_masks)
        w = w + outer_penalty

        tp = (w * prob * gt_mask).sum(dim=(1, 2))
        fp = (w * prob * (1.0 - gt_mask)).sum(dim=(1, 2))
        fn = (w * (1.0 - prob) * gt_mask).sum(dim=(1, 2))

        tversky = (tp + self.eps) / (
            tp + self.alpha * fp + self.beta * fn + self.eps
        )

        loss = 1.0 - tversky

        return loss.mean()
class PopovWeightedDiceLoss(nn.Module):

    def __init__(
        self,
        vessel_pos_weight: float = 2.0,
        edge_boost: float = 4.0,
        bg_weight: float = 0.3,
        smooth: float = 1e-6,
    ):
        super().__init__()
        self.vessel_pos_weight = float(vessel_pos_weight)
        self.edge_boost = float(edge_boost)
        self.bg_weight = float(bg_weight)
        self.smooth = float(smooth)

    def forward(self, pred_probs, gt):
        """
        pred_probs:  (B,1,H,W) fg only
        gt: (B,1,H,W) or (B,H,W), values 0/1 
        """

        # Foreground probability from 2-class softmax
     
        edge = popov_edge_map(gt)[:, 0]

        w = torch.full_like(gt, self.bg_weight)
        w = torch.where(gt > 0.5, torch.full_like(w, self.vessel_pos_weight), w)
        w = w + self.edge_boost * edge

        dims = (1, 2)
        intersection = (w * pred_probs * gt).sum(dim=dims)
        denom = (w * pred_probs).sum(dim=dims) + (w * gt).sum(dim=dims)

        dice = (2.0 * intersection + self.smooth) / (denom + self.smooth)
        loss = 1.0 - dice.mean()
        return loss
    
class GnnFocalLoss(nn.Module):
    """Multi-class focal loss with per-class weights. Node-level (GNN)."""
    def __init__(self, weight=None, gamma=1.5, reduction="mean",ignore_index=100):
        super().__init__()
        if(weight is not None):
            weight = torch.tensor(weight)
        else:
            weight = None
        self.register_buffer("weight", weight if weight is not None else None)
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        keep = target != self.ignore_index          # drop bg nodes entirely
        if keep.sum() == 0:
            return logits.sum() * 0.0                # no real nodes this batch
        logits, target = logits[keep], target[keep]

        logp = F.log_softmax(logits, dim=1)
        logp_t = logp.gather(1, target[:, None]).squeeze(1)
        p_t = logp_t.exp()
        focal = (1.0 - p_t) ** self.gamma * (-logp_t)

        if self.weight is not None:
            focal = focal * self.weight[target]

        return focal.mean() if self.reduction == "mean" else focal.sum()
    
# class GnnLoss(nn.Module):
#     def __init__(self,args):
#         super(GnnLoss,self).__init__()
#         if(args["ce_w"] is not None):
#             w = torch.tensor(args["ce_w"],device="cuda")
#             self.ce_loss_fn = nn.CrossEntropyLoss(weight=w)
#             print("w is set for ce : ",w)
#         else:
#             self.ce_loss_fn = nn.CrossEntropyLoss()

        
#         if(args["focal_coef"]["alpha"] is not None):
#             alpha = torch.tensor(args["focal_coef"]["alpha"]).cuda()
#         else:
#             alpha = None
#         print(f"focal w is set to {alpha}")
#         # self.fce_loss_fn = GnnFocalLoss(
#         #     gamma=args["focal_coef"]["gamma"],
#         #     weight=alpha,
#         #     to_onehot_y=True, use_softmax=True,
#         # )
#         self.fce_loss_fn = GnnFocalLoss(
#             gamma=args["focal_coef"]["gamma"],
#             weight=alpha,
#             reduction = "mean",
#             ignore_index = 0 if args["focal_coef"]["ignore_bg"] else 100
#         )
#         self.loss_type = args["loss_type"]
#         print(f"loss is set to {self.loss_type}")
#         # self.ce_loss_fn = sigmoid_focal_loss(gamma=2)
#     def forward(self,y_pred,y_true):
#         y_true = y_true.reshape(-1)
#         if(self.loss_type=="focal"):
#             # print(y_pred.shape)
#             # print(torch.unique(y_true))
#             ce_loss = self.fce_loss_fn(
#                 y_pred,
#                 y_true
#             )
#         else:
#             ce_loss = self.ce_loss_fn(
#                 y_pred,
#                 y_true
#             )
            
#         loss_dict = {
#             "total loss" : ce_loss
#         }
#         return ce_loss , loss_dict

class NewFocalLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.gamma = args["gamma"]
        self.extra_w = args.get("extra_w", 1)
        self.ignore_bg = args.get("ignore_bg", False)     # drop class-0 nodes from the loss
        self.bg_index = args.get("bg_index", 0)
 
        alpha = args.get("alpha")
        if alpha is not None and not torch.is_tensor(alpha):
            alpha = torch.tensor(alpha, dtype=torch.float32,device="cuda")
        self.register_buffer("alpha", alpha)
 
    def forward(self, inputs, targets, hard_mask=None, ignore_mask=None,
                node_weights=None):
        targets = targets.long().reshape(-1)
        ce = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce)
        loss = (1 - pt) ** self.gamma * ce
        if self.alpha is not None:
            loss = self.alpha[targets] * loss
 
        w = torch.ones_like(loss)
        if hard_mask is not None:
            w = torch.where(hard_mask.reshape(-1) == 1,
                            torch.full_like(w, float(self.extra_w)), w)
        if node_weights is not None:
            w = w * node_weights.reshape(-1).to(loss.dtype)
        if ignore_mask is not None:
            w = w * (~ignore_mask.reshape(-1).bool()).to(loss.dtype)
        if self.ignore_bg:
            w = w * (targets != self.bg_index).to(loss.dtype)   # <- background excluded
 
        l = (loss * w).sum() / w.sum().clamp(min=1e-8)
        return l, {"total loss": l}
