
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSemanticSegmentation

CLS3_IDX = 3
FPN_CHANNELS = 512      # == UperNetConfig.hidden_size (verified default)
N_LATERALS = 3          # len(in_channels) - 1 for swin-base


class DilatedContext(nn.Module):
    """Propagate the 1-ch class-3 probability map to a wide receptive field.

    RF for stacked 3x3 dilated convs = 1 + 2*sum(dilations).
      (1,2,4,8,16,32,64,128) -> 511 px   (the old 1x1 head was 1 px)

    GroupNorm rather than BatchNorm: batch sizes here are small and BN running
    stats on <=4 images are unreliable at eval.
    """

    def __init__(self, in_ch=1, ch=16, dilations=(1, 2, 4, 8, 16, 32, 64, 128)):
        super().__init__()
        layers, c_prev = [], in_ch
        for d in dilations:
            layers += [
                nn.Conv2d(c_prev, ch, 3, padding=d, dilation=d, bias=False),
                nn.GroupNorm(4, ch),
                nn.ReLU(inplace=True),
            ]
            c_prev = ch
        self.body = nn.Sequential(*layers)
        self.out_channels = ch
        self.dilations = dilations

    def receptive_field(self):
        return 1 + 2 * sum(self.dilations)

    def forward(self, x):
        return self.body(x)


def coord_channels(x):
    """(B,_,H,W) -> (B,2,H,W) normalised row/col grids in [-1,1]."""
    b, _, h, w = x.shape
    ys = torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype).view(1, 1, h, 1).expand(b, 1, h, w)
    xs = torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype).view(1, 1, 1, w).expand(b, 1, h, w)
    return torch.cat([ys, xs], dim=1)


class SwinUperNetCascade(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model = AutoModelForSemanticSegmentation.from_pretrained(
            "openmmlab/upernet-swin-base", local_files_only=True
        )

        class_count = args["class_count"]                              # 6
        bottleneck = args["swin"].get("bottleneck_channels", 32)
        self.detach_anchor = args["swin"].get("detach_anchor", True)
        self.use_coord = args["swin"].get("use_coord", True)
        self.class_count = class_count

        # compresser REPLACES the ADE20K classifier (last op of decode_head),
        # so decode_head now emits `bottleneck` channels instead of 150.
        self.model.decode_head.classifier = nn.Conv2d(FPN_CHANNELS, bottleneck, kernel_size=1)
        self.model.auxiliary_head = None      # unused; don't waste compute

        coord_c = 2 if self.use_coord else 0
        self.conv_main_vessels = nn.Conv2d(bottleneck + coord_c, class_count - 2, kernel_size=1)

        self.anchor_ctx = DilatedContext(in_ch=1, ch=16)
        self.conv_4_16_vessels = nn.Conv2d(
            bottleneck + coord_c + self.anchor_ctx.out_channels, 2, kernel_size=1
        )

        self.deep_super_vision = args["deep_super_vision"]
        if self.deep_super_vision:
            self.dsv0_conv = nn.Conv2d(FPN_CHANNELS, class_count, kernel_size=1)
            self.dsv1_conv = nn.Conv2d(FPN_CHANNELS, class_count, kernel_size=1)
            self.dsv2_conv = nn.Conv2d(FPN_CHANNELS, class_count, kernel_size=1)
            self.dsv_convs = [self.dsv0_conv, self.dsv1_conv, self.dsv2_conv]

    # NOTE: no hooks, no self.features, no shared mutable state.

    def forward(self, x, side=None):
        backbone_out = self.model.backbone.forward_with_filtered_kwargs(x)
        """
        torch.Size([1, 128, 128, 128])
        torch.Size([1, 256, 64, 64])
        torch.Size([1, 512, 32, 32])
        torch.Size([1, 1024, 16, 16])
        """
        feature_maps = backbone_out.feature_maps          # tuple, 1/4 .. 1/32
        head = self.model.decode_head

        
        # ---- main path -----------------------------------------------------
        logits = head(feature_maps)                       # (B, bottleneck, h/4, w/4)
        compressed = F.interpolate(
            logits, size=x.shape[2:], mode="bilinear", align_corners=False
        )                                                 # (B, bottleneck, H, W)

        if self.use_coord:
            compressed = torch.cat([compressed, coord_channels(compressed)], dim=1)

        main_outs = self.conv_main_vessels(compressed)    # (B,4,H,W) [bg,1,2,3]
        
        p3 = F.softmax(main_outs, dim=1)[:, CLS3_IDX:CLS3_IDX + 1]
        if self.detach_anchor:
            p3 = p3.detach()

        anchor = self.anchor_ctx(p3)                      # RF = 511 px
        others_outs = self.conv_4_16_vessels(torch.cat([anchor, compressed], dim=1))

        out = torch.cat([main_outs, others_outs], dim=1)  # (B,6,H,W)

        # ---- deep supervision ----------------------------------------------
        outs = []
        if self.deep_super_vision:
            # rebuild the laterals explicitly instead of hooking them
            outs = [
                self.dsv_convs[i](head.lateral_convs[i](feature_maps[i]))
                for i in range(N_LATERALS)
            ]
            # still at 1/4, 1/8, 1/16 resolution -- resize targets in the loss.

        return [out] + outs


# ---------------------------------------------------------------------------
# Run this on best_model (the deepcopy), not just on model.
# ---------------------------------------------------------------------------
@torch.no_grad()
def assert_input_dependent(model, device="cuda", size=512):
    model.eval()
    a = torch.randn(1, 3, size, size, device=device)
    b = torch.randn(1, 3, size, size, device=device)
    ya, yb = model(a)[0], model(b)[0]
    d = (ya - yb).abs().max().item()
    print(f"max |f(a) - f(b)| = {d:.6f}")
    assert d > 1e-4, (
        "MODEL OUTPUT IS CONSTANT w.r.t. INPUT -- every image will predict "
        "identically. Do not train/report until this passes."
    )
    print("OK: output depends on input.")


@torch.no_grad()
def assert_deepcopy_safe(model, device="cuda", size=512):
    """The check that would have caught this bug at its source."""
    import copy
    model.eval()
    clone = copy.deepcopy(model).eval()
    a = torch.randn(1, 3, size, size, device=device)
    b = torch.randn(1, 3, size, size, device=device)

    same_input = (model(a)[0] - clone(a)[0]).abs().max().item()
    varies = (clone(a)[0] - clone(b)[0]).abs().max().item()

    print(f"model(a) vs deepcopy(a)      : {same_input:.6f}  (want ~0)")
    print(f"deepcopy(a) vs deepcopy(b)   : {varies:.6f}  (want >0)")
    assert same_input < 1e-3, "deepcopy does not reproduce the original's output"
    assert varies > 1e-4, "DEEPCOPY IS INPUT-INDEPENDENT -- stale hook/shared state"
    print("OK: deepcopy is safe.")

if __name__ == "__main__":
    args = {
        "swin":{
            "encoder_out_channels":150
        },
        "image_shape" : (512,512),
        "in_c":3,
        "deep_super_vision":True,
        "class_count":6
    }
    model = SwinUperNetCascade(args).to("cuda")
    inp = torch.rand((1,3,args["image_shape"][0],args["image_shape"][1])).to("cuda")
    outs = model(inp)
    # for out in outs:
    #     print(out.shape)