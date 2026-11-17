
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# DropBlock (used inside RCFM1, per the paper which uses dropblock instead of
# dropout in the encoding path -- consistent with SA-UNet / SD-UNet lineage)
# ---------------------------------------------------------------------------
class DropBlock2D(nn.Module):
    """Regularization that drops contiguous square regions of a feature map.

    Ref: Ghiasi et al., "DropBlock: A regularization method for convolutional
    networks." The paper's RCFM1 places a dropblock after the first conv.
    """

    def __init__(self, drop_prob: float = 0.1, block_size: int = 7):
        super().__init__()
        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x

        # gamma controls the number of dropped activation centers
        gamma = self.drop_prob / (self.block_size ** 2)
        mask = (torch.rand_like(x[:, :1, :, :]) < gamma).float()

        # expand each dropped center into a block_size x block_size square
        mask = F.max_pool2d(
            mask,
            kernel_size=self.block_size,
            stride=1,
            padding=self.block_size // 2,
        )
        mask = 1.0 - mask
        # normalize so expected activation magnitude is preserved
        count = mask.numel()
        keep = mask.sum()
        out = x * mask * (count / (keep + 1e-7))
        return out


# ---------------------------------------------------------------------------
# RCFM1 : Residual Convolution Fusion Module for the ENCODER path (Fig. 3a)
# ---------------------------------------------------------------------------
class RCFM1(nn.Module):
    """Encoder-path residual fusion block.

    Left branch :  3x3 Conv -> DropBlock -> Activation -> BN -> 3x3 Conv
    Right branch:  1x1 Conv -> BN
    Output      :  element-wise sum of the two branches.

    The dropblock in the left branch helps the encoder converge quickly and
    focus on the main segmentation target (Section 3.3).
    """

    def __init__(self, in_channels: int, out_channels: int,
                 drop_prob: float = 0.1, block_size: int = 7):
        super().__init__()

        # --- left branch ---
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.dropblock = DropBlock2D(drop_prob, block_size)
        self.act = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)

        # --- right branch (shortcut) ---
        self.shortcut_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.shortcut_bn = nn.BatchNorm2d(out_channels)

        # final activation after fusion
        self.out_act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left = self.conv1(x)
        left = self.dropblock(left)
        left = self.act(left)
        left = self.bn1(left)
        left = self.conv2(left)

        right = self.shortcut_conv(x)
        right = self.shortcut_bn(right)

        return self.out_act(left + right)


# ---------------------------------------------------------------------------
# RCFM2 : Residual Convolution Fusion Module for the DECODER path (Fig. 3b)
# ---------------------------------------------------------------------------
class RCFM2(nn.Module):
    """Decoder-path residual fusion block.

    Left branch :  (3x3 Conv -> Activation -> BN) x 2     [no dropblock]
    Right branch:  1x1 Conv -> BN
    Output      :  element-wise sum of the two branches.

    The decoder recovers/aggregates information that is already filtered, so
    dropping pixels is unnecessary; dropblock is therefore omitted (Section 3.3).
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        # --- left branch: two (conv -> act -> bn) stages ---
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.act1 = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.act2 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # --- right branch (shortcut) ---
        self.shortcut_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.shortcut_bn = nn.BatchNorm2d(out_channels)

        self.out_act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left = self.bn1(self.act1(self.conv1(x)))
        left = self.bn2(self.act2(self.conv2(left)))

        right = self.shortcut_bn(self.shortcut_conv(x))

        return self.out_act(left + right)


# ---------------------------------------------------------------------------
# LESA : Location-Enhanced Spatial Attention (Section 3.4, Fig. 4)
# ---------------------------------------------------------------------------
class LESA(nn.Module):
    """Location-Enhanced Spatial Attention with a cascaded waterfall structure.

    Given input X of shape (B, C, H, W):
      1. Encode along three directions by average pooling:
           A_{H,C} : pool over width  W  -> per-row    descriptor  (B, C, H, 1)
           B_{W,C} : pool over height H  -> per-column descriptor  (B, C, 1, W)
           C_{W,H} : pool over channel C -> spatial    descriptor  (B, 1, H, W)
      2. Expand the spatial (channel-compressed) descriptor back along C, then
         concatenate the three directional descriptors and fuse with a 1x1 conv.
      3. Split the fused tensor back into three directional attention maps
         (I_{H,C}, J_{W,C}, K_{W,H}), pass each through ReLU/sigmoid gating.
      4. Output  Y = X * I * J * K   (broadcast element-wise multiplication).

    This couples horizontal, vertical, and spatial information so the module
    captures long-distance dependencies between coronary-vessel pixels.

    Implementation note: the paper concatenates descriptors of differing spatial
    extents into a 1x1-conv fusion. To realize this concretely while preserving
    the directional-attention semantics, the three pooled descriptors are each
    projected to a common spatial form, fused, and then re-projected into the
    three directional gates. The final gating multiplications follow Eq. for Y.
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(channels // reduction, 8)

        # 1x1 conv that fuses the concatenated directional descriptors.
        # We concatenate along channel dim: A-branch(C) + B-branch(C) + C-branch(1)
        self.fuse = nn.Conv2d(channels * 2 + 1, mid, kernel_size=1, bias=False)
        self.fuse_bn = nn.BatchNorm2d(mid)
        self.fuse_act = nn.ReLU(inplace=True)

        # Directional attention generators (the "split" outputs I, J, K).
        # I_{H,C}: row-wise (H) gate over channels
        self.gen_h = nn.Conv2d(mid, channels, kernel_size=1, bias=True)
        # J_{W,C}: column-wise (W) gate over channels
        self.gen_w = nn.Conv2d(mid, channels, kernel_size=1, bias=True)
        # K_{W,H}: spatial gate (single channel, broadcast over C)
        self.gen_s = nn.Conv2d(mid, 1, kernel_size=1, bias=True)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        # --- 1. directional encodings ---
        # A_{H,C}: average over width -> (B, C, H, 1)
        a = x.mean(dim=3, keepdim=True)
        # B_{W,C}: average over height -> (B, C, 1, W)
        bb = x.mean(dim=2, keepdim=True)
        # C_{W,H}: average over channels -> (B, 1, H, W)  (spatial compression)
        cc = x.mean(dim=1, keepdim=True)

        # --- 2. bring descriptors to a common (H, W) grid and concatenate ---
        a_full = a.expand(b, c, h, w)            # broadcast row descriptor over W
        b_full = bb.expand(b, c, h, w)           # broadcast col descriptor over H
        c_full = cc                              # already (B, 1, H, W)

        concat = torch.cat([a_full, b_full, c_full], dim=1)  # (B, 2C+1, H, W)

        # --- fuse via 1x1 conv (the cascaded "waterfall" fusion) ---
        f = self.fuse_act(self.fuse_bn(self.fuse(concat)))   # (B, mid, H, W)

        # --- 3. split into directional attention maps I, J, K ---
        i_hc = self.gen_h(f).mean(dim=3, keepdim=True)       # (B, C, H, 1)
        j_wc = self.gen_w(f).mean(dim=2, keepdim=True)       # (B, C, 1, W)
        k_wh = self.gen_s(f)                                 # (B, 1, H, W)

        i_hc = self.sigmoid(i_hc)
        j_wc = self.sigmoid(j_wc)
        k_wh = self.sigmoid(k_wh)

        # --- 4. Y = X * I * J * K  (broadcasting handles the shapes) ---
        y = x * i_hc * j_wc * k_wh
        return y


# ---------------------------------------------------------------------------
# Down / Up sampling helpers
# ---------------------------------------------------------------------------
class Down(nn.Module):
    """Two-step downsampling: RCFM1 then max-pool (halves H and W)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.rcfm = RCFM1(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        feat = self.rcfm(x)      # feature kept for the skip connection
        down = self.pool(feat)   # downsampled feature for the next stage
        return feat, down


class Up(nn.Module):
    """Upsampling via transposed convolution, concat skip, then RCFM2."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels,
                                     kernel_size=2, stride=2)
        self.rcfm = RCFM2(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        # pad in case of odd spatial sizes so concat aligns
        dy = skip.size(2) - x.size(2)
        dx = skip.size(3) - x.size(3)
        if dy != 0 or dx != 0:
            x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        x = torch.cat([skip, x], dim=1)
        return self.rcfm(x)


# ---------------------------------------------------------------------------
# SARC-UNet : full network (Fig. 2)
# ---------------------------------------------------------------------------
class SARC_UNet(nn.Module):
    """SARC-UNet.

    Channel progression matches Fig. 2 (base width = 16):
        input(3) -> 16 -> 32 -> 64 -> 128(bottleneck) -> 64 -> 32 -> 16 -> output(1)

    Encoder uses RCFM1 + downsampling; the LESA attention sits at the
    bottleneck between encoder and decoder; the decoder uses RCFM2 with skip
    connections; a final 1x1 conv produces the segmentation map.
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 2,
                 base_channels: int = 16):
        super().__init__()
        c1 = base_channels        # 16
        c2 = base_channels * 2    # 32
        c3 = base_channels * 4    # 64
        c4 = base_channels * 8    # 128

        # --- Encoder ---
        self.down1 = Down(in_channels, c1)   # skip: 16
        self.down2 = Down(c1, c2)            # skip: 32
        self.down3 = Down(c2, c3)            # skip: 64

        # --- Bottleneck (RCFM1 -> LESA -> RCFM2 as in Fig. 2 bottom row) ---
        self.bottleneck_in = RCFM1(c3, c4)   # 64 -> 128
        self.lesa = LESA(c4)                 # location-enhanced spatial attention
        self.bottleneck_out = RCFM2(c4, c4)  # 128 -> 128

        # --- Decoder ---
        self.up3 = Up(c4, skip_channels=c3, out_channels=c3)  # 128 -> 64, skip 64
        self.up2 = Up(c3, skip_channels=c2, out_channels=c2)  # 64  -> 32, skip 32
        self.up1 = Up(c2, skip_channels=c1, out_channels=c1)  # 32  -> 16, skip 16

        # --- Output head (Conv 1x1 in Fig. 2) ---
        self.out_conv = nn.Conv2d(c1, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1, x = self.down1(x)   # s1: (B,16,H,W)
        s2, x = self.down2(x)   # s2: (B,32,H/2,W/2)
        s3, x = self.down3(x)   # s3: (B,64,H/4,W/4)

        x = self.bottleneck_in(x)   # (B,128,H/8,W/8)
        x = self.lesa(x)            # spatial attention
        x = self.bottleneck_out(x)  # (B,128,H/8,W/8)

        x = self.up3(x, s3)     # (B,64,H/4,W/4)
        x = self.up2(x, s2)     # (B,32,H/2,W/2)
        x = self.up1(x, s1)     # (B,16,H,W)
        if(self.training):
            return [self.out_conv(x)],torch.tensor(0.0,device=x.device)  # (B,num_classes,H,W) -- apply sigmoid externally
        else:
            return [self.out_conv(x)] 


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model = SARC_UNet(in_channels=3, num_classes=1, base_channels=16)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"SARC-UNet parameters: {n_params:,}")

    # Paper uses 512x512 inputs; use 256 here just to keep the check light.
    dummy = torch.randn(2, 3, 256, 256)
    out = model(dummy)
    print(f"Input : {tuple(dummy.shape)}")
    print(f"Output: {tuple(out.shape)}")   # expect (2, 1, 256, 256)
    assert out.shape == (2, 1, 256, 256)
    print("Forward pass OK.")