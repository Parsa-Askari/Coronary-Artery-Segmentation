import torch.nn as nn
import torch
import torch.nn.functional as F
###IE###
###SS###
def crop_dims():
    pass
class Conv(nn.Module):
    def __init__(self,in_c , out_c,p):
        super(Conv,self).__init__()
        self.layers =  nn.Sequential( 
            nn.Conv2d(
                in_channels = in_c , 
                out_channels = out_c ,
                kernel_size=3, 
                stride = 1 ,
                padding = p
            ),
            nn.InstanceNorm2d(out_c, eps=1e-5, affine=True),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True)
        )
    def forward(self,x):
        return self.layers(x)
class DownsampleConv(nn.Module):
    def __init__(self,in_c , out_c):
        super(DownsampleConv,self).__init__()
        self.layers =  nn.Sequential( 
            nn.Conv2d(
                in_channels = in_c , 
                out_channels = out_c ,
                kernel_size=3, 
                stride = 2,
                padding=1
            ),
            nn.InstanceNorm2d(out_c, eps=1e-5, affine=True),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True)
        )
    def forward(self,x):
        return self.layers(x)
class EncoderBlock(nn.Module):
    def __init__(self,in_c , out_c,p=1):
        super(EncoderBlock,self).__init__()
        self.layers = nn.Sequential(
            Conv(in_c = in_c , out_c = out_c , p=p),
            Conv(in_c = out_c , out_c=out_c ,p=p)
        )
        self.pool =  DownsampleConv(in_c = out_c , out_c=out_c )
    def forward(self,x):
        z = self.layers(x)
        return z , self.pool(z)

class DecoderBlock(nn.Module):
    def __init__(self,in_c ,out_c , f_int_scale,class_count,gate_c = None , attention=False,dsv=False):
        super(DecoderBlock,self).__init__()
        self.dsv=dsv
        self.conv1 = Conv(in_c=in_c , out_c=out_c,p=1)
        self.conv2 = Conv(in_c = out_c , out_c = out_c , p=1)
        self.upsampler = nn.ConvTranspose2d(
            in_channels = out_c , 
            out_channels = out_c//2 ,
            kernel_size=2 ,
            stride=2
        )

        if(self.dsv):
            self.dsv_block = nn.Conv2d(in_channels=out_c,out_channels=class_count,kernel_size=1)
        #in_c * 2 = gate_c
        if(attention):
            self.gate = AttentionGate(gate_in_c=gate_c ,f_int_scale=f_int_scale,skip_in_c=in_c//2)
        self.attention=attention
    def forward(self,x_in,x_skip,x_gate):
        if(self.attention):
            x_skip = self.gate(x_skip , x_gate)
        if(x_in.shape[2] != x_skip.shape[2] or x_in.shape[3] != x_skip.shape[3]):
            x_skip = crop_dims(x_in,x_skip)
        
        x = torch.cat([x_skip,x_in],dim=1)
        z = self.conv1(x)
        gate_z = self.conv2(z)
        upsampled_z = self.upsampler(gate_z)
        if(self.dsv):
            dsv_out = self.dsv_block(gate_z)
            
            if(self.attention):
               
                return upsampled_z , gate_z , dsv_out
            else:
                return upsampled_z , None, dsv_out 
        else:
            if(self.attention):
                return upsampled_z , gate_z , None
            else:
                return upsampled_z , None , None

class BottleNeck(nn.Module):
    def __init__(self,in_c , out_c,p,attention=False):
        super(BottleNeck,self).__init__()
        self.conv1 = Conv(in_c=in_c , out_c=out_c,p=p)
        self.conv2 = Conv(in_c = out_c , out_c = out_c , p=p)
        self.upsampler = nn.ConvTranspose2d(
            in_channels = out_c , 
            out_channels = out_c//2 ,
            kernel_size=2 ,
            stride=2
        )
        self.attention=attention
    def forward(self,x):
        z = self.conv1(x)
        gate_z = self.conv2(z)
        upsampled_z = self.upsampler(gate_z)
        if(self.attention):
            return upsampled_z , gate_z
        return upsampled_z , None

class AttentionGate(nn.Module):
    def __init__(self,gate_in_c,skip_in_c,f_int_scale=2,f_int=None,scaler="sigmoid"):
        super(AttentionGate,self).__init__()
        f_int = min(gate_in_c//f_int_scale,skip_in_c//f_int_scale) if f_int==None else f_int
        f_int = 1 if f_int == 0 else f_int
        self.conv_gate = nn.Conv2d(in_channels = gate_in_c , out_channels = f_int , 
                                   kernel_size = 1)
        self.conv_skip = nn.Conv2d(in_channels = skip_in_c , out_channels = f_int , 
                                   kernel_size = 1)
        self.relu = nn.ReLU(inplace=True)
        self.conv_shrink = nn.Conv2d(in_channels = f_int , out_channels = 1 ,
                                     kernel_size = 1)
        if(scaler =="sigmoid"):
            self.scaler = nn.Sigmoid()    
    def forward(self,x_skip,x_gate):
        x_gate_int = self.conv_gate(x_gate)
        x_skip_int = self.conv_skip(x_skip)
        # x_skip_int = crop_dims(x_gate_int,x_skip_int)
        if x_skip_int.shape[2:] != x_gate_int.shape[2:]:
            x_skip_int = F.interpolate(
                x_skip_int, 
                size=x_gate_int.shape[2:], 
                mode="bilinear", 
                align_corners=False
            )
        
        added_x = x_skip_int + x_gate_int
        relu_x = self.relu(added_x)
        shrinked_x = self.conv_shrink(relu_x)
        sig_x = self.scaler(shrinked_x)
        # padded_x = padd_dims(x_skip , sig_x)
        if sig_x.shape[2:] != x_skip.shape[2:]:
            padded_x = F.interpolate(
                sig_x, 
                size=x_skip.shape[2:], 
                mode="bilinear", 
                align_corners=False
            )

        return padded_x*x_skip

class Head(nn.Module):
    def __init__(self,in_c ,out_c ,class_count ,f_int_scale, 
        gate_c = None , attention=False):

        super(Head,self).__init__()
        self.conv1 = Conv(in_c=in_c , out_c=out_c,p=1)
        self.conv2 = Conv(in_c = out_c , out_c = out_c , p=1)
        self.conv1x1 = nn.Conv2d(
            in_channels = out_c , 
            out_channels = class_count ,
            kernel_size=1
        )
        
        if(attention):
            self.gate = AttentionGate(
                gate_in_c=gate_c , 
                f_int_scale=f_int_scale,
                skip_in_c=in_c//2
            )
        self.attention=attention
    def forward(self,x_in,x_skip,x_gate):
        if(self.attention):
            x_skip = self.gate(x_skip , x_gate)
        if(x_in.shape[2] != x_skip.shape[2] or x_in.shape[3] != x_skip.shape[3]):
            x_skip = crop_dims(x_in,x_skip)
            
        x = torch.cat([x_skip,x_in],dim=1)
        z = self.conv1(x)
        gate_z = self.conv2(z)
        
        class_feature_maps = self.conv1x1(gate_z) 
        
        return class_feature_maps