import torch
import torch.nn as nn
import numpy as np
from torchvision import models
import torch.nn.functional as F

class ConvLSTMBlock(nn.Module):
    def __init__(self,feature_map_size,hidden_size,kernel_size,padding):
        super(ConvLSTMBlock,self).__init__()

        self.conv = nn.Conv2d(
            in_channels=feature_map_size+hidden_size,
            out_channels=hidden_size*4,
            kernel_size=kernel_size,
            padding = padding
        )
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
    def forward(self,x,h,c):
        """
        h : batch_size , d , h' , w'
        c : batch_size , d , h' , w'
        x : batch_size , d , h' , w'
        """
        
        conv_outs = self.conv(torch.cat([x,h],dim=1))
        
        # batch_size , d , h' , w'
        i_conv , f_conv , o_conv , g_conv = torch.chunk(conv_outs,chunks=4 , dim=1)
        
        forget_gate = self.sigmoid(f_conv)
        input_gate = self.sigmoid(i_conv) * self.tanh(g_conv)
        output_gate = self.sigmoid(o_conv)

        c = c*forget_gate + input_gate

        h = output_gate*(self.tanh(c))
        
        return h , c
class ConvLSTM(nn.Module):
    def __init__(self,feature_map_size,hidden_size,kernel_size,padding,stack_size=2,device="cuda"):
        super(ConvLSTM,self).__init__()
        if(hidden_size is None):
            hidden_size = feature_map_size
        models_stack = [
            ConvLSTMBlock(
                feature_map_size=feature_map_size,
                hidden_size=hidden_size,
                kernel_size=kernel_size,
                padding=padding
            )
        ]
        for i in range(stack_size-1):
            models_stack+=[ConvLSTMBlock(
                feature_map_size=hidden_size,
                hidden_size=hidden_size,
                kernel_size=kernel_size,
                padding=padding
            )]
        
        self.models_stack = nn.ModuleList(models_stack)
        self.stack_size=stack_size
        self.hidden_size = hidden_size
        self.feature_map_size = feature_map_size
        self.device=device
    def forward_fn(self, x, H, C):
        model_inp = x
        for i, model in enumerate(self.models_stack):
            h, c = model(model_inp, H[i], C[i])
            H[i], C[i] = h, c
            model_inp = h
        return H, C
    
    def forward(self, x, seq_len):
        B, _, Hh, Ww = x.shape

        H = [x.new_zeros((B, self.hidden_size, Hh, Ww)) for _ in range(self.stack_size)]
        C = [x.new_zeros((B, self.hidden_size, Hh, Ww)) for _ in range(self.stack_size)]

        outputs = []
        for _ in range(seq_len):
            H, C = self.forward_fn(x, H, C)
            outputs.append(H[-1])

        return torch.stack(outputs, dim=0)

class FCN(nn.Module):
    def __init__(self):
        super(FCN,self).__init__()
        resnet = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT,
            replace_stride_with_dilation=[False, True, True]
        )
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4 

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

class FullConvLSTM(nn.Module):
    def __init__(self,args):
        super(FullConvLSTM,self).__init__()
        feature_map_size = args["feature_map_size"]
        hidden_size = args["hidden_size"]
        padding = args["padding"]
        kernel_size = args["kernel_size"]
        stack_size = args["stack_size"]
        class_count = args["class_count"]
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.fcn = FCN()
        
        self.conv_lstm_model = ConvLSTM(
            feature_map_size=feature_map_size,
            hidden_size=hidden_size,
            kernel_size=kernel_size,
            padding=padding,
            stack_size=stack_size,
            device=device
        )

        self.stop_head = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(hidden_size,1)
        )
        self.mask_head = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_size,
                out_channels=hidden_size,
                kernel_size=1,
                bias=True
            ), 
            nn.ConvTranspose2d(hidden_size , hidden_size , kernel_size=4 ,stride=2,padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_size, 48, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(48, class_count, kernel_size=4, stride=2, padding=1)
        )
        self.hidden_size = hidden_size
    def forward(self,x,seq_len):
        """
        inputs : imgs 
            - imgs : (batch_size,C,H,W)
        outputs : stop_label , pred_mask
            - stop_label : (batch_size x seq_len)
            - pred_mask : (batch_size x seq_len,class_count,H,W)
        """
        batch_size , image_channels , h , w = x.shape

        feature_maps = self.fcn(x) #B x d x H//8 x W//8
        
        heads_inputs = self.conv_lstm_model(feature_maps,seq_len) #seq len x B x d x H//8 x W//8
        heads_inputs = heads_inputs.reshape(-1,*heads_inputs.shape[2:])

        stop_label = self.stop_head(heads_inputs) #seq len*B 
        pred_mask = self.mask_head(heads_inputs) #seq len*B x class_count x H x W

        return stop_label , pred_mask