import torch.nn as nn
import torch
import torch.nn.functional as F
###IE###
from .nnunet_blocks import EncoderBlock , BottleNeck , DecoderBlock , Head
###SS###
class nnUnet(nn.Module):
    def __init__(self,args,encoder_channel_settings=None,decoder_channel_settings=None):
        super(nnUnet,self).__init__()
        
        in_c = args["in_c"]
        class_count = args["class_count"]
        attention = args["attention"]
        image_shape = args["image_shape"]
        base_channel = args["base_channel"]
        f_int_scale = args["f_int_scale"]
        max_channels = args["max_channels"]
        input_channels = args["input_channels"]
        h = image_shape[0]
        w = image_shape[1]
        
        max_pool_count = 0
        co=0
        
        while(w>4 and h>4):
            w/=2
            h/=2
            co+=1
        print(f"number of layers : {co}")

        # create encoder settings 
        if(encoder_channel_settings is None):
            self.encoder_channel_settings = [base_channel]
            for i in range(co-1):
                new_c =min(self.encoder_channel_settings[i]*2,max_channels)
                self.encoder_channel_settings +=[new_c]
        else :
            self.encoder_channel_settings = encoder_channel_settings
        
        # create bottleneck settings
        self.bottle_neck_channel_setting = self.encoder_channel_settings[-1]*2
        # create decoder settings 
        if(decoder_channel_settings is  None):
            self.decoder_channel_settings =[]
            for i in range(co-1):
                self.decoder_channel_settings = [self.encoder_channel_settings[i]*2] +  self.decoder_channel_settings
        else :
            self.decoder_channel_settings = decoder_channel_settings

        
        # build encoder
        self.encoders = nn.ModuleList()
        for i in range(co):
            output_channels = self.encoder_channel_settings[i]
            self.encoders.append(EncoderBlock(in_c=input_channels,out_c=output_channels , p=1))
            input_channels = output_channels
        # build bottleneck
        self.bottle_neck = BottleNeck(in_c = output_channels ,out_c = self.bottle_neck_channel_setting , p=1,attention = attention)
        #build decoder
        input_channels = self.bottle_neck_channel_setting
        self.decoders = []
        for i in range(co-1):
            output_channels = self.decoder_channel_settings[i]
            self.decoders = [
                DecoderBlock(
                    in_c = input_channels , 
                    out_c=output_channels , 
                    gate_c = input_channels , 
                    attention = attention,
                    f_int_scale=f_int_scale
                )] + self.decoders
            input_channels = output_channels

        self.decoders = nn.ModuleList(self.decoders)
        self.attention = attention
        
        self.head = Head(
            in_c = input_channels , 
            out_c=input_channels//2 ,
            class_count = class_count,
            gate_c = input_channels , 
            attention = False,
            f_int_scale=f_int_scale
        )
        print("encoder settings : ", self.encoder_channel_settings)
        print("bottle-neck settings : ", self.bottle_neck_channel_setting)
        print("decoder settings : ", self.decoder_channel_settings)
        print("head settings : ",class_count)
    def forward(self,x):
        skips = []
        for encoder in self.encoders : 
            skip , out = encoder(x)
            skips += [skip]
            x = out
        x_in,gate_in = self.bottle_neck(x)
        
        for i in range(len(self.decoders) - 1, -1, -1):
            decoder = self.decoders[i]
            skip = skips[i+1]
            x_out,gate_out = decoder(x_in,skip,gate_in)
            x_in=x_out
            gate_in=gate_out
        # print(x_in.shape)
        return self.head(
            x_in = x_in,
            x_skip = skips[0],
            x_gate = gate_in
        )
# args = {
#     "base_path" : "../arcade/nnUnet_dataset/syntax",
#     "in_c" : 1,
#     "base_channel" :32,
#     "image_shape" : (512,512),
#     "class_count" : 26 ,
#     "attention" : True,
#     "batch_size" : 5,
#     "num_workers" : 4,
#     "device" : "cuda" if torch.cuda.is_available() else "cpu",
#     "lr" : 0.0001,
#     "momentum" : 0.9,
#     "epcohs":1,
#     "f_int_scale" : 2,
#     "full_report_cycle" : 10,
#     "max_channels":512,
#     "input_channels":1,
#     "output_base_path" : "./outputs",
#     "name" : "NoAttention"
# }
# class_map = {
#     0: '1',1: '2', 2: '3',3: '4',
#     4: '5',5: '6',6: '7',7: '8',
#     8: '9',9: '9a',10: '10',11: '10a',
#     12: '11',13: '12',14: '12a',15: '13',
#     16: '14',17: '14a',18: '15',19: '16',
#     20: '16a',21: '16b',22: '16c',
#     23: '12b',24: '14b'
# }
# model = nnUnet(args)
# ls = torch.ones((1,1,512,512)).float()
# print(model(ls).shape)