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
        self.deep_super_vision = args["deep_super_vision"]
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
                    f_int_scale=f_int_scale,
                    dsv = self.deep_super_vision,
                    class_count=class_count
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
        
        outputs = []
        # print(len(self.decoders))
        for i in range(len(self.decoders) - 1, -1, -1):
            # print("2")
            decoder = self.decoders[i]
            skip = skips[i+1]
            x_out,gate_out,dsv_out = decoder(x_in,skip,gate_in)
            
            if(dsv_out!=None):
                outputs = [dsv_out] + outputs
            x_in=x_out
            gate_in=gate_out
        # print(x_in.shape)
        outputs = [self.head(
            x_in = x_in,
            x_skip = skips[0],
            x_gate = gate_in
        )] + outputs
        return outputs
if __name__ == "__main__":
    args = {
        "base_path" : "../arcade/nnUnet_dataset/syntax",
        "in_c" : 1,
        "base_channel" :32,
        "image_shape" : (512,512),
        "class_count" : 26 ,
        "attention" : True,
        "k":40,
        "batch_size" : 10,
        "num_workers" : 10,
        "device" : "cuda" if torch.cuda.is_available() else "cpu",
        "lr" : 0.01,
        "momentum" : 0.99,
        "weight_decay" : 3e-5,
        "epcohs":30,
        "f_int_scale" : 2,
        "full_report_cycle" : 10,
        "max_channels":512,
        "input_channels":1,
        "loss_type":"dice loss",
        "alpha":0.75,
        "beta":0.25,
        "gamma":1.00,
        "f_gamma":2.0,
        "f_loss_scale":1,
        "loss_coefs":{"CE":1.0,"Second":1.0},
        "output_base_path" : "./outputs",
        "name" : "Attention7-AllClass",
        "deep_super_vision" : True
    }
    class_map = {
        1: '1',2: '2', 3: '3',4: '4',
        5: '5',6: '6',7: '7',8: '8',
        9: '9',10: '9a',11: '10',12: '10a',
        13: '11',14: '12',15: '12a',16: '13',
        17: '14',18: '14a',19: '15',20: '16',
        21: '16a',22: '16b',23: '16c',
        24: '12b',25: '14b'
    }
    model = nnUnet(args).to("cuda")
    ls = torch.ones((10,1,512,512)).float().to("cuda")
    outs = model(ls)
    # for out in outs:
    #     print(out.shape)

    # """
    # torch.Size([10, 32, 256, 256])
    # torch.Size([10, 64, 128, 128])
    # torch.Size([10, 128, 64, 64])
    # torch.Size([10, 256, 32, 32])
    # torch.Size([10, 512, 16, 16])
    # torch.Size([10, 512, 8, 8])
    # torch.Size([10, 512, 4, 4])
    # """