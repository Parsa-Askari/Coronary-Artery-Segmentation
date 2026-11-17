import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
###IE###
###SS###


class CBAM_BLOCK(nn.Module):
    def __init__(self, c, ratio=2, spatial_kernel=7):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(output_size=1)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        
        self.mlp = nn.Sequential(
            nn.Conv2d(c, c//ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(c//ratio, c, kernel_size=1, bias=False)
        )
        
        self.conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=spatial_kernel,
            padding="same",
            bias=False
        )
        
        # Separate instances to isolate hooks cleanly
        self.channel_sigmoid = nn.Sigmoid()
        self.spatial_sigmoid = nn.Sigmoid()

    def _channel_wise_attention(self, x):
        x_max = self.mlp(self.maxpool(x))
        x_avg = self.mlp(self.avgpool(x))
        return self.channel_sigmoid(x_max + x_avg)

    def _spatial_attention(self, x):
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x_mean = torch.mean(x, dim=1, keepdim=True)
        x_cat = torch.cat([x_max, x_mean], dim=1)
        return self.spatial_sigmoid(self.conv(x_cat))

    def forward(self, f):
        f = f * self._channel_wise_attention(f)
        f = f * self._spatial_attention(f)
        return f
    
class MOE_BLOCK(nn.Module):
    def __init__(self,in_c,out_c,k,s=1,p=0):
        super(MOE_BLOCK,self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_c,
            out_channels=out_c,
            stride = s,
            padding = p,
            kernel_size = k,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
    def forward(self,x):
        z = self.conv(x)
        z = self.bn(z)
        z = self.relu(z)
        return z
    
class MOE_MODULE(nn.Module):
    def __init__(self,rp_in_channels,rp_out_channels,dockers_channels,
                 resample_base_size = None,router_counts=4,
                 expert_counts=3,top_k = 2):
        super(MOE_MODULE,self).__init__()
        self.feature_extractor = nn.Sequential(
            MOE_BLOCK(
                in_c = rp_in_channels,
                out_c = rp_out_channels,
                k = 1
            ),
            MOE_BLOCK(
                in_c = rp_out_channels,
                out_c = rp_out_channels,
                k = 1
            )
        )
        self.routers = nn.Sequential(
            nn.Linear(in_features = rp_out_channels,out_features=expert_counts*router_counts)
        )
        self.experts = nn.ModuleList([
            nn.Sequential(
                MOE_BLOCK(
                    in_c = rp_out_channels,
                    out_c = rp_out_channels,
                    k = 1
                ),
                MOE_BLOCK(
                    in_c = rp_out_channels,
                    out_c = rp_out_channels,
                    k = 1
                )
            ) for i in range(1,expert_counts+1)
        ])
        self.docker = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels=rp_out_channels,
                    out_channels=dockers_channels[i+1]["out_c"],
                    kernel_size=1,
                    bias=False
                ),
                nn.ReLU(inplace=True)
            ) for i in range(router_counts)
        ])
        
        self.router_counts = router_counts
        self.expert_counts = expert_counts
        self.top_k = top_k
        self.resample_base_size = resample_base_size

    def _router_balance_loss(self,g_probs):
        """
            - g_probs : (B, router_counts, expert_counts)
        """
        avg_expert_prob = g_probs.mean(dim=0)
        loss = (avg_expert_prob*torch.log(avg_expert_prob*self.expert_counts + 1e-8)).sum(dim=-1).mean()
        return loss
    def _handle_experts(self,E,topk_idx,topk_probs):
        """
            - E : (2*B,rp_out_channels,H,W)
            - topk_idx : (B,2)
            - topk_probs : (B,2)
        """
        B, _ = topk_idx.shape
        _, C, H, W = E.shape

        
        expert_results = torch.zeros_like(E)
        for expert_key,expert in enumerate(self.experts):
            expert_chosen_mask = (topk_idx==expert_key).reshape(-1)
            sample_indices = expert_chosen_mask.nonzero().flatten()
            if(len(sample_indices)>0):
                expert_input = E[sample_indices]
                result = expert(expert_input)
                expert_results = expert_results.index_add(dim = 0, source = result,index=sample_indices)
            elif self.training:
                random_index = torch.randint(0,E.shape[0],(1,),device=E.device)
                expert_input = E[random_index]
                result = expert(expert_input)
                expert_results = expert_results.index_add(dim = 0, source = result,index=random_index)
        # [B, K] -> [B*K, 1, 1, 1]
        weights = topk_probs.reshape(B * self.top_k, 1, 1, 1)
        expert_results = expert_results * weights
        expert_results = expert_results.view(B, self.top_k, C, H, W)
        output = expert_results.sum(dim=1) #B,rp_out_channels,H,W

        return output
    def forward(self,skips):
        """
        skips : {i : (B,C_i,H//(2*i),W//(2*i)) for i in (1,depth)}
        """

        """
                resampling
        """
        E = []
        if(self.resample_base_size):
            new_H,new_W = self.resample_base_size
        else:
            B,C,new_H,new_W = skips[2].shape 

        for d , skip in sorted(skips.items()):
            new_s = F.interpolate(skip,size=(new_H,new_W),mode="bilinear", align_corners=False)
            E.append(new_s)
        """
                router results
        """
        E = torch.cat(E,dim=1)
        E = self.feature_extractor(E) #(B,rp_out_channels,H,W)
        E_max_pooled = torch.flatten(
            F.adaptive_max_pool2d(E,output_size=1),
            start_dim=1
        ) # (B,rp_out_channels)
        G = self.routers(E_max_pooled) #(B,expert_counts*router_counts)
        G = G.view(-1,self.router_counts,self.expert_counts)#(B,router_counts,expert_counts)

        g_probs = F.softmax(G,dim=-1)
        b_loss = self._router_balance_loss(g_probs=g_probs)

        topk_vals, topk_idx = torch.topk(G, k=self.top_k, dim=-1) # (B,router_counts,2),(B,router_counts,2) each index = the chosen expert
        topk_probs = F.softmax(topk_vals,dim=2)

        B,rp_out_channels,H,W = E.shape
        E = E.unsqueeze(1).expand(B,self.top_k,rp_out_channels,H,W).reshape(-1,rp_out_channels,H,W)
        expert_results = []
        for r_index in range(self.router_counts):
            result = self._handle_experts(E,topk_idx[:,r_index],topk_probs[:,r_index])
            expert_results.append(result)
                    
        """
                docker computation
        """
        
        # print(self.proj.keys())
        results = {}
        for i, key in enumerate(sorted(skips.keys())):
            out = expert_results[i]
            out = F.interpolate(out,size=skips[key].shape[2:],mode="bilinear", align_corners=False)
            moe_out = self.docker[i](out)
            results[key] = moe_out
            
        return results , b_loss

class UnetBlocks(nn.Module):
    def __init__(self,in_c,out_c,k,s=1,p=0):
        super(UnetBlocks,self).__init__()
        self.layers =  nn.Sequential( 
            nn.Conv2d(
                in_channels = in_c , 
                out_channels = out_c ,
                kernel_size=k, 
                stride = s ,
                padding = p,
                bias=False
            ),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        return self.layers(x)

class EncoderBlocks(nn.Module):
    def __init__(self,in_c,out_c,cbam_args,dropout_args,normalization,
                 do_downsample=True,
                 conv_co=4,bias=False):
        super(EncoderBlocks,self).__init__()
        self.start_conv =  nn.Sequential(
                nn.Conv2d(
                    in_channels = in_c,
                    out_channels=out_c,
                    kernel_size=3,
                    padding=1,
                    stride = 2 if do_downsample else 1,
                    bias=bias
                ),
                normalization(out_c,affine= True),
                # nn.GroupNorm(out_c // 4, out_c),
                nn.ReLU(),
                nn.Dropout2d(p = dropout_args["p"]) if dropout_args["use"] else nn.Identity(),

                nn.Conv2d(in_channels = out_c,out_channels=out_c,kernel_size=3,padding=1,bias=bias),
                normalization(out_c,affine= True),
                # nn.GroupNorm(out_c // 4, out_c),
                CBAM_BLOCK(
                    c=out_c,
                    ratio=cbam_args["ratio"],
                    spatial_kernel=cbam_args["spatial_kernel"]
                ) if cbam_args["use_cbam"] else nn.Identity()
                
            )

        self.enc_convs = nn.ModuleDict({
                f"enc_conv_({i})":nn.Sequential(
                    nn.Conv2d(in_channels = out_c,out_channels=out_c,kernel_size=3,padding=1,bias=bias),
                    normalization(out_c,affine= True),
                    # nn.GroupNorm(out_c // 4, out_c),
                    nn.ReLU(),
                    nn.Dropout2d(p = dropout_args["p"]) if dropout_args["use"] else nn.Identity(),

                    nn.Conv2d(in_channels = out_c,out_channels=out_c,kernel_size=3,padding=1,bias=bias),
                    normalization(out_c,affine= True),
                    # nn.GroupNorm(out_c // 4, out_c),
                    CBAM_BLOCK(
                        c=out_c,
                        ratio=cbam_args["ratio"],
                        spatial_kernel=cbam_args["spatial_kernel"]
                    ) if cbam_args["use_cbam"] else nn.Identity(),
           

                ) for i in range(conv_co-1)
            }
        )
        
        self.second_relu = nn.ReLU()

        if(do_downsample):
            self.res_conv = nn.Sequential(
                nn.Conv2d(
                    in_channels = in_c,
                    out_channels = out_c,
                    kernel_size = 1,
                    stride = 2,
                    padding = 0,
                    bias=bias
                ),
                normalization(out_c,affine= True)
                # nn.GroupNorm(out_c // 4, out_c)
            )

        

        self.do_downsample = do_downsample
    def forward(self,x,img=None):
        if(self.do_downsample):
            res = self.res_conv(x)
        else:
            res = x   

        z = self.start_conv(x)

        z = z + res
        z = self.second_relu(z)
        for i in range(len(self.enc_convs)):
            layer = self.enc_convs[f"enc_conv_({i})"]
            z_new = layer(z)
            z_new = z_new + z
            z_new = self.second_relu(z_new)
            z = z_new

        
        return z , z    

class Encoder(nn.Module):
    def __init__(self,encoder_channels,cbam_args,dropout_args,norm):
        """
            encoder_channels : [channels]
        """
        super(Encoder,self).__init__()
        self._set_temps()

        normalization = getattr(nn,norm)
        bias = True if norm=="InstanceNorm2d" else False

        self.base_conv = nn.Sequential(
            nn.Conv2d(
                in_channels = encoder_channels[0]["in_c"],
                out_channels = encoder_channels[0]["out_c"],
                kernel_size = 3,
                padding = 1,
                stride = 1,
                bias=bias
            ),
            normalization(encoder_channels[0]["out_c"],affine= True),
            # nn.GroupNorm(encoder_channels[0]["out_c"] // 4, encoder_channels[0]["out_c"]),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p = dropout_args["p"]) if dropout_args["use"] else nn.Identity(),
        )

        self.maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        # self.temp_encoder = nn.Sequential(
        #     nn.Conv2d(in_channels=3,out_channels=64,kernel_size=1),
        #     normalization(64,affine= True),
        #     nn.ReLU(inplace=True)
        # )

        # self.processor_encoder = nn.Sequential(
        #     nn.Conv2d(64+encoder_channels[0]["out_c"],
        #               out_channels=encoder_channels[0]["out_c"],kernel_size=3,padding=1),
        #     normalization(encoder_channels[0]["out_c"],affine= True),
        #     nn.ReLU(inplace=True)
        # )

        # self.base_conv = EncoderBlocks(
        #     in_c = encoder_channels[0]["in_c"] , 
        #     out_c = encoder_channels[0]["out_c"],
        #     do_downsample = False,
        #     conv_co = 2,
        #     cbam_args = cbam_args,
        #     normalization = normalization,
        #     bias = bias
        # )
        # self.maxpool = nn.Sequential(
        #     nn.Conv2d(
        #         in_channels = encoder_channels[0]["out_c"],
        #         out_channels=encoder_channels[0]["out_c"],
        #         kernel_size=3,
        #         padding=1,
        #         stride = 2,
        #         bias=bias
        #     ),
        #     normalization(encoder_channels[0]["out_c"]),
        #     nn.ReLU(),
        # )

        self.model = nn.ModuleDict({
            f"encoder_block_[{i}]":EncoderBlocks(
                in_c = encoder_channels[i]["in_c"] , 
                out_c = encoder_channels[i]["out_c"],
                do_downsample = encoder_channels[i]["do_downsample"],
                conv_co = encoder_channels[i]["conv_co"],
                cbam_args = cbam_args,
                dropout_args = dropout_args,
                normalization = normalization,
                bias = bias

            ) for i in range(1,len(encoder_channels))
        })
        # resnet = models.resnet34(pretrained=False)
        # self.filters_resnet = [64, 64, 128, 256, 512]
        # self.filters_decoder = [32, 64, 128, 256, 512]
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(3, 64, 3, 1, 1, bias=True),
        #     nn.BatchNorm2d(self.filters_resnet[0]),
        #     nn.ReLU(inplace=True)
        # )
        # self.maxpool = nn.MaxPool2d(2, 2)

        # self.conv2 = resnet.layer1 
        # self.conv3 = resnet.layer2 
        # self.conv4 = resnet.layer3 
        # self.conv5 = resnet.layer4

    def _set_temps(self):
        first = (np.load("./dataset/syntax/labels/train/411.npy")!=0).astype(int)
        second = (np.load("./dataset/syntax/labels/train/500.npy")!=0).astype(int)
        third = (np.load("./dataset/syntax/labels/train/69.npy")!=0).astype(int)
        l_template_mask = np.stack([first,second,third],axis=0)
        r_first = (np.load("./dataset/syntax/labels/train/933.npy")!=0).astype(int)
        r_template_mask = np.stack([r_first,r_first,r_first],axis=0)

        r_template_mask = torch.from_numpy(r_template_mask).float()
        l_template_mask = torch.from_numpy(l_template_mask).float()
        self.temps = torch.stack([r_template_mask,l_template_mask],dim=0).cuda()
    def forward(self,img,sides):
        """
            - img : (B,1,H,W)
        """
        
        # temp_batch = self.temps[sides]
        inp = img

        skip_o = self.base_conv(inp)
        # temp_o = self.temp_encoder(temp_batch)
        # skip_o = self.processor_encoder(torch.cat([temp_o,skip_o],dim=1))


        out = self.maxpool(skip_o)
        skips = {1:skip_o}
        for i in range(len(self.model)):
            out , skip= self.model[f"encoder_block_[{i+1}]"](out)
      
            if(i!=len(self.model)-1):
                skips[i+2]= skip
        
        
        return skips , out
        # e1 = self.conv1(img)           # (B, 64, 224, 224)
        # e1_maxp = self.maxpool(e1)   # (B, 64, 112, 112)
        # e2 = self.conv2(e1_maxp)     # (B, 64, 112, 112)
        # e3 = self.conv3(e2)          # (B, 128, 56, 56)
        # e4 = self.conv4(e3)          # (B, 256, 28, 28)
        # e5 = self.conv5(e4) 
        # skips = {1:e1,2:e2,3:e3,4:e4}
        # return skips , e5

# class DecoderBlocks(nn.Module):
#     def __init__(self, in_c, skip_c, out_c, cbam_args, norm,
#                  use_skip_proc=True, use_attn_gate=True, attn_floor=0.3,
#                  deep_fuse=True):
#         super().__init__()
#         normalization = getattr(nn, norm)
#         bias = True if norm == "InstanceNorm2d" else False
#         up_c = in_c // 2
 
#         self.up_conv = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
#             nn.Conv2d(in_c, up_c, 1, bias=bias),
#             normalization(up_c,affine= True), nn.ReLU(inplace=True),
#             # nn.GroupNorm(up_c // 4, up_c),
            
#         )
 
#         self.use_skip_proc = use_skip_proc
#         if use_skip_proc:
#             self.skip_proc = nn.Sequential(
#                 nn.Conv2d(skip_c, skip_c, 3, padding=1, bias=bias),
#                 normalization(skip_c,affine= True),
#                 # nn.GroupNorm(skip_c // 4, skip_c), 
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(skip_c, skip_c, 3, padding=1, bias=bias),
#                 normalization(skip_c,affine= True),
#                 # nn.GroupNorm(skip_c // 4, skip_c)
#             )
#             self.skip_relu = nn.ReLU(inplace=True)
 
#         self.use_attn_gate = use_attn_gate
#         self.attn_floor = attn_floor
#         if use_attn_gate:
#             f_int = max(out_c // 2, 16)
#             self.Wg = nn.Conv2d(up_c, f_int, 1, bias=True)
#             self.Wx = nn.Conv2d(skip_c, f_int, 1, bias=True)
#             self.psi = nn.Conv2d(f_int, 1, 1, bias=True)
#             self.attn_relu = nn.ReLU(inplace=True)
 
#         cat_c = skip_c + up_c
#         layers = [
#             nn.Conv2d(cat_c, out_c, 3, padding=1, bias=bias),
#             normalization(out_c,affine= True), 
#             # nn.GroupNorm(out_c // 4, out_c), 
#             nn.ReLU(inplace=True)
#         ]
#         if deep_fuse:
#             layers += [
#                 nn.Conv2d(out_c, out_c, 3, padding=1, bias=bias),
#                 normalization(out_c,affine= True)
#                 # nn.GroupNorm(out_c // 4, out_c), 
#             ]
#         self.convs = nn.Sequential(*layers)
 
#         self.res_proj = (
#             nn.Sequential(
#                 nn.Conv2d(cat_c, out_c, 1, bias=bias),
#                 normalization(out_c,affine= True)
#                 # nn.GroupNorm(out_c // 4, out_c), 
#                 )if cat_c != out_c else nn.Identity())
#         self.relu = nn.ReLU(inplace=True)
#         self.cbam_block = (CBAM_BLOCK(c=out_c, ratio=cbam_args["ratio"],
#                                       spatial_kernel=cbam_args["spatial_kernel"])
#                            if cbam_args["use_cbam"] else nn.Identity())
 
#     def _attention(self, skip, up):
#         a = self.attn_relu(self.Wg(up) + self.Wx(skip))
#         a = torch.sigmoid(self.psi(a))                      # (B,1,H,W)
#         a = self.attn_floor + (1.0 - self.attn_floor) * a   # floor-protected
#         return skip * a
 
#     def forward(self, skip, inp):
#         up = self.up_conv(inp)
#         if self.use_skip_proc:
#             skip = self.skip_relu(self.skip_proc(skip) + skip)
#         if self.use_attn_gate:
#             skip = self._attention(skip, up)
#         z = torch.cat([skip, up], dim=1)
#         out = self.relu(self.convs(z) + self.res_proj(z))
#         return self.cbam_block(out)
 
 
# class Decoder(nn.Module):

#     def __init__(self, decoder_channels, cbam_args, dsv, class_count, norm,
#                  fine_blocks=3, deep_fuse_all=True):
#         super().__init__()
#         self.model = nn.ModuleDict()
#         if dsv:
#             self.dsv_convs = nn.ModuleDict()
#         for i, channels in decoder_channels.items():
#             fine = i <= fine_blocks
#             self.model[f"decoder_block_[{i}]"] = DecoderBlocks(
#                 in_c=channels["in_c"], out_c=channels["out_c"],
#                 skip_c=channels["skip_c"], cbam_args=cbam_args, norm=norm,
#                 use_skip_proc=fine, use_attn_gate=fine, deep_fuse=deep_fuse_all,
#             )
#             if dsv and i != max(decoder_channels.keys()):
#                 self.dsv_convs[f"dsv_block_[{i}]"] = nn.Conv2d(
#                     channels["out_c"], class_count, kernel_size=1)
#         self.dsv = dsv
 
#     def forward(self, x, skips):
#         inp = x
#         key = len(self.model)
#         outs = []
#         while key != 0:
#             inp = self.model[f"decoder_block_[{key}]"](inp=inp, skip=skips[key])
#             if self.dsv and f"dsv_block_[{key}]" in self.dsv_convs:
#                 outs = [self.dsv_convs[f"dsv_block_[{key}]"](inp)] + outs
#             key -= 1
#         return [inp] + outs
class DecoderBlocks(nn.Module):
    def __init__(self,in_c,skip_c,out_c,cbam_args,norm):
        super(DecoderBlocks,self).__init__()
        up_c = in_c // 2

        normalization = getattr(nn,norm)
        bias = True if norm=="InstanceNorm2d" else False
        self.up_conv = nn.Sequential(
            # nn.ConvTranspose2d(
            #     in_channels = in_c,
            #     out_channels = in_c //2,
            #     kernel_size = 2,
            #     stride = 2 
            # ),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_c,in_c // 2,kernel_size=1),
            normalization(in_c//2),
            nn.ReLU(inplace=True)
        )
        
        self.downsample = nn.Sequential(
            nn.Conv2d(skip_c+ (in_c//2), out_c, kernel_size=1, bias=bias),
            normalization(out_c),
        ) if skip_c+ (in_c//2) != out_c else nn.Identity()
        
        # self.skip_denoise = nn.Sequential(
        #     nn.Conv2d(skip_c, up_c, kernel_size=1, bias=bias),
        #     normalization(up_c),
        #     nn.ReLU(inplace=True)

            
        # )

        self.convs = nn.Sequential(
            nn.Conv2d(skip_c+ (in_c//2), out_c,kernel_size = 3,padding = 1,bias=bias),
            normalization(out_c),
            # nn.ReLU(inplace=True),

            # nn.Conv2d(out_c,out_c,kernel_size = 3,padding = 1,bias=bias),
            # normalization(out_c)
        )

        self.relu = nn.ReLU(inplace=True)
        self.cbam_block = CBAM_BLOCK(
            c=out_c,
            ratio=cbam_args["ratio"],
            spatial_kernel=cbam_args["spatial_kernel"]
        ) if cbam_args["use_cbam"] else nn.Identity()
    def forward(self,skip,inp):
        z = self.up_conv(inp)
        # skip = self.skip_denoise(skip)
        # skip = self.cbam_block(skip)
        z1 = torch.cat([skip,z],dim=1)
        res_z = self.downsample(z1)
        z1 = self.convs(z1)
        
        z1 = self.relu(z1 + res_z)
        return self.cbam_block(z1)
    
class Decoder(nn.Module):
    def __init__(self,decoder_channels,cbam_args,dsv,class_count,norm):
        super(Decoder,self).__init__()
        self.model = nn.ModuleDict()
        if(dsv):
            self.dsv_convs = nn.ModuleDict()
        for i,channels in decoder_channels.items():
            self.model[f"decoder_block_[{i}]"] = DecoderBlocks(
                in_c = channels["in_c"],
                out_c = channels["out_c"],
                skip_c = channels["skip_c"],
                cbam_args = cbam_args,
                norm=norm
            )
            if(dsv and i!=max(decoder_channels.keys())):
                
                self.dsv_convs[f"dsv_block_[{i}]"] = nn.Conv2d(
                    in_channels=channels["out_c"],
                    out_channels= class_count,
                    kernel_size=1,
                )
        self.dsv = dsv
    def forward(self,x,skips):
        inp = x
        key = len(self.model)
        outs = []
        while key !=0 :
            model = self.model[f"decoder_block_[{key}]"] 
            inp = model(
                inp = inp,
                skip = skips[key]
            )
            if(self.dsv and f"dsv_block_[{key}]" in self.dsv_convs):
                dsv_out = self.dsv_convs[f"dsv_block_[{key}]"](inp)
                outs = [dsv_out]+ outs
            key-=1
        outs = [inp] + outs
        return outs
    
class MoeUnet(nn.Module):

    def __init__(self,args):
        super(MoeUnet,self).__init__()
        model_args = args["moe_unet"]
        encoder_out_channels = model_args["encoder_out_channels"] # 32 , 64 , 128 , 256

        resample_base_size = model_args["resample_base_size"]
        rp_out_channels = model_args["rp_out_channels"]
        router_counts = model_args["router_counts"]
        expert_counts = model_args["expert_counts"]
        train_moe = model_args["train_moe"]
        top_k = model_args["top_k"]
        cbam_args = model_args["cbam_args"]
        dropout_args = model_args["dropout_args"]
        norm = model_args["norm"]
        

        deep_super_vision = args["deep_super_vision"]
        class_count = args["class_count"]
        image_ch=args["in_c"]

        normalization = getattr(nn,norm)
        bias = True if norm=="InstanceNorm2d" else False

        


        encoder_channels,dockers_channels,decoder_channels,rp_in_channels = self._get_network_settings(
            encoder_out_channels = encoder_out_channels,
            image_ch=image_ch,
            rp_out_channels = rp_out_channels
            
        )

        self.MainEncoder = Encoder(
            encoder_channels=encoder_channels,
            cbam_args =cbam_args,
            dropout_args = dropout_args,
            norm=norm
        )
        
        # self.MainMoeModule = MOE_MODULE(
        #     rp_in_channels=rp_in_channels,
        #     dockers_channels = dockers_channels,
        #     resample_base_size = resample_base_size,
        #     rp_out_channels = rp_out_channels,
        #     router_counts = router_counts,
        #     expert_counts = expert_counts,
        #     top_k = top_k
        # )
        print(decoder_channels)
        self.MainDecoder = Decoder(
            decoder_channels=decoder_channels,
            cbam_args = cbam_args,
            dsv = deep_super_vision,
            class_count = class_count,
            norm = norm
        )
        self.Head = nn.Sequential(
            nn.Conv2d(
                in_channels=encoder_out_channels[0][0],
                out_channels=encoder_out_channels[0][0]//2,
                kernel_size=1,
                bias=bias
            ),
            normalization(encoder_out_channels[0][0]//2,affine= True),
            # nn.GroupNorm(encoder_out_channels[0][0]//8, encoder_out_channels[0][0]//2), 
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=encoder_out_channels[0][0]//2,
                out_channels=class_count,
                kernel_size=1
            )

        )
        self.train_moe = train_moe
    def _side_loss(self,gt_sides):
        loss = F.binary_cross_entropy_with_logits(
            input=self.side_pred.reshape(-1),
            target=gt_sides
        )
        return loss
    def _get_network_settings(
            self,
            encoder_out_channels,
            image_ch,
            rp_out_channels
        ):
        encoder_channels = {
            0:{
                "in_c":image_ch,
                "out_c":encoder_out_channels[0][0],
                "conv_co":0,
                "do_downsample":False}
        }
        dockers_channels = {}
        decoder_channels = {}
        for i,setting in enumerate(encoder_out_channels) : 
            print(setting)
            c,conv_co = setting
            in_c = c if i == 0 else encoder_out_channels[i-1][0]
            out_c = c
            do_downsample = True if i!=0 else False
            encoder_channels[i+1]={
                "in_c":in_c,
                "out_c":out_c,
                "conv_co":conv_co,
                "do_downsample":do_downsample
            }
        print(encoder_channels)
        rp_in_channels = 0
        for i in encoder_channels:
            if(i==max(encoder_channels.keys())):
                continue
                
            dockers_channels[i+1] = {
                "c_in":rp_out_channels,
                "out_c":encoder_channels[i]["out_c"],
            }
            rp_in_channels += encoder_channels[i]["out_c"]
        for key in dockers_channels:
            skip_c = dockers_channels[key]["out_c"]
            in_c = encoder_channels[key]["out_c"]
            out_c = encoder_channels[key]["in_c"]
            decoder_channels[key] = {"skip_c":skip_c,"in_c":in_c,"out_c":out_c}

        return encoder_channels , dockers_channels , decoder_channels , rp_in_channels
    
    def forward(self,x,branch=None):
        skips,out = self.MainEncoder(x,branch)
    
        decoder_outs = self.MainDecoder(skips=skips,x=out)

        decoder_outs[0] = self.Head(decoder_outs[0])
        
        return decoder_outs
    


    
if __name__ == "__main__":
    # s = torch.zeros((2,3,112,112))
    # new_s = torch.flatten(F.adaptive_max_pool2d(s,output_size=1),start_dim=1)
    # print(new_s.shape)
    # x = {
    #     1:torch.zeros((2,32,112,112)).cuda(),
    #     2:torch.zeros((2,64,124,124)).cuda(),
    #     3:torch.zeros((2,128,64,64)).cuda(),
    #     4:torch.zeros((2,224,32,32)).cuda(),
    #     5:torch.zeros((2,448,16,16)).cuda(),
    # }
    # model = MOE_MODULE(
    #     rp_in_channels=32+64+128+224+448,
    #     rp_out_channels=120,
    #     encoder_channels = [32,64,128,224,448],
    #     router_counts=5
    # ).cuda()
    # results = model(x)
    # for key,r in results.items():
    #     print(key,r.shape)
    x = torch.zeros((1,1,512,512)).cuda()
    args = {
        "moe_unet" : {
            "encoder_out_channels" : [[64,2],[128,2],[256,2],[512,2],[512,2],[512,2],[512,2]],
            "cbam_args":{"use_cbam":True,"ratio":16,"spatial_kernel":7},
            "resample_base_size" : None,
            "rp_out_channels" : 64,
            "router_counts":4,
            "expert_counts":3,
            "top_k":2,
            "train_moe":False,
            "norm" : "InstanceNorm2d"#InstanceNorm2d
        },
        "in_c" : 1,
        "class_count" : 2,
        "deep_super_vision":True
    }
    unet = MoeUnet(args).cuda()

    outs,_ = unet(x)
    print(len(outs))

    print(_)
    for o in outs:
        print(o.shape)
    # print(out[0].shape)
    # x = torch.zeros((2,10,512,512)).cuda()
    # model = CBAM_BLOCK(c=10).cuda()
    # print(model(x).shape)
    # pool = nn.AdaptiveAvgPool2d(1)
    # print(pool(x).shape)
