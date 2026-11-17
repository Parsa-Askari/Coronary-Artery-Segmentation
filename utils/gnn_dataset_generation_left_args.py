import torch
###IE###
from .helpers import left_mapping_function,get_collapse
###SS###

def return_gnn_maps (class_count):
    """ 
        class_count 4 : LDA , LCX , 5
    """
    if(class_count == 4):
        root_locations = {
            1 : "left",
            2 : "left",
            3 : "topleft"
        }
        index_maps = {
            1:[2,3,4,5,6,7], #6,7,8,9,9a,10
            2:[9,10,11,12,13,14,15,16,17], #11,12,12a,13,14,15,12b,14b
            3:[1],#5,6,11
        }
        extra_index_maps = {
            1:[],
            2:[],
            3:[]
        }
        root_idx_dict = {
            
        }
    return root_locations , index_maps,extra_index_maps,root_idx_dict

args = {
    "paths":{
        "base_path" : "./dataset/syntax/",
        "root_path" : "./dataset/syntax/topology_dataset/labels/",
        "output_base_path" : "./outputs",
        "images_folder" : "origianl_preprocessed", # new_preprocessed , origianl_preprocessed , images
        "clahe_folder" : "origianl_preprocessed",
        "generated_folder" : "generated_mask_probs",
        "labels_folder" : "labels",
        "ignore_labels_folder" : "k_fold_preds/2026-08-01_21-19-26/extra_labels",
        "bootstrap_folder" : "bootstrap",
        "hard_vessel_folder" : "hard_vessels_mask",
        "filter_folder" : "filters",
        "bootstrap_seq":"./bootstrap_sequence",
        "topology_folder" : "topology_dataset/labels",
        "side_folder":"./data/",
        "border_cords_folder" : "boarder_removed",
        "skel_folder" : "recall_skel"
        
    },
    "memory_unet":{
        "layer_count":4,
        "channel_max":256,
        "downsampling":"maxpool",
        "dropout" : False,
        "normaliztion":"InstanceNorm2d",#InstanceNorm2d
        "activation" : "LeakyReLU",
        "first_layer_out_c" : 32,
    },
    "swin":{
        "encoder_out_channels":150,
        "warmup_ep" : 0
    },
    "moe_unet" : {                 # 256,   128,    64,    32,     16,     8,      4
        # "encoder_out_channels" : [[128,2],[256,3],[512,5],[1024,2],[1024,3]],
        "encoder_out_channels":[[128,2],[256,3],[512,6],[1024,3],[1024,4]],
        "cbam_args":{"use_cbam":True,"ratio":16,"spatial_kernel":7},
        "dropout_args":{"use":True,"p":0.2},
        "resample_base_size" : None,
        "rp_out_channels" : 64,
        "router_counts":5,
        "expert_counts":3,
        "top_k":2,
        "train_moe":False,
        "norm" : "InstanceNorm2d"#InstanceNorm2d
    },
    "dsc_net":{
        "kernel_size" : 9,
        "extend_scope": 1,
        "if_offset" : True,
        "dim" : 1,
        "numer" : 16
    },
    "socre_params":{
        "p":0.4,
        "r":0.6,
        "choice_score" : "dice"
    },
    "image_shape" : (512,512),
    "in_c":3,
    "deep_super_vision":False,
    "bootstrap":False,
    "training_mode":"multi", # binary - multi - cleaning
    "binary_training_mode" : "partial",# full - partial
    "side_mode":"left", #both - left - right
    "report_class_wise":True,
    "attention" : False,
    "batch_size" : 1,
    "num_workers" : 0,
    "device" : "cuda" if torch.cuda.is_available() else "cpu",
    "epochs":50,
    "f_int_scale" : 2,
    "new_vessel_width":2,
    "full_report_cycle" : 10,
    "max_channels":512,
    "loss_type":"tversky",
    "name" : "model[{}]-{}-{}-{}class-{}epo-aux_branchpoint-nobias-cbam-upsample",
    "mask_ce_weights":None,
    "layer_count":5,
    "cut_borders":True
}
old_class_map = {
    1: '5',2: '6',3: '7',4: '8',
    5: '9',6: '9a',7: '10',8: '10a',
    9: '11',10: '12',11: '12a',12: '13',
    13: '14',14: '14a',15: '15',
    16: '12b',17: '14b'
}
cnn_classes = []
# class_map = {
#     1: '5', 2: '6', 3: '7', 4: '8', 5: '11', 6: '13'
# }
collapse , class_map = get_collapse(option=0)
print(class_map)
print(collapse)
root_locations , index_maps,extra_index_maps,root_idx_dict = return_gnn_maps(class_count=4)
# root_locations = {
#     1 : "topleft",
#     2 : "topleft",
#     3 : "topleft",
#     4 : "topleft",
#     5 : "topleft",
#     6 : "topleft",
#     7 : "topleft",
#     8 : "topleft",
#     9 : "topleft",
#     10 : "topleft",
#     11 : "topleft",
#     12 : "topleft",
# }

# index_maps = {
#     1:[1,10], #5,12 
#     2:[2,5], #6,9
#     3:[3,6,7],#7,9a,10
#     4:[4,8], #8
#     5:[9,11,16],#11,12a,12b
#     6:[12,13,14,17,15],#13,14,14a,14b,15
# }
# extra_index_maps = {
#     1:[2,9], #6,11
#     2:[1,3,7], #5,7,9a
#     3:[4,2],#8,6
#     4:[3], #7
#     5:[1,10,12],#5,12,13
#     6:[9,11,16],#11,12a,12b
# }
extra_index_indx = {
    i:[len(index_maps[i])+j+1 for j in range(len(extra_index_maps[i]))] for i in index_maps
}
for key in index_maps:
    index_maps[key] += extra_index_maps[key]

# index_maps = {
#     1:[1,10, 2,9], #5,12,6,11
#     2:[2,5, 1,3,6], #6,9,5,7,9a
#     3:[3,6,7, 4,2,4,8],#7,9a,10,6,8,10a
#     4:[4,8, 3], #8,10a,7
#     5:[9,11,16, 1,12,10,13],#11,12a,12b,5,13,12,14
#     6:[12,13,14,17,15, 9,11,16],#13,14,14a,14b,15,11,12a,12b
# }

# index_maps = {
#     1:[1], #5
#     2:[10],#12
#     3:[2],#6
#     4:[5],#9
#     5:[3],#7
#     6:[6,7],#9a , 10
#     7:[4,8],#8,10a
#     8:[9],#11
#     9:[11,16],#12a,12b
#     10:[12], #13
#     11:[13,14,17,15] #14,14a,14b,15
# }


validator_class_maps = {}
for i,ls in index_maps.items():
    validator_class_maps[i]={0:"bg"}
    for j,idx in enumerate(ls):
        validator_class_maps[i][j+1] = old_class_map[idx]

print(validator_class_maps)

reversed_map = {v: k for k, v in class_map.items()}

lut = left_mapping_function(
    old_class_map = old_class_map,
    new_class_map = class_map,
    collapse = collapse
)
args["class_count"] = len(old_class_map)+1

dataset_keys = {
    "x":0,
    "y":1,
    "geo_root":2,
    "euc_root":3,
    "man_root":4,
    "reachable":5,
    "degree":6,
    "radius":7,
    "branch_depth":8
}
base_key_size= len(dataset_keys) 
for i in range(len(class_map)+1):
    dataset_keys[f"prob_{i}"]=i+base_key_size