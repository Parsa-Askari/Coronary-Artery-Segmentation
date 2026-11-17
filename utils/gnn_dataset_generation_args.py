import torch
###IE###
###SS###
args = {
    "paths":{
        "base_path" : "./dataset/syntax/",
        "root_path" : "./dataset/syntax/topology_dataset/labels/",
        "output_base_path" : "./outputs",
        "images_folder" : "origianl_preprocessed", # new_preprocessed , origianl_preprocessed , images
        "clahe_folder" : "origianl_preprocessed",
        "generated_folder" : "generated_mask_probs",
        "labels_folder" : "labels",
        "bootstrap_folder" : "bootstrap",
        "hard_vessel_folder" : "hard_vessels_mask",
        "filter_folder" : "filters",
        "bootstrap_seq":"./bootstrap_sequence",
        "topology_folder" : "topology_dataset/labels",
        "side_folder":"./data/",
        "border_cords_folder" : "boarder_removed",
        "skel_folder" : "new_skels",
        "gnn_dataset_masks" : "gnn_dataset_augmented_images"
        
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
        "encoder_out_channels":150
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
    "socre_params":{
        "p":0.4,
        "r":0.6,
        "choice_score" : "dice"
    },
    "image_shape" : (512,512),
    "in_c":3,
    "deep_super_vision":True,
    "bootstrap":False,
    "training_mode":"multi", # binary - multi - cleaning
    "binary_training_mode" : "partial",# full - partial
    "side_mode":"right", #both - left - right
    "report_class_wise":True,
    "attention" : False,
    "batch_size" : 4,
    "num_workers" : 4,
    "device" : "cuda" if torch.cuda.is_available() else "cpu",
    "epochs":5,
    "f_int_scale" : 2,
    "new_vessel_width":2,
    "full_report_cycle" : 10,
    "max_channels":512,
    "loss_type":"tversky",
    "name" : "model[{}]-{}-{}-{}class-{}epo-aux_branchpoint-nobias-cbam-upsample",
    "mask_ce_weights":None,
    "layer_count":5,
    "cut_borders":True,
    ### Loss Functions
    "mask_ce_conf":{
        "imp_coef":1.0
    },
    "mask_frangi_ce_conf": {
        "expert_vessel_pos_weight": 4.0,
        "model_vessel_pos_weight": 3.0,
        "outer_boundary_boost": 4.5,
        "unlabeled_vessel_weight": 0.06,
        "bg_weight": 0.2,
        "kernel_size": 3,
        "imp_coef": 1.0,
        "use_expert_boundary_only": True
    },
    "mask_frangi_dice_conf": {
        "expert_vessel_pos_weight": 4.0,
        "model_vessel_pos_weight": 3.0,
        "outer_boundary_boost": 4.5,
        "unlabeled_vessel_weight": 0.06,
        "kernel_size": 3,
        "bg_weight": 0.3,
        "imp_coef": 1.0,
        "alpha": 0.2,
        "beta": 0.8,
        "use_expert_boundary_only": True
    },
    "ce_loss_conf":{
        "pos_weight":5.0,
        "bg_weight":1.0,
        "small_vessel_weights":10.0,
        "imp_coef":1.0
    },
    "fce_conf":{
        "gamma":2.0,
        "alpha":None
    },
    "tversky_conf":{
        "alpha":0.5,
        "beta":0.5,
        "gamma":1.00,
        "weights":None,
        "imp_coef":1.0
    },
    "aux_loss_conf":{
        "imp_coef" : 0.1
    },
    "cl_dice_conf":{
        "imp_coef":1.0,
        "k":12
    },
    ### Optimizers
    "sgd_conf":{
        "lr" : 0.1,
        "momentum" : 0.99,
        "weight_decay" : 3e-5,
        "nesterov":True
    },
    "adam_conf":{
        "lr" : 0.01,
        "weight_decay" : 3e-5,
    },
    "adamw_conf":{
        "lr" : 0.0001,
        "weight_decay" : 3e-5,
    }
}


if(args["training_mode"] == "binary" or args["training_mode"]=="cleaning"):
    args["class_count"] = 2 
    class_map = {
        1:"fg"
    }

else:
    """
        sides : 
            0 - right side
            1 - left side
    """
    if(args["side_mode"] == "left"):
        args["class_count"] = 18
        class_map = {
            1: '5',2: '6',3: '7',4: '8',
            5: '9',6: '9a',7: '10',8: '10a',
            9: '11',10: '12',11: '12a',12: '13',
            13: '14',14: '14a',15: '15',
            16: '12b',17: '14b'
        }
    elif(args["side_mode"] == "right"):
        args["class_count"] = 9
        class_map = {
            1: '1',2: '2', 3: '3',4: '4',
            5:'16',6: '16a',7: '16b',8: '16c'
        }
        # args["class_count"] = 6
        # class_map = {
        #     1: '1',2: '2', 3: '3',4: '4',
        #     5:'16'
        # }
    else:
        args["class_count"] = 26
        class_map = {
            1: '1',2: '2', 3: '3',4: '4',
            5: '5',6: '6',7: '7',8: '8',
            9: '9',10: '9a',11: '10',12: '10a',
            13: '11',14: '12',15: '12a',16: '13',
            17: '14',18: '14a',19: '15',20: '16',
            21: '16a',22: '16b',23: '16c',
            24: '12b',25: '14b'
        }