'''
Config Proto
naming convention: 	backbonename-layernum(数字)-method(大写)-备注(小写)
'''

# about model
name = "resnet-50-arc"
description = "将最后一个bn去掉"
# about data
data_settings = {
    "training" :{
        "batch_size" : 512, # 768
        "num_workers" : 4,
        "num_class" : 86876, # 86876, 180855 46346（filter）# TODO
        # num_class : 10575 # 86876, 180855 46346（filter）# TODO

        "loader_settings" :
            {
                # lmdb_path = "/home/ubuntu/data4/lk/data/lmdb_default" # TODO
                "lmdb_path" : "/home/ubuntu/data4/lk/data/lmdb_mask_augu_full", # TODO
                # lmdb_path = "/home/ubuntu/data4/lk/data/lmdb_mask_augu_full_casia" # TODO
                "num" :  3923399, # 3923399 6753545 3074286（filter）3913769 # TODO
                # datanum :  493733 # 3923399 6753545 3074286（filter）3913769 # TODO
                
                "max_reader" : 4,        
                "augu_paral": False,
                "ldm68": False,
                "augu_rate": 0,
                "preproc": None,
                "shuffle": True,
            }
        
    }
    
}
## common setting for training and testing
common_settings={
    "backbone":{
        "num" : 1,
        "settings": [
            {
                "backbone_model_name" : "resnet50", # SimpleResnet_split_abla_36
                "args" :{
                    "input_size":[112,112],
                    # "cl": True,
                    "fc_num":2,
                },
                # "resume_net_model" : "/home/ubuntu/data2/lk/recognition/pytorch_new/snapshot/SR_36_ddp_sphereNorm_mix/backbone_10_checkpoint.pth",
                "resume_net_model" : None,
            },
        ]
    },
    "classifier":{
        "num": 2,
        "settings": [
            {
            "classifier_model_name" : "ArcFace", # MarginCosineProduct_sub MarginCosineProduct
            "args" : {
                      "in_features":256,
                      "out_features":data_settings["training"]["num_class"],
                      },
            # "resume_net_classifier" : "/home/ubuntu/data2/lk/recognition/pytorch_new/snapshot/SR_36_ddp_sphereNorm_mix/classifier_status_10_checkpoint.pth",
            "resume_net_classifier" : None,
            "alpha":0.5,
            },
            {
            "classifier_model_name" : "ArcFace", # MarginCosineProduct_sub
            "args" : {
                      "in_features":256,
                      "out_features":data_settings["training"]["num_class"],
                      },
            # "resume_net_classifier" : "/home/ubuntu/data2/lk/recognition/pytorch_new/snapshot/SR_36_ddp_sphereNorm_mix/classifier_status_10_checkpoint.pth",
            "resume_net_classifier" : None,
            "alpha":0.5,
            },
        ]
    },
}
# about gpu
gpu_settings={
    "no_cuda" : False,
    "gpu_num" : 2
}

# about log
log_settings ={
    "training":{
        "log_path" : "./logs/res_50_fc_bothPure_seperate_nofinalbn.log",
        "log_pic_path" : "./logs/pic/res_50_fc_bothPure_seperate_nofinalbn/",
        "save_path" : 'snapshot/res_50_fc_bothPure_seperate_nofinalbn/',
        "log_interval" : 100,
           
    },
    "testing" :{
        "result_path" : "./result/result.txt"
    },
}



other_settings = {
    "resume": False,
    "resume_net_optimizer":None,
    # about LR policy
    "start_epoch" : 1,
    "max_epoch" : 25,

    "lr" : 0.05,
    # step_size =[60000, 140000, 180000,300000] # 6726601/512=13138
    # step_size =[77000, 150000, 200000,300000] # 3923399/512=7662
    "base":"epoch",
    "step_size" :[10, 20, 30], # 3923399/512=7662
    # step_size =[ 210000 ,450000] # 6726601/512=13138
    # step_size = [ 50000 ,150000, 250000, 350000, 400000]

    # about SGD
    "momentum" : 0.9,
    "gama" : 0.1,
    "weight_decay" : 5e-4,
}

environ_settings = {
    "rank" : -1, # 表示进程序号，用于进程间通信，可以用于表示进程的优先级。我们一般设置 rank:0 的主机为 master 节点
    "dist_url":"env://", # 使用环境变量
    "world_size":-1, # 总共有几个 Worker
    "gpu" : None,
    "dist_backend" : 'nccl',
    "distributed" : True,
    "master_port" : 22345, # 多个训练任务同时进行ddp训练的时候，保证两个任务的该参数值不同
    "multiprocessing_distributed":False,
    "SEED" : 1337,
}

