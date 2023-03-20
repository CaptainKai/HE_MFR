'''
Config Proto
naming convention: 	backbonename-layernum(数字)-method(大写)-备注(小写)
'''

# about model
name = "hpda-res50"
backbone_model_name = "HPDA_res50"
classify_model_name = "MarginCosineProduct" # MarginCosineProduct
# resume_net_model = "/home/ubuntu/data2/lk/recognition/pytorch/snapshot/hpda_hbp_sb_31/backbone_65_checkpoint.pth"
resume_net_model = "/home/ubuntu/data2/lk/recognition/pytorch_new/snapshot/hpda_res50_31_test/backbone_120_checkpoint.pth"
# resume_net_model = None
# resume_net_classifier = None
# resume_net_classifier = "/home/ubuntu/data2/lk/recognition/pytorch/snapshot/hpda_hbp_sb_31/classifier_status_65_checkpoint.pth"
resume_net_classifier = "/home/ubuntu/data2/lk/recognition/pytorch_new/snapshot/hpda_res50_31_test/classifier_status_120_checkpoint.pth"

# about gpu
no_cuda = False
gpu_num = 2

# about file
lmdb_path = "/home/ubuntu/data4/lk/data/lmdb_mask_augu_full" # TODO

# about data
batch_size = 1
datanum =  3923399 # 3923399 6753545 3074286（filter）3913769 # TODO
num_class = 86876 # 86876, 180855 46346（filter）# TODO
lmdb_workers = 4
num_workers = 4

# about LR policy
max_epoch = 38

lr = 0.1
base="epoch"
step_size =[15, 30, 45, 60, 75, 90, 105] # 3923399/512=7662

# about SGD
momentum = 0.9
gama = 0.1
weight_decay = 5e-4
