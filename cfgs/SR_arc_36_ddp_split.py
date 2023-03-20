'''
Config Proto
naming convention: 	backbonename-layernum(数字)-method(大写)-备注(小写)
'''

# about model
# name = "resnet50_split_layer2_MixArgu"
name = "SResnet_split_layer4"
# backbone_model_name = "resnet50_split" 
backbone_model_name = "SimpleResnet_split_36"
classify_model_name = "ArcFace" # MarginCosineProduct
# resume_net_model = "/home/ubuntu/data2/lk/recognition/pytorch_new/snapshot/res_50_ddp_split/backbone_49_checkpoint.pth"
resume_net_model = None
resume_net_classifier = None
# resume_net_classifier = "/home/ubuntu/data2/lk/recognition/pytorch_new/snapshot/res_50_ddp_split/classifier_status_49_checkpoint.pth"

# about gpu
no_cuda = False
gpu_num = 1

# about file
log_interval = 100
log_path = "./logs/SResnet_36_split_drop_0.1_Arc.log"
log_pic_path = "./logs/pic/SResnet_36_split_drop_0.1_Arc/"
save_path = 'snapshot/SResnet_36_split_drop_0.1_Arc/'
# lmdb_path = "/home/ubuntu/data4/lk/data/lmdb_default" # TODO
lmdb_path = "/home/ubuntu/data4/lk/data/lmdb_mask_augu_full" # TODO

# about data
batch_size = 512 # 768
# batch_size = 512 # 768
datanum =  3923399 # 3923399 6753545 3074286（filter）3913769 # TODO
num_class = 86876 # 86876, 180855 46346（filter）# TODO
lmdb_workers = 4
num_workers = 4

# about LR policy
start_epoch = 1
# max_epoch = 65
max_epoch = 38

lr = 0.05
# step_size =[60000, 140000, 180000,300000] # 6726601/512=13138
# step_size =[77000, 150000, 200000,300000] # 3923399/512=7662
base="epoch"
step_size =[10, 20, 30, 40, 50,60] # 3923399/512=7662
# step_size =[ 210000 ,450000] # 6726601/512=13138
# step_size = [ 50000 ,150000, 250000, 350000, 400000]

# about SGD
momentum = 0.9
gama = 0.1
weight_decay = 5e-4

rank = -1 # 表示进程序号，用于进程间通信，可以用于表示进程的优先级。我们一般设置 rank=0 的主机为 master 节点
dist_url="env://" # 使用环境变量
world_size=-1
gpu = None
dist_backend = 'nccl'
distributed = True
master_port = 32345 # 多个训练任务同时进行ddp训练的时候，保证两个任务的该参数值不同
# master_port = 22345 # 多个训练任务同时进行ddp训练的时候，保证两个任务的该参数值不同
multiprocessing_distributed=False
SEED = 1337
