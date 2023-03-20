'''
Config Proto
naming convention: 	backbonename-layernum(数字)-method(大写)-备注(小写)
'''

# about model
# name = "resnet50_split_layer2_MixArgu"
name = "SResnet_split"
# backbone_model_name = "resnet50_split" 
backbone_model_name = "SimpleResnet_36" # SimpleResnet_fullsplit_36
# classify_model_name = "MarginCosineProduct" # DCFace
# resume_net_model = "/home/ubuntu/data2/lk/recognition/pytorch_new/snapshot/res_50_ddp_split/backbone_49_checkpoint.pth"
margin = 0.2
resume_net_model = None
resume_net_classifier = None
# resume_net_classifier = "/home/ubuntu/data2/lk/recognition/pytorch_new/snapshot/res_50_ddp_split/classifier_status_49_checkpoint.pth"

# about gpu
no_cuda = False
gpu_num = 1

# about file
log_interval = 100
log_path = "./logs/SResnet_36_triplet.log"
log_pic_path = "./logs/pic/SResnet_36_triplet/"
save_path = 'snapshot/SResnet_36_triplet/'
# lmdb_path = "/home/ubuntu/data4/lk/data/lmdb_default" # TODO
lmdb_path = "/home/ubuntu/data4/lk/data/lmdb_mask_augu_full" # TODO
# lmdb_path = "/home/ubuntu/data4/lk/data/lmdb_mask_augu_full_casia" # TODO

# about data
triplet_batch_size = 544 # 768
# batch_size = 512 # 768
datanum = 3923399  # 3923399 6753545 3074286（filter）3913769 493733# TODO
num_class = 86876 # 86876, 180855 46346（filter）10575# TODO
id_dict_path = None #"/home/ubuntu/data4/lk/data/lmdb_mask_augu_full/id-dict.npy"
training_triplets_path = "/home/ubuntu/data2/lk/recognition/pytorch_new/temp_datasets/generated_triplets/epoch_1_training_triplets_2560000_identities_32_batch_544.npy"
lmdb_workers = 4
num_workers = 4

num_human_identities_per_batch = 32
use_semihard_negatives = False
iterations_per_epoch = 5000

# about LR policy
start_epoch = 1
# max_epoch = 65
max_epoch = 150

lr = 0.1
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