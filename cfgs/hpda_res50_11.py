'''
Config Proto
naming convention: 	backbonename-layernum(数字)-method(大写)-备注(小写)
'''

# about model
name = "hpda-res50"
backbone_model_name = "HPDA_res50"
classify_model_name = "MarginCosineProduct" # MarginCosineProduct
resume_net_model = "/home/ubuntu/data2/lk/recognition/pytorch_new/snapshot/hpda_res50_11/backbone_35_checkpoint.pth"
# resume_net_model = None
# resume_net_classifier = None
resume_net_classifier = "/home/ubuntu/data2/lk/recognition/pytorch_new/snapshot/hpda_res50_11/classifier_status_35_checkpoint.pth"

# about file
log_interval = 100
log_path = "./logs/hpda_res50_11_continue.log"
log_pic_path = "./logs/pic/hpda_res50_11_continue/"
save_path = 'snapshot/hpda_res50_11_continue/'
lmdb_path = "/home/ubuntu/data4/lk/data/lmdb_default" # TODO
# lmdb_path = "/home/ubuntu/data4/lk/data/lmdb_mask_augu_full" # TODO

# about gpu
no_cuda = False
gpu_num = 2

# about data
batch_size = 512
datanum =  3923399 # 3923399 6753545 3074286（filter）3913769 # TODO
num_class = 86876 # 86876, 180855 46346（filter）# TODO
lmdb_workers = 4
num_workers = 4

# about LR policy
start_epoch = 1
max_epoch = 70

lr = 0.1
base="epoch"
# step_size =[10, 20, 30] # 3923399/512=7662
step_size =[15, 30, 38, 46, 52, 60] # 3923399/512=7662
# step_size =[5,10,15] # 3923399/512=7662

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
master_port = 22349 # 多个训练任务同时进行ddp训练的时候，保证两个任务的该参数值不同
multiprocessing_distributed=False
SEED = 1337

