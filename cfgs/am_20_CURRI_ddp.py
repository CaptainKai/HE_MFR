'''
Config Proto
naming convention: 	backbonename-layernum(数字)-method(大写)-备注(小写)
'''

# about model
name = "amsoft-20"
backbone_model_name = "SimpleResnet_20"
classify_model_name = "CurricularFace" #MarginCosineProduct
# resume_net_model = "/home/ubuntu/data2/lk/recognition/pytorch/snapshot/SimpleResnet_20_more/amsoft_33_checkpoint.pth"
resume_net_model = None
resume_net_classifier = None
# resume_net_classifier = "/home/ubuntu/data2/lk/recognition/pytorch/snapshot/SimpleResnet_20_more/amsoft-classify_33_checkpoint.pth"

# about gpu
no_cuda = False
gpu_num = 2

# about file
log_interval = 100
log_path = "./logs/am_20_CURRI_ddp_test.log"
log_pic_path = "./logs/pic/am_20_CURRI_ddp_test/"
save_path = 'snapshot/am_20_CURRI_ddp_test/'
lmdb_path = "/home/ubuntu/data4/lk/data/lmdb_default" # TODO

# about data
batch_size = 512
datanum =  3923399 # 3923399 6753545 3074286（filter）3913769 # TODO
num_class = 86876 # 86876, 180855 46346（filter）# TODO
lmdb_workers = 4
num_workers = 4

# about LR policy
max_epoch = 38

lr = 0.1
# step_size =[60000, 140000, 180000,300000] # 6726601/512=13138
# step_size =[77000, 150000, 200000,300000] # 3923399/512=7662
base="epoch"
step_size =[10, 20, 30] # 3923399/512=7662
# step_size =[ 210000 ,450000] # 6726601/512=13138
# step_size = [ 50000 ,150000, 250000, 350000, 400000]

# about SGD
momentum = 0.9
gama = 0.1
weight_decay = 5e-4

rank = -1 # 表示进程序号，用于进程间通信，可以用于表示进程的优先级。我们一般设置 rank=0 的主机为 master 节点
dist_url="env://"
world_size=-1
gpu = None # specify your GPU ids
dist_backend = 'nccl'
distributed = True
multiprocessing_distributed=False
SEED = 1337
