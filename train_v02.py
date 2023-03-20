import os
import time
import datetime
import torch
import torch.utils.data
import torch.optim
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

# import pycuda.driver as cuda
# cuda.init()
# cudnn.benchmark = True # 加速声明，让内置的 cuDNN 的 auto-tuner 自动寻找最适合当前配置的高效算法 
# # 需要注意的是，添加了这行代码之后，测试或者finetune也需要添加这行代码
# 引入tensorboardX,tqdm
# 全局的loss 改为epoch内
# 修复会覆盖log文件的bug
## 常用的训练版本，未引入分布式训练
import sys
import numpy as np
import argparse

from utils import Logger, Timer, compute_accuracy_multi, adjust_learning_rate
from util.utils import AverageMeter
from tensorboardX import SummaryWriter
from tqdm import tqdm

_t = {'forward_pass': Timer(), 'backward_pass': Timer(), 'data_pass': Timer(), "train_pass": Timer()}

iteration = 0

def train(train_loader, model, classifier, criterion, optimizer, epoch, writer):
    loss_display = AverageMeter()
    accuracy_display = AverageMeter()
    global iteration
    model.train()
    global lr

    _t['data_pass'].tic()
    _t['train_pass'].tic()

    # for data, target in tqdm(iter(train_loader)):
    for batch_idx, (data, target) in  tqdm(enumerate(train_loader, 1)):
        
        _t['data_pass'].toc()
        
        iteration += 1
        lr = adjust_learning_rate(optimizer, configs.lr, iteration, epoch, configs.step_size, lr, configs.gama, base=configs.base)

        data, target = data.to(device), torch.from_numpy(np.array(target)).to(device)
        # compute output
        _t["forward_pass"].tic()
        # mask_list, output = model(data)# output is variable type
        output = model(data)# output is variable type
        # print("get output")# 第一次forward过程会消耗一部分时间做类似于初始化
        output, origin_output = classifier(output, target)
        # output = classifier(output, target)
        _t["forward_pass"].toc()

        
        loss = criterion(output, target)
        loss_display.update(loss.data.item(), data.size(0))

        _t["backward_pass"].tic()
        # compute gradient and do SGD step
        optimizer.zero_grad()# 梯度归0
        loss.backward()# 回传
        optimizer.step()# 根据梯度更新参数
        _t["backward_pass"].toc()
        _t['data_pass'].tic()

        with torch.no_grad():
            output_cls = output
            accuracy = compute_accuracy_multi(output_cls,target)
            accuracy_display.update(accuracy)

        if iteration % configs.log_interval == 0:
            _t['train_pass'].toc()
            _t['train_pass'].tic()
            # loss_display /= configs.log_interval
            # INFO = ' Margin: {:.4f}, Scale: {:.2f}'.format(classifier.m, classifier.s)
            INFO = ''
            
            # back to result after FC instead of after amsoft evaluation
            # with torch.no_grad():
            #     s=30.0
            #     m=0-(0-0.35)
            #     one_hot = torch.zeros_like(output)
            #     one_hot.scatter_(1, target.view(-1, 1), 1.0)
            #     output_cls = torch.div(output,s)+torch.mul(one_hot,m)
            # output_cls = output

            # accuracy = compute_accuracy_multi(output_cls,target)

            eta = _t["train_pass"].diff * ((configs.max_epoch - epoch + 1) * len(train_loader) - batch_idx) / configs.log_interval
            
            message = 'Train Epoch: {} [{}/{} ({:.0f}%)]{}, Loss: {:.6f}, Elapsed time: {:.4f}s, {} iters, Accuracy:{:.2f}, lr:{}, eta: {}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                iteration, loss_display.avg, _t["train_pass"].diff, configs.log_interval, accuracy_display.avg, lr, str(datetime.timedelta(seconds=eta))) + INFO
            
            writer.add_scalar("Training_Loss_iteration", loss_display.avg, iteration)
            writer.add_scalar("Training_Accuracy_iteration", accuracy_display.avg, iteration)
            
            logger.print(message)           
            # loss_display = 0.0
    writer.add_scalar("Training_Loss_epoch", loss_display.avg, epoch)
    writer.add_scalar("Training_Accuracy_epoch", accuracy_display.avg, epoch)


parser = argparse.ArgumentParser(description='PyTorch amsoft training')
# DATA
parser.add_argument('--config_path', type=str, default='./cfgs/amDensenet_121.py',
                    help='config path') # TOOD
args = parser.parse_args()

import imp
configs = imp.load_source("config", args.config_path)

# initial process
if os.path.exists(configs.log_path):
    recreate = input("%s log file exists, clean? Y/N: "%(configs.log_path))
    if recreate.lower()=="y":
        f=open(configs.log_path, "w")
        f.close()
        logger = Logger(configs.log_path)
    elif  recreate.lower()!="y" and recreate.lower()!="n":
        print("'%s' common invalid!"%(recreate))
        exit(0)
    else:
        print("resume a training!")
        logger = Logger(configs.log_path, mode="continue")
else:
    logger = Logger(configs.log_path)
logger.print([(x, configs.__dict__[x]) for x in list(configs.__dict__.keys()) if "__" not in x])
writer = SummaryWriter(configs.log_pic_path)
configs.cuda = not configs.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if configs.cuda else "cpu")

# logger.print('start loading model')

# init the model
import models
# model = models.model_zoo.get_model(configs.backbone_model_name, input_size=[112,112])
model = models.model_zoo.get_model(configs.backbone_model_name)
classifier = models.model_zoo.get_model(configs.classify_model_name, in_features=512, out_features=configs.num_class)
optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': classifier.parameters()}],
                            lr=configs.lr,
                            momentum=configs.momentum,
                            weight_decay=configs.weight_decay)

if configs.resume_net_model is not None:
    logger.print('Loading resume (model) network...')
    state_dict = torch.load(configs.resume_net_model)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:] # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    logger.print("resume net (model) loaded")

if configs.resume_net_classifier is not None:
    logger.print('Loading resume (classifier) network...')
    checkpoint = torch.load(configs.resume_net_classifier)
    configs.start_epoch = checkpoint['EPOCH']
    print("start epoch: %d"%(configs.start_epoch))
    optimizer.load_state_dict(checkpoint['OPTIMIZER'])#断点恢复：动量和l2惩罚项
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda()
    # state_dict = torch.load(configs.resume_net_classifier)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    state_dict = checkpoint['CLASSIFIER']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:] # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    classifier.load_state_dict(new_state_dict)
    logger.print("resume net (classifier) loaded")

if not configs.no_cuda:
    device_ids=[x for x in range(configs.gpu_num)]
    model = torch.nn.DataParallel(model,device_ids=device_ids).cuda()
    classifier = torch.nn.DataParallel(classifier, device_ids=device_ids).cuda()

if not os.path.exists(configs.save_path):
    os.makedirs(configs.save_path)

train_transform = transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
        ])
import dataprocess.dataloader_lmdb as dataloader_lmdb
# dataloader_lmdb = imp.load_source("dataloader", configs.dataloader_python)

dataset = dataloader_lmdb.ImageList(lmdb_path=configs.lmdb_path, max_reader=configs.lmdb_workers,num=configs.datanum, 
                    format_transform=train_transform,shuffle=True, preproc=None)              
train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=configs.batch_size, 
        num_workers=configs.num_workers, pin_memory=True, drop_last=True, shuffle=True) # timeout=1

logger.print('length of train Database: ' + str(len(train_loader.dataset)))
logger.print('Number of Identities: ' + str(configs.num_class))

criterion = torch.nn.CrossEntropyLoss().to(device)# log_softmax NLLloss
# model and classifier is put into optimizer so that their parameters can be update

logger.print("start training")
lr = configs.lr
for epoch in range(configs.start_epoch, configs.max_epoch+1):

    train(train_loader, model, classifier, criterion, optimizer, epoch, writer)
    
    if epoch%5==0 and epoch<=15 or epoch>15:
        model.module.save(configs.save_path + 'backbone_' + str(epoch) + '_checkpoint.pth')
        # classifier.module.save(configs.save_path + 'amsoft-classify_' + str(epoch) + '_checkpoint.pth')
        save_dict = {'EPOCH': epoch+1,
                    'CLASSIFIER': classifier.module.state_dict(),
                    'OPTIMIZER': optimizer.state_dict()}#?!!
        torch.save(save_dict, configs.save_path + 'classifier_status_' + str(epoch) + '_checkpoint.pth')

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=configs.batch_size, 
        num_workers=configs.num_workers, pin_memory=True, drop_last=True, shuffle=True) # timeout=1

dataset.close()
logger.close()






