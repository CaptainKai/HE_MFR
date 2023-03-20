'''
@author：Xinghua Meng, Kai Li
本文件为使用datadistributeparallel进行训练的示例，以人脸识别训练为背景，需要修改或注意的部分已经用# TODO标记
'''
import builtins
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

import models
import datetime

from tqdm import tqdm
import os
import sys
import time
import numpy as np
import scipy
import pickle

import argparse


def main():
    parser = argparse.ArgumentParser(description='PyTorch amsoft training')
    parser.add_argument('--config_path', type=str, default='./cfgs/am_20_ddp.py',
                        help='config path') # TOOD 参数解析，本实例中使用imp解析参数文件，读者可以直接使用parser.add_argument解析参数
    parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training') # 必须有
    args = parser.parse_args()

    # TODO 参数解析部分，读者可改为自己的代码
    import imp
    configs = imp.load_source("config", args.config_path)
    configs.local_rank=args.local_rank
    os.environ['MASTER_PORT'] = str(configs.master_port) ## TODO for multi training task
    
    if configs.dist_url == "env://" and configs.world_size == -1:
        try:
            configs.world_size = int(os.environ["WORLD_SIZE"])
        except:
            print("dp mode")
    configs.distributed = configs.world_size > 1 or configs.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()

    if configs.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        configs.world_size = ngpus_per_node * configs.world_size

        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        print("using mp")
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, configs))
    else:
        # Simply call main_worker function
        main_worker(configs.gpu, ngpus_per_node, configs)
    
def main_worker(gpu, ngpus_per_node, cfg):
    cfg.gpu = gpu
    if cfg.local_rank != -1:
        cfg.gpu = cfg.local_rank
    
    # TODO log文件权柄的生成代码（根据local_rank参数生成两个log文件），读者可改为自己的代码
    logger, writer = writer_logger(cfg.log_path, cfg.log_pic_path, cfg.local_rank, cfg.resume_net_model)
    logger.print([(x, cfg.__dict__[x]) for x in list(cfg.__dict__.keys()) if "__" not in x])
    
    SEED = cfg.SEED # random seed for reproduce results
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    if cfg.distributed:
        if cfg.dist_url == "env://" and cfg.rank == -1:
            cfg.rank = int(os.environ["RANK"])
        if cfg.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            cfg.rank = cfg.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=cfg.dist_backend, init_method=cfg.dist_url,
                                world_size=cfg.world_size, rank=cfg.rank)
    
    # TODO 模型初始化代码，读者可改为自己的代码
    model = models.model_zoo.get_model(cfg.backbone_model_name)
    classifier = models.model_zoo.get_model(cfg.classify_model_name, in_features=512, out_features=cfg.num_class)
    optimizer = torch.optim.SGD([{'params': model.parameters(), 'weight_decay': cfg.weight_decay}, {'params': classifier.parameters(), 'weight_decay': cfg.weight_decay}],
                        lr=cfg.lr,
                        momentum=cfg.momentum,
                        )
    # TODO 模型载入代码，读者可改为自己的代码
    if cfg.resume_net_model is not None:
        logger.print('Loading resume (model) network...')
        state_dict = torch.load(cfg.resume_net_model)
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

    if cfg.resume_net_classifier is not None:
        logger.print('Loading resume (classifier) network...')
        checkpoint = torch.load(cfg.resume_net_classifier, map_location=torch.device('cpu'))
        cfg.start_epoch = checkpoint['EPOCH']
        logger.print("start epoch: %d"%(cfg.start_epoch))
        optimizer.load_state_dict(checkpoint['OPTIMIZER'])#断点恢复：动量和l2惩罚项
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    if cfg.distributed:
                        state[k] = v.cuda(cfg.gpu)# TODO 必须指定GPU
                    else:
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
    
    if cfg.distributed and (cfg.resume_net_classifier or cfg.resume_net_model):
        # Use a barrier() to make sure that all processes have finished reading the
        # checkpoint
        dist.barrier()

    # TODO 模型载入GPU，读者可改为自己的代码（人脸训练任务有两个模型需要载入）
    if cfg.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if cfg.gpu is not None:
            torch.cuda.set_device(cfg.gpu)
            model.cuda(cfg.gpu)
            classifier.cuda(cfg.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            cfg.batch_size = int(cfg.batch_size / ngpus_per_node)
            cfg.num_workers = int((cfg.num_workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.gpu])
            classifier = torch.nn.parallel.DistributedDataParallel(classifier, device_ids=[cfg.gpu])
            logger.print("data balance")
        else:
            model.cuda()
            classifier.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
            classifier = torch.nn.parallel.DistributedDataParallel(classifier)
            logger.print("use all available GPUs")
    elif cfg.gpu is not None:
        torch.cuda.set_device(cfg.gpu)
        model = model.cuda(cfg.gpu)
        classifier = classifier.cuda(cfg.gpu)
        logger.print('Use one GPU')
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()
        classifier = torch.nn.DataParallel(classifier).cuda()
        logger.print('Use DP')   

    # TODO 数据格式更改代码，读者可改为自己的代码
    train_transform = transforms.Compose([
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
            ])
    # TODO 数据集初始化代码，读者可改为自己的代码
    dataset_train = dataloader_lmdb.ImageList(lmdb_path=cfg.lmdb_path, max_reader=cfg.lmdb_workers,num=cfg.datanum, 
                        format_transform=train_transform,shuffle=False, preproc=None)
    
    if cfg.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=cfg.batch_size, shuffle = (train_sampler is None), num_workers=cfg.num_workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    # TODO loss初始化，读者可改为自己的代码
    loss = torch.nn.CrossEntropyLoss()
    
    if not os.path.exists(cfg.save_path):
        os.makedirs(cfg.save_path)
    
    # TODO 循环训练代码，读者可改为自己的代码
    lr = cfg.lr
    for epoch in range(1, cfg.max_epoch + 1):
        np.random.seed(epoch)# TODO 必须有
        if cfg.distributed:# TODO 必须有
            train_sampler.set_epoch(epoch)
        # adjust_learning_rate(optimizer, epoch, cfg)
        lr = adjust_learning_rate(optimizer, cfg.lr, 0, epoch, cfg.step_size, lr, cfg.gama, base=cfg.base)

        #train for one epoch
        train(train_loader, model, classifier, loss, optimizer, epoch, cfg, writer, logger)

        # TODO 此处判断语句的作用为：仅主进程保存模型
        if cfg.rank % ngpus_per_node == 0 or cfg.rank==-1: # TODO cfg.rank的值代表当前进程的编号，编程时一般将该值为0（或-1）的进程当成主进程
            if epoch%5==0 and epoch<=15 or epoch>15:
                logger.print("Save Checkpoint...")
                logger.print("=" * 60)

                model.module.save(cfg.save_path + 'backbone_' + str(epoch) + '_checkpoint.pth')
                save_dict = {'EPOCH': epoch + 1,
                            'CLASSIFIER': classifier.module.state_dict(),
                            'OPTIMIZER': optimizer.state_dict()}
                torch.save(save_dict, cfg.save_path + 'classifier_status_' + str(epoch) + '_checkpoint.pth')
                # Use a barrier() to make sure that process 1 loads the model after process
                # 0 saves i
                dist.barrier()


def reduce_tensor(rt):
    # sum the tensor data across all machines
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    return rt


# TODO 以下代码和训练任务本身相关，读者参考即可
def train(train_loader, backbone, head, criterion, optimizer, epoch, cfg, writer, logger):
    backbone.train()  # set to training mode
    head.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for batch, (inputs, labels) in tqdm(enumerate(train_loader, 1)):# tqdm: 进度条

        inputs = inputs.cuda(cfg.gpu, non_blocking=True) # TODO 最好指定GPU
        labels = labels.cuda(cfg.gpu, non_blocking=True)       

        features = backbone(inputs)
        outputs, original_logits = head(features, labels)

       
        loss = criterion(outputs, labels)

 
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

           
        # measure accuracy and record loss
        # print(torch.cuda.device_count())
        factor = 1
        prec1, prec5 = accuracy(original_logits.data, labels, topk = (1, 5))

        # TODO 以下代码的作用是同步进程间信息，读者可以根据自己的需要决定是否使用该部分代码
        if cfg.distributed:
            loss = reduce_tensor(loss.clone().detach_())
            prec1 = reduce_tensor(prec1)
            prec5 = reduce_tensor(prec5)
            torch.cuda.synchronize() # wait every process finish above transmission
            factor = torch.cuda.device_count()

        losses.update(loss.data.item()/factor, inputs.size(0))
        top1.update(prec1.data.item()/factor, inputs.size(0))
        top5.update(prec5.data.item()/factor, inputs.size(0))
        
        if ((batch + 1) % cfg.log_interval == 0) or batch == 0:

            logger.print('Epoch {}/{} Batch {}/{} eta: {}\t'
                            'Training Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                            'Training Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                            'Training Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                                epoch + 1, cfg.max_epoch, batch + 1, len(train_loader), str(datetime.timedelta(seconds=eta)), loss = losses, top1 = top1, top5 = top5))
            logger.print("=" * 60)
            sys.stdout.flush() # TODO 即使更新terminal的显示信息，读者可以根据自己的需要决定是否使用该部分代码
    epoch_loss = losses.avg
    epoch_acc = top1.avg
    eta = _t["train_pass"].diff * ((cfg.max_epoch - epoch + 1) * len(train_loader) - batch) / cfg.log_interval
    logger.print('Epoch: {}/{} eta: {}\t''Training Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            'Training Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
            'Training Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch + 1, cfg.max_epoch, str(datetime.timedelta(seconds=eta)), loss = losses, top1 = top1, top5 = top5))
    logger.print("=" * 60)
    
    sys.stdout.flush()
    if cfg.rank <= 0: # TODO 仅主进程进行相关操作
        writer.add_scalar("Training_Loss", epoch_loss, epoch + 1)
        writer.add_scalar("Training_Accuracy", epoch_acc, epoch + 1)
        writer.add_scalar("Top1", top1.avg, epoch+1)
        writer.add_scalar("Top5", top5.avg, epoch+1)

if __name__ == '__main__':
    main()
