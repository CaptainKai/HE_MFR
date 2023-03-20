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
import dataprocess.dataloader_lmdb as dataloader_lmdb
from utils import writer_logger, Timer, compute_accuracy_multi, adjust_learning_rate
from util.utils import get_time, AverageMeter, accuracy
import datetime

from tqdm import tqdm
import os
import sys
import time
import numpy as np
import scipy
import pickle

import argparse
'''
修改了存储model时 barrier导致无法进行下去的问题
修改了log文件过多，输出log过多的问题
对epoch的问题进行了修正

用于训练split版本，因为分类器有三个，所以目前的分类器resume没法用
'''

def main():
    parser = argparse.ArgumentParser(description='PyTorch amsoft training')
    parser.add_argument('--config_path', type=str, default='./cfgs/res_50_ddp_split_mask.py',
                        help='config path') # TOOD
    parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training') # 必须有
    args = parser.parse_args()

    import imp
    configs = imp.load_source("config", args.config_path)
    configs.local_rank=args.local_rank

    os.environ['MASTER_PORT'] = str(configs.master_port) # for multi training task
    
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
    
    logger, writer = writer_logger(cfg.log_path, cfg.log_pic_path, cfg.local_rank, cfg.resume_net_model)
    model = models.model_zoo.get_model(cfg.backbone_model_name, input_size=[112, 112])
    
    if cfg.rank <= 0:
        logger.print([(x, cfg.__dict__[x]) for x in list(cfg.__dict__.keys()) if "__" not in x])
        logger.print(model)
    classifier1 = models.model_zoo.get_model(cfg.classify_model_name, in_features=256, out_features=cfg.num_class)
    classifier2 = models.model_zoo.get_model(cfg.classify_model_name, in_features=256, out_features=cfg.num_class)
    classifier = models.model_zoo.get_model(cfg.classify_model_name, in_features=512, out_features=cfg.num_class)
    optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': classifier.parameters()}, {'params': classifier1.parameters()}, {'params': classifier2.parameters()}],
                        weight_decay=cfg.weight_decay,
                        lr=cfg.lr,
                        momentum=cfg.momentum,
                        )
    
    if cfg.resume_net_model is not None:
        if cfg.rank <= 0:
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
        if cfg.rank <= 0:
            logger.print("resume net (model) loaded")

    if cfg.resume_net_classifier is not None:
        if cfg.rank <= 0:
            logger.print('Loading resume (classifier) network...')
        checkpoint = torch.load(cfg.resume_net_classifier, map_location=torch.device('cpu'))
        cfg.start_epoch = checkpoint['EPOCH']
        if cfg.rank <= 0:
            logger.print("start epoch: %d"%(cfg.start_epoch))
        optimizer.load_state_dict(checkpoint['OPTIMIZER'])#断点恢复：动量和l2惩罚项
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    if cfg.distributed:
                        state[k] = v.cuda(cfg.gpu)
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

        state_dict = checkpoint['CLASSIFIER1']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            head = k[:7]
            if head == 'module.':
                name = k[7:] # remove `module.`
            else:
                name = k
            new_state_dict[name] = v
        classifier1.load_state_dict(new_state_dict)

        state_dict = checkpoint['CLASSIFIER2']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            head = k[:7]
            if head == 'module.':
                name = k[7:] # remove `module.`
            else:
                name = k
            new_state_dict[name] = v
        classifier2.load_state_dict(new_state_dict)
        
        if cfg.rank <= 0:
            logger.print("resume net (classifier) loaded")
    
    if cfg.distributed and (cfg.resume_net_classifier or cfg.resume_net_model) and ngpus_per_node>1:
        # Use a barrier() to make sure that all processes have finished reading the
        # checkpoint
        dist.barrier()


    if cfg.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if cfg.gpu is not None:
            torch.cuda.set_device(cfg.gpu)
            model.cuda(cfg.gpu)
            classifier.cuda(cfg.gpu)
            classifier1.cuda(cfg.gpu)
            classifier2.cuda(cfg.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            cfg.batch_size = int(cfg.batch_size / ngpus_per_node)
            cfg.num_workers = int((cfg.num_workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.gpu])
            classifier = torch.nn.parallel.DistributedDataParallel(classifier, device_ids=[cfg.gpu])
            classifier1 = torch.nn.parallel.DistributedDataParallel(classifier1, device_ids=[cfg.gpu])
            classifier2 = torch.nn.parallel.DistributedDataParallel(classifier2, device_ids=[cfg.gpu])
            if cfg.rank <= 0:
                logger.print("data balance")
        else:
            model.cuda()
            classifier.cuda()
            classifier1.cuda()
            classifier2.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
            classifier = torch.nn.parallel.DistributedDataParallel(classifier)
            classifier1 = torch.nn.parallel.DistributedDataParallel(classifier1)
            classifier2 = torch.nn.parallel.DistributedDataParallel(classifier2)
            if cfg.rank <= 0:
                logger.print("use all available GPUs")
    elif cfg.gpu is not None:
        torch.cuda.set_device(cfg.gpu)
        model = model.cuda(cfg.gpu)
        classifier = classifier.cuda(cfg.gpu)
        classifier1 = classifier1.cuda(cfg.gpu)
        classifier2 = classifier2.cuda(cfg.gpu)
        if cfg.rank <= 0:
            logger.print('Use one GPU')
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()
        classifier = torch.nn.DataParallel(classifier).cuda()
        classifier1 = torch.nn.DataParallel(classifier1).cuda()
        classifier2 = torch.nn.DataParallel(classifier2).cuda()
        if cfg.rank <= 0:
            logger.print('Use DP')   

    train_transform = transforms.Compose([
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
            ])
    
    dataset_train = dataloader_lmdb.ImageList(lmdb_path=cfg.lmdb_path, max_reader=cfg.lmdb_workers,num=cfg.datanum, 
                        format_transform=train_transform, augu_paral=False, ldm68=True, augu_rate=1,shuffle=False)
    if cfg.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=cfg.batch_size, shuffle = (train_sampler is None), num_workers=cfg.num_workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    # loss = torch.nn.CrossEntropyLoss().cuda(cfg.gpu)
    loss = torch.nn.CrossEntropyLoss()
    
    if not os.path.exists(cfg.save_path):
        os.makedirs(cfg.save_path)
    
    # train
    lr = cfg.lr
    for epoch in range(cfg.start_epoch, cfg.max_epoch + 1):
        np.random.seed(epoch)
        if cfg.distributed:
            train_sampler.set_epoch(epoch)
        # adjust_learning_rate(optimizer, epoch, cfg)
        lr = adjust_learning_rate(optimizer, cfg.lr, 0, epoch, cfg.step_size, lr, cfg.gama, base=cfg.base)

        #train for one epoch
        train(train_loader, model, [classifier, classifier1, classifier2], loss, optimizer, epoch, cfg, writer, logger)

        if cfg.rank % ngpus_per_node == 0 or cfg.rank==-1:
            if epoch%5==0 and epoch<=25 or epoch>25:
                logger.print("Save Checkpoint...")
                logger.print("=" * 60)

                model.module.save(cfg.save_path + 'backbone_' + str(epoch) + '_checkpoint.pth')
                save_dict = {'EPOCH': epoch,
                            'CLASSIFIER': classifier.module.state_dict(),
                            'CLASSIFIER1': classifier1.module.state_dict(),
                            'CLASSIFIER2': classifier2.module.state_dict(),
                            'OPTIMIZER': optimizer.state_dict()}
                torch.save(save_dict, cfg.save_path + 'classifier_status_' + str(epoch) + '_checkpoint.pth')
                logger.print("Save done!")
                logger.print("=" * 60)
                # Use a barrier() to make sure that process 1 loads the model after process
                # 0 saves i
        if cfg.distributed and ngpus_per_node>1:
            dist.barrier()
        # pass


def reduce_tensor(rt):
    # sum the tensor data across all machines
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt


def train(train_loader, backbone, head, criterion, optimizer, epoch, cfg, writer, logger):
    backbone.train()  # set to training mode
    head[0].train()
    head[1].train()
    head[2].train()
    losses = AverageMeter()
    losses2 = AverageMeter()
    losses3 = AverageMeter()
    total_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    top1_down = AverageMeter()
    top1_up = AverageMeter()

    _t = {'forward_pass': Timer(), 'backward_pass': Timer(), 'data_pass': Timer(), "train_pass": Timer()}
    _t['data_pass'].tic()
    _t['train_pass'].tic()
    for batch, (inputs, labels) in tqdm(enumerate(train_loader, 1)):# tqdm: 进度条
        _t['data_pass'].toc()
        # compute output
        inputs = inputs.cuda(cfg.gpu, non_blocking=True)
        labels = labels.cuda(cfg.gpu, non_blocking=True)
        # inputs, labels = inputs.to(cfg['gpu']), torch.from_numpy(np.array(labels)).to(cfg['gpu'])
        
        _t["forward_pass"].tic()
        features_up, features_down = backbone(inputs)
        outputs_down, original_logits_down = head[2](features_down, labels)
        outputs_up, original_logits_up = head[1](features_up, labels)
        outputs, original_logits = head[0](torch.cat([features_up, features_down], 1), labels)
        _t["forward_pass"].toc()
       
        loss = criterion(outputs, labels)
        loss2 = criterion(outputs_up, labels)
        loss3 = criterion(outputs_down, labels)
        # total_loss = loss + 0.1*loss2 + 0.1*loss3
        # total_loss = loss + 0.01*loss2 + 0.01*loss3
        total_loss = (loss + 1*loss2 + 1*loss3)/3
        _t["backward_pass"].tic()
        # compute gradient and do SGD step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        _t["backward_pass"].toc()
        _t['data_pass'].tic()
           
        # measure accuracy and record loss
        # print(torch.cuda.device_count())
        factor = 1
        prec1, prec5 = accuracy(original_logits.data, labels, topk = (1, 5))
        prec1_down, prec5_down = accuracy(original_logits_down.data, labels, topk = (1, 5))
        prec1_up, prec5_up = accuracy(original_logits_up.data, labels, topk = (1, 5))
        if cfg.distributed:
            loss = reduce_tensor(loss.clone().detach_())
            loss2 = reduce_tensor(loss2.clone().detach_())
            loss3 = reduce_tensor(loss3.clone().detach_())
            total_loss = reduce_tensor(total_loss.clone().detach_())
            
            prec1 = reduce_tensor(prec1)
            prec5 = reduce_tensor(prec5)
            prec1_down = reduce_tensor(prec1_down)
            prec1_up = reduce_tensor(prec1_up)
            # prec1 = reduce_tensor(torch.from_numpy(prec1))
            # prec5 = reduce_tensor(torch.from_numpy(prec5))
            torch.cuda.synchronize() # wait every process finish above transmission
            factor = torch.cuda.device_count()

        losses.update(loss.data.item()/factor, inputs.size(0))
        losses2.update(loss2.data.item()/factor, inputs.size(0))
        losses3.update(loss3.data.item()/factor, inputs.size(0))
        total_losses.update(total_loss.data.item()/factor, inputs.size(0))
        
        top1.update(prec1.data.item()/factor, inputs.size(0))
        top5.update(prec5.data.item()/factor, inputs.size(0))
        top1_up.update(prec1_up.data.item()/factor, inputs.size(0))
        top1_down.update(prec1_down.data.item()/factor, inputs.size(0))
        
        if ( ((batch + 1) % cfg.log_interval == 0) or batch == 0 ) and cfg.rank <= 0:
            logger.print("time cost, forward:{}, backward:{}, data cost:{} "
            .format(str(_t['forward_pass'].average_time), str(_t['backward_pass'].average_time), str(_t['data_pass'].average_time)))
            logger.print("=" * 60)
            _t['train_pass'].toc()
            _t['train_pass'].tic()
            eta = _t["train_pass"].diff * ((cfg.max_epoch - epoch + 1) * len(train_loader) - batch) / cfg.log_interval
            logger.print('Epoch {}/{} Batch {}/{} eta: {}\t'
                            'Training Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                            'Training Loss2 {loss2.val:.4f} ({loss2.avg:.4f})\t'
                            'Training Loss3 {loss3.val:.4f} ({loss3.avg:.4f})\t'
                            'Training Total_Loss {total.val:.4f} ({total.avg:.4f})\t'
                            'Training Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                            'Training Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                            'Training Prec@1_up {top1_up.val:.3f} ({top1_up.avg:.3f})\t'
                            'Training Prec@1_down {top1_down.val:.3f} ({top1_down.avg:.3f})\t'.format(
                                epoch, cfg.max_epoch, batch + 1, len(train_loader), str(datetime.timedelta(seconds=eta)), loss = losses, top1 = top1, top5 = top5,
                                loss2=losses2, loss3=losses3, total=total_losses,
                                top1_up=top1_up, top1_down=top1_down))
            logger.print("=" * 60)
            # sys.stdout.flush()
    epoch_loss = losses.avg
    epoch_loss2 = losses2.avg
    epoch_loss3 = losses3.avg
    epoch_total_loss = total_losses.avg
    epoch_acc = top1.avg
    eta = _t["train_pass"].diff * ((cfg.max_epoch - epoch + 1) * len(train_loader) - batch) / cfg.log_interval
    if cfg.rank <= 0:
        logger.print('Epoch: {}/{} eta: {}\t''Training Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Training Loss2 {loss2.val:.4f} ({loss2.avg:.4f})\t'
                'Training Loss3 {loss3.val:.4f} ({loss3.avg:.4f})\t'
                'Training Total_Loss {total.val:.4f} ({total.avg:.4f})\t'
                'Training Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                'Training Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, cfg.max_epoch, str(datetime.timedelta(seconds=eta)), loss = losses, top1 = top1, top5 = top5,
                    loss2=losses2, loss3=losses3, total=total_losses,
                    top1_up=top1_up, top1_down=top1_down))
        logger.print("=" * 60)
    
    # sys.stdout.flush()
    if cfg.rank <= 0:
        writer.add_scalar("Training_Loss", epoch_loss, epoch)
        writer.add_scalar("Training_Loss2", epoch_loss2, epoch)
        writer.add_scalar("Training_Loss3", epoch_loss3, epoch)
        writer.add_scalar("Training_Total_Loss", epoch_total_loss, epoch)
        writer.add_scalar("Training_Accuracy", epoch_acc, epoch)
        writer.add_scalar("Top1", top1.avg, epoch)
        writer.add_scalar("Top5", top5.avg, epoch)
        writer.add_scalar("Top1_up", top1_up.avg, epoch)
        writer.add_scalar("Top1_down", top1_down.avg, epoch)

if __name__ == '__main__':
    main()
