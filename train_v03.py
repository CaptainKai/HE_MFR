import torch
import torchvision.transforms as transforms

import torch.distributed as dist
import torch.multiprocessing as mp

import models
import dataprocess.dataloader_lmdb as dataloader_lmdb
from utils import writer_logger, Timer, compute_accuracy_multi, adjust_learning_rate
from util.utils import AverageMeter, accuracy
import datetime

from tqdm import tqdm
import os
import numpy as np

import argparse
'''
2022年6月28日20:17:47
最新版训练脚本，试图尽可能简化脚本
'''

def main():
    parser = argparse.ArgumentParser(description='PyTorch amsoft training')
    parser.add_argument('--config_path', type=str, default='./cfgs/SR_36_ddp_pipline.py',
                        help='config path') # TOOD
    parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training') # 必须有
    args = parser.parse_args()

    import imp
    configs = imp.load_source("config", args.config_path)
    configs.local_rank=args.local_rank
    
    environ_settings = configs.environ_settings
    
    os.environ['MASTER_PORT'] = str(environ_settings["master_port"]) # for multi training task
    
    if environ_settings["dist_url"] == "env://" and environ_settings["world_size"] == -1:
        try:
            environ_settings["world_size"] = int(os.environ["WORLD_SIZE"])
        except:
            print("dp mode")
    environ_settings["distributed"] = environ_settings["world_size"] > 1 or environ_settings["multiprocessing_distributed"]
    ngpus_per_node = torch.cuda.device_count()

    if environ_settings["multiprocessing_distributed"]:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        environ_settings["world_size"] = ngpus_per_node * environ_settings["world_size"]

        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        print("using mp")
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, configs))
    else:
        # Simply call main_worker function
        main_worker(environ_settings["gpu"], ngpus_per_node, configs)
    
def main_worker(gpu, ngpus_per_node, cfg):
    
    environ_settings = cfg.environ_settings
    common_settings = cfg.common_settings
    log_settings = cfg.log_settings["training"]
    data_settings = cfg.data_settings["training"]
    other_settings = cfg.other_settings
    
    ## 1. environ for worker
    
    environ_settings["gpu"] = gpu
    if cfg.local_rank != -1:
        environ_settings["gpu"] = environ_settings.local_rank
    
    SEED = environ_settings["SEED"] # random seed for reproduce results
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    if environ_settings["distributed"]:
        if environ_settings["dist_url"] == "env://" and environ_settings["rank"] == -1:
            environ_settings["rank"] = int(os.environ["RANK"])
        if environ_settings["multiprocessing_distributed"]:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            environ_settings["rank"] = environ_settings["rank"] * ngpus_per_node + gpu
        dist.init_process_group(backend=environ_settings["dist_backend"], init_method=environ_settings["dist_url"],
                                world_size=environ_settings["world_size"], rank=environ_settings["rank"])
    
    logger, writer = writer_logger(log_settings["log_path"], log_settings["log_pic_path"], cfg.local_rank, common_settings["backbone"]["settings"][0]["resume_net_model"])
    
    if environ_settings["rank"] <= 0:
        logger.print([(x, cfg.__dict__[x]) for x in list(cfg.__dict__.keys()) if "__" not in x])
    
    ## 2. training initial
    model = []
    classifier = []
    backbone_settings = common_settings["backbone"]["settings"]
    for i in range(common_settings["backbone"]["num"]):
        model.append(
            models.model_zoo.get_model(backbone_settings[i]["backbone_model_name"], **backbone_settings[i]["args"])
        )
        if backbone_settings[i]["resume_net_model"] is not None:
            if environ_settings["rank"] <= 0:
                logger.print('Loading resume (model) network...')
            state_dict = torch.load(backbone_settings[i]["resume_net_model"])["model_%d"%(i)]
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
            model[i].load_state_dict(new_state_dict)
            if environ_settings["rank"] <= 0:
                logger.print("resume net (model) loaded")
    
    classifier_settings = common_settings["classifier"]["settings"]
    for i in range(common_settings["classifier"]["num"]):
        classifier.append(
            models.model_zoo.get_model(classifier_settings[i]["classifier_model_name"], **classifier_settings[i]["args"])
        )
        if classifier_settings[i]["resume_net_classifier"] is not None:
            if environ_settings["rank"] <= 0:
                logger.print('Loading resume (classifier) network...')
            state_dict = torch.load(classifier_settings[i]["resume_net_classifier"])["classifier_%d"%(i)]
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
            classifier[i].load_state_dict(new_state_dict)
            if environ_settings["rank"] <= 0:
                logger.print("resume net (classifier) loaded")
    
    optimizer = torch.optim.SGD([{'params': x.parameters()} for x in model]+
                                 [{'params': x.parameters()} for x in classifier],
                        weight_decay=other_settings["weight_decay"],
                        lr=other_settings["lr"],
                        momentum=other_settings["momentum"],
                        )
    ## 3. resume & model broadcast
    if other_settings["resume"] is True:
        if environ_settings["rank"] <= 0:
            logger.print('Loading resume (optimizer) network...')
        checkpoint = torch.load(other_settings["resume_net_optimizer"], map_location=torch.device('cpu'))
        other_settings["start_epoch"] = checkpoint['EPOCH']
        if environ_settings["rank"] <= 0:
            logger.print("start epoch: %d"%(other_settings["start_epoch"]))
        optimizer.load_state_dict(checkpoint['OPTIMIZER'])#断点恢复：动量和l2惩罚项
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    if environ_settings["distributed"]:
                        state[k] = v.cuda(environ_settings["gpu"])
                    else:
                        state[k] = v.cuda()
        
        if environ_settings["rank"] <= 0:
            logger.print("resume net (optimizer) loaded")
    
    if environ_settings["distributed"] and other_settings["resume"] and ngpus_per_node>1:
        # Use a barrier() to make sure that all processes have finished reading the
        # checkpoint
        dist.barrier()


    if environ_settings["distributed"]:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if environ_settings["gpu"] is not None:
            torch.cuda.set_device(environ_settings["gpu"])
            for i in range(common_settings["backbone"]["num"]):
                model[i].cuda(environ_settings["gpu"])
                model[i] = torch.nn.parallel.DistributedDataParallel(model[i], device_ids=[environ_settings["gpu"]])
            for i in range(common_settings["classifier"]["num"]):
                classifier[i].cuda(environ_settings["gpu"])
                classifier[i] = torch.nn.parallel.DistributedDataParallel(classifier[i], device_ids=[environ_settings["gpu"]])
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            data_settings["batch_size"] = int(data_settings["batch_size"] / ngpus_per_node)
            data_settings["num_workers"] = int((data_settings["num_workers"] + ngpus_per_node - 1) / ngpus_per_node)
            if environ_settings["rank"] <= 0:
                logger.print("data balance")
        else:
            for i in range(common_settings["backbone"]["num"]):
                model[i].cuda()
                model[i] = torch.nn.parallel.DistributedDataParallel(model[i])
            for i in range(common_settings["classifier"]["num"]):
                classifier[i].cuda()
                classifier[i] = torch.nn.parallel.DistributedDataParallel(classifier[i])
            if environ_settings["rank"] <= 0:
                logger.print("use all available GPUs")
    elif environ_settings["gpu"] is not None:
        torch.cuda.set_device(environ_settings["gpu"])
        for i in range(common_settings["backbone"]["num"]):
            model[i].cuda(environ_settings["gpu"])
        for i in range(common_settings["classifier"]["num"]):
            classifier[i].cuda(environ_settings["gpu"])
        if environ_settings["rank"] <= 0:
            logger.print('Use one GPU')
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        for i in range(common_settings["backbone"]["num"]):
            model[i]=torch.nn.DataParallel(model[i]).cuda()
        for i in range(common_settings["classifier"]["num"]):
            classifier[i]=torch.nn.DataParallel(classifier[i]).cuda()
        if environ_settings["rank"] <= 0:
            logger.print('Use DP')   

    ## 4. data settings
    train_transform = transforms.Compose([
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
            ])
    
    dataset_train = dataloader_lmdb.ImageList(format_transform=train_transform, **data_settings["loader_settings"])
                        # format_transform=train_transform,shuffle=False, preproc=None)
    if environ_settings["distributed"]:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=data_settings["batch_size"], shuffle = (train_sampler is None), num_workers=data_settings["num_workers"], pin_memory=True, sampler=train_sampler, drop_last=True)

    loss = []
    # loss = torch.nn.CrossEntropyLoss().cuda(environ_settings["gpu"])
    loss.append(torch.nn.CrossEntropyLoss())
    # loss2= torch.nn.LogSoftmax(dim=1)
    # loss3= torch.nn.NLLLoss()
    if environ_settings["rank"] <= 0:
        logger.print(loss)
    
    if not os.path.exists(log_settings["save_path"]):
        os.makedirs(log_settings["save_path"])
    
    ## 5. train
    lr = other_settings["lr"]
    for epoch in range(other_settings["start_epoch"], other_settings["max_epoch"] + 1):
        np.random.seed(epoch)
        if environ_settings["distributed"]:
            train_sampler.set_epoch(epoch)
        # adjust_learning_rate(optimizer, epoch, cfg)
        lr = adjust_learning_rate(optimizer, other_settings["lr"], 0, epoch, other_settings["step_size"], lr, other_settings["gama"], base=other_settings["base"])

        #train for one epoch
        # train(train_loader, model, [classifier1, classifier2], [loss1,loss2,loss3], optimizer, epoch, cfg, writer, logger)
        train(train_loader, model, classifier, loss, optimizer, epoch, cfg, writer, logger)

        if environ_settings["rank"] % ngpus_per_node == 0 or environ_settings["rank"]==-1:
            if epoch%5==0 and epoch<=25 or epoch>25:
                logger.print("Save Checkpoint...")
                logger.print("=" * 60)
                for i in range(common_settings["backbone"]["num"]):
                    model[i].module.save(log_settings["save_path"] + 'backbone_%d_%d_checkpoint.pth'%(i, epoch))
                for i in range(common_settings["classifier"]["num"]):
                    classifier[i].module.save(log_settings["save_path"] + 'classifier_%d_%d_checkpoint.pth'%(i, epoch))
                save_dict = {'EPOCH': epoch,
                            'OPTIMIZER': optimizer.state_dict()}
                torch.save(save_dict, log_settings["save_path"] + 'optimizer_' + str(epoch) + '_checkpoint.pth')
                logger.print("Save done!")
                logger.print("=" * 60)
                # Use a barrier() to make sure that process 1 loads the model after process
                # 0 saves i
        if environ_settings["distributed"] and ngpus_per_node>1:
            dist.barrier()
        # pass


def reduce_tensor(rt):
    # sum the tensor data across all machines
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt


def train(train_loader, backbone, head, criterion, optimizer, epoch, cfg, writer, logger):
    common_settings = cfg.common_settings
    environ_settings = cfg.environ_settings
    log_settings = cfg.log_settings["training"]
    other_settings = cfg.other_settings
    loss_recorder = [] # for training loss
    measurement = [] # for testing measurement
    for i in range(common_settings["backbone"]["num"]):
        backbone[i].train()
    for i in range(common_settings["classifier"]["num"]):
        head[i].train()
        loss_recorder.append(AverageMeter()) # for each loss
        measurement.append(AverageMeter())
    if common_settings["classifier"]["num"]==1:
        measurement.append(AverageMeter())
    loss_recorder.append(AverageMeter()) # for total loss


    _t = {'forward_pass': Timer(), 'backward_pass': Timer(), 'data_pass': Timer(), "train_pass": Timer()}
    _t['data_pass'].tic()
    _t['train_pass'].tic()
    for batch, (inputs, labels, classes, diff_location) in tqdm(enumerate(train_loader, 1)):# tqdm: 进度条
        _t['data_pass'].toc()
        # compute output
        inputs = inputs.cuda(environ_settings["gpu"], non_blocking=True)
        labels = labels.cuda(environ_settings["gpu"], non_blocking=True)
        classes = classes.cuda(environ_settings["gpu"], non_blocking=True)
        # diff_location = diff_location.cuda(environ_settings["gpu"], non_blocking=True)
        # inputs, labels = inputs.to(cfg['gpu']), torch.from_numpy(np.array(labels)).to(cfg['gpu'])
        
        _t["forward_pass"].tic()
        # features_up, features_down = backbone(inputs)
        features = backbone[0](inputs)## single output
        # features_down = features_down*(1-classes.unsqueeze(1))
        outputs, original_logits = head[0](features, labels) ## single output
        ## multi output
        # outputs_up, original_logits_up = head[0](features_up, labels) 
        # outputs_down, original_logits_down = head[1](features_down, labels)
        
        # outputs, original_logits = head[0](torch.cat([features_up, features_down], 1), labels)
        _t["forward_pass"].toc()
        # mask_num = max( (1-classes).sum(), 1)
        # loss = criterion(outputs, labels)
        loss2 = criterion[0](outputs, labels)
        # loss2 = criterion[0](outputs_up, labels)
        # loss3 = criterion[0](outputs_down, labels)
        
        # log_softmax_down = criterion[1](outputs_down)*(1-classes.unsqueeze(1))
        # loss3 = criterion[2](log_softmax_down, labels)*(classes.shape[0]**1) / (mask_num**1)
        # total_loss = loss + 0.1*loss2 + 0.1*loss3
        # total_loss = loss + 0.01*loss2 + 0.01*loss3
        # total_loss = (loss + 1*loss2 + 1*loss3)/3
        # total_loss = (loss + 2*loss3)/3
        
        # total_loss = common_settings["classifier"]["settings"][0]["alpha"]*loss2 + \
        #             common_settings["classifier"]["settings"][1]["alpha"]*loss3
        total_loss = loss2
        
        # total_loss = loss
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
        
        # prec1_down, prec5_down = accuracy(original_logits_down.data, labels, topk = (1, 5))
        # prec1_up, prec5_up = accuracy(original_logits_up.data, labels, topk = (1, 5))
        if environ_settings["distributed"]:
            # loss = reduce_tensor(loss.clone().detach_())
            loss2 = reduce_tensor(loss2.clone().detach_())
            # loss3 = reduce_tensor(loss3.clone().detach_())
            total_loss = reduce_tensor(total_loss.clone().detach_())
            
            prec1 = reduce_tensor(prec1)
            prec5 = reduce_tensor(prec5)
            
            # prec1_down = reduce_tensor(prec1_down)
            # prec1_up = reduce_tensor(prec1_up)
            
            # prec1 = reduce_tensor(torch.from_numpy(prec1))
            # prec5 = reduce_tensor(torch.from_numpy(prec5))
            torch.cuda.synchronize() # wait every process finish above transmission
            factor = torch.cuda.device_count()

        # losses.update(loss.data.item()/factor, inputs.size(0))
        loss_recorder[0].update(loss2.data.item()/factor, inputs.size(0))
        # loss_recorder[1].update(loss3.data.item()/factor, inputs.size(0))
        loss_recorder[-1].update(total_loss.data.item()/factor, inputs.size(0))
        
        # top1.update(prec1.data.item()/factor, inputs.size(0))
        # top5.update(prec5.data.item()/factor, inputs.size(0))
        measurement[0].update(prec1.data.item()/factor, inputs.size(0))
        measurement[1].update(prec5.data.item()/factor, inputs.size(0))
        # measurement[0].update(prec1_up.data.item()/factor, inputs.size(0))
        # measurement[1].update(prec1_down.data.item()/factor, inputs.size(0))
        
        if ( ((batch + 1) % log_settings["log_interval"] == 0) or batch == 0 ) and environ_settings["rank"] <= 0:
            logger.print("time cost, forward:{}, backward:{}, data cost:{} "
            .format(str(_t['forward_pass'].average_time), str(_t['backward_pass'].average_time), str(_t['data_pass'].average_time)))
            logger.print("=" * 60)
            _t['train_pass'].toc()
            _t['train_pass'].tic()
            eta = _t["train_pass"].diff * ((other_settings["max_epoch"] - epoch + 1) * len(train_loader) - batch) / log_settings["log_interval"]
            logger.print('Epoch {}/{} Batch {}/{} eta: {}\t'
                            'Training Loss1 {loss1.val:.4f} ({loss1.avg:.4f})\t'
                            # 'Training Loss2 {loss2.val:.4f} ({loss2.avg:.4f})\t'
                            # 'Training Loss3 {loss3.val:.4f} ({loss3.avg:.4f})\t'
                            'Training Total_Loss {total.val:.4f} ({total.avg:.4f})\t'
                            'Training Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                            'Training Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                            # 'Training Prec@1_up {top1_up.val:.3f} ({top1_up.avg:.3f})\t'
                            # 'Training Prec@1_down {top1_down.val:.3f} ({top1_down.avg:.3f})\t'
                            .format(
                                epoch, other_settings["max_epoch"], batch + 1, len(train_loader), str(datetime.timedelta(seconds=eta)),
                                loss1=loss_recorder[0],
                                # top1 = top1, top5 = top5,
                                # loss2=loss_recorder[1], 
                                # loss3=losses3,
                                total=loss_recorder[-1],
                                top1=measurement[0], 
                                top5=measurement[1],
                                # top1_up=measurement[0], 
                                # top1_down=measurement[1],
                                ))
            logger.print("=" * 60)
            # sys.stdout.flush()
    epoch_loss1 = loss_recorder[0].avg
    # epoch_loss2 = loss_recorder[1].avg
    # epoch_loss3 = losses3.avg
    epoch_total_loss = loss_recorder[-1].avg
    # epoch_acc = top1.avg
    epoch_acc_1 = measurement[0].avg
    epoch_acc_5 = measurement[1].avg
    # epoch_acc_up = measurement[0].avg
    # epoch_acc_down = measurement[1].avg
    eta = _t["train_pass"].diff * ((other_settings["max_epoch"] - epoch + 1) * len(train_loader) - batch) / log_settings["log_interval"]
    if environ_settings["rank"] <= 0:
        logger.print('Epoch: {}/{} eta: {}\t'
                'Training Loss1 {loss1.val:.4f} ({loss1.avg:.4f})\t'
                # 'Training Loss2 {loss2.val:.4f} ({loss2.avg:.4f})\t'
                # 'Training Loss3 {loss3.val:.4f} ({loss3.avg:.4f})\t'
                'Training Total_Loss {total.val:.4f} ({total.avg:.4f})\t'
                'Training Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                'Training Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                # 'Training Prec@1_up {top1_up.val:.3f} ({top1_up.avg:.3f})\t'
                # 'Training Prec@1_down {top1_down.val:.3f} ({top1_down.avg:.3f})\t'
                .format(
                    epoch, other_settings["max_epoch"], str(datetime.timedelta(seconds=eta)),
                    loss1 = loss_recorder[0],
                    #  top1 = top1, top5 = top5,
                    # loss2=loss_recorder[1], 
                    # loss3=losses3,
                    total=loss_recorder[-1],
                    top1=measurement[0], 
                    top5=measurement[1],
                    # top1_up=measurement[0], 
                    # top1_down=measurement[1],
                    ))
        logger.print("=" * 60)
    
    # sys.stdout.flush()
    if environ_settings["rank"]<= 0:
        writer.add_scalar("Training_Loss1", epoch_loss1, epoch)
        # writer.add_scalar("Training_Loss2", epoch_loss2, epoch)
        # writer.add_scalar("Training_Loss3", epoch_loss3, epoch)
        writer.add_scalar("Training_Total_Loss", epoch_total_loss, epoch)
        # writer.add_scalar("Training_Accuracy", epoch_acc, epoch)
        writer.add_scalar("Top1", epoch_acc_1, epoch)
        writer.add_scalar("Top5", epoch_acc_5, epoch)
        # writer.add_scalar("Top1_up", epoch_acc_up, epoch)
        # writer.add_scalar("Top1_down", epoch_acc_down, epoch)

if __name__ == '__main__':
    main()
