### 基于train_v3 编写一个训练器类
from unittest import result
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

import imp

def reduce_tensor(rt):
    # sum the tensor data across all machines
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt


class Trainer():
    def __init__(self, local_rank, config_path):
        self.config = imp.load_source("config", config_path)
        self.config.local_rank = local_rank
        
        self.environ_settings = self.config.environ_settings
        self.common_settings = self.config.common_settings
        self.log_settings = self.config.log_settings["training"]
        self.data_settings = self.config.data_settings["training"]
        self.other_settings = self.config.other_settings
        self.backbone_settings = self.common_settings["backbone"]["settings"]
        self.classifier_settings = self.common_settings["classifier"]["settings"]
        
        
        # self.run()
        
    def run(self):
        
        os.environ['MASTER_PORT'] = str(self.environ_settings["master_port"]) # for multi training task
        
        if self.environ_settings["dist_url"] == "env://" and self.environ_settings["world_size"] == -1:
            try:
                self.config.environ_settings["world_size"] = int(os.environ["WORLD_SIZE"])
            except:
                print("dp mode")
        self.config.environ_settings["distributed"] = self.environ_settings["world_size"] > 1 or self.environ_settings["multiprocessing_distributed"]
        ngpus_per_node = torch.cuda.device_count()

        if self.environ_settings["multiprocessing_distributed"]:
            # Since we have ngpus_per_node processes per node, the total world_size
            # needs to be adjusted accordingly
            self.config.environ_settings["world_size"] = ngpus_per_node * self.environ_settings["world_size"]

            # Use torch.multiprocessing.spawn to launch distributed processes: the
            # main_worker process function
            print("using mp")
            mp.spawn(self.pipeline, nprocs=ngpus_per_node, args=(ngpus_per_node))
        else:
            # Simply call main_worker function
            self.pipeline(self.environ_settings["gpu"], ngpus_per_node)
    
    def pipeline(self, gpu, ngpus_per_node):
        
        self.environ_set(gpu, ngpus_per_node) ## 有可能出问题，这一步的目的其实是开启 group，可能在函数中开启group会出问题 #TODO
        self.network_set(ngpus_per_node)
        self.data_set()
        self.train(ngpus_per_node)
        
    
    def environ_set(self, gpu, ngpus_per_node):
        
        self.config.environ_settings["gpu"] = gpu
        if self.config.local_rank != -1:
            self.config.environ_settings["gpu"] = self.config.local_rank
        
        SEED = self.environ_settings["SEED"] # random seed for reproduce results
        torch.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        if self.environ_settings["distributed"]:
            if self.environ_settings["dist_url"] == "env://" and self.environ_settings["rank"] == -1:
                self.config.environ_settings["rank"] = int(os.environ["RANK"])
            if self.environ_settings["multiprocessing_distributed"]:
                # For multiprocessing distributed training, rank needs to be the
                # global rank among all the processes
                self.config.environ_settings["rank"] = self.environ_settings["rank"] * ngpus_per_node + gpu
            dist.init_process_group(backend=self.environ_settings["dist_backend"], init_method=self.environ_settings["dist_url"],
                                    world_size=self.environ_settings["world_size"], rank=self.environ_settings["rank"])
        
        self.logger, self.writer = writer_logger(self.log_settings["log_path"], self.log_settings["log_pic_path"], self.config.local_rank, self.common_settings["backbone"]["settings"][0]["resume_net_model"])
        
        if self.environ_settings["rank"] <= 0:
            self.logger.print([(x, self.config.__dict__[x]) for x in list(self.config.__dict__.keys()) if "__" not in x])

    def network_set(self, ngpus_per_node):
        
        self.model = []
        self.plugModule = []
        self.classifier = []
        for i in range(self.common_settings["backbone"]["num"]):
            self.model.append(
                models.model_zoo.get_model(self.backbone_settings[i]["backbone_model_name"], **self.backbone_settings[i]["args"])
            )
            
            if self.environ_settings["rank"] <= 0:
                    self.logger.print(self.model[i])
            
            if self.backbone_settings[i]["resume_net_model"] is not None:
                if self.environ_settings["rank"] <= 0:
                    self.logger.print('Loading resume (model) network...')
                state_dict = torch.load(self.backbone_settings[i]["resume_net_model"])["model_%d"%(i)]
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
                self.model[i].load_state_dict(new_state_dict)
                if self.environ_settings["rank"] <= 0:
                    self.logger.print("resume net (model) loaded")
                
        for i in range(self.common_settings["classifier"]["num"]):
            self.classifier.append(
                models.model_zoo.get_model(self.classifier_settings[i]["classifier_model_name"], **self.classifier_settings[i]["args"])
            )
            if self.classifier_settings[i]["resume_net_classifier"] is not None:
                if self.environ_settings["rank"] <= 0:
                    self.logger.print('Loading resume (classifier) network...')
                state_dict = torch.load(self.classifier_settings[i]["resume_net_classifier"])["classifier_%d"%(i)]
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
                self.classifier[i].load_state_dict(new_state_dict)
                if self.environ_settings["rank"] <= 0:
                    self.logger.print("resume net (classifier) loaded")
        
        self.optimizer = torch.optim.SGD(
                            [{'params': x.parameters()} for x in self.model]+
                            [{'params': x.parameters()} for x in self.classifier]
                            ,
                            weight_decay=self.other_settings["weight_decay"],
                            lr=self.other_settings["lr"],
                            momentum=self.other_settings["momentum"],
                            )
        
        if self.other_settings["resume"] is True:
            if self.environ_settings["rank"] <= 0:
                self.logger.print('Loading resume (optimizer) network...')
            checkpoint = torch.load(self.other_settings["resume_net_optimizer"], map_location=torch.device('cpu'))
            self.other_settings["start_epoch"] = checkpoint['EPOCH']
            if self.environ_settings["rank"] <= 0:
                self.logger.print("start epoch: %d"%(self.other_settings["start_epoch"]))
            self.optimizer.load_state_dict(checkpoint['OPTIMIZER'])#断点恢复：动量和l2惩罚项
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        if self.environ_settings["distributed"]:
                            state[k] = v.cuda(self.environ_settings["gpu"])
                        else:
                            state[k] = v.cuda()
            
            if self.environ_settings["rank"] <= 0:
                self.logger.print("resume net (optimizer) loaded")
        
        if self.environ_settings["distributed"] and self.other_settings["resume"] and ngpus_per_node>1:
            # Use a barrier() to make sure that all processes have finished reading the
            # checkpoint
            dist.barrier()


        if self.environ_settings["distributed"]:
            # For multiprocessing distributed, DistributedDataParallel constructor
            # should always set the single device scope, otherwise,
            # DistributedDataParallel will use all available devices.
            if self.environ_settings["gpu"] is not None:
                torch.cuda.set_device(self.environ_settings["gpu"])
                for i in range(self.common_settings["backbone"]["num"]):
                    self.model[i].cuda(self.environ_settings["gpu"])
                    self.model[i] = torch.nn.parallel.DistributedDataParallel(self.model[i], device_ids=[self.environ_settings["gpu"]], find_unused_parameters=True)
                for i in range(self.common_settings["classifier"]["num"]):
                    self.classifier[i].cuda(self.environ_settings["gpu"])
                    self.classifier[i] = torch.nn.parallel.DistributedDataParallel(self.classifier[i], device_ids=[self.environ_settings["gpu"]], find_unused_parameters=True)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs we have
                self.config.data_settings["training"]["batch_size"] = int(self.config.data_settings["training"]["batch_size"] / ngpus_per_node)
                self.config.data_settings["training"]["num_workers"] = int((self.config.data_settings["training"]["num_workers"] + ngpus_per_node - 1) / ngpus_per_node)
                if self.environ_settings["rank"] <= 0:
                    self.logger.print("data balance")
            else:
                for i in range(self.common_settings["backbone"]["num"]):
                    self.model[i].cuda()
                    self.model[i] = torch.nn.parallel.DistributedDataParallel(self.model[i], find_unused_parameters=True)
                for i in range(self.common_settings["classifier"]["num"]):
                    self.classifier[i].cuda()
                    self.classifier[i] = torch.nn.parallel.DistributedDataParallel(self.classifier[i], find_unused_parameters=True)
                if self.environ_settings["rank"] <= 0:
                    self.logger.print("use all available GPUs")
        elif self.environ_settings["gpu"] is not None:
            torch.cuda.set_device(self.environ_settings["gpu"])
            for i in range(self.common_settings["backbone"]["num"]):
                self.model[i].cuda(self.environ_settings["gpu"])
            for i in range(self.common_settings["classifier"]["num"]):
                self.classifier[i].cuda(self.environ_settings["gpu"])
            if self.environ_settings["rank"] <= 0:
                self.logger.print('Use one GPU')
        else:
            # DataParallel will divide and allocate batch_size to all available GPUs
            for i in range(self.common_settings["backbone"]["num"]):
                self.model[i]=torch.nn.DataParallel(self.model[i]).cuda()
            for i in range(self.common_settings["classifier"]["num"]):
                self.classifier[i]=torch.nn.DataParallel(self.classifier[i]).cuda()
            if self.environ_settings["rank"] <= 0:
                self.logger.print('Use DP')
    
    def data_set(self):
        
        train_transform = transforms.Compose([
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
            ])
    
        self.dataset_train = dataloader_lmdb.ImageList(format_transform=train_transform, **self.data_settings["loader_settings"])
                            # format_transform=train_transform,shuffle=False, preproc=None)
        if self.environ_settings["distributed"]:
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.dataset_train)
        else:
            self.train_sampler = None
        self.train_loader = torch.utils.data.DataLoader(self.dataset_train, batch_size=self.data_settings["batch_size"], shuffle = (self.train_sampler is None), num_workers=self.data_settings["num_workers"], pin_memory=True, sampler=self.train_sampler, drop_last=True)

        self.loss = []
        # loss = torch.nn.CrossEntropyLoss().cuda(self.environ_settings["gpu"])
        self.loss.append(torch.nn.CrossEntropyLoss()) ## ! TODO loss是需要手动修改的
        self.loss.append(torch.nn.MSELoss())
        # self.loss.append(torch.nn.LogSoftmax(dim=1))
        # self.loss.append(torch.nn.NLLLoss())
        
        
        if self.environ_settings["rank"] <= 0:
            self.logger.print(self.loss)
            self.logger.print(self.loss_func)
        
        if not os.path.exists(self.log_settings["save_path"]):
            os.makedirs(self.log_settings["save_path"])
            
    def loss_func(self, lossList, **kwargs):
        result=0
        for i in range(self.common_settings["classifier"]["num"]):
            result += self.classifier_settings[i]["alpha"]*lossList[i]
        return result

    def printInfo(self, *args, **kwargs):
        str = 'Epoch {0[0]}/{0[1]} Batch {0[2]}/{0[3]} eta: {0[4]}\t'.format(*args) ## 因为输入的是列表，所以修改了{}内的内容
        strDict = {}
        for i in range(len(kwargs["loss_recorder"])-1):
            str+='Training Loss%d {loss%d.val:.4f} ({loss%d.avg:.4f})\t' %(i+1, i+1, i+1)
            strDict["loss%d"%(i+1)]=kwargs["loss_recorder"][i]
        str += 'Training Total_Loss {total.val:.4f} ({total.avg:.4f})\t'
        strDict["total"]=kwargs["loss_recorder"][-1]
        if self.common_settings["classifier"]["num"]==1:
            str +='Training Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
            str +='Training Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
        else:
            str +='Training Prec@1_up {top1.val:.3f} ({top1.avg:.3f})\t'
            str +='Training Prec@1_down {top5.val:.3f} ({top5.avg:.3f})\t'
        strDict["top1"]=kwargs["measurement"][0]
        strDict["top5"]=kwargs["measurement"][1]
        return str.format(**strDict)
    def train(self, ngpus_per_node):
        lr = self.other_settings["lr"]
        for epoch in range(self.other_settings["start_epoch"], self.other_settings["max_epoch"] + 1):
            np.random.seed(epoch)
            if self.environ_settings["distributed"]:
                self.train_sampler.set_epoch(epoch)
            # adjust_learning_rate(optimizer, epoch, cfg)
            lr = adjust_learning_rate(self.optimizer, self.other_settings["lr"], 0, epoch, self.other_settings["step_size"], lr, self.other_settings["gama"], base=self.other_settings["base"])

            #train for one epoch
            # train(train_loader, model, [classifier1, classifier2], [loss1,loss2,loss3], optimizer, epoch, cfg, writer, logger)
            self.train_epoch(epoch)

            if self.environ_settings["rank"] % ngpus_per_node == 0 or self.environ_settings["rank"]==-1:
                if epoch%5==0 and epoch<=25 or epoch>25:
                    self.logger.print("Save Checkpoint...")
                    self.logger.print("=" * 60)
                    for i in range(self.common_settings["backbone"]["num"]):
                        self.model[i].module.save(self.log_settings["save_path"] + 'backbone_%d_%d_checkpoint.pth'%(i, epoch))
                    for i in range(self.common_settings["classifier"]["num"]):
                        self.classifier[i].module.save(self.log_settings["save_path"] + 'classifier_%d_%d_checkpoint.pth'%(i, epoch))
                    save_dict = {'EPOCH': epoch,
                                'OPTIMIZER': self.optimizer.state_dict()}
                    torch.save(save_dict, self.log_settings["save_path"] + 'optimizer_' + str(epoch) + '_checkpoint.pth')
                    self.logger.print("Save done!")
                    self.logger.print("=" * 60)
                    # Use a barrier() to make sure that process 1 loads the model after process
                    # 0 saves i
            if self.environ_settings["distributed"] and ngpus_per_node>1:
                dist.barrier()
    
    def train_epoch(self, epoch):
        loss_recorder = [] # for training loss # 和lossLost长度一致
        measurement = [] # for testing measurement # 长度和validationList长度一致
        for i in range(self.common_settings["backbone"]["num"]):
            self.model[i].train()
        for i in range(self.common_settings["classifier"]["num"]):
            self.classifier[i].train()
            loss_recorder.append(AverageMeter()) # for each loss
            measurement.append(AverageMeter())
        if self.common_settings["classifier"]["num"]==1:
            measurement.append(AverageMeter())
        if "plugin" in self.backbone_settings[0]["args"].keys() and self.backbone_settings[0]["args"]["plugin"]=="PCM_AM":
            loss_recorder.append(AverageMeter())
        loss_recorder.append(AverageMeter()) # for total loss


        _t = {'forward_pass': Timer(), 'backward_pass': Timer(), 'data_pass': Timer(), "train_pass": Timer()}
        _t['data_pass'].tic()
        _t['train_pass'].tic()
        # for batch, (inputs, labels, classes, diff_location) in tqdm(enumerate(self.train_loader, 1)):# tqdm: 进度条
        for batch, data_batch in tqdm(enumerate(self.train_loader, 1)):# tqdm: 进度条
            _t['data_pass'].toc()
            # compute output
            inputs = data_batch[0].cuda(self.environ_settings["gpu"], non_blocking=True)
            labels = data_batch[1].cuda(self.environ_settings["gpu"], non_blocking=True)
            if self.data_settings["loader_settings"]["ldm68"]:
                classes = data_batch[2].cuda(self.environ_settings["gpu"], non_blocking=True)
            if self.data_settings["loader_settings"]["augu_paral"]:
                maskedInputs= data_batch[-2].cuda(self.environ_settings["gpu"], non_blocking=True)
                inputs = torch.cat((inputs, maskedInputs), dim=0) # ! 注意前半段是正常，后半段是口罩

            _t["forward_pass"].tic() # ! 要求augu的参数必须得有！
            featuresList = list(self.model[0](inputs, self.data_settings["loader_settings"]["augu_paral"])) # tuple转list； # ! model内没有做口罩和正常的区分
            # featuresList = self.model[0](inputs, self.data_settings["loader_settings"]["augu_paral"]) # tuple转list； # ! model内没有做口罩和正常的区分
            
            if "norm" in self.backbone_settings[0]["args"].keys() and self.backbone_settings[0]["args"]["norm"]==False:
                ## TODO 比较硬的执行flatten命令
                featuresList[0] = torch.flatten(featuresList[0], start_dim=1)
                featuresList[1] = torch.flatten(featuresList[1], start_dim=1)
            # features_down = features_down*(1-classes.unsqueeze(1))
            ## multi output
            outputList = []
            if self.common_settings["classifier"]["num"]==1:
                outputList.append(self.classifier[0](featuresList, labels)) # outputs, original_logits
            else:
                for i in range(self.common_settings["classifier"]["num"]):
                    outputList.append(self.classifier[i](featuresList[i], labels)) # outputs_up, original_logits_up

            _t["forward_pass"].toc()
            lossList=[]
            lossList.append( self.loss[0](outputList[0][0], labels) )
            if self.common_settings["classifier"]["num"]==2:
                lossList.append( self.loss[0](outputList[1][0], labels) )
            
            total_loss = self.loss_func(lossList)
            if "plugin" in self.backbone_settings[0]["args"].keys() and self.backbone_settings[0]["args"]["plugin"]=="PCM_AM":
                lossList.append( self.loss[-1](featuresList[2], torch.full_like(featuresList[2], 1)) )
                total_loss += lossList[-1]

            # normal_num = max( (1-classes).sum(), 1)
            # log_softmax_down = self.loss[1](outputList[1][0])*(1-classes.unsqueeze(1))
            # lossList.append( self.loss[2](log_softmax_down, labels)*(classes.shape[0]**1) / (normal_num**1) )
            
            # total_loss = loss + 0.1*loss2 + 0.1*loss3
            # total_loss = loss + 0.01*loss2 + 0.01*loss3
            # total_loss = (loss + 1*loss2 + 1*loss3)/3
            # total_loss = (loss + 2*loss3)/3

            
            lossList.append(total_loss)
            
            # total_loss = loss
            _t["backward_pass"].tic()
            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            _t["backward_pass"].toc()
            _t['data_pass'].tic()
            
            # measure accuracy and record loss
            # print(torch.cuda.device_count())
            factor = 1
            
            validationList = []
            for i in range(self.common_settings["classifier"]["num"]):
                validationList.append(accuracy(outputList[i][1].data, labels, topk = (1, 5)))
                # prec1_up, prec5_up = accuracy(original_logits_up.data, labels, topk = (1, 5))
            if self.environ_settings["distributed"]:
                for i in range(len(lossList)):
                    lossList[i] = reduce_tensor(lossList[i].clone().detach_()) # each loss
                for i in range(len(validationList)):
                    validationList[i][0] = reduce_tensor(validationList[i][0].clone().detach_()) # each loss
                    validationList[i][1] = reduce_tensor(validationList[i][1].clone().detach_()) # each loss
                
                # prec1 = reduce_tensor(torch.from_numpy(prec1))
                # prec5 = reduce_tensor(torch.from_numpy(prec5))
                torch.cuda.synchronize() # wait every process finish above transmission
                factor = torch.cuda.device_count()

            # loss_recorder[-1].update(total_loss.data.item()/factor, inputs.size(0))
            for i in range(len(lossList)):
                loss_recorder[i].update(lossList[i].data.item()/factor, inputs.size(0))
            
            # top1.update(prec1.data.item()/factor, inputs.size(0))
            # top5.update(prec5.data.item()/factor, inputs.size(0))
            if self.common_settings["classifier"]["num"]==1:
                measurement[0].update(validationList[0][0].data.item()/factor, inputs.size(0))
                measurement[1].update(validationList[0][1].data.item()/factor, inputs.size(0))
            else:
                for i in range(len(validationList)):
                    measurement[i].update(validationList[i][0].data.item()/factor, inputs.size(0))
            
            if ( ((batch + 1) % self.log_settings["log_interval"] == 0) or batch == 0 ) and self.environ_settings["rank"] <= 0:
                self.logger.print("time cost, forward:{}, backward:{}, data cost:{} "
                .format(str(_t['forward_pass'].average_time), str(_t['backward_pass'].average_time), str(_t['data_pass'].average_time)))
                self.logger.print("=" * 60)
                _t['train_pass'].toc()
                _t['train_pass'].tic()
                eta = _t["train_pass"].diff * ((self.other_settings["max_epoch"] - epoch + 1) * len(self.train_loader) - batch) / self.log_settings["log_interval"]
                self.logger.print(self.printInfo([epoch, self.other_settings["max_epoch"], batch + 1, len(self.train_loader), str(datetime.timedelta(seconds=eta))], 
                                                 loss_recorder=loss_recorder, measurement=measurement))
                self.logger.print("=" * 60)
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
        eta = _t["train_pass"].diff * ((self.other_settings["max_epoch"] - epoch + 1) * len(self.train_loader) - batch) / self.log_settings["log_interval"]
        if self.environ_settings["rank"] <= 0:
            self.logger.print(self.printInfo([epoch, self.other_settings["max_epoch"], batch + 1, len(self.train_loader), str(datetime.timedelta(seconds=eta))], 
                                                 loss_recorder=loss_recorder, measurement=measurement))
            self.logger.print("=" * 60)
        
        # sys.stdout.flush()
        if self.environ_settings["rank"]<= 0:
            self.writer.add_scalar("Training_Loss1", epoch_loss1, epoch)
            # writer.add_scalar("Training_Loss2", epoch_loss2, epoch)
            # writer.add_scalar("Training_Loss3", epoch_loss3, epoch)
            self.writer.add_scalar("Training_Total_Loss", epoch_total_loss, epoch)
            # writer.add_scalar("Training_Accuracy", epoch_acc, epoch)
            self.writer.add_scalar("Top1", epoch_acc_1, epoch)
            self.writer.add_scalar("Top5", epoch_acc_5, epoch)
            # writer.add_scalar("Top1_up", epoch_acc_up, epoch)
            # writer.add_scalar("Top1_down", epoch_acc_down, epoch)
 