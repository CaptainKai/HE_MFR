import logging
import sys

import time

import os
from tensorboardX import SummaryWriter

def print_with_time(string):
    print(time.strftime("%Y-%m-%d %H:%M:%S ", time.localtime()) + string)

formater = logging.Formatter('%(asctime)s: %(message)s')

class Logger(object):
    
    def __init__(self, log_path, name="train_log", formater=formater, mode="new"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        if mode=="new":
            self.file_handler = logging.FileHandler(log_path, mode="w")
        elif mode=="continue":
            self.file_handler = logging.FileHandler(log_path, mode="a")
        else:
            print("logger init wrong!")
            exit(0)
        self.file_handler.setFormatter(formater)
        self.file_handler.setLevel(logging.DEBUG)
        self.console_handler = logging.StreamHandler(sys.stdout)
        self.console_handler.formatter=formater
        self.console_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(self.file_handler)
        self.logger.addHandler(self.console_handler)

    def print(self, message):
        # print_with_time(message)
        if isinstance(message,list):
            for x in message:
                self.logger.info(x)
        else:        
            self.logger.info(message)
        
    def close(self):
        self.logger.removeHandler(self.file_handler)
        self.logger.removeHandler(self.console_handler)
        

class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff

    def clear(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

import torch
import numpy as np
def compute_accuracy_multi(prob_cls, gt_cls):
    '''
    多label的准确率计算
    '''
    with torch.no_grad():
        predict_cls = torch.argmax(prob_cls,1)
        correct = (np.array(predict_cls.cpu())==np.array(gt_cls.cpu()))
        # correct = (predict_cls==gt_cls)# torch vision
        size = gt_cls.size()[0]
        # return torch.div(torch.mul(torch.sum(correct),float(1.0)),float(size))
        return sum(correct)*1.0/float(size)


## 学习率调整函数，目前只实现了multysteps学习率调整策略
def adjust_learning_rate(optimizer, init_lr, iteration, epoch, step_size, lr, gama=0.1, base="iteration"):
    """Sets the learning rate to the initial LR decayed"""
    temp_lr = init_lr
    if base=="iteration":
        for milestone in step_size: 
            # Multistep(step)
            temp_lr *= float(gama) if iteration >= milestone else 1.    
    elif base=="epoch":
        for milestone in step_size: 
            # Multistep(step)
            temp_lr *= float(gama) if epoch >= milestone else 1.
    if temp_lr!=lr:
        print_with_time('Adjust learning rate to {}'.format(temp_lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = temp_lr
    return temp_lr


def writer_logger(log_path, log_pic_path, local_rank, resume_net_model):
    '''
    @resume_net_model: 用于判断是新创建Log文件还是追加内容
    '''
    log_path = log_path[:-4] + '_' + str(local_rank) + '.log'
    # log_pic_path = log_pic_path[:-1] + '_' +str(local_rank) + '/'
    if local_rank==0 or local_rank==-1:
        if os.path.exists(log_path):
            if resume_net_model is None:
                f = open(log_path, "w")
                f.close()
                logger = Logger(log_path)
            else:
                print("resume a training!")
                logger = Logger(log_path, mode="continue")
        else:
            logger = Logger(log_path)
        writer = SummaryWriter(log_pic_path)
    else:
        writer = None
        logger = None
    
    return logger, writer
