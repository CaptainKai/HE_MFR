# python3
import lmdb
from PIL import Image
import six
import os
import cv2
import numpy as np
from lmdb_init import LMDB

'''
用于测试create_lmdb脚本生成的lmdb文件是否有效、以及一些简单的测试(时间消耗)
# @result 看起来没有问题
'''
import argparse
#note type can be str,list,bool,int,dict,float
parser = argparse.ArgumentParser(description='read lmdb')
# DATA
parser.add_argument('--lmdb_path', type=str, default='/home/ubuntu/data3/lk/amsoft_pytorch/data/lmdb_default',
                    help='the lmdb path for saving lmdb file')                  
parser.add_argument('--save', type=bool, default=False,
                    help='determing whether save aligned pic or not')
parser.add_argument('--save_path', type=str, default='./aligned_pic',
                    help='the root path for saving aligned pic')                   
args = parser.parse_args()
print(args)

save = args.save
save_path = args.save_path
lmdb_path = args.lmdb_path
lmdb_instance = LMDB(lmdb_path,num=3923399)
import time
time_test_cost_start = time.time()
print(lmdb_instance.num)
# for i in range(45678,45678+1+512):
#     image,label = lmdb_instance[i]
    # print(label)
    # break
image,label = lmdb_instance[0]
print(label)
time_test_cost_end = time.time()
time_test = time_test_cost_end-time_test_cost_start
print(time_test)
# image.show()
lmdb_instance.closelmdb()
