# coding: utf-8
'''
lmdb基本类的整理
最重要的长度分三种模式获取
经过修改，目前只能使用numpy格式读取
@author: likai
'''
import lmdb
import chardet

import random

import os
import numpy as np
from .utils import getlmdb_caffe, getlmdb_stream # TODO
from .utils import getlmdb_numpy_image, getlmdb_numpy_label
import time # for time test


LMDB_MAP_SIZE = 1073741824*1024 # 500G #1099511627776 1T
class LMDB():
    '''对lmdb的一个封装''' 
    def __init__(self, lmdb_path="./lmdb_data/default", max_reader=1,num=None,key_list=None,method="numpy"):
        '''
        @lmdb_path: 存放lmdb文件的路径,当文件不存在时会自动进行创建
        @max_reader: 最大同时操作lmdb的线程个数，默认为1
        @num: 提前设置数据集长度，减少遍历数据集获取长度的过程
        @key_list：存储key的文件路径，已经用默认规则取代该参数,即该参数可有可无
        @method：读取数据集的方式，可设置为numpy, stream, caffe
        '''
        super(LMDB, self).__init__()
        self.lmdb_path = lmdb_path
        if not os.path.exists(self.lmdb_path):
            print("lmdb path does not exist, create path: %s"%(self.lmdb_path))
            os.makedirs(self.lmdb_path)
            lmdbfiles = []
        else:
            lmdbfiles = os.listdir(lmdb_path)
        self.num=0
        if len(lmdbfiles)!=0:
            self.enviroment = lmdb.open(lmdb_path, max_readers=max_reader,
                lock=False, readahead=False, meminit=False)# lock=false 可以同时进行读写操作
            self.get_num(num=num,key_list=key_list)
            self.description = "both keys of labels and values are marked by image-%%09d and label-%%09d;there are total %d data"%self.num
            print(self.description)            
        if len(lmdbfiles)==0:
            self.enviroment = lmdb.open(lmdb_path, map_size=LMDB_MAP_SIZE)
            self.description = "both keys of labels and values are marked by image-%%09d and label-%%09d;there are total %d data"%self.num
        self.lmdb_txn = self.enviroment.begin()# 用于lmdb读取
        if method=="numpy":
            self.getitem_image = getlmdb_numpy_image
            self.getitem_label = getlmdb_numpy_label
        elif method=="stream":
            self.getitem_all = getlmdb_stream# 对应文件直接读写
        elif method=="caffe":
            self.getitem_all = getlmdb_caffe# 对应caffe脚本生成lmdb
        else:
            print("please make sure your method are call for numpy;stream;caffe！")
            exit(0)
    
    def get_num(self,num=None,two=True,key_list=None,log_interval=2000):
        '''
        @num：直接提供数据集长度
        @two: 数据和label是否分开存储，如果是，则长度为遍历次数的一半
        @key_list: 存储用于取数据的关键词的文件
        @log_interval：仅在需要对数据集进行遍历的时候生效，每读取@log_interval次进行一次提示
        '''
        if num is not None:
            self.num=num
        else:
            i=0
            if key_list is not None:
                with open(key_list,'r') as key_file:

                    keyline = key_file.readline()
                    while keyline:
                        i = i+1
                        keyline = key_file.readline()
            else:
                with self.enviroment.begin(write=False) as txn:
                    for key, value in txn.cursor():
                        if i%log_interval ==0:
                            print('reading %d'%(i))
                            # print(time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()))
                        i+=1
            if two:   
                self.num = max(0,i)//2
            else:
                self.num = max(0,i)

    def __len__(self):
        return self.num
 
    def getitem(self, index, ldm=False, ldm68=False):
        '''
        @index: 数据的index
        @ldm: 是否读取5点标签
        @ldm68: 是否读取68点标签, 该输入会覆盖上一个输入的效果
        '''
        lmdb_txn = self.lmdb_txn
        index_id = index

        img = self.getitem_image(lmdb_txn, index, 'image')
        label = self.getitem_label(lmdb_txn, index, 'label')
        if ldm and not ldm68:
            ldm = self.getitem_label(lmdb_txn, index, 'ldm')
            return img, label, ldm
        elif not ldm and ldm68:
            ldm68 = self.getitem_label(lmdb_txn, index, 'ldm-full')
            return img,label,ldm68
        elif ldm and ldm68:
            ldm = self.getitem_label(lmdb_txn, index, 'ldm')
            ldm68 = self.getitem_label(lmdb_txn, index, 'ldm-full')
            return img,label,ldm68
        else:
            return img,label


    def get_label(self, index):
        lmdb_txn = self.lmdb_txn
        label = self.getitem_label(lmdb_txn, index, 'label')
        return label
    
    def get_image(self, index, ldm=False, ldm68=False):
        lmdb_txn = self.lmdb_txn
        img = self.getitem_image(lmdb_txn, index, 'image')
        if ldm and not ldm68:
            ldm = self.getitem_label(lmdb_txn, index, 'ldm')
            return img, ldm, None
        elif not ldm and ldm68:
            ldm68 = self.getitem_label(lmdb_txn, index, 'ldm-full')
            return img, None, ldm68
        elif ldm and ldm68:
            ldm = self.getitem_label(lmdb_txn, index, 'ldm')
            ldm68 = self.getitem_label(lmdb_txn, index, 'ldm-full')
            return img, ldm, ldm68
        else:
            return img, None, None


    
    def write_data(self, keys, values,log_interval=100,record=True):
        '''
        批量输入数据，数据格式要求如下
        @keys：string类型的数组
        @values:字节流型数组(必须是字节流，无法改为numpy数组，因为编码需要结合文件后缀)
        # IMP: 添加对输入类型的判断
        # REP:(len(keys)%2!=0)
        # IMP: 好像整个代码没有对输入是none的判断之类的
        '''

        # REP：assert(len(keys)!=len(values)) # open
        if len(keys)!=len(values):
            print("key length is not equal to value length！%d:%d"%(len(keys),len(values)))
            return False
        i = 0
        txn = self.enviroment.begin(write=True)
        for key in keys:
            txn.put(key.encode(), values[i])
            # if i%log_interval==0:
            #     print("processing %d/%d done"%(i,num))
            i+=1
        txn.commit()
        if record:
            self.num+=len(keys)
            self.description = "both  keys of labels and values are marked by image-%%09d and label-%%09d;there are total %d data"%self.num
    
    def gen_keylist(self, keyfilepath):
        keyfile=open(keyfilepath,'w')
        txn=self.enviroment.begin(write=False)
        i=0
        for key,value in txn.cursor():
            print(key.decode())
            keyfile.write(key.decode()+"\n")
            i+=1
        print("total key num is %d"%(i))
        keyfile.close()
        
    def closelmdb(self):
        self.enviroment.close()

## TC sample
if __name__ == "__main__":
    # lmdb_path = '/home/ubuntu/data3/lk/amsoft_dataedit_train/data/lmdb_data/msra_train_lmdb'
    # lmdb_test = LMDB(lmdb_path, 3)
    # lmdb_test.caffe_readall()

    # lmdb_path = "/home/ubuntu/data2/lk/amsoft_pytorch/data/amsoft_train/amsoft_lmdb"
    # filelist = "/home/ubuntu/data1/lk/CelebA/Anno/train_amsoft_id.txt"
    # keylist = "/home/ubuntu/data2/lk/amsoft_pytorch/data/amsoft_train/keylist.txt"
    # lmdb_test = LMDB(lmdb_path,3)
    # lmdb_test.write_filelist2lmdb(filelist)
    # print(lmdb_test.description)
    # lmdb_test.save_keylist(keylist)
    # # lmdb_test.read_all()
    # lmdb_test.closelmdb()

    lmdb_path = "/home/ubuntu/data4/lk/data/lmdb_mask_argu"
    datanum = 3923399
    test_lmdb = LMDB(lmdb_path,num=datanum)
    img, label = test_lmdb.getitem(0)
    img.show()
    # test_lmdb.gen_keylist("test.txt")
    test_lmdb.closelmdb()
    # pass
    
