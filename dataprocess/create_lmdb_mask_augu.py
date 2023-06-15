# python3
import lmdb
from PIL import Image
import six
import os
import cv2
import numpy as np
from lmdb_init import LMDB
from utils import findNonreflectiveSimilarity

from wearmask import *
import random

'''
根据/home/ubuntu/data3/lk/amsoft_dataedit_train/data/data_gen/train_data_gen.py 修改
删除标准点探索的相关代码
封装为脚本，使用.sh运行
主要功能: 根据输入参数，产生lmdb文件
#note 先实现单数据源输入，再实现多数据源生成，训练
#note 目前只实现固定关键点，之后添加动态关键点
#note 目前没有实现在准备数据的阶段对数据进行增广
具体功能：
对原图进行对齐，并且存储对齐之后的五点，68点标签
'''
import argparse
#note type can be str,list,bool,int,dict,float
parser = argparse.ArgumentParser(description='create lmdb for amsoft training')
# DATA
parser.add_argument('--root_path', type=str, default='/home/ubuntu/data1/lk',
                    help='path to root path of images')
# parser.add_argument('--gt_paths', type=list, default=['/home/ubuntu/data1/lk/celebrity_msra_lmk/msra_lmk_68'],
parser.add_argument('--gt_paths', type=list, default=['/home/ubuntu/data2/lk/amsoft_pytorch/amsoft_pytorch/data/result/Dets_casia_68_rect_id_fixBug'],
# parser.add_argument('--gt_paths', type=list, default=['/home/ubuntu/data2/lk/amsoft_pytorch/amsoft_pytorch/dataprocess/test_68'],
                    help='path to txt of groundtruth')# 
parser.add_argument('--std_point', type=list, default=[39.2946, 69.5318, 62.0252, 29.5493, 65.7200, \
                                                        21.6963, 14.5014, 56.7366, 87.3655, 80.2041],
                    help='the standard point for aligning face（format: x1,x2...y1,y2...')
parser.add_argument('--lmdb_path', type=str, default='/home/ubuntu/data4/lk/data/lmdb_mask_augu_full_casia',
                    help='the lmdb path for saving lmdb file')
parser.add_argument('--single', type=bool, default=True,
                    help='determing whether output single or multiple imdb files')
parser.add_argument('--save_interval', type=int, default=25000,
                    help='the interval for save data array')                    
parser.add_argument('--save', type=bool, default=False,
                    help='determing whether save aligned pic or not')
parser.add_argument('--save_path', type=str, default='./aligned_pic',
                    help='the root path for saving aligned pic')
parser.add_argument('--unit_size', type=int, default=112,
                    help='the pic size which is the value of width and height')                    
args = parser.parse_args()
print(args)

root_path = args.root_path
gt_paths = args.gt_paths
single = args.single
save = args.save
save_path = args.save_path
save_interval = args.save_interval
lmdb_path = args.lmdb_path
unit_size = args.unit_size

face_maker = FaceMasker()

std_point = [float(x) for x in args.std_point]
std = std_point
# std = np.zeros(10) #[x1,x2...y1,y2...]=>[x1,y1,...]
# for i in range(5):
#     std[i]=std_point[i*2]
# for i in range(5):
#     std[i+5]=std_point[i*2+1]
if os.path.exists(lmdb_path):
    lmdb_files = os.listdir(lmdb_path)
    for lmdb_file in lmdb_files:
        os.remove(os.path.join(lmdb_path, lmdb_file))
lmdb_instance = LMDB(lmdb_path)
import time
import datetime

label_keys = []
image_keys = []
ldm_keys = [] # TODO
ldm68_keys = [] # TODO
label_values = []
image_values = []
ldm_values = [] # TODO
ldm68_values = [] # TODO

cnt = 0

for gt_path in gt_paths:
    print(gt_path)
    gt_file = open(gt_path, 'r')
    gt=gt_file.readline()
    print(gt)
    print(cnt)

    make_t0 = time.time()

    while gt:
        line = gt.strip("\n").split()
        if cnt % 50000 == 0 and cnt>0:
            make_t1 = time.time()
            print("processing %d"%(cnt))
            time_cost = make_t1-make_t0
            print('time consume:{}'.format(str(datetime.timedelta(seconds=time_cost))))
            make_t0 = time.time()
        pic_name = line[0]
        # print(pic_name)
        id = int(line[1])
        # landmark = [float(x) for x in line[2:12]]
        landmark = [float(x) for x in line[6:16]]#

        pic_path = pic_name # TODO
        # pic_path = os.path.join(root_path, pic_name)
        if not os.path.exists(pic_path):
            print(pic_path)
            gt=gt_file.readline()
            continue
        pic_frame = cv2.imread(pic_path)

        src = np.zeros(10) #[x1,y1,...]=>[x1,x2...y1,y2...]
        for i in range(5):
            src[i]=landmark[i*2]
        for i in range(5):
            src[i+5]=landmark[i*2+1]
        # # TC1: 点的可视化
        # for i in range(5):
        #     cv2.circle(pic_frame, (int(src[i]),int(src[i+5])), 1, (0,0,255), 2)
        # for i in range(5):
        #     cv2.circle(pic_frame, (int(std[i]),int(std[i+5])), 1, (255,0,0), 2)
        # cv2.imshow("origin",pic_frame)
        # cv2.waitKey(0)
        # 获取仿射矩阵，eg: M = cv2.getAffineTransform(src,dst)
        M = findNonreflectiveSimilarity(src, std)
        res = cv2.warpAffine(pic_frame,M,(int(unit_size),int(unit_size)))
        # cv2.imshow("frame1",res)
        # cv2.waitKey(0)
        # exit(0)
        # print(src)
        new_p = []
        for i in range(5):
            x = M[0,0]*landmark[2*i] + M[0,1]*landmark[2*i+1] + M[0,2]
            y = M[1,0]*landmark[2*i] + M[1,1]*landmark[2*i+1] + M[1,2]
            # new_p.append([x, y])
            new_p.append(x)
            new_p.append(y)
        
        if save:
            dest_path = os.path.join(save_path,pic_name)
            if not os.path.exists(dest_path):
                os.makedirs(dest_path)
            cv2.imwrite(dest_path, res)
        
        image_key = "image-%09d"%(cnt)
        label_key = "label-%09d"%(cnt)

        ldm_key = "ldm-%09d"%(cnt)# TODO

        image_keys.append(image_key)
        label_keys.append(label_key)

        ldm_keys.append(ldm_key)# TODO

        img_encode = cv2.imencode("."+pic_name.split(".")[-1], res)
        str_encode = img_encode[1].tostring()
        label = str(id)

        ldm = " ".join([str(x) for x in new_p])# TODO

        image_values.append(str_encode)
        label_values.append(label.encode())

        ldm_values.append(ldm.encode())# TODO

        ## mask argumentation
        # color_index = random.randint(0,5)
        # point_68 = [int(x) for x in line[12:]]
        point_68 = [float(x) for x in line[16:]]
        new_point_68=[]
        for i in range(68):
            x = M[0,0]*point_68[2*i] + M[0,1]*point_68[2*i+1] + M[0,2]
            y = M[1,0]*point_68[2*i] + M[1,1]*point_68[2*i+1] + M[1,2]
            x = x if x>=0 else 0
            y = y if y>=0 else 0
            x = x if x<=int(unit_size) else int(unit_size)
            y = y if y<=int(unit_size) else int(unit_size)

            new_point_68.append(int(x))
            new_point_68.append(int(y))
        point_68 = new_point_68
        # for i in range(68):
        #     cv2.circle(res, (point_68[2*i],point_68[2*i+1]), 1, (255,0,0), 2)
        # cv2.imshow("origin",res)
        # cv2.waitKey(0)
        # exit(0)
        # masked_image, success = face_maker.mask(pic_frame[:,:,::-1], color_index, point_68) # BGR->RGB
        # masked_image, success = face_maker.mask(res[:,:,::-1], color_index, point_68) # BGR->RGB
        # if success:
        #     # res = cv2.warpAffine(np.asarray(masked_image)[:,:,::-1], M, (int(unit_size),int(unit_size))) # RGB->BGR
        # res = np.asarray(masked_image)[:,:,::-1]
        # image_key = "image-%09d"%(cnt)
        # label_key = "label-%09d"%(cnt)
        ldm68_key = "ldm-full-%09d"%(cnt)# TODO
        ldm68_keys.append(ldm68_key)# TODO
        ldm68_value = " ".join([str(x) for x in point_68])
        ldm68_values.append(ldm68_value.encode())

        cnt+=1

        if cnt%save_interval==0:
            lmdb_instance.write_data(image_keys,image_values,record=True)
            lmdb_instance.write_data(label_keys,label_values,record=False)
            lmdb_instance.write_data(ldm_keys,ldm_values,record=False)# TODO
            lmdb_instance.write_data(ldm68_keys,ldm68_values,record=False)# TODO
            label_keys = []
            image_keys = []
            ldm_keys = []# TODO
            ldm68_keys = []# TODO
            label_values = []
            image_values = []
            ldm_values = []# TODO
            ldm68_values = []# TODO
        gt=gt_file.readline()
    if len(image_keys)!=0:
        lmdb_instance.write_data(image_keys,image_values,record=True)
        lmdb_instance.write_data(label_keys,label_values,record=False)
        lmdb_instance.write_data(ldm_keys,ldm_values,record=False)# TODO
        lmdb_instance.write_data(ldm68_keys,ldm68_values,record=False)# TODO
        label_keys = []
        image_keys = []
        ldm_keys = []# TODO
        ldm68_keys = []# TODO
        label_values = []
        image_values = []
        ldm_values = []# TODO
        ldm68_values = []# TODO
    gt_file.close()
print(lmdb_instance.description)       
lmdb_instance.closelmdb()