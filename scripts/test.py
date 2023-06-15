from PIL import Image
import numpy as np

from torchvision.transforms import functional as F
import torchvision.transforms as transforms
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn


def cos_distance(f1,f2):
	# 输入的特征均没有进行归一化
    with torch.no_grad():# 避免计算的时候计算梯度
        distance = f1.dot(f2) / (f1.norm() * f2.norm() + 1e-5)
        return distance

import cv2
# 余弦距离计算-numpy版本
def cos_distance_numpy(f1,f2):
	# 输入的特征均没有进行归一化
    aN = cv2.norm(f1)
    bN = cv2.norm(f2)

    dt = np.dot(f2,f1)
    similarity = dt / (aN*bN)

    return similarity

def l2_distance(f1,f2):
    with torch.no_grad():
        distance = (f1-f2).norm()
    return distance

# pair计算MLS
def pair_MLS_score(x1, x2, sigma_sq1=None, sigma_sq2=None):
    if sigma_sq1 is None:
        x1, x2 = np.array(x1), np.array(x2)
        assert sigma_sq2 is None, 'either pass in concated features, or mu, sigma_sq for both!'
        # D = int(x1.shape[1] / 2)
        # mu1, sigma_sq1 = x1[:,:D], x1[:,D:]
        # mu2, sigma_sq2 = x2[:,:D], x2[:,D:]
        mu1,sigma_sq1=x1[:512],x1[512:]
        mu2,sigma_sq2=x2[:512],x2[512:]
    else:
        x1, x2 = np.array(x1), np.array(x2)
        sigma_sq1, sigma_sq2 = np.array(sigma_sq1), np.array(sigma_sq2)
        mu1, mu2 = x1, x2
    sigma_sq_mutual = sigma_sq1 + sigma_sq2
    # dist = np.sum(np.square(mu1 - mu2) / sigma_sq_mutual + np.log(sigma_sq_mutual), axis=1)
    dist = np.sum(np.square(mu1 - mu2) / sigma_sq_mutual + np.log(sigma_sq_mutual))
    return dist


def MLS_score_cuda(x1, x2, sigma_sq1=None, sigma_sq2=None):
    if sigma_sq1 is None:
        x1, x2 = np.array(x1), np.array(x2)
        assert sigma_sq2 is None, 'either pass in concated features, or mu, sigma_sq for both!'
        # D = int(x1.shape[1] / 2)
        # mu1, sigma_sq1 = x1[:,:D], x1[:,D:]
        # mu2, sigma_sq2 = x2[:,:D], x2[:,D:]
        mu1,sigma_sq1=x1[:512],x1[512:]
        mu2,sigma_sq2=x2[:512],x2[512:]
    else:
        x1, x2 = np.array(x1), np.array(x2)
        sigma_sq1, sigma_sq2 = np.array(sigma_sq1), np.array(sigma_sq2)
        mu1, mu2 = x1, x2
    sigma_sq_mutual = sigma_sq1 + sigma_sq2
    # dist = np.sum(np.square(mu1 - mu2) / sigma_sq_mutual + np.log(sigma_sq_mutual), axis=1)
    dist = np.sum(np.square(mu1 - mu2) / sigma_sq_mutual + np.log(sigma_sq_mutual))
    return dist

# PFE模型提取图片特征
def extractDeepFeature_PFE(img, model,flop=False,transform=None):
    if transform is not None:
        img = transform(img)
    img = img.unsqueeze(0).to('cuda')
    if flop:
        img_ = transform(F.hflip(img))
        img_ = img_.unsqueeze(0).to('cuda')
        ft = torch.cat((model(img), model(img_)), 1)[0].to('cpu')
        return ft
    else:
        feature, conv_final = model(img)
        # print(feature)
        return feature[0], conv_final[0]

def extractDeepFeature(img, model,flop=False, transform=None):
    img = transform(img)
    img = img.unsqueeze(0).to('cuda')
    if flop:
        img_ = transform(F.hflip(img))
        img_ = img_.unsqueeze(0).to('cuda')
        ft = torch.cat((model(img), model(img_)), 1)[0]
        return ft
    else:
        return model(img)[0]
# 计算正确率
def get_accuracy(score_matrix,gt_matrix,threshold):
    pos_index = np.where(score_matrix>=threshold)[0]

    TP = np.where(gt_matrix[pos_index]>=1)[0].shape[0]
    FP = pos_index.shape[0]-TP
    return TP,FP


import json
import os
def rank_test(img_list_file=None, noise_file=None, tag="pytorch", rank_N=1,img_root="/home/ubuntu/Work/lk/test_data/facescrub_transform_lzj",noise_root="/home/ubuntu/Work/lk/test_data/megaface_transform_lzj/megaface"):
    # 读取测试文件路径与标签
    face_crub_file_path = "/home/ubuntu/Work/lk/test_data/megaface-devkit/templatelists/facescrub_uncropped_features_list.json"
    face_crub_file = open(face_crub_file_path, "r")
    face_crub_json = json.load(face_crub_file)
    face_crub_id = face_crub_json["id"]
    face_crub_path = face_crub_json["path"]
	
	# 读取噪音文件的路径
    megaface_file_path = "/home/ubuntu/Work/lk/test_data/megaface-devkit/templatelists/megaface_features_list.json_10000_1"
    megaface_file = open(megaface_file_path, "r")
    megaface_json = json.load(megaface_file)
    megaface_path = megaface_json["path"]
    ## 可以开多线程
    gallery = []

    # 通过文件路径读取噪音文件，形成噪音矩阵（10000*512）
    for bin_path in megaface_path:
        bin_path = os.path.join(noise_root, bin_path)
        bin_path = bin_path + "_" + tag + ".bin"
        if(not os.path.isfile(bin_path)):
            print(bin_path)
            exit(0)
            continue
        feature = np.float32(np.fromfile(bin_path,dtype=np.float32))
        gallery.append(feature/cv2.norm(feature))
    gallery_len = len(gallery)
    
    print("gallery ready: %d"%(gallery_len))

    # 通过文件路径读取测试文件，形成测试矩阵（n*512)
    probe=[]
    for bin_path in face_crub_path:
        bin_path = os.path.join(img_root, bin_path)
        # print(bin_path)
        bin_path = bin_path + "_" + tag +".bin"
        if(not os.path.isfile(bin_path)):
            print(bin_path)
            bin_path=bin_path.replace(".jpg", ".png")
            if (not os.path.isfile(bin_path)):
                print(bin_path)
                exit(0)
                continue
        feature = np.float32(np.fromfile(bin_path,dtype=np.float32))
        probe.append(feature/cv2.norm(feature))
    probe_len = len(probe)
    print("probe ready: %d"%(probe_len))
    
    now_id = face_crub_id[0]
    now_index = 0

    # face_crub_len = len(face_crub_id)
    acc=0# 记录正确率
    compare_num = 0

    for i in range(probe_len):
        if now_id!=face_crub_id[i] or i==probe_len-1:
            # print(now_index,i)
            if i==probe_len-1:
                end_index=i+1
            else:
                end_index=i
            for j in range(now_index,end_index):
                if j==now_index:
                    probe_array = np.array(probe[now_index+1:end_index])
                    # gallery.append(probe[0])
                    gallery_array = np.array(np.concatenate((gallery,[probe[now_index]]),axis=0))
                elif j==end_index-1:
                    probe_array = np.array(probe[now_index:end_index-1])
                    # gallery.append(probe[temp_len])
                    gallery_array = np.array(np.concatenate((gallery,[probe[end_index-1]]),axis=0))
                else:
                    probe_array = np.array(np.concatenate((probe[now_index:j],probe[j+1:end_index]),axis=0))
                    # gallery.append(probe[j])
                    gallery_array = np.array(np.concatenate((gallery,[probe[j]]),axis=0))
                # print(probe_array.shape)
                # print(j,probe_array.size)
                score=[]
                score = np.dot(gallery_array, probe_array.transpose())

                score_index_sorted = np.argsort(score, axis=0)# 100001*(id_num-1)
                gt = gallery_len

                expected_index = score_index_sorted[-rank_N:][:]

                acc+=np.where(expected_index==gt)[0].shape[0]
                
                # gallery.pop()
                compare_num +=end_index-1-now_index
                # print(acc/float(compare_num))
                # break

            now_index = i
            now_id = face_crub_id[i]
        if (i+1)%200 ==0:
            print("processing %d/%d"%(i,probe_len))
            print(acc/float(compare_num))

    print("rank1: %f"%(acc/float(compare_num)))


def rank_test_opt(img_list_file=None, noise_file=None, tag="faceboxes", rank_N=1,img_root="/home/ubuntu/data2/lk/facebox/pytorch_version/FaceBoxes_landmark/data/Facecrub/aligned",noise_root="/home/ubuntu/data2/lk/facebox/pytorch_version/FaceBoxes_landmark/data/Megaface/aligned"):
    '''
    不使用并行，用空间换时间
    '''
    # 读取测试文件路径与标签
    face_crub_file_path = "/home/ubuntu/Work/lk/test_data/megaface-devkit/templatelists/facescrub_uncropped_features_list.json"
    face_crub_file = open(face_crub_file_path, "r")
    face_crub_json = json.load(face_crub_file)
    face_crub_id = face_crub_json["id"]
    face_crub_path = face_crub_json["path"]

    # 读取噪音文件的路径
    megaface_file_path = "/home/ubuntu/Work/lk/test_data/megaface-devkit/templatelists/megaface_features_list.json_10000_1"
    megaface_file = open(megaface_file_path, "r")
    megaface_json = json.load(megaface_file)
    megaface_path = megaface_json["path"]

    # 通过文件路径读取噪音文件，形成噪音矩阵（10000*512）
    gallery = []
    for bin_path in megaface_path:
        bin_path = os.path.join(noise_root, bin_path)
        bin_path = bin_path + "_" + tag + ".bin"
        if(not os.path.isfile(bin_path)):
            print(bin_path)
            exit(0)
            continue
        feature = np.float32(np.fromfile(bin_path,dtype=np.float32))[4:]
        gallery.append(feature/cv2.norm(feature))
    gallery_len = len(gallery)
    
    print("gallery ready: %d"%(gallery_len))

    # 通过文件路径读取测试文件，形成测试矩阵（n*512）
    probe=[]
    for bin_path in face_crub_path:
        bin_path = os.path.join(img_root, bin_path)
        # print(bin_path)
        bin_path = bin_path + "_" + tag +".bin"
        if(not os.path.isfile(bin_path)):
            print(bin_path)
            exit(0)
            continue
        feature = np.float32(np.fromfile(bin_path,dtype=np.float32))[4:]
        probe.append(feature/cv2.norm(feature))
    probe_len = len(probe)
    print("probe ready: %d"%(probe_len))
    
    now_id = face_crub_id[0]
    now_index = 0

    # face_crub_len = len(face_crub_id)
    acc=0 # 记录正确率
    compare_num = 0 # 记录比较次数

    # 矩阵数据类型转换（转为array，方便使用numpy进行矩阵计算）
    gallery_array = np.array(gallery)
    probe_array = np.array(probe)
    
    # 预先获得噪音和测试矩阵的余弦距离得分，使用矩阵计算进行加速，以下分别称为噪音矩阵和测试矩阵
    gallery_score = np.dot(gallery_array, probe_array.transpose())
    probe_score = np.dot(probe_array, probe_array.transpose())

    # 通过测试文件的id将测试矩阵分为同id下的矩阵块
    test_list = []# 用于记录同id文件的index（开始和结束index）
    for i in range(probe_len):
        if now_id!=face_crub_id[i] or i==(probe_len-1):
            if i==probe_len-1:
                end_index = i+1
            else:
                end_index = i
            test_list.append([now_index,end_index])
            now_index=i
            now_id=face_crub_id[i]
    
    # 对每个id进行测试
    for index_pair in test_list:
        now_index = index_pair[0]
        end_index = index_pair[1]
        print(now_index,end_index)
        for i in range(now_index, end_index):
            # 将挑选出来的矩阵列加入噪声矩阵之后
            score = np.array(np.concatenate((gallery_score[:,now_index:end_index],[probe_score[i,now_index:end_index]]),axis=0))
            # print(score.shape)
            # 将分数矩阵进行排序
            score_index_sorted = np.argsort(score, axis=0) # 100001*(id_num-1)
            gt = gallery_len

            expected_index = score_index_sorted[-rank_N:][:]
            # 记录正确识别的次数
            acc+=np.where(expected_index==gt)[0].shape[0]-1
            
            # 记录总共的比较次数
            compare_num +=end_index-1-now_index

    print("rank1: %f"%(acc/float(compare_num)))


def rank_test_pfe(img_list_file=None, noise_file=None, tag="pfe", rank_N=1,img_root="/home/ubuntu/Work/lk/test_data/facescrub_transform_lzj",noise_root="/home/ubuntu/Work/lk/test_data/megaface_transform_lzj/megaface"):
    '''
    使用pfe的测试代码（分数计算方式不一样）
    '''
    face_crub_file_path = "/home/ubuntu/Work/lk/test_data/megaface-devkit/templatelists/facescrub_uncropped_features_list.json"
    face_crub_file = open(face_crub_file_path, "r")
    face_crub_json = json.load(face_crub_file)
    face_crub_id = face_crub_json["id"]
    face_crub_path = face_crub_json["path"]

    megaface_file_path = "/home/ubuntu/Work/lk/test_data/megaface-devkit/templatelists/megaface_features_list.json_10000_1"
    megaface_file = open(megaface_file_path, "r")
    megaface_json = json.load(megaface_file)
    megaface_path = megaface_json["path"]
    ## 可以开多线程
    gallery = []

    for bin_path in megaface_path:
        bin_path = os.path.join(noise_root, bin_path)
        bin_path = bin_path + "_" + tag + ".bin"
        if(not os.path.isfile(bin_path)):
            print(bin_path)
            exit(0)
            continue
        feature = np.float32(np.fromfile(bin_path,dtype=np.float32))
        gallery.append(feature)
    gallery_len = len(gallery)
    
    print("gallery ready: %d"%(gallery_len))

    probe=[]
    for bin_path in face_crub_path:
        bin_path = os.path.join(img_root, bin_path)
        # print(bin_path)
        bin_path = bin_path + "_" + tag +".bin"
        if(not os.path.isfile(bin_path)):
            print(bin_path)
            exit(0)
            continue
        feature = np.float32(np.fromfile(bin_path,dtype=np.float32))
        probe.append(feature)
    probe_len = len(probe)
    print("probe ready: %d"%(probe_len))
    
    now_id = face_crub_id[0]
    now_index = 0

    # face_crub_len = len(face_crub_id)
    acc=0
    compare_num = 0

    gallery_array = np.array(gallery)
    probe_array = np.array(probe)
    import time
    # print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

    # gallery_score = np.dot(gallery_array, probe_array.transpose())
    # probe_score = np.dot(probe_array, probe_array.transpose())
    mu1,sigma_sq1=gallery_array[:,4:512+4],gallery_array[:,512+4:]
    mu2,sigma_sq2=probe_array[:,4:512+4],probe_array[:,512+4:]
    # print(sigma_sq1.shape)
    x1 = mu1.reshape(-1,1,512)
    # y1 = mu2.reshape(1,-1,512)
    sigma_sq_X1 = sigma_sq1.reshape(-1,1,512)
    # sigma_sq_Y1 = sigma_sq2.reshape(1,-1,512)

    # sigma_sq_fuse1 = sigma_sq_X1 + sigma_sq_Y1

    # print(time.strftime('%Y-%m-%d %H:%M',time.localtime(time.time())))
    # print(sigma_sq_fuse1.shape)
    # exit(0)
    
    # gallery_score = np.sum((x1-y1)**2/(1e-10 + sigma_sq_fuse1)+np.log(sigma_sq_fuse1), axis=2)
    # print(time.strftime('%Y-%m-%d %H:%M',time.localtime(time.time())))
    # print(gallery_score.shape)
    # x2 = mu2.reshape(-1,1,512)
    # sigma_sq_X2 = sigma_sq2.reshape(-1,1,512)

    # sigma_sq_fuse2 = sigma_sq_X2 + sigma_sq_Y1

    # probe_score = np.sum((x2-y1)**2/(1e-10 + sigma_sq_fuse2)+np.log(sigma_sq_fuse2), axis=2)
    # probe_score = np.sum((x2-y1)**2/(1e-10 + sigma_sq_fuse2)+np.log(sigma_sq_fuse2), axis=2)
    # print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    # print(probe_score.shape)

    test_list = []
    for i in range(probe_len):
        if now_id!=face_crub_id[i] or i==(probe_len-1):
            if i==probe_len-1:
                end_index = i+1
            else:
                end_index = i
            test_list.append([now_index,end_index])
            now_index=i
            now_id=face_crub_id[i]
    
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

    for index_pair in test_list:

        now_index = index_pair[0]
        end_index = index_pair[1]

        # print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
        print(now_index,end_index)

        y2 = mu2[now_index:end_index,:].reshape(1,-1,512)
        sigma_sq_Y2=sigma_sq2[now_index:end_index,:].reshape(1,-1,512)
        sigma_sq_fuse1 = sigma_sq_X1 + sigma_sq_Y2
        gallery_score = np.sum((x1-y2)**2/(1e-10 + sigma_sq_fuse1)+np.log(sigma_sq_fuse1), axis=2)

        # print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
        # print(gallery_score.shape)
        
        for i in range(now_index, end_index):

            x2 = mu2[i].reshape(-1,1,512)
            sigma_sq_X2 = sigma_sq2[i].reshape(-1,1,512)
            sigma_sq_fuse2 = sigma_sq_X2 + sigma_sq_Y2
            probe_score = np.sum((x2-y2)**2/(1e-10 + sigma_sq_fuse2)+np.log(sigma_sq_fuse2), axis=2)
            
            # print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
            # print(probe_score.shape)
            # print(probe_score)
            
            score = np.array(np.concatenate((gallery_score,probe_score),axis=0))
            # print(score.shape)
            score_index_sorted = np.argsort(score, axis=0) # 100001*(id_num-1)
            gt = gallery_len

            expected_index = score_index_sorted[:rank_N][:]
            # print(expected_index)
            # print(score[expected_index])

            acc+=np.where(expected_index==gt)[0].shape[0]-1

            # gallery.pop()
            compare_num +=end_index-1-now_index
            # print(acc, compare_num)
            # print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
            # print("acc/compare= %f"%(acc/float(compare_num)))
            # exit(0)


    print("rank1: %f"%(acc/float(compare_num)))
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))


import queue
import threading
import time

exitFlag = 0


class Recorder():
    def __init__(self):
        self.index_range = []
        self.record = [ [] ]
    def append_index(self, index_range):
        self.index_range.append(index_range)
        self.record.append([])
    def append_record(self, index_list, score_list):
        self.record[-1].append( [index_list, score_list] )


def to_count_dict(lst):
    dic={}
    for i in lst:
        if i not in dic:
            dic[i] = 1
        else:
            dic[i] += 1
    return dic


class myThread (threading.Thread):
    def __init__(self, probe_score, gallery_score,q,lock, rank_N=1,exitFlag=0):
        threading.Thread.__init__(self)
        self.probe_score = probe_score
        self.gallery_score = gallery_score
        self.q = q
        self.rank_N = rank_N
        self.acc = 0
        self.compare_num = 0
        self.lock = lock
        self.exitFlag = exitFlag

        # self.recorder = Recorder()
    
    def run(self):
        self.process_data()

    def process_data(self):
        queueLock = self.lock
        workQueue = self.q
        q = self.q
        gallery_score = self.gallery_score
        probe_score = self.probe_score
        rank_N = self.rank_N
        while not self.exitFlag:
            queueLock.acquire()
            if not workQueue.empty():
                index_pair = q.get()

                queueLock.release()
                now_index = index_pair[0]
                end_index = index_pair[1]
                gallery_len = gallery_score[0].shape[0]
                print(now_index,end_index)

                # self.recorder.append_index(index_pair)
                for i in range(now_index, end_index):
                    expected_index_list = []
                    expected_score_list= []
                    for j in range(len(gallery_score)):
                        score = np.array(np.concatenate((gallery_score[j][:,now_index:end_index],[probe_score[j][i,now_index:end_index]]),axis=0))
                        # print(score.shape)
                        # print(score)
                        score_index_sorted = np.argsort(score, axis=0) # 10001*(id_num-1)
                        # print(score_index_sorted.shape)
                        score_sorted = np.sort(score, axis=0)
                        # score_sorted = score[score_index_sorted]
                        # print(score_sorted.shape)
                        
                        gt = gallery_len

                        expected_index = score_index_sorted[-(rank_N*1):][:]
                        expected_score = score_sorted[-(rank_N*1):][:]
                        
                        expected_index_list.append(expected_index)
                        expected_score_list.append(expected_score)
                # print(expected_index_list)
                    expected_index_list=np.stack(expected_index_list, axis=2).squeeze(axis=0).transpose()# (rank_N*N)*(id_num-1)
                    expected_score_list=np.stack(expected_score_list, axis=2).squeeze(axis=0).transpose()# (rank_N*N)*(id_num-1)
                    # print(expected_index_list.shape, expected_score_list.shape)
                    # exit(0)
                    # expected_index_list = expected_index_list.astype(np.int64)
                    w,h =  expected_index_list.shape
                    expected_index_count_list = []
                    for id in range(h):
                        count_dict = to_count_dict(expected_index_list[:,id])
                        count_dict_sorted = sorted(count_dict.items(), key=lambda x: x[1])
                        expected_index_count = [int(x[0]) for x in count_dict_sorted if x[1]>1] # ensure voting
                        if len(expected_index_count)<rank_N:
                            score_index_sorted = np.argsort(-expected_score_list[:,id], axis=0)
                            for index in expected_index_list[:,id][score_index_sorted]:
                                if index not in expected_index_count:
                                    expected_index_count.append(index)
                                    if len(expected_index_count)==rank_N:
                                        break
                        expected_index_count_list.append(expected_index_count)
                    expected_index_list_final = np.stack(expected_index_count_list, axis=1)
                    # print(expected_index_list_final.shape)
                    # queueLock.acquire()
                    self.acc+=np.where(expected_index_list_final==gt)[0].shape[0]-1
                    self.compare_num +=end_index-1-now_index
                    # queueLock.release()
                    

                # index_top5 = score_index_sorted[-5:][:]
                # self.recorder.append_record(index_top5, np.take_along_axis(score, index_top5,axis=0))
                
            else:
                queueLock.release()
                # time.sleep(1)
    
    def exit(self):
        self.exitFlag=1


def feature_prun(feature, mode, separate=False):
    '''
    out: [array1,...]
    '''
    if separate:
        length = feature.shape[0]
        feature_list = []
        for i in range((length-1)//256):
            feature_list.append(feature[4+i*256:4+(i+1)*256])
        return feature_list
    if mode==0:
        feature = feature[4:]
    elif mode==1:
        feature = feature[4:256+4]
    elif mode==2:
        feature = feature[256+4:]
    elif mode==3:
        feature = feature[4:256+4] + feature[256+4:]
    elif mode==4:
        feature = feature[512+4:]
    return [feature]


def read_feature(path, tag):
    bin_path = path + "_" + tag +".bin"
    if(not os.path.isfile(bin_path)):
        print(bin_path)
        return None
    feature = np.float32(np.fromfile(bin_path,dtype=np.float32))
    return feature


class Rank_Test():
    def __init__(self, img_list_file=None, noise_file=None, tag="pytorch",
    img_root="/home/ubuntu/Work/lk/test_data/facescrub_transform_lzj",
    noise_root="/home/ubuntu/Work/lk/test_data/megaface_transform_lzj/megaface",
    face_crub_file_path=None, megaface_file_path=None, mode=0, separate=False):
        if not face_crub_file_path:
            self.face_crub_file_path = "/home/ubuntu/Work/lk/test_data/megaface-devkit/templatelists/facescrub_uncropped_features_list.json"
            # face_crub_file_path = "/home/ubuntu/Work/lk/test_data/megaface-devkit/templatelists/facescrubv1_no_noisy_combined.json"
        else:
            self.face_crub_file_path = face_crub_file_path
        
        if not megaface_file_path:
            self.megaface_file_path = "/home/ubuntu/Work/lk/test_data/megaface-devkit/templatelists/megaface_features_list_edit.json_10000_1"
        else:
            self.megaface_file_path = megaface_file_path
        
        self.face_crub_id, self.face_crub_path = self.parse_face_crub()
        self.megaface_path = self.parse_megaface()
        
        self.gallery = self.gen_feature_lsit(noise_root[0], self.megaface_path, tag=tag, mode=mode, separate=separate)
        self.gallery_len = self.gallery[0].shape[0]
        print("gallery ready: %d"%(self.gallery_len))
        
        self.probe = self.gen_feature_lsit(img_root[0], self.face_crub_path, tag=tag, mode=mode, separate=separate)
        self.probe_len = self.probe[0].shape[0]
        print("probe ready: %d"%(self.probe_len))
        
        self.gallery_mask = None
        self.probe_mask = None
        if len(noise_root)>1:
            self.gallery_mask = self.gen_feature_lsit(noise_root[1], self.megaface_path, tag=tag, mode=mode, separate=separate)
            self.probe_mask = self.gen_feature_lsit(img_root[1], self.face_crub_path, tag=tag, mode=mode, separate=separate)
        
        self.gen_score_matrix()
        self.gen_test_list()
            
        
    def parse_face_crub(self):
        face_crub_file = open(self.face_crub_file_path, "r")
        face_crub_json = json.load(face_crub_file)
        face_crub_id = face_crub_json["id"]
        face_crub_path = face_crub_json["path"]
        return face_crub_id, face_crub_path

    def parse_megaface(self):
        megaface_file = open(self.megaface_file_path, "r")
        megaface_json = json.load(megaface_file)
        megaface_path = megaface_json["path"]
        return megaface_path
    
    def gen_feature_lsit(self, root_path, file_list_path, tag, mode, separate=False):
        '''
        out: @feature_list: [array1, array2]
        '''
        feature_list = [[]]
        for bin_path in file_list_path:
            bin_path = os.path.join(root_path, bin_path)
            feature = read_feature(bin_path, tag=tag)
            if feature is None:
                continue
            feature = feature_prun(feature, mode, separate=separate)
            if len(feature_list)<len(feature):
                feature_list = [ [] for x in feature]
            for i in range(len(feature)):
                feature_list[i].append(feature[i]/cv2.norm(feature[i]))
        for i in range(len(feature_list)): # to array
            feature_list[i] = np.array(feature_list[i])
        return feature_list

    def gen_score_matrix(self):
        
        self.gallery_nm_score_list = []
        self.gallery_mm_score_list = []
        self.gallery_full_score_list = []
        self.probe_nm_score_list = []
        self.probe_mm_score_list = []
        self.probe_full_score_list = []
        self.gallery_score_list = []
        self.probe_score_list = []
        
        if self.gallery_mask is not None:
            for i in range(len(self.probe)):
                self.probe_nm_score_list.append(np.dot(self.probe[i], self.probe_mask[i].transpose()))
                self.probe_mm_score_list.append(np.dot(self.probe_mask[i], self.probe_mask[i].transpose()))

                self.gallery_nm_score_list.append(np.dot(self.gallery[i], self.probe_mask[i].transpose()))
                self.gallery_mm_score_list.append(np.dot(self.gallery_mask[i], self.probe_mask[i].transpose()))

                self.gallery_full_score_list.append((self.gallery_nm_score_list[i]+self.gallery_mm_score_list[i])/2)
                self.probe_full_score_list.append((self.probe_nm_score_list[i]+self.probe_mm_score_list[i])/2)
        
        else:
            for i in range(len(self.probe)):
                self.probe_score_list.append(np.dot(self.probe[i], self.probe[i].transpose()))
                self.gallery_score_list.append(np.dot(self.gallery[i], self.probe[i].transpose()))

    def gen_test_list(self):
        now_id = self.face_crub_id[0]
        now_index = 0
        self.test_list = []
        for i in range(self.probe_len):
            if now_id!=self.face_crub_id[i] or i==(self.probe_len-1):
                if i==self.probe_len-1:
                    end_index = i+1
                else:
                    end_index = i
                self.test_list.append([now_index,end_index])
                now_index=i
                now_id=self.face_crub_id[i]


def multi_thread_fun(rank_test, rank_N):
    acc=0
    compare_num = 0
    queueLock = threading.Lock()
    workQueue = queue.Queue(len(rank_test.test_list))
    threads = []

    thread_num = 12

    # 创建新线程
    for i in range(thread_num):
        if len(rank_test.probe_nm_score_list)==0:
            thread = myThread(rank_test.probe_score_list, rank_test.gallery_score_list, workQueue, queueLock, rank_N=rank_N)
        else:
            # thread = myThread(rank_test.probe_nm_score_list, rank_test.gallery_nm_score_list, workQueue, queueLock, rank_N=rank_N)# 口罩对正常
            # thread = myThread(probe_mm_score,gallery_mm_score, workQueue, queueLock, rank_N=rank_N) # 口罩对口罩（gallery）
            thread = myThread(rank_test.probe_full_score_list,rank_test.gallery_full_score_list, workQueue, queueLock, rank_N=rank_N)

        thread.start()
        threads.append(thread)

    # 填充队列
    queueLock.acquire()
    for index_pair in rank_test.test_list:
        # print(index_pair)
        workQueue.put(index_pair)
    queueLock.release()

    # 等待队列清空
    while not workQueue.empty():
        pass

    # 通知线程是时候退出
    for t in threads:
        t.exit()

    # 等待所有线程完成
    for t in threads:
        t.join()
        acc+=t.acc
        compare_num += t.compare_num
    return acc, compare_num

def rank_test_MPI(img_list_file=None, noise_file=None, tag="pytorch", rank_N=1,
    img_root="/home/ubuntu/Work/lk/test_data/facescrub_transform_lzj",
    noise_root="/home/ubuntu/Work/lk/test_data/megaface_transform_lzj/megaface",
    face_crub_file_path=None, megaface_file_path=None, mode=0):
    '''
    多线程版本，队列
    '''
    if not face_crub_file_path:
        face_crub_file_path = "/home/ubuntu/Work/lk/test_data/megaface-devkit/templatelists/facescrub_uncropped_features_list.json"
        # face_crub_file_path = "/home/ubuntu/Work/lk/test_data/megaface-devkit/templatelists/facescrubv1_no_noisy_combined.json"
    else:
        face_crub_file_path = face_crub_file_path
    face_crub_file = open(face_crub_file_path, "r")
    face_crub_json = json.load(face_crub_file)
    face_crub_id = face_crub_json["id"]
    face_crub_path = face_crub_json["path"]
    
    if not megaface_file_path:
        # megaface_file_path = "/home/ubuntu/Work/lk/test_data/megaface-devkit/templatelists/megaface_features_list.json_10000_1"
        megaface_file_path = "/home/ubuntu/Work/lk/test_data/megaface-devkit/templatelists/megaface_features_list_edit.json_10000_1"
    else:
        megaface_file_path = megaface_file_path
    megaface_file = open(megaface_file_path, "r")
    megaface_json = json.load(megaface_file)
    megaface_path = megaface_json["path"]

    ## 可以开多线程
    gallery = []

    for bin_path in megaface_path:
        bin_path = os.path.join(noise_root, bin_path)
        feature = read_feature(bin_path)
        if feature is None:
            continue
        feature = feature_prun(feature, mode)
        gallery.append(feature/cv2.norm(feature))
    
    gallery_len = len(gallery)
    
    print("gallery ready: %d"%(gallery_len))

    probe=[]
    for bin_path in face_crub_path:
        bin_path = os.path.join(img_root, bin_path)
        # print(bin_path)
        feature = read_feature(bin_path)
        if feature is None:
            continue
        feature = feature_prun(feature, mode)
        probe.append(feature/cv2.norm(feature))
    probe_len = len(probe)
    print("probe ready: %d"%(probe_len))
    
    now_id = face_crub_id[0]
    now_index = 0

    # face_crub_len = len(face_crub_id)
    acc=0
    compare_num = 0

    gallery_array = np.array(gallery)
    probe_array = np.array(probe)
    # print(gallery_array, probe_array)
    
    gallery_score = np.dot(gallery_array, probe_array.transpose())
    probe_score = np.dot(probe_array, probe_array.transpose())

    test_list = []
    for i in range(probe_len):
        if now_id!=face_crub_id[i] or i==(probe_len-1):
            if i==probe_len-1:
                end_index = i+1
            else:
                end_index = i
            test_list.append([now_index,end_index])
            now_index=i
            now_id=face_crub_id[i]

    queueLock = threading.Lock()
    workQueue = queue.Queue(len(test_list))
    threads = []

    thread_num = 12

    # 创建新线程
    for i in range(thread_num):
        thread = myThread([probe_score],[gallery_score], workQueue, queueLock, rank_N=rank_N)
        thread.start()
        threads.append(thread)

    # 填充队列
    queueLock.acquire()
    for index_pair in test_list:
        # print(index_pair)
        workQueue.put(index_pair)
    queueLock.release()

    # 等待队列清空
    while not workQueue.empty():
        pass

    # 通知线程是时候退出
    for t in threads:
        t.exit()

    # 等待所有线程完成
    for t in threads:
        t.join()
        acc+=t.acc
        compare_num += t.compare_num
        # print(acc, compare_num)
    print("tag: %s, rank %d: %f"%(tag, rank_N, acc/float(compare_num)))


def rank_test_MPI_mask(img_list_file=None, noise_file=None, tag="pytorch", rank_N=1,
    img_root="/home/ubuntu/Work/lk/test_data/facescrub_transform_lzj",
    noise_root="/home/ubuntu/Work/lk/test_data/megaface_transform_lzj/megaface",
    face_crub_file_path=None, megaface_file_path=None, mode=0):
    '''
    多线程版本，队列
    '''
    if not face_crub_file_path:
        face_crub_file_path = "/home/ubuntu/Work/lk/test_data/megaface-devkit/templatelists/facescrub_uncropped_features_list.json"
        # face_crub_file_path = "/home/ubuntu/Work/lk/test_data/megaface-devkit/templatelists/facescrubv1_no_noisy_combined.json"
    else:
        face_crub_file_path = face_crub_file_path
    face_crub_file = open(face_crub_file_path, "r")
    face_crub_json = json.load(face_crub_file)
    face_crub_id = face_crub_json["id"]
    face_crub_path = face_crub_json["path"]
    
    if not megaface_file_path:
        megaface_file_path = "/home/ubuntu/Work/lk/test_data/megaface-devkit/templatelists/megaface_features_list_edit.json_10000_1"
    else:
        megaface_file_path = megaface_file_path
    megaface_file = open(megaface_file_path, "r")
    megaface_json = json.load(megaface_file)
    megaface_path = megaface_json["path"]

    '''
    准备特征值矩阵
    '''
    ## 可以开多线程
    gallery = []

    for bin_path in megaface_path:
        bin_path = os.path.join(noise_root[0], bin_path)
        feature = read_feature(bin_path)
        if feature is None:
            continue
        feature = feature_prun(feature, mode)
        gallery.append(feature/cv2.norm(feature))
    
    gallery_mask = []

    for bin_path in megaface_path:
        bin_path = os.path.join(noise_root[1], bin_path)
        feature = read_feature(bin_path)
        if feature is None:
            continue
        feature = feature_prun(feature, mode)
        gallery_mask.append(feature/cv2.norm(feature))
    
    gallery_len = len(gallery)
    print("gallery ready: %d"%(gallery_len))

    probe=[]
    for bin_path in face_crub_path:
        bin_path = os.path.join(img_root[0], bin_path)
        feature = read_feature(bin_path)
        if feature is None:
            continue
        feature = feature_prun(feature, mode)
        probe.append(feature/cv2.norm(feature))
    
    probe_mask=[]
    for bin_path in face_crub_path:
        bin_path = os.path.join(img_root[1], bin_path)
        feature = read_feature(bin_path)
        if feature is None:
            continue
        feature = feature_prun(feature, mode)
        probe_mask.append(feature/cv2.norm(feature))
    
    probe_len = len(probe)
    print("probe ready: %d"%(probe_len))
    
    now_id = face_crub_id[0]
    now_index = 0

    # face_crub_len = len(face_crub_id)
    acc=0
    compare_num = 0

    gallery_array = np.array(gallery)
    probe_array = np.array(probe)
    gallery_mask_array = np.array(gallery_mask)
    probe_mask_array = np.array(probe_mask)
    # print(gallery_array, probe_array)
    
    gallery_nm_score = np.dot(gallery_array, probe_mask_array.transpose())
    gallery_mm_score = np.dot(gallery_mask_array, probe_mask_array.transpose())
    gallery_full_score = (gallery_nm_score+gallery_mm_score)/2
    # gallery_full_score = gallery_nm_score
    # probe_score = np.dot(probe_array, probe_array.transpose())
    probe_nm_score = np.dot(probe_array, probe_mask_array.transpose())
    probe_mm_score = np.dot(probe_mask_array, probe_mask_array.transpose())
    probe_full_score = (probe_nm_score+probe_mm_score)/2


    test_list = []
    for i in range(probe_len):
        if now_id!=face_crub_id[i] or i==(probe_len-1):
            if i==probe_len-1:
                end_index = i+1
            else:
                end_index = i
            test_list.append([now_index,end_index])
            now_index=i
            now_id=face_crub_id[i]

    queueLock = threading.Lock()
    workQueue = queue.Queue(len(test_list))
    threads = []

    thread_num = 12

    # 创建新线程
    for i in range(thread_num):
        # thread = myThread(probe_full_score,gallery_full_score, workQueue, queueLock, rank_N=rank_N)
        thread = myThread([probe_nm_score],[gallery_nm_score], workQueue, queueLock, rank_N=rank_N)# 口罩对正常
        # thread = myThread(probe_mm_score,gallery_mm_score, workQueue, queueLock, rank_N=rank_N) # 口罩对口罩（gallery）
        thread.start()
        threads.append(thread)

    # 填充队列
    queueLock.acquire()
    for index_pair in test_list:
        # print(index_pair)
        workQueue.put(index_pair)
    queueLock.release()

    # 等待队列清空
    while not workQueue.empty():
        pass

    # 通知线程是时候退出
    for t in threads:
        t.exit()

    # 等待所有线程完成
    for t in threads:
        t.join()
        acc+=t.acc
        compare_num += t.compare_num
        # print(acc, compare_num)
    print("rank1: %f"%(acc/float(compare_num)))


def rank_test_MPI_final(img_list_file=None, noise_file=None, tag="pytorch", rank_N=1,
    img_root="/home/ubuntu/Work/lk/test_data/facescrub_transform_lzj",
    noise_root="/home/ubuntu/Work/lk/test_data/megaface_transform_lzj/megaface",
    face_crub_file_path=None, megaface_file_path=None, mode=0, separate=False):
    
    RT = Rank_Test(img_list_file=img_list_file, noise_file=noise_file, tag=tag,
    img_root=img_root,noise_root=noise_root,
    face_crub_file_path=face_crub_file_path, megaface_file_path=megaface_file_path, mode=mode, separate=separate)
    
    acc, compare_num = multi_thread_fun(RT, rank_N)

    # print(acc, compare_num)
    print("rank1: %f"%(acc/float(compare_num)))

import multiprocessing
def rank_compare_OMP(probe,gallery,now_index, end_index,rank_N=1):
	# 多进程版本
    acc=0
    gallery_len=len(gallery)
    compare_num=0
    # print("start OMP (%d,%d)"%(now_index,end_index))
    for j in range(now_index,end_index):
        if j==now_index:
            probe_array = np.array(probe[now_index+1:end_index])
            # gallery.append(probe[0])
            gallery_array = np.array(np.concatenate((gallery,[probe[now_index]]),axis=0))
        elif j==end_index-1:
            probe_array = np.array(probe[now_index:end_index-1])
            # gallery.append(probe[temp_len])
            gallery_array = np.array(np.concatenate((gallery,[probe[end_index-1]]),axis=0))
        else:
            probe_array = np.array(np.concatenate((probe[now_index:j],probe[j+1:end_index]),axis=0))
            # gallery.append(probe[j])
            gallery_array = np.array(np.concatenate((gallery,[probe[j]]),axis=0))
        # print(probe_array.shape)
        # print(j,probe_array.size)
        score=[]
        score = np.dot(gallery_array, probe_array.transpose())

        score_index_sorted = np.argsort(score, axis=0) # 100001*(id_num-1)
        gt = gallery_len

        expected_index = score_index_sorted[-rank_N:][:]

        acc+=np.where(expected_index==gt)[0].shape[0]
        
        # gallery.pop()
        compare_num +=end_index-1-now_index
    # print(acc/float(compare_num))

    # print(now_index,end_index,acc)
    return acc
    # q.put(acc,block=False)

# pytorch
def rank_test_omp(img_list_file=None, noise_file=None, tag="pytorch_repeat", rank_N=1,
    img_root="/home/ubuntu/Work/lk/test_data/facescrub_transform_lzj_repeat",
    noise_root="/home/ubuntu/Work/lk/test_data/megaface_transform_lzj_repeat/megaface",
    face_crub_file_path=None, megaface_file_path=None):
    #多进程版本，进程池
    if not face_crub_file_path:
        face_crub_file_path = "/home/ubuntu/Work/lk/test_data/megaface-devkit/templatelists/facescrub_uncropped_features_list.json"
    else:
        face_crub_file_path = face_crub_file_path
    face_crub_file = open(face_crub_file_path, "r")
    face_crub_json = json.load(face_crub_file)
    face_crub_id = face_crub_json["id"]
    face_crub_path = face_crub_json["path"]
    
    if not megaface_file_path:
        megaface_file_path = "/home/ubuntu/Work/lk/test_data/megaface-devkit/templatelists/megaface_features_list.json_10000_1"
    else:
        megaface_file_path = megaface_file_path
    megaface_file = open(megaface_file_path, "r")
    megaface_json = json.load(megaface_file)
    megaface_path = megaface_json["path"]
    ## 可以开多线程
    gallery = []

    for bin_path in megaface_path:
        bin_path = os.path.join(noise_root, bin_path)
        bin_path = bin_path + "_" + tag + ".bin"
        if(not os.path.isfile(bin_path)):
            print(bin_path)
            exit(0)
            continue
        feature = np.float32(np.fromfile(bin_path,dtype=np.float32))
        # print(feature)
        gallery.append(feature/cv2.norm(feature))
    gallery_len = len(gallery)
    
    print("gallery ready: %d"%(gallery_len))

    probe=[]
    for bin_path in face_crub_path:
        bin_path = os.path.join(img_root, bin_path)
        # print(bin_path)
        bin_path = bin_path + "_" + tag +".bin"
        if(not os.path.isfile(bin_path)):
            print(bin_path)
            exit(0)
            continue
        feature = np.float32(np.fromfile(bin_path,dtype=np.float32))
        probe.append(feature/cv2.norm(feature))
    probe_len = len(probe)
    print("probe ready: %d"%(probe_len))

    now_id = face_crub_id[0]
    now_index = 0

    # face_crub_len = len(face_crub_id)

    # q = multiprocessing.Queue()
    acc=[]
    pool = multiprocessing.Pool(processes = 12)
    compare_num=0
    ## 修改为一次性输入，而不是重新创建
    ## 输出结果不需要保持原有的顺序
    for i in range(probe_len):
        if now_id!=face_crub_id[i] or i==(probe_len-1):
            if i==probe_len-1:
                end_index = i+1
            else:
                end_index = i
            acc.append(pool.apply_async(rank_compare_OMP, (probe,gallery,now_index, end_index,)))
            # print(now_index,acc)
            # print(now_index,end_index)
            compare_num+=(end_index-1-now_index)*(end_index-now_index)
            # print(compare_num)
            now_index=i
            now_id=face_crub_id[i]
        # if (i)%200 ==0:
        #     print("processing %d/%d"%(i,probe_len))

    pool.close()
    pool.join()
    # print(acc)
    # P = np.sum(acc)
    P=0
    # print(compare_num)
    for ac in acc:
        P+=ac.get()
        # print(P)
    print("rank1: %f"%(P/float(compare_num)))


def pair_test(test_file_path, model, best=True, threshold=None,save_file_path=None,criterion=cos_distance):
    test_file = open(test_file_path,"r")
    test_lines = test_file.readlines()
    score_matrix = []
    gt_matrix = []

    transform = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
    ])
    count = 0
    ## 可以开多线程
    for test_line in test_lines:
        # print(count)
        fileA,fileB,gt = test_line.strip().split()
        fileA = os.path.join('/home/ubuntu/data2/lk/amsoft_pytorch/amsoft_pytorch/data/pairtest/zhuhai_0322_v04/aligned', fileA)
        # fileA = os.path.join('/home/ubuntu/data2/lk/amsoft_pytorch/amsoft_pytorch/data/pairtest/lfw_transform_lzj/images', fileA)
        fileB = os.path.join('/home/ubuntu/data2/lk/amsoft_pytorch/amsoft_pytorch/data/pairtest/zhuhai_0322_v04/aligned', fileB)
        # fileB = os.path.join('/home/ubuntu/data2/lk/amsoft_pytorch/amsoft_pytorch/data/pairtest/lfw_transform_lzj/images', fileB)
        im1 = Image.open(fileA).convert('RGB')
        im2 = Image.open(fileB).convert('RGB')
        
        # f1,cf1= extractDeepFeature_PFE(im1,model,transform=transform)
        f1= extractDeepFeature(im1,model,transform=transform)
        # f2,cf2 = extractDeepFeature(im2,model,transform=transform)
        f2= extractDeepFeature(im2,model,transform=transform)
        
        f1 = f1.to("cpu")
        f2 = f2.to("cpu")
        score = criterion(f1,f2)
        
        score_matrix.append(score.numpy())
        gt_matrix.append(int(gt))

        # count+=1
        # print(score_matrix)
        # print(gt_matrix)
    
    score_matrix = np.array(score_matrix)
    gt_matrix = np.array(gt_matrix)

    TP=0
    FP=0
    # print(np.where(gt_matrix>=1)) (array()，)
    total_P = (np.where(gt_matrix>=1))[0].shape[0]
    # print(total_P)
    total_F = (gt_matrix).shape[0]-total_P
    if not best:
        assert threshold is not None
        TP,FP = get_accuracy(score_matrix,gt_matrix,threshold)
        TN = total_F - FP
        print("TP: %d, FP: %d, TP/total_P: %f, TN/total_F: %F"%(TP, FP, TP/float(total_P), TN/float(total_F)))
        print("accuracy:%f"%((TN+TP)/float((gt_matrix).shape[0])))
    else:
        thresholds = score_matrix[np.argsort(score_matrix)]
        acc = []
    ## 可以开多线程
        for threshold in thresholds:
            TP,FP = get_accuracy(score_matrix,gt_matrix,threshold)
            TN = total_F - FP
            print(threshold,TP,FP,TN)
            accuracy = (TN+TP)/float((gt_matrix).shape[0])
            print(threshold, accuracy)
            acc.append(accuracy)
        best_index = np.argsort(acc)
        print("best acc:%f, threshold: %f"%(acc[best_index[-1]], thresholds[best_index[-1]]))
    
# def pair_test_pre()

if __name__ == '__main__':
    # import sys
    # sys.path.append('../')
    # import timer
    # face_path = "/home/ubuntu/Work/lk/test_data/facescrub_transform_lzj_repeat"
    # face_path = "/home/ubuntu/Work/lk/test_data/facescrub_Meng_arcface"
    # face_path = "/home/ubuntu/Work/lk/test_data/facescrub_Meng_lzj"
    face_path = "/home/ubuntu/data3/lk/test_data/facescrub_Meng_std"
    
    # face_path_mask = "/home/ubuntu/Work/lk/test_data/facescrub_mask_lzj"
    face_path_mask = "/home/ubuntu/data3/lk/test_data/facescrub_mask_std"
    # face_path = "/home/ubuntu/data2/lk/amsoft_pytorch/amsoft_pytorch/data/weibo_large/new_diff_nodiff/aligned"
    # face_path = "/home/ubuntu/data2/lk/amsoft_pytorch/amsoft_pytorch/data/baidu_hd/new_diff_nodiff/aligned"
    
    # megaface_path = "/home/ubuntu/Work/lk/test_data/megaface_transform_lzj_repeat/megaface"
    megaface_path = "/home/ubuntu/data3/lk/test_data/megaface_mtcnn_transform_std"
    # megaface_path = "/home/ubuntu/Work/lk/test_data/megaface_mtcnn_transform_arcface"
    # megaface_path = "/home/ubuntu/Work/lk/test_data/megaface_transform_arcface"
    # megaface_path_mask = "/home/ubuntu/Work/lk/test_data/megaface_mask_lzj"
    megaface_path_mask = "/home/ubuntu/data3/lk/test_data/megaface_mask_mtcnn_transform_std"
    
    tag = "FocusFace"
    # tag = "FFR"
    # tag = "am_36_20"
    # tag = "res_50_ddp_20"
    mode = 0 # mode=0:full; 1:up, 2:down
    separate=False

    print(tag, mode)
    rank_test_MPI_final(tag=tag, img_root=[face_path], noise_root=[megaface_path], rank_N=1, mode=mode, separate=separate)
    rank_test_MPI_final(tag=tag, img_root=[face_path, face_path_mask], noise_root=[megaface_path, megaface_path_mask], rank_N=1, mode=mode, separate=separate)
