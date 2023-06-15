import numpy as np
import os
# import json
import sys
import cv2
sys.path.append("/home/ubuntu/data2/lk/recognition/pytorch_new/")
# from util.metrics import ROC_by_mat, ROC
from util.metrics_cuda import ROC_by_mat, ROC

# normal_list = "/home/ubuntu/data1/lk/lfw_transform.txt" # TODO
# normal_list = "/home/ubuntu/data1/lk/lfw_mask_transform.txt"
normal_list = "/home/ubuntu/data3/data/lfw_112_align_v4_masked_new.txt"
# normal_list = "/home/ubuntu/data3/data/lfw_112_align_v4_masked_new_mask.txt"
# normal_list = "/home/ubuntu/data1/lk/lfw_transform_std.txt"
# normal_list = "/home/ubuntu/data3/data/spider/baidu/image_hd_filter/img_list_0714.txt"
# normal_list = "/home/ubuntu/data3/data/spider/baidu/image_hd_filter/img_list_mask_0714.txt"
# mask_list = "/home/ubuntu/data1/lk/lfw_mask_transform.txt"
# mask_list = "/home/ubuntu/data1/lk/lfw_transform.txt"
# mask_list = "/home/ubuntu/data3/data/lfw_112_align_v4_masked_new.txt"
mask_list = "/home/ubuntu/data3/data/lfw_112_align_v4_masked_new_mask.txt"
# mask_list = "/home/ubuntu/data1/lk/lfw_transform_std.txt"
# mask_list = "/home/ubuntu/data1/lk/lfw_mask_transform_std.txt"
# mask_list = "/home/ubuntu/data3/data/spider/baidu/image_hd_filter/img_list_mask_0714.txt"
# mask_list = "/home/ubuntu/data3/data/spider/baidu/image_hd_filter/img_list_0714.txt"
# tag = "SResnet_36_split_noDrop_noLA_b3_init_repeat_20" #  am_36_20 # TODO
# tag = "PCM_AM" #  am_36_20
# tag = "FFR" #  am_36_20
tag = "FocusFace" #  am_36_20
# tag = "am_36_ddp_mask_20" #  am_36_20
# tag = "res_arc_50_20" #  am_36_20

abs_flag = "abs" # None "abs"
fold = None # None 1
print(tag,fold, abs_flag, normal_list, mask_list)

# 存图片，txt，直接输出auc table
normal_file = open(normal_list, "r")
mask_file = open(mask_list, "r")

base_file_path = "/home/ubuntu/data1/lk/lfw_mask"
result_base_file_path = "/home/ubuntu/data2/lk/recognition/pytorch_new/scripts/lfw_result"
result_base_file_path = os.path.join(result_base_file_path, tag+str(fold)+"alignV4") # TODO
if not os.path.exists(result_base_file_path):
    os.makedirs(result_base_file_path)

if abs_flag is not None:
    result_file_path = result_base_file_path+"/%d_%s_nm.txt"%(0,abs_flag) # TODO nn nm
else:
    result_file_path = result_base_file_path+"/%d_nm.txt"%0
if not os.path.exists(result_file_path):
    normal_feature_list = []
    normal_id = []
    for pic_path in normal_file.readlines():
        pic_path = pic_path.strip()
        if not os.path.isfile(pic_path):
            print("no exists", pic_path)
            continue
        id = pic_path.split("/")[-2]
        normal_id.append(id)
        feature_path = pic_path+ "_" + tag + ".bin"
        if(not os.path.isfile(feature_path)):
            print(feature_path)
            exit(0)
            continue
        feature = np.float32(np.fromfile(feature_path,dtype=np.float32)) # TODO
        normal_feature_list.append(feature)
    normal_len = len(normal_feature_list)
    print("normal: ", normal_len)

    mask_feature_list = []
    mask_id = []
    for pic_path in mask_file.readlines():
        pic_path = pic_path.strip()
        if not os.path.isfile(pic_path):
            print("no exists", pic_path)
            continue
        id = pic_path.split("/")[-2]
        mask_id.append(id)
        feature_path = pic_path+ "_" + tag + ".bin"
        if(not os.path.isfile(feature_path)):
            print(feature_path)
            exit(0)
            continue
        feature = np.float32(np.fromfile(feature_path,dtype=np.float32))
        mask_feature_list.append(feature)
    mask_len = len(mask_feature_list)
    print("mask: ", mask_len)

    mask_feature_array = np.array(mask_feature_list)
    normal_feature_array = np.array(normal_feature_list)
    score_matrix = np.dot(mask_feature_array, normal_feature_array.transpose())

    if abs_flag is not None:
        score_matrix = np.abs(score_matrix)

    id_matrix = np.zeros((mask_len, normal_len))
    for i in range(mask_len):
        for j in range(normal_len):
            if mask_id[i]==normal_id[j] or mask_id[i]==(normal_id[j]+" 口罩"):
            # if mask_id[i] in normal_id[j] or normal_id[j] in mask_id[i]:
                id_matrix[i,j]=1
    id_matrix = id_matrix!=0

    TAR_list = []
    FAR_list = []
    threshold_list = []

    if fold is None:
        threshold = np.linspace(0, 1, 100000)
        if mask_list!=normal_list:
            TARs, FARs, thresholds = ROC_by_mat(score_matrix, id_matrix, thresholds=threshold)
        else:
            TARs, FARs, thresholds = ROC_by_mat(score_matrix, id_matrix, triu_k=0, thresholds=threshold)
        TAR_list.append(TARs)
        FAR_list.append(FARs)
        threshold_list.append(thresholds)
    # print(TARs, FARs, thresholds)

    # 根据提前分好的list取对
    else:
        
        for i in range(fold):
            random_file_path = base_file_path+"_%d"%(i)
            result_file = open(random_file_path, "r")
            random_list = [int(x.strip()) for x in result_file.readlines()]
            random_list_array = np.array(random_list)
            # test = score_matrix[random_list_array,:][:, random_list_array]
            # print(test.shape, test.ndim)
            if mask_list!=normal_list:
                TARs, FARs, thresholds = ROC_by_mat(score_matrix[random_list_array,:][:, random_list_array], id_matrix[random_list_array,:][:, random_list_array])
            else:
                TARs, FARs, thresholds = ROC_by_mat(score_matrix[random_list_array,:][:, random_list_array], id_matrix[random_list_array,:][:, random_list_array], triu_k=0)
            print(TARs, FARs, thresholds)
            if abs_flag is not None:
                result_file_path = result_base_file_path+"/%d_%s_nm.txt"%(i,abs_flag)
            else:
                result_file_path = result_base_file_path+"/%d_nm.txt"%i
            result_file = open(result_file_path, "w")
            for j in range(len(TARs)):
                result_file.write("%f %f %f\n"%(TARs[j], FARs[j], thresholds[j]))
            result_file.close()
            TAR_list.append(TARs)
            FAR_list.append(FARs)
            threshold_list.append(thresholds)
        TAR_list = np.array(TAR_list)
        FAR_list = np.array(FAR_list)

else:
    ## TODO ! 如果已经保存中间结果，则直接读取
    TAR_list = []
    FAR_list = []
    threshold_list = []
    result_file = open(result_file_path, "r")
    TARs, FARs, thresholds = [],[],[]
    for result in result_file.readlines():
        result = result.strip().split()
        TARs.append(float(result[0]))
        FARs.append(float(result[1]))
        thresholds.append(float(result[2]))
    TAR_list.append(TARs)
    FAR_list.append(FARs)
    TAR_list = np.array(TAR_list)
    FAR_list = np.array(FAR_list)
    threshold_list.append(thresholds)
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
# print(TARs[0], FARs[0])
plt.clf()
plt.figure()
plt.xlabel("FPR", fontsize = 14)
plt.ylabel("TPR", fontsize = 14)
plt.title("ROC Curve", fontsize = 14)
# fpr = np.flipud(FAR_list[0])
# tpr = np.flipud(TAR_list[0])
fpr = FAR_list[0]
tpr = TAR_list[0]
threshold = threshold_list[0]
try:
    auc_score = auc(fpr, tpr)
    print(auc_score)
except Exception as e:
    print(e)
plt.xlim([10**-6, 10**-1])
# plt.ylim([0.1, 1.0])
plt.plot(fpr, tpr )
# buf = io.BytesIO()
# plt.savefig(buf, format = 'jpeg')
# buf.seek(0)
# plt.show()
if abs_flag is not None:
    plt.savefig(result_base_file_path+"/"+str(fold)+"_nm_abs.jpg")
else:
    plt.savefig(result_base_file_path+"/"+str(fold)+"_nm.jpg")

tpr_fpr_row = []
x_labels = [10**-6, 10**-5, 10**-4,10**-3, 10**-2, 10**-1]
for fpr_iter in np.arange(len(x_labels)):
    # print(x_labels[fpr_iter],fpr)
    # print(x_labels[fpr_iter],fpr)
    # fpr = list(fpr)
    delta = abs(fpr-x_labels[fpr_iter])
    _, min_index = min(list(zip(delta, range(len(fpr)))))
    tpr_fpr_row.append([x_labels[fpr_iter], '%.4f' % tpr[min_index],'%f' %threshold[min_index]])
print(tpr_fpr_row)

