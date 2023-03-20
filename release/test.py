from PIL import Image
import numpy as np

from torchvision.transforms import functional as F
import torchvision.transforms as transforms
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from torch.nn.modules.dropout import AlphaDropout, Dropout2d
# import pycuda.driver as cuda
# cuda.init()
# cudnn.benchmark = True

import cv2

transform = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
    ])

def extractDeepFeature(img, model,flop=False, transform=transform, mode=1):
    exit(0)
    img = transform(img)
    img = img.unsqueeze(0).to('cuda')
    if flop:
        img_ = transform(F.hflip(img))
        img_ = img_.unsqueeze(0).to('cuda')
        ft = torch.cat((model(img), model(img_)), 1)[0].to('cpu')
        return ft
    else:
        # return model(img)[0].to('cpu') # origin
        # return model.forward_split(img, split_index=int(56.7366-4))[0].to('cpu') # origin
        if mode==3:
            return model(img)[1][0].to('cpu') # 0:UP 1:DOWN
        elif mode==2:
            return model(img)[0][0].to('cpu') # 0:UP 1:DOWN
        elif mode==1:
        ### full
            result = model(img)[:2]
            result = torch.cat(result, 1)
            return result[0].to('cpu')

def extractDeepFeature_ldm(img, ldm, model,flop=False, transform=transform):
    img = transform(img)
    img = img.unsqueeze(0).to('cuda')
    ldm = torch.from_numpy(np.array(ldm)).to('cuda')
    return model(img, ldm)[0].to('cpu')  


def save_feature_list(testlist, suffix, model, ldm_input=False, log_interval=1000,save_path=None,multi_gpus=True):
    import time
    print(time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()))
    
    with open(testlist) as f:
        import struct
        import os
        imgfiles = f.readlines()
        i=0
        for imgfile in imgfiles:
            imgfile = imgfile.strip()
            if not os.path.exists(imgfile):
                print(imgfile, "no exists!")
                continue
            with open(imgfile, 'rb') as f:
                img =  Image.open(f).convert('RGB')
            if img.size[0]!=112 or img.size[1]!=112:
                print("resize", imgfile)
                img = img.resize((112,112))

            if not ldm_input:
                feature = extractDeepFeature(img, model)# TODO
                # print(feature)
                # exit(0)
            else:
                ldm_file_path = imgfile+".landmark"
                if not os.path.exists:
                    ldm = [112]*10
                ldm_file = open(ldm_file_path, "r")
                ldm_line = ldm_file.readline() # TODO
                ldm_line = ldm_line.strip()
                # #ldm_str = (".".join( ldm_line.split('.')[1:] )).split(' ')[1:]
                # #print(ldm_line)
                ldm_str = ldm_line.split(' ')[-10:]
                # #print(ldm_str)
                ldm = [ int(float(x)) for x in ldm_str]
                if ldm[0]==-1:
                    ldm = [112]*10 # TODO
                # img_draw = cv2.imread(imgfile)
                # for i in range(5):
                #     cv2.circle(img_draw, (ldm[2*i], ldm[2*i+1]), 2, [0,0,255], 2 )
                # cv2.imshow('test', img_draw)
                # cv2.waitKey(0)
                # exit(0)

                feature = extractDeepFeature_ldm(img, [ldm], model)# TODO
            
            feature_name = imgfile.split('/')[-1]
            img_root = imgfile.split(feature_name)[0]
            feature_name = feature_name + '_' + suffix + '.bin'
            if not save_path:
                save_root = img_root
            else:
                save_root = save_path
            result_path = os.path.join(save_root, feature_name)
            if not os.path.exists(save_root):
                os.makedirs(save_root)
            result_file = open(result_path,'wb')
            # write_bin
            feature /= feature.norm()
            # feature=feature.detach().numpy()
            result_file.write(struct.pack('4i', len(feature), 1, 4, 5))
            result_file.write(struct.pack('%df'%len(feature), *feature))
            result_file.close()
            if i%log_interval==0:
                print("%d done"%(i))
            i+=1
    
    print(time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()))


if __name__ == '__main__':
    
    testlist3 = "/home/ubuntu/data3/data/AR_warp_zip/img_list_aligned.txt" # TODO 文件列表txt：每行存放对齐之后的图片完整路径

    
    suffix = "SResnet_36_split_noDrop_noLA_b3_init_20_repeat_down" # cfg name + 备注 TODO 特征的后缀，用于区分不同模型提取得到的特征
    
    model_name = "SimpleResnet_split_abla_36" # 模型名称，应该和models/model_zoo.py文件中对应
    
    model_path = "./snapshot/SResnet_36_split_noDrop_noLA_b3_init_repeat/backbone_20_checkpoint.pth" # 模型存放的路径
    
    multi_gpus = False
    
    import models
    model = models.get_model(model_name, input_size=[112,112])
    # model = models.get_model(model_name)
    
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if multi_gpus:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = model.to(device)
    model.eval()
    # def apply_dropout(m):
    #     if type(m) == Dropout2d:
    #         m.train()
    # model.apply(apply_dropout)
    print("%s model load done, test tag is %s"%(model_path, suffix))

    print(testlist3)
    save_feature_list(testlist3, suffix, model)

