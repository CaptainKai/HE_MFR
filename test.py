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

def extractDeepFeature(img, model,flop=False, transform=transform):
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
        # return model(img)[0][0].to('cpu') # 0:UP 1:DOWN
        ### full
        result = model(img)[:]
        # print(print(result))
        # print(result[0].to('cpu'))
        # result[1] = result[1]*0.01
        result = torch.cat(result, 1)
        # print(result.shape)
        # exit(0)

        return result[0].to('cpu')

def extractDeepFeature_ldm(img, ldm, model,flop=False, transform=transform):
    img = transform(img)
    img = img.unsqueeze(0).to('cuda')
    ldm = torch.from_numpy(np.array(ldm)).to('cuda')
    return model(img, ldm)[0].to('cpu')  

def pair_test():
    model = amsoft.AmsoftBackbone().to('cuda')
    model.load_state_dict(torch.load('/home/ubuntu/data3/lk/amsoft_pytorch/snapshot/filtamsoft_33_checkpoint.pth'))#/home/ubuntu/data3/lk/amsoft_pytorch/snapshot/filtamsoft_27_checkpoint.pth
    model.eval()
    with open('test.txt') as f:
        pairs_lines = f.readlines()

    with torch.no_grad():
        for pair in pairs_lines:
            file1 = pair.strip().split()[0]
            file2 = pair.strip().split()[1]
            with open(file1, 'rb') as f:
                img1 =  Image.open(f).convert('RGB')
            with open(file2, 'rb') as f:
                img2 =  Image.open(f).convert('RGB')
            f1 = extractDeepFeature(img1, model)
            f2 = extractDeepFeature(img2, model)
            distance = f1.dot(f2) / (f1.norm() * f2.norm() + 1e-5)
            print("distance is %f"%(distance))


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
    # test()/home/ubuntu/data3/lk/megaface_test/megalist_10000_txt/megaface_10000_list_transform.txt
    # /home/ubuntu/data3/lk/megaface_test/facescrublist_txt/facescrub_list_transform.txt
    # /home/ubuntu/data2/lk/facebox/pytorch_version/FaceBoxes_landmark/data/Megaface/megaface_10000_list_aligned.txt
    # /home/ubuntu/data2/lk/facebox/pytorch_version/FaceBoxes_landmark/data/Facecrub/facescrub_list_aligned.txt
    testlist = "/home/ubuntu/Work/lk/test_data/facescrub_mask_list_transform.txt"
    # testlist = "/home/ubuntu/Work/lk/test_data/facescrub_list_transform_Meng.txt"
    # testlist = "/home/ubuntu/Work/lk/test_data/facescrub_list_transform_Meng_arcface.txt"
    # testlist = "/home/ubuntu/data2/lk/amsoft_pytorch/amsoft_pytorch/data/pairtest/lfw_transform_lzj/lfw_filelist.txt"
    # testlist2 = "/home/ubuntu/Work/lk/test_data/megaface_10000_list_transform_repeat.txt"
    # testlist2 = "/home/ubuntu/Work/lk/test_data/megaface_10000_list_transform_arcface.txt"
    # testlist2 = "/home/ubuntu/Work/lk/test_data/megaface_10000_list_transform_arcface_mtcnn.txt"
    testlist2 = "/home/ubuntu/Work/lk/test_data/megaface_10000_mask_list_transform.txt"
    # testlist = "/home/ubuntu/data2/lk/amsoft_pytorch/amsoft_pytorch/data/weibo_large/new_diff_nodiff/img_list_normal.txt"
    # testlist = "/home/ubuntu/data2/lk/amsoft_pytorch/amsoft_pytorch/data/weibo_large/new_diff_nodiff/img_list_mask.txt"
    # testlist = "/home/ubuntu/data2/lk/amsoft_pytorch/amsoft_pytorch/data/baidu_hd/img_list_normal.txt"
    # testlist = "/home/ubuntu/data2/lk/amsoft_pytorch/amsoft_pytorch/data/baidu_hd/new_diff_nodiff/img_list_mask.txt"
    # testlist = "/home/ubuntu/data2/lk/amsoft_pytorch/amsoft_pytorch/data/baidu_hd/new_diff_nodiff/img_list_normal.txt"

    # testlist3 = "/home/ubuntu/data3/data/AR_warp_zip/img_list_aligned.txt"
    # testlist3 = "/home/ubuntu/data1/lk/lfw_transform.txt"
    # testlist3 = "/home/ubuntu/data3/data/spider/baidu/image_hd_filter/img_list_mask_0714.txt"
    # testlist3 = "/home/ubuntu/data1/lk/lfw_mask_transform.txt"
    # suffix = "res_arc_50_20" # cfg name + 备注
    # suffix = "ArcfaceResnet_50" # cfg name + 备注
    # suffix = "SResnet_36_split_noDrop_noLA_b1_20_init_up" # cfg name + 备注
    # suffix = "am_36_20"
    # suffix = "res_50_ddp_split_layer2_mask_0.5_fixBug_20_down"
    # suffix = "res_50_mix_20"
    # suffix = "res_sph2_50_30"
    # suffix = "res_50_arc_noNorm_20"
    # suffix = "SResnet_36_split_noDrop_noLA_b3_LR_20" # 左右特征分支
    # suffix = "SR36_sphereNorm_20"
    # suffix = "SR36_sphereNorm_mix_25"
    # suffix = "SR36_mix_25_25"
    # suffix = "sr_36_b3_cl_seperate_28"
    suffix = "sr_36_fc_bothMix_seperate_30"
    # suffix = "SR36_sphereface2_25"
    # suffix = "res_50_split_noDrop_noLA_f1_20"
    # suffix = "am_20_half" # cfg name + 备注
    # suffix = "SResnet_36_split_noDrop_noLA_b3_init_tri_fixBug_4loss_0.35_0.2_fixBug_25" # cfg name + 备注
    # suffix = "SR_36_ddp_multiTask_40" # cfg name + 备注
    # suffix = "SResnet_36_split_noDrop_noLA_b3_init_classSeparate_fixBug_25"
    # suffix = "SResnet_36_split_noDrop_noLA_b3_init_tri_3branch_25"
    # suffix = "SResnet_36_split_noDrop_noLA_b3_3loss_1_25"
    # suffix = "SimpleResnet_fullsplit_36_noLA_20"
    # suffix = "ArcfaceResnet_101" # cfg name + 备注
    # suffix = "SimpleResnet_36"
    # suffix = "dense121"
    
    # # pair_test()
    # model_name = "densenet_121"
    # model_name = "resnet50"
    # model_name = "resnet50_split"
    # model_name = "ResNet50_split_abla"
    # model_name = "SimpleResnet_fullsplit_36"
    # model_name = "SimpleResnet_split_abla_36"
    # model_name = "SimpleResnet_split_abla_36_fusion"
    model_name = "SimpleResnet_36"
    # model_name = "SimpleResnet_20"
    # model_name = "SimpleResnet_atten_split_abla_36"
    # model_name = "SimpleResnet_36_multiTask"
    
    # model_name = "resnet101"
    # model_path = "/home/ubuntu/data2/lk/amsoft_pytorch/amsoft_pytorch/snapshot/msra-gt-mask-arguementation-0.5-resnest50_test/amsoft_25_checkpoint.pth"
    # model_path = "/home/ubuntu/data2/lk/recognition/pytorch_new/snapshot/res_50_ddp/backbone_20_checkpoint.pth"
    # model_path = "/home/ubuntu/data2/lk/recognition/pytorch_new/snapshot/res_50_mix/backbone_20_checkpoint.pth"
    # model_path = "/home/ubuntu/data2/lk/recognition/pytorch_new/snapshot/res_arc_50_mask/backbone_20_checkpoint.pth"
    # model_path = "/home/ubuntu/data2/lk/recognition/pytorch_new/snapshot/res_50_ddp_split_layer2_fixBug/backbone_30_checkpoint.pth"
    # model_path = "/home/ubuntu/data2/lk/recognition/pytorch_new/snapshot/res_50_ddp_split_layer2_mask_0.5_fixBug/backbone_30_checkpoint.pth"
    # model_path = "/home/ubuntu/data2/lk/recognition/pytorch_new/snapshot/res_50_split_noDrop_noLA_f1/backbone_20_checkpoint.pth"
    # model_path = "/home/ubuntu/data2/lk/recognition/pytorch_new/snapshot/res_arc_50/backbone_20_checkpoint.pth"
    # model_path = "/home/ubuntu/data2/lk/recognition/pytorch_new/snapshot/res_50_arc_noNorm/backbone_0_20_checkpoint.pth"
    # model_path = "/home/ubuntu/data2/lk/recognition/pytorch_new/snapshot/SR_36_ddp_multiTask/backbone_40_checkpoint.pth"
    # model_path = "/home/ubuntu/data2/lk/recognition/pytorch_new/snapshot/SResnet_36_split_noDrop_noLA_b3_init_justSeparate/backbone_30_checkpoint.pth"
    # model_path = "/home/ubuntu/data2/lk/recognition/pytorch_new/snapshot/SResnet_36_split_noDrop_noLA_b3_init_tri_fixBug_4loss_0.35_0.2_fixBug/backbone_25_checkpoint.pth"
    # model_path = "/home/ubuntu/data2/lk/recognition/pytorch_new/snapshot/SResnet_36_split_noDrop_noLA_b3_init_repeat/backbone_20_checkpoint.pth"
    # model_path = "/home/ubuntu/data2/lk/recognition/pytorch_new/snapshot/SResnet_36_split_noDrop_noLA_b3_init_classSeparate_fixBug/backbone_25_checkpoint.pth"
    # model_path = "/home/ubuntu/data2/lk/recognition/pytorch_new/snapshot/SResnet_36_split_noDrop_noLA_b3_init_tri_3branch/backbone_25_checkpoint.pth"
    # model_path = "/home/ubuntu/data2/lk/recognition/pytorch_new/snapshot/SResnet_36_split_noDrop_noLA_b3_3loss_1/backbone_25_checkpoint.pth"
    # model_path = "/home/ubuntu/data2/lk/recognition/pytorch_new/snapshot/SR_36_ddp_sphereNorm/backbone_20_checkpoint.pth"
    # model_path = "/home/ubuntu/data2/lk/recognition/pytorch_new/snapshot/SR_36_ddp_sphereNorm_mix_continue/backbone_25_checkpoint.pth"
    # model_path = "/home/ubuntu/data2/lk/recognition/pytorch_new/snapshot/SR_36_ddp_mix/backbone_0_25_checkpoint.pth"
    # model_path = "/home/ubuntu/data2/lk/recognition/pytorch_new/snapshot/sr_36_b3_cl_seperate/backbone_0_28_checkpoint.pth"
    model_path = "/home/ubuntu/data2/lk/recognition/pytorch_new/snapshot/sr_36_fc_bothMix_seperate/backbone_0_30_checkpoint.pth"
    # model_path = "/home/ubuntu/data2/lk/recognition/pytorch_new/snapshot/SResnet_36_split_noDrop_noLA_b3_LR/backbone_20_checkpoint.pth"
    # model_path = "/home/ubuntu/data2/lk/recognition/pytorch_new/snapshot/SResnet_36_split_noDrop_noLA_fullsplit/backbone_20_checkpoint.pth"
    # model_path = "/home/ubuntu/data2/lk/recognition/pytorch_new/snapshot/Arcface_Resnet101_35.pth"
    # model_path = "/home/ubuntu/data2/lk/recognition/pytorch_new/snapshot/am_20_ddp_test/backbone_381_checkpoint.pth"
    # model_path = "/home/ubuntu/data2/lk/recognition/pytorch_new/snapshot/amDensenet_121_test/backbone_38_checkpoint.pth"
    multi_gpus = False
    
    import models
    # fusion_mode = models.model_zoo.get_model("MLB", d=256, c=256, active_mode=0)
    model = models.get_model(model_name, input_size=[112,112], fc_num=2)
    # model = models.get_model(model_name, input_size=[112,112], three_branch=True)
    # model = models.get_model(model_name)
    
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if multi_gpus:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = model.to(device)
    model.eval()
    def apply_dropout(m):
        if type(m) == Dropout2d:
            m.train()
    model.apply(apply_dropout)
    print("%s model load done, test tag is %s"%(model_path, suffix))
    print(testlist)
    save_feature_list(testlist, suffix, model)
    save_feature_list(testlist2, suffix, model)
    print(testlist2)

    # print(testlist3)
    # save_feature_list(testlist3, suffix, model)

    # testlist3="/home/ubuntu/data3/lk/amsoft_pytorch/test3.txt"
    # accuracy_test(testlist3)

