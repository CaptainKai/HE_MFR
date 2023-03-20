from wearmask import *

import argparse

parser = argparse.ArgumentParser(description='create lmdb for amsoft training')
# DATA
# parser.add_argument('--root_path', type=str, default='/home/ubuntu/data1/lk/facecrub_new/face_test/data/facescrub',
# parser.add_argument('--root_path', type=str, default='/home/ubuntu/data1/lk/facecrub_new/face_test/data/megaface',
# parser.add_argument('--root_path', type=str, default='/home/ubuntu/data1/lk',
parser.add_argument('--root_path', type=str, default='/home/ubuntu/data1/lk/lfw',
                    help='path to root path of images')
# parser.add_argument('--gt_paths', type=list, default=['/home/ubuntu/data1/lk/celebrity_msra_lmk/msra_lmk', '/home/ubuntu/data1/lk/celebrity_msra_lmk/celebrity_lmk'],
# parser.add_argument('--gt_paths', type=list, default=['/home/ubuntu/data1/lk/facecrub_new/face_test/data/facescrub_list.txt'],
# parser.add_argument('--gt_paths', type=list, default=['/home/ubuntu/data1/lk/facecrub_new/face_test/data/megaface_10000_list.txt'],
# parser.add_argument('--gt_paths', type=list, default=['/home/ubuntu/data2/lk/amsoft_pytorch/amsoft_pytorch/dataprocess/test.txt'],
parser.add_argument('--gt_paths', type=list, default=['/home/ubuntu/data2/lk/amsoft_pytorch/amsoft_pytorch/data/result/Dets_lfw.txt'],
                    help='path to txt of groundtruth')#                   
args = parser.parse_args()
print(args)

root_path = args.root_path
gt_paths = args.gt_paths

face_maker = FaceMasker()

import time
import datetime
import dlib


for gt_path in gt_paths:
    cnt = 0

    print(gt_path)
    gt_file = open(gt_path, 'r')
    gt=gt_file.readline()
    print(gt)
    print(cnt)
    dest_file = open(gt_path.split('.')[0]+'_68_rect', 'w')
    make_t0 = time.time()

    while gt:
        # line = gt.strip("\n")# TODO
        line = gt.strip("\n").split()# TODO
        if cnt % 10000 == 0 and cnt>0:
            make_t1 = time.time()
            print("processing %d"%(cnt))
            time_cost = make_t1-make_t0
            print('time consume:{}'.format(str(datetime.timedelta(seconds=time_cost))))
            make_t0 = time.time()
        pic_name = line[0]# TODO
        # pic_name = line# TODO
        # print(pic_name)
        # id = int(line[1])
        # landmark = [float(x) for x in line[2:12]] # TODO
        landmark = [float(x) for x in line[5:15]] # TODO

        pic_path = os.path.join(root_path, pic_name)
        if not os.path.exists(pic_path):
            print(pic_path)
            gt=gt_file.readline()
            continue
        pic_frame = cv2.imread(pic_path)
        h,w,c = pic_frame.shape

        face_w = min( int(max(landmark[0::2])*1.2) - int(min(landmark[0::2])/1.8) , int(max(landmark[1::2])*1.3) - int(min(landmark[1::2])/1.8) )
        # TODO
        # bbox = dlib.rectangle(int(min(landmark[0::2])/1.8), int(min(landmark[1::2])/1.8), min(w, int(max(landmark[0::2])*1.2)), min(h, int(max(landmark[1::2])*1.3)))
        # bbox = dlib.rectangle(int(min(landmark[0::2])/1.8), int(min(landmark[1::2])/1.8), int(max(landmark[0::2])*1.2), int(max(landmark[1::2])*1.3))
        # bbox = dlib.rectangle(0,0, int(max(landmark[0::2])*1.5), int(max(landmark[1::2])*1.5))
        # bbox = dlib.rectangle(0,0, face_w, face_w)
        bbox = dlib.rectangle(int(min(landmark[0::2])/1.8), int(min(landmark[1::2])/1.8), min(w, int(min(landmark[0::2])/1.8)+face_w), min(h, int(max(landmark[1::2])/1.8)+face_w))

        points_68 = face_maker.detect_ldm(pic_frame, [bbox])# TODO
        # points_68 = face_maker.detect_ldm(pic_frame)# TODO
        if len(points_68)>0:
            if len(points_68)//68//2>1:
                print('too many face', pic_name)

            points_68_str = " ".join([ str(x) for x in points_68])

            new_line = gt.strip()+' '+points_68_str# TODO
            dest_file.write(new_line+'\n')
        else:
            print('no face', pic_name)
            dest_file.write(gt)
        # exit(0)
        cnt+=1
        gt=gt_file.readline()
    
    dest_file.close()
    gt_file.close()
