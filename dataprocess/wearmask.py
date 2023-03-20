# -*- coding: utf-8 -*-
import os
import sys

import argparse
import numpy as np
import cv2

import math
import dlib
from PIL import Image, ImageFile

__version__ = '0.3.0'
'''
修改自thirdparty版本，删除了cli函数，因为不适应了
'''

IMAGE_DIR = '/home/ubuntu/data2/lk/thirdparty/mask/self-built-masked-face-recognition-dataset/wear_mask_to_face/images'
# IMAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'images')
DEFAULT_IMAGE_PATH = os.path.join(IMAGE_DIR, 'white-mask-n95.png')# 1 ok
BLACK_IMAGE_PATH = os.path.join(IMAGE_DIR, 'black-mask.png')# 3 no
BLUE_IMAGE_PATH = os.path.join(IMAGE_DIR, 'blue-mask.png')# 2 ok
GS_IMAGE_PATH = os.path.join(IMAGE_DIR, 'green-mask-surgery.png') # 4 no
GN_IMAGE_PATH = os.path.join(IMAGE_DIR, 'green-msk-n95.png') # 5 ok
# RED_IMAGE_PATH = os.path.join(IMAGE_DIR, 'red-mask.png')
WHITE_IMAGE_PATH = os.path.join(IMAGE_DIR, 'default-mask.png')
mask_path_list = [DEFAULT_IMAGE_PATH, BLUE_IMAGE_PATH, GN_IMAGE_PATH, BLACK_IMAGE_PATH, DEFAULT_IMAGE_PATH, BLACK_IMAGE_PATH, WHITE_IMAGE_PATH]


from skimage import io

class FaceMasker:
    KEY_FACIAL_FEATURES = ('nose_bridge', 'chin')# 鼻梁，下巴

    def __init__(self, mask_path_list=mask_path_list, model='hog', predictor="/home/ubuntu/data2/lk/thirdparty/mask/self-built-masked-face-recognition-dataset/shape_predictor_68_face_landmarks.dat"):
        # self.mask_path = mask_path
        self.mask_path_list = mask_path_list
        self.model = model
        self._face_img = None
        self._mask_img = None
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor)
    
    def detect_ldm(self, image_frame, bbox=None):
        face_image_np = image_frame
        if not bbox:
            dets = self.detector(face_image_np, 1)
        else:
            dets = bbox
        # print(dets)
        face_landmark=[]
        
        for k, d in enumerate(dets):
            # print(d)
            shape = self.predictor(face_image_np, d)
            # cv2.rectangle(face_image_np,(d.left(),d.top()),(d.right(),d.bottom()),(255,255,255))
            # print(d.right()-d.left())
            # print(d.bottom()-d.top())
            # cv2.imshow('test', face_image_np)
            # cv2.waitKey(0)

            # landmark = np.matrix([[p.x, p.y] for p in shape.parts()])
            for p in shape.parts():
                face_landmark.append(p.x)
                face_landmark.append(p.y)
            #     cv2.circle(face_image_np, (p.x,p.y), 1,[0,0,255], 1)
            #     cv2.imshow('test', face_image_np)
            # cv2.waitKey(0)
        return face_landmark
    

    def detect_full(self, image_frame, bbox=None):
        '''
        输出检测器得到的所有（自定义格式的）结果
        '''
        face_image_np = image_frame
        # face_image_np = cv2.cvtColor(face_image_np, cv2.COLOR_BGR2GRAY)
        if not bbox:
            dets = self.detector(face_image_np, 1)
        else:
            dets = bbox
        # print(dets)
        face_landmark = []
        face_bbox = []
        
        for k, d in enumerate(dets):
            # print(d)
            shape = self.predictor(face_image_np, d)
            # cv2.rectangle(face_image_np,(d.left(),d.top()),(d.right(),d.bottom()),(255,255,255))
            # print(d.right()-d.left())
            # print(d.bottom()-d.top())
            # cv2.imshow('test', face_image_np)
            # cv2.waitKey(0)

            # landmark = np.matrix([[p.x, p.y] for p in shape.parts()])
            for p in shape.parts():
                face_landmark.append(p.x)
                face_landmark.append(p.y)
            #     cv2.circle(face_image_np, (p.x,p.y), 1,[0,0,255], 1)
            #     cv2.imshow('test', face_image_np)
            # cv2.waitKey(0)
            face_bbox.append([d.left(), d.top(), d.right(), d.bottom()])
        return face_bbox, face_landmark
    
    def mask(self, image_frame, color_index, landmark=None):
        face_image_np = image_frame
        # face_image_np = io.imread((face_path))
        
        face_landmarks = []

        if not landmark:
            landmark = self.detect_ldm(face_image_np)
            # dets = self.detector(face_image_np, 1)
            # for k, d in enumerate(dets):
                # face_landmark={}
            #     shape = self.predictor(face_image_np, d)
            #     landmark = np.matrix([[p.x, p.y] for p in shape.parts()])
                # face_landmark['chin'] =  [ [landmark.part(i).x, landmark.part(i).y]  for i in range(4,12) ]
                # face_landmark['nose_bridge'] = [ [landmark.part(i).x, landmark.part(i).y]  for i in range(27,31) ]
                # face_landmarks.append(face_landmark)
            #     # # left eye, right eye, nose, left mouth, right mouth
            #     # # order = [36, 45, 30, 48, 54]
            #     # for j in range(landmark.shape[0]):
            #     #     x = shape.part(j).x
            #     #     y = shape.part(j).y
            #     #     cv2.circle(face_image_np, (x,y), 1,[0,0,255], 1)
            #     #     cv2.putText(face_image_np, str(j), (x, y+1),
            #     #                 cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
            #     # cv2.imshow('test', face_image_np)
            #     # # cv2.waitKey(0)
            # # if len(dets)==0:
            # #     print(face_path)
            # #     return
        # else:
        #     landmark = []
        for d in range(len(landmark)//68//2):
            face_landmark={}
            face_landmark['chin'] = [ [ landmark[2*i], landmark[2*i+1] ] for i in range(4+d*68*2, 12+d*68*2)] # 8 点
            face_landmark['nose_bridge'] = [ [ landmark[2*i], landmark[2*i+1] ] for i in range(27+d*68*2, 31+d*68*2)] # 4 点
            face_landmarks.append(face_landmark)
            # for j in range(68):
            #     x = landmark[2*j]
            #     y = landmark[2*j+1]
            #     cv2.circle(face_image_np, (x,y), 1,[0,0,255], 1)
            #     cv2.putText(face_image_np, str(j), (x, y+1),
            #                 cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
        #     cv2.imshow('test', face_image_np)
        #     cv2.waitKey(0)
        
        # self._face_img = Image.fromarray(face_image_np) # TODO
        self._face_img = face_image_np
        self._mask_img = Image.open(self.mask_path_list[color_index])
        # self._mask_img = self._mask_img.convert('RGB')
        # print(self._mask_img.mode, self._face_img.mode)
        # print()
        # self._face_img.show()
        found_face = False
        for face_landmark in face_landmarks:
            found_face = True
            # check whether facial features meet requirement
            for facial_feature in self.KEY_FACIAL_FEATURES:
                if facial_feature not in list(face_landmark.keys()):
                    found_face = False
                    break

            if found_face:  
                # print("start mssk") 
                diff_location = self._mask_face(face_landmark)
                # self._face_img.show()

        return self._face_img, found_face, diff_location

    def _mask_face(self, face_landmark):
        nose_bridge = face_landmark['nose_bridge']
        nose_point = nose_bridge[len(nose_bridge) * 1 // 4]# 4//4=1 不是最上面的点，而是第二个点
        nose_v = np.array(nose_point)

        chin = face_landmark['chin']
        chin_len = len(chin)
        chin_bottom_point = chin[chin_len // 2] # 8//2=4
        chin_bottom_v = np.array(chin_bottom_point)
        chin_left_point = chin[chin_len // 8]
        chin_right_point = chin[chin_len * 7 // 8]

        # split mask and resize
        width = self._mask_img.width
        height = self._mask_img.height
        # width_ratio = 1.2
        width_ratio = 1.8
        new_height = int(np.linalg.norm(nose_v - chin_bottom_v))

        # left
        mask_left_img = self._mask_img.crop((0, 0, width // 2, height))
        mask_left_width = self.get_distance_from_point_to_line(chin_left_point, nose_point, chin_bottom_point)
        mask_left_width = max(mask_left_width, 1.0)
        mask_left_width = int(mask_left_width * width_ratio)
        mask_left_img = mask_left_img.resize((mask_left_width, new_height))

        # right
        mask_right_img = self._mask_img.crop((width // 2, 0, width, height))
        mask_right_width = self.get_distance_from_point_to_line(chin_right_point, nose_point, chin_bottom_point)
        mask_right_width = max(mask_right_width, 1.0)
        mask_right_width = int(mask_right_width * width_ratio)
        
        mask_right_img = mask_right_img.resize((mask_right_width, new_height))
        

        # merge mask
        size = (mask_left_img.width + mask_right_img.width, new_height)
        mask_img = Image.new('RGBA', size)
        mask_img.paste(mask_left_img, (0, 0), mask_left_img) # source img; upper right corner; mask
        mask_img.paste(mask_right_img, (mask_left_img.width, 0), mask_right_img)
        # mask_img.show()

        # rotate mask
        angle = np.arctan2(chin_bottom_point[1] - nose_point[1], chin_bottom_point[0] - nose_point[0])
        rotated_mask_img = mask_img.rotate(angle, expand=True)

        # calculate mask location
        center_x = (nose_point[0] + chin_bottom_point[0]) // 2
        center_y = (nose_point[1] + chin_bottom_point[1]) // 2

        offset = mask_img.width // 2 - mask_left_img.width
        radian = angle * np.pi / 180
        box_x = center_x + int(offset * np.cos(radian)) - rotated_mask_img.width // 2
        box_y = center_y + int(offset * np.sin(radian)) - rotated_mask_img.height // 2

        # mask_img.show()
        # print(mask_img.mode, mask_img.format)
        # add mask
        self._face_img.paste(mask_img, (box_x, box_y), mask=mask_img) # TODO 这里的应该用rotated_mask_img 但因为差不太多，所以为了保持变量一致，不进行改变
        # self._face_img.paste(rotated_mask_img, (box_x, box_y), mask=rotated_mask_img)

        # self._face_img.show()
        # self._save("/home/ubuntu/data2/lk/recognition/pytorch_new/dataprocess/test.png")

        ## find the top location
        r,g,b,a = mask_img.split()
        a_content = a.load()
        mask_width, mask_height = mask_img.size
        diff_location=[0,0]
        for y in range(mask_height):
            for x in range(mask_width):
                if a_content[x, y]!=0:
                    diff_location[0]=box_y+y
                    diff_location[1]=box_x+x
                    break
            if a_content[x, y]!=0:
                break
        return diff_location



    def _save(self, savepath):
        self._face_img.save(savepath)
        print('Save to %s'%(savepath))

    @staticmethod
    def get_distance_from_point_to_line(point, line_point1, line_point2):
        distance = np.abs((line_point2[1] - line_point1[1]) * point[0] +
                          (line_point1[0] - line_point2[0]) * point[1] +
                          (line_point2[0] - line_point1[0]) * line_point1[1] +
                          (line_point1[1] - line_point2[1]) * line_point1[0]) / \
                   np.sqrt((line_point2[1] - line_point1[1]) * (line_point2[1] - line_point1[1]) +
                           (line_point1[0] - line_point2[0]) * (line_point1[0] - line_point2[0]))
        return int(distance)


if __name__ == '__main__':
    dataset_path = '/home/ubuntu/data2/lk/amsoft_pytorch/amsoft_pytorch/data/facecrub_local'
    save_dataset_path = '/home/ubuntu/data2/lk/amsoft_pytorch/amsoft_pytorch/data/facecrub_local_mask'
    for root, dirs, files in os.walk(dataset_path, topdown=False):
        for name in files:
            new_root = root.replace(dataset_path, save_dataset_path)
            # if not os.path.exists(new_root):
            #     os.makedirs(new_root)
            # deal
            imgpath = os.path.join(root, name)
            save_imgpath = os.path.join(new_root, name)
            cli(imgpath,save_imgpath)
