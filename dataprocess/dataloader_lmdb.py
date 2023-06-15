from .lmdb_init import LMDB
import random

import time
'''
实例化lmdb_init文件中实现的LMDB,编写用于pytorch训练的dataloader自定义类
'''
from .wearmask import *
face_maker = FaceMasker()

import os
import torch
# import sys
# sys.path.append("/home/ubuntu/data2/lk/thirdparty/mask/MaskTheFace")
# import MaskFace
# face_maker = MaskFace.FaceMasker()

def get_mask(img, ldm68, rate, valid=True):
    res = img
    p = random.random()
    if p<=rate and valid:    
        color_index = random.randint(0,6) # include end point
        # color_index = 0 # include end point
        # color_index = 1 # -1,1,0,2,3
        # print("color_index:", color_index)
        ldm68 = [int(x) for x in ldm68.split(" ")]
        # [r,g,b]=img.split()
        # from PIL import Image
        # input_img = Image.merge("RGB", (b,g,r))
        input_img = res.copy()
        try:
            masked_image, success, diff_location = face_maker.mask(input_img, color_index, ldm68)
        except:
            success = False
            masked_image = res
            diff_location = [-1,-1]
        # if success:
        #     res = masked_image
    elif p>rate:
        success = False
        masked_image = res
        diff_location = [-1,-1]
    else:
        success = True
        masked_image = res
        diff_location = [-1,-1]
    return masked_image, success, diff_location


class ImageList():

    def __init__(self, lmdb_path="./lmdb_data/default", max_reader=1,num=6726601, \
                format_transform=None, preproc=get_mask, augu_rate=0, augu_paral=False, ldm=False, ldm68=False, shuffle=True):
        '''
        数据集类
        @preproc: 对图像进行（口罩）增广
        @augu_rate: 图像增广的概率
        @augu_paral: 增广数据是否并行输出，该选项以上一个选项存在为前提
        @ldm：是否获取5点关键点
        @ldm68：是否获取68点关键点，该选项目前仅用于增广
        '''
        self.dataset = LMDB(lmdb_path,max_reader,num)
        print("get the specified lmdb dataset")
        self.num=self.dataset.num
        self.indexlist=[x for x in range(self.num)]
        self.format_transform = format_transform
        self.preproc = preproc
        self.ldm = ldm
        self.ldm68 = ldm68
        self.augu_rate = float(augu_rate)
        self.augu_paral = augu_paral
        
        if not( (self.preproc and self.ldm68 and self.augu_rate) or \
            (not self.preproc and not self.ldm68 and not self.augu_rate) ): # mask augu，如果要增广，则三个参数都必须有值，否则，必须都为None/False
            print("setting error! 1")
            exit(0)
        if self.augu_paral and self.augu_rate!=1: # mapping network
            print("setting error! 2")
            exit(0)
        if self.augu_paral and not self.ldm68: # mapping network
            print("setting error! 3")
            exit(0)
        
        if shuffle:
            random.shuffle(self.indexlist)
    
    def __len__(self):
        return self.num
    
    def __getitem__(self, index_id):
        index = self.indexlist[index_id]
        
        # 一旦ldm68为TRUE，则代表着需要进行口罩增广
        if self.ldm68 and not self.ldm:
            img,label,ldm68 = self.dataset.getitem(index, ldm=self.ldm, ldm68=self.ldm68)
            new_img, flag, diff_location = self.preproc(img, ldm68, self.augu_rate)
            if self.augu_rate>=1:
                while not flag:
                    index = self.indexlist[random.randint(0, self.num-1)]
                    img,label,ldm68 = self.dataset.getitem(index, ldm=self.ldm, ldm68=self.ldm68)
                    new_img, flag, diff_location = self.preproc(img, ldm68, self.augu_rate)
            # img.show()
            # new_img.show()
            # exit(0)
        elif self.ldm and self.ldm68:
            img, label, ldm, ldm68 = self.dataset.getitem(index, ldm=self.ldm, ldm68=self.ldm68)
            new_img, flag, diff_location = self.preproc(img, ldm68, self.augu_rate)
            if self.augu_rate>=1:
                while not flag:
                    index = self.indexlist[random.randint(0, self.num-1)]
                    img,label,ldm68 = self.dataset.getitem(index, ldm=self.ldm, ldm68=self.ldm68)
                    new_img, flag, diff_location = self.preproc(img, ldm68, self.augu_rate)
        elif self.ldm:
            img,label,ldm = self.dataset.getitem(index, ldm=self.ldm, ldm68=self.ldm68)
        else:
            img,label = self.dataset.getitem(index, ldm=self.ldm, ldm68=self.ldm68)
    
        if self.format_transform is not None:
            img = self.format_transform(img)
            if self.ldm68:
                new_img = self.format_transform(new_img)
        
        # 是否同时并行输出增广数据
        if self.augu_paral:
            if self.ldm:
                return img, int(label), ldm, int(flag), new_img, diff_location
            else:
                return img, int(label), int(flag), new_img, diff_location
        else:
            if self.ldm68:
                if self.ldm:
                    return new_img, int(label), ldm, int(flag), diff_location
                else:
                    return new_img, int(label), int(flag), diff_location
            else:
                if self.ldm:
                    return img, int(label),ldm
                else:
                    return img, int(label)

    def close(self):
        self.dataset.closelmdb()

from tqdm import tqdm
class TripletList():
    def __init__(self, num_triplets, epoch, lmdb_path="./lmdb_data/default", max_reader=1,num=3923399, id_num=86876, \
                format_transform=None, preproc=get_mask, num_human_identities_per_batch=32,
                 triplet_batch_size=544, training_triplets_path=None, id_dict_path=None, ldm=False, ldm68=False, augu_rate=0):
        """
        Args:

        # root_dir: Absolute path to dataset.
        # training_dataset_csv_path: Path to csv file containing the image paths inside the training dataset folder.
        num_triplets: Number of triplets required to be generated.
        epoch: Current epoch number (used for saving the generated triplet list for this epoch).
        num_generate_triplets_processes: Number of separate Python processes to be created for the triplet generation
                                          process. A value of 0 would generate a number of processes equal to the
                                          number of available CPU cores.
        num_human_identities_per_batch: Number of set human identities per batch size.
        triplet_batch_size: Required number of triplets in a batch.
        training_triplets_path: Path to a pre-generated triplet numpy file to skip the triplet generation process (Only
                                 will be used for one epoch).
        transform: Required image transformation (augmentation) settings.
        """

        # Modified here to set the data types of the dataframe columns to be suitable for other datasets other than the
        #  VggFace2 dataset (Casia-WebFace in this case because of the identities starting with numbers automatically
        #  forcing the 'name' column as being of type 'int' instead of type 'object')
        # self.df = pd.read_csv(training_dataset_csv_path, dtype={'id': object, 'name': object, 'class': int})
        # self.root_dir = root_dir
        self.lmdb_path = lmdb_path
        self.dataset = LMDB(lmdb_path,max_reader,num)
        print("get the specified lmdb dataset")
        self.num=self.dataset.num
        self.indexlist=[x for x in range(self.num)]
        self.format_transform = format_transform
        self.preproc = preproc
        self.ldm = ldm
        self.ldm68 = ldm68
        self.augu_rate = float(augu_rate)
        
        self.num_triplets = num_triplets
        self.num_human_identities_per_batch = num_human_identities_per_batch
        self.triplet_batch_size = triplet_batch_size
        # self.epoch = epoch
        # self.transform = transform

        # Modified here to bypass having to use pandas.dataframe.loc for retrieving the class name
        #  and using dataframe.iloc for creating the face_classes dictionary
        self.id_num = id_num

        if id_dict_path is None:
            self.face_classes = self.make_dictionary_for_face_class()
        else:
            print("Loading id-dict file ...")
            self.face_classes = dict(np.load(id_dict_path, allow_pickle=True)).item()

        if training_triplets_path is None:
            self.reset_triplets(epoch)
        else:
            print("Loading pre-generated triplets file ...")
            self.training_triplets = np.load(training_triplets_path, allow_pickle=True)

    def make_dictionary_for_face_class(self):
        """
            face_classes = {'class0': [class0_id0, ...], 'class1': [class1_id0, ...], ...}
        """
        face_classes = dict()
        for index in range(self.num):
            label = self.dataset.get_label(index)
            label = int(label)
            if label not in face_classes:
                face_classes[label] = []
            # Instead of utilizing the computationally intensive pandas.dataframe.iloc() operation
            face_classes[label].append(index)
        print("Saving Id-Dict file in datasets directory ...")
        np.save('{}/id-dict.npy'.format(
                self.lmdb_path
            ),
            face_classes
        )
        print("Id-Dict file Saved!\n")
        return face_classes

    def generate_triplets(self, epoch):
        triplets = []
        classes = [x for x in range(self.id_num)]

        print("\nGenerating {} triplets ...".format(self.num_triplets))
        num_training_iterations_per_process = self.num_triplets / self.triplet_batch_size
        progress_bar = tqdm(range(int(num_training_iterations_per_process)))  # tqdm progress bar does not iterate through float numbers

        for training_iteration in progress_bar:

            """
            For each batch: 
                - Randomly choose set amount of human identities (classes) for each batch
            
                  - For triplet in batch:
                      - Randomly choose anchor, positive and negative images for triplet loss
                      - Anchor and positive images in pos_class
                      - Negative image in neg_class
                      - At least, two images needed for anchor and positive images in pos_class
                      - Negative image should have different class as anchor and positive images by definition
            """
            classes_per_batch = np.random.choice(classes, size=self.num_human_identities_per_batch, replace=False)

            for triplet in range(self.triplet_batch_size):

                pos_class = np.random.choice(classes_per_batch)
                neg_class = np.random.choice(classes_per_batch)
                # print(self.face_classes.shape, pos_class)

                while len(self.face_classes[pos_class]) < 2:
                    pos_class = np.random.choice(classes_per_batch)

                while pos_class == neg_class:
                    neg_class = np.random.choice(classes_per_batch)

                if len(self.face_classes[pos_class]) == 2:
                    ianc, ipos = np.random.choice(2, size=2, replace=False)

                else:
                    ianc = np.random.randint(0, len(self.face_classes[pos_class]))
                    ipos = np.random.randint(0, len(self.face_classes[pos_class]))

                    while ianc == ipos:
                        ipos = np.random.randint(0, len(self.face_classes[pos_class]))

                ineg = np.random.randint(0, len(self.face_classes[neg_class]))

                triplets.append(
                    [
                        self.face_classes[pos_class][ianc],
                        self.face_classes[pos_class][ipos],
                        self.face_classes[neg_class][ineg],
                        pos_class,
                        neg_class,
                    ]
                )

        print("Saving training triplets list in 'temp_datasets/generated_triplets' directory ...")
        np.save('temp_datasets/generated_triplets/epoch_{}_training_triplets_{}_identities_{}_batch_{}.npy'.format(
                epoch, self.num_triplets, self.num_human_identities_per_batch, self.triplet_batch_size
            ),
            triplets
        )
        print("Training triplets' list Saved!\n")

        return triplets

    def reset_triplets(self, epoch):
        training_triplets_path = 'temp_datasets/generated_triplets/epoch_{}_training_triplets_{}_identities_{}_batch_{}.npy'.format(
                epoch, self.num_triplets, self.num_human_identities_per_batch, self.triplet_batch_size
            )
        if os.path.exists(training_triplets_path):
            print("Loading pre-generated triplets file ...")
            self.training_triplets = np.load(training_triplets_path, allow_pickle=True)
        else:
            self.training_triplets = self.generate_triplets(epoch)

    def __getitem__(self, idx):

        anc_id, pos_id, neg_id, pos_class, neg_class = self.training_triplets[idx]

        if self.ldm68 :
            img, ldm, ldm68 = self.dataset.get_image(anc_id, ldm=self.ldm, ldm68=self.ldm68)
            new_img, flag, diff_location = self.preproc(img, ldm68, self.augu_rate)
            if self.augu_rate>=1:
                while not flag:
                    anc_id, pos_id, neg_id, pos_class, neg_class = self.training_triplets[random.randint(0, len(self.training_triplets)-1)]
                    img, ldm, ldm68 = self.dataset.get_image(anc_id, ldm=self.ldm, ldm68=self.ldm68)
                    new_img, flag, diff_location = self.preproc(img, ldm68, self.augu_rate)
        
        else:
            img, ldm, ldm68 = self.dataset.get_image(anc_id, ldm=self.ldm, ldm68=self.ldm68)
        
        anc_img = new_img if self.ldm68 else img
        pos_img, pos_ldm, pos_ldm68 = self.dataset.get_image(pos_id, ldm=self.ldm, ldm68=self.ldm68)
        neg_img, neg_ldm, neg_ldm68 = self.dataset.get_image(neg_id, ldm=self.ldm, ldm68=self.ldm68)

        pos_class = torch.from_numpy(np.array([pos_class]).astype('long'))
        neg_class = torch.from_numpy(np.array([neg_class]).astype('long'))

        sample = {
            'anc_img': anc_img,
            'pos_img': pos_img,
            'neg_img': neg_img,
            'pos_class': pos_class,
            'neg_class': neg_class
        }

        if self.format_transform:
            sample['anc_img'] = self.format_transform(sample['anc_img'])
            sample['pos_img'] = self.format_transform(sample['pos_img'])
            sample['neg_img'] = self.format_transform(sample['neg_img'])

        return sample

    def __len__(self):
        return len(self.training_triplets)


