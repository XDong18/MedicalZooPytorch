import glob
import os
import os.path as osp

import numpy as np
import torch
from torch.utils.data import Dataset
import json

import nibabel as nib


import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset

import lib.augment3D as augment3D
import lib.utils as utils
from lib.medloaders import medical_image_process as img_loader
from lib.medloaders.medical_loader_utils import get_viz_set, create_sub_volumes


# class Pancreas(Dataset):
#     def __init__(self,
#                 args,
#                 # mode,
#                 dataset_path='./datasets',
#                 anno_file,
#                 phase)
#         self.phase = phase
#         self.data_prefix = dataset_path
#         self.anno_file = anno_file
#         self._load_data_annotations()
#         self.augmentation = args.augmentation
#         if self.augmentation:
#             self.transform = augment3D.ComposeTransforms(
#                 transforms=[augment3D.RandomFlip(),])
    
#     def __len__(self):
#         if self.phase=='train':
#             return self.annotations['numTraining'] 
#         elif self.phase=='val':

#         elif self.phase=='test':
#             return self.annotations['numTest'] 

#     def __getitem__(self, idx):
#         img = self.data['image']
#         label = self.data['label']
#         [img], label = self.transform([img], label)
#         return torch.FloatTensor(img.copy()).unsqueeze(0), torch.LongTensor(label.copy())

#         # return self.pipeline(self.data[idx])

#     # def pipeline(self, info):
#     #     t = []
#     #     t.append(transform.RandomCrop(256))
#     #     t.append(transform.ToTensor())
#     #     self.transform = transform.Compose(t)
#     #     temp_data = [info['image'], info['label']]
#     #     temp_data = list(self.transform(*temp_data))
#     #     return {'image': temp_data[0], 'label': temp_data[1]}

#     def _load_data_annotations(self, anno_file):
#         self.annotations = {}
#         with open(anno_file) as f:
#             annotations = json.load(f)
#         self.annotations = annotations
#         self.data = []
#         if self.phase=='train':
#             self.ct_names = self.annotations['training']
#         else:
#             self.ct_names = self.annotations['test']

#         for ct_name in self.ct_names:
#             info = {}
#             if self.phase=='train':
#                 ct_obj = nib.load(osp.join(self.data_prefix, ct_name['image']))
#                 label_obj = nib.load(osp.join(self.data_prefix, ct_name['label']))
#                 info['image'] = np.array(ct_obj.dataobj)
#                 info['label'] = np.array(label_obj.dataobj)   
#             else:
#                 ct_obj = nib.load(osp.join(self.data_prefix, ct_name))
#                 info['image'] = np.array(ct_obj.dataobj)
#                 info['label'] = None
            
#             self.data.append(info)

    

class Pancreas(Dataset):
    def __init__(self,
                args,
                mode,
                split_id=1,
                samples=1000,
    ):
        self.mode = mode
        self.root = args.path
        self.CLASSES = 3

        # TODO fix 1. full_vol_dim is not a constanr
        #           2. dimension scenquence is [h, w, s]???
        self.full_vol_dim = (512, 512, 256)  # slice, width, height 

        self.threshold = args.threshold
        self.normalization = args.normalization
        self.augmentation = args.augmentation
        self.crop_size = args.dim

        self.list = []
        self.samples = samples
        self.split_id = split_id

        self.full_volume = None
    
        # self.save_name = self.root + '/iseg_2017/iSeg-2017-Training/iseg2017-list-' + mode + '-samples-' + str(
        #     samples) + '.txt'
        if self.augmentation:
            self.transform = augment3D.RandomChoice(
                transforms=[augment3D.GaussianNoise(mean=0, std=0.01), augment3D.RandomFlip(),
                            augment3D.ElasticTransform()], p=0.5)
        
        # if load:
        #     ## load pre-generated data
        #     self.list = utils.load_list(self.save_name)
        #     list_IDsT1 = sorted(glob.glob(os.path.join(self.training_path, '*T1.img')))
        #     self.affine = img_loader.load_affine_matrix(list_IDsT1[0])
        #     return
        
        self._load_data_annotations(args.anno_file)
        self.affine = img_loader.load_affine_matrix(self.train_img_list[0])

        # make sub volume list
        subvol = '_vol_' + str(self.crop_size[0]) + 'x' + str(self.crop_size[1]) + 'x' + str(self.crop_size[2])
        self.sub_vol_path = self.root + '/generated/' + mode + subvol + '/' 
        utils.make_dirs(self.sub_vol_path)
        self._make_sub_volume_list()
    
    def __len__(self):
        return len(self.list)
         
    def __getitem__(self, index):
        img_path, seg_path = self.list[index]
        t, s = np.load(img_path), np.load(seg_path)

        if self.mode == 'train' and self.augmentation:
            print('augmentation reee')
            [augmented_t,], augmented_s = self.transform([t1,], s)

            return torch.FloatTensor(augmented_t.copy()).unsqueeze(0), torch.FloatTensor(augmented_s.copy())

        return torch.FloatTensor(t).unsqueeze(0), torch.FloatTensor(s)

    def _load_data_annotations(self, anno_file):
        self.annotations = {}
        with open(anno_file) as f:
            annotations = json.load(f)
        self.annotations = annotations
        self.data = []
        self.train_data = self.annotations['training']
        # self.test_data = self.annotations['test']

        self.train_img_list = sorted([
            osp.join(self.root, data['image'][2:]) for data in self.train_data
        ])

        self.train_seg_list = sorted([
            osp.join(self.root, data['label'][2:]) for data in self.train_data
        ])

        # SELF.OUT: self.annotations, self.train_data, self.train_img_list, self.train_seg_list

    def _make_sub_volume_list(self):
        if self.mode == 'train':

            train_split_list = self.train_img_list[:self.split_id]
            # list_IDsT2 = list_IDsT2[:split_id]
            label_split_list = self.train_seg_list[:self.split_id]

            self.list = create_sub_volumes(train_split_list, label_split_list, dataset_name="pancreas",
                                           mode=self.mode, samples=self.samples, full_vol_dim=self.full_vol_dim,
                                           crop_size=self.crop_size,
                                           sub_vol_path=self.sub_vol_path, th_percent=self.threshold,
                                           normalization=self.normalization)


        elif self.mode == 'val':
            # utils.make_dirs(self.sub_vol_path)
            train_split_list = self.train_img_list[self.split_id:]
            label_split_list = self.train_seg_list[self.split_id:]
            self.list = create_sub_volumes(train_split_list, label_split_list, dataset_name="pancreas",
                                           mode=self.mode, samples=self.samples, full_vol_dim=self.full_vol_dim,
                                           crop_size=self.crop_size,
                                           sub_vol_path=self.sub_vol_path, th_percent=self.threshold,
                                           normalization=self.normalization)

            # self.full_volume = get_viz_set(list_IDsT1, list_IDsT2, labels, dataset_name="iseg2017")
            # SELF.OUT = self.list


                
            
            



