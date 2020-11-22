import os, json, logging
import cv2, pydicom
import torch
import numpy as np
from glob import glob
from PIL import Image
import seaborn as sns
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from cfg import *

class DatasetCOCO(Dataset):
    def __init__(self, json_file, image_root, transform = None):
        self.image_root = image_root
        self.json_file = json_file
        self.transform = transform
        ############################################
        self.cate_num = 2
        self.coco = COCO(self.json_file)
        self.category_ids = [i for i in self.coco.getCatIds()]
        self.image_ids = list(set([ j  for cid in self.coco.getCatIds() for j in self.coco.catToImgs[cid] ]))
        self.lens = len(self.image_ids)

        logging.info(f'Creating dataset with {self.lens} examples')

    def __len__(self):
        return self.lens

    def __getitem__(self, i):

        image_id = self.image_ids[i]
        ############################################
        image_info = self.coco.loadImgs(image_id)[0]
        w, h = image_info["width"], image_info["height"]
        image_path = os.path.join(self.image_root, image_info["file_name"])
        img = Image.open(image_path).convert('RGB')
        ############################################
        annIds = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(annIds)
        ############################################
        mask = [[] for i in range(self.cate_num)]
        coeff = [i+1 for i in range(self.cate_num)]
        for ann in anns:
            mask[ann["category_id"]].append(self.coco.annToMask(ann))
        # (n_classes, h, w)
        # masks = np.array([np.max(np.array(mask[a]), 0) * coeff[a] if len(mask[a]) > 1 else np.array(mask[a]) * coeff[a] for a in range(len(mask))])
        masks = np.zeros((h,w))
        for a in range(len(mask)):
            if len(mask[a]) > 1:
                masks = np.maximum(masks, np.max(np.array(mask[a]), 0) * coeff[a] )
        masks = Image.fromarray(masks)
        ############################################
        if self.transform:
            img = self.transform(img)
            masks = self.transform(masks)

        return {'image': img, 'mask': masks }

class DatasetMR(Dataset):
    def __init__(self, dir_mr, sets, transform = None):
        self.dir_mr = dir_mr
        self.transform = transform
        self.images = []
        self.ground_files = []
        for s in sets:
            images_path = os.path.join(dir_mr,"images/")
            image_list = glob(images_path + "pat"+ str(s) +"*")
            for str1 in image_list:
                self.images.append(str1)
        for str1 in self.images:
            self.ground_files.append(str1.replace("/images/","/final_annotations_bmp/").replace(str1.split("_")[-1],"") + ".bmp")

        logging.info(f'Creating dataset with {len(self.images)} examples')

    def __len__(self):
        return len(self.images)


    def __getitem__(self, i):
        in_file = self.images[i]
        ground_file = self.ground_files[i]

        in_data = Image.open(in_file)
        mask = Image.open(ground_file)
        if self.transform:
            in_data = self.transform(in_data)
            mask = self.transform(mask)
        # (in_data.shape) (1,H,W)
        # (out_data.shape) (1,H,W)
        # (mask.shape) (H,W)
        return {'image': in_data, 'mask': mask }



class wsiDataset(Dataset):
    def __init__(self, images, masks, transform = None):
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image, mask= self.images[index],self.masks[index]
        image = Image.open(image).convert('RGB')
        #mask = Image.open(mask).convert('RGB')
        mask = Image.fromarray(np.load(mask))
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        # (image.shape) (3,H,W)
        # (mask.shape) (H,W)
        return {'image': image, 'mask': mask }
