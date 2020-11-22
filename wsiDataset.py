import numpy as np
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import torch
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import torchvision.utils
import random
from torch.autograd import Variable
import torch.nn as nn
import torchvision.models as models
from torch.optim import lr_scheduler
import pretrainedmodels
import joblib
from numpy.random import seed


class NetworkDataset(Dataset):
    def __init__(self,images,masks,transform=None):

        self.images = images
        self.masks = masks
        self.transform = transform
        
    def __getitem__(self,index):
        image, label= self.images[index],self.masks[index]
        image = Image.open(image).convert('RGB')
        label = joblib.load(label)
        label = 1-label[0:1,:,:]
        label = torch.from_numpy(label)
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.images)


def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path) 
        print (path+' 创建成功')
        return True
    else:
        print (path+' 目录已存在')
        return False

def convert(train):
    path=[]
    for i in range(len(train)):
        path.append(train[i].replace("_img_","_mask_").replace("patch","patch_oh")[:-3]+"pkl")
    return path
    

def splitdataset(fold):
    img_path = '/data3/digestpath_img_patch'
    mask_path = '/data3/digestpath_mask_patch_oh'
    dictionary=joblib.load("../dictionary.pkl")
    wsi = []
    wsi_label = []
    for level in os.listdir(img_path):
        for image in os.listdir(os.path.join(img_path,level)):
            wsi.append(os.path.join(img_path,level,image)) 
            if level=="normal":
                wsi_label.append(0)
            elif level =="low level":
                wsi_label.append(1)
            else:
                wsi_label.append(2)
    kf = StratifiedKFold(n_splits=fold,shuffle=True,random_state=0)
    train_dataset=[]
    val_dataset=[]
    test_dataset=[]
    num=1
    for train_index , test_index in kf.split(wsi,wsi_label):
        print("Fold"+str(num))
        new_train_index, val_index , _ , _ = train_test_split(train_index,train_index,test_size=0.2,random_state=0)
        train_img_patch_fold=[]
        val_img_patch_fold=[]
        test_img_patch_fold=[]
        train_label_patch_fold=[]
        val_label_patch_fold=[]
        test_label_patch_fold=[]
        filter_train_patch_name=[]
        train_gjb=[]
        train_djb=[]
        train_normal=[]
        for index in list(new_train_index):
            # print(index)
            for patch_name in os.listdir(os.path.join(wsi[index])):
                if dictionary[os.path.join(wsi[index],patch_name)] == 0:
                    train_normal.append(os.path.join(wsi[index],patch_name))
                elif dictionary[os.path.join(wsi[index],patch_name)] == 1:
                    train_djb.append(os.path.join(wsi[index],patch_name))
                else:
                    train_gjb.append(os.path.join(wsi[index],patch_name))
        train_normal=random.sample(train_normal, max(len(train_gjb),len(train_djb)))
#         train_img_patch_fold=train_normal+train_djb+train_gjb
#         train_label_patch_fold=convert(train_normal)+convert(train_djb)+convert(train_gjb)
        train_img_patch_fold=train_djb+train_gjb
        train_label_patch_fold=convert(train_djb)+convert(train_gjb)
        print("Fold"+str(num)+"训练集统计：")
        print("正常图片数："+str(len(train_normal)))
        print("低级别图片数："+str(len(train_djb)))
        print("高级别图片数："+str(len(train_gjb)))
        val_gjb=[]
        val_djb=[]
        val_normal=[]
        for index in list(val_index):
            # print(index)
            for patch_name in os.listdir(os.path.join(wsi[index])):
                if dictionary[os.path.join(wsi[index],patch_name)] == 0:
                    val_normal.append(os.path.join(wsi[index],patch_name))
                elif dictionary[os.path.join(wsi[index],patch_name)] == 1:
                    val_djb.append(os.path.join(wsi[index],patch_name))
                else:
                    val_gjb.append(os.path.join(wsi[index],patch_name))
        val_normal=random.sample(val_normal, max(len(val_gjb),len(val_djb)))
#         val_img_patch_fold=val_normal+val_djb+val_gjb
#         val_label_patch_fold=convert(val_normal)+convert(val_djb)+convert(val_gjb)
        val_img_patch_fold=val_djb+val_gjb
        val_label_patch_fold=convert(val_djb)+convert(val_gjb)
        print("Fold"+str(num)+"验证集统计：")
        print("正常图片数："+str(len(val_normal)))
        print("低级别图片数："+str(len(val_djb)))
        print("高级别图片数："+str(len(val_gjb)))

        test_gjb=0
        test_djb=0
        test_normal=0
        for index in list(test_index):
            print(wsi[index])
            for patch_name in os.listdir(os.path.join(wsi[index])):
                if dictionary[os.path.join(wsi[index],patch_name)] == 0:
                    test_normal+=1
                elif dictionary[os.path.join(wsi[index],patch_name)] == 1:
                    test_djb+=1
                else:
                    test_gjb+=1
                test_img_patch_fold.append(os.path.join(wsi[index],patch_name))
        test_label_patch_fold=convert(test_img_patch_fold)
        print("Fold"+str(num)+"测试集统计：")
        print("正常图片数："+str(test_normal))
        print("低级别图片数："+str(test_djb))
        print("高级别图片数："+str(test_gjb))


        train_dataset_fold = NetworkDataset(train_img_patch_fold, 
                                train_label_patch_fold,
                                transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.7350986935007351, 0.735139274726245, 0.7351281576640633],
                                                 std=[0.19376948189891913, 0.1938536818511742, 0.19386376721220433])
                                                                          ]))
        val_dataset_fold = NetworkDataset(val_img_patch_fold, 
                               val_label_patch_fold,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.7350986935007351, 0.735139274726245, 0.7351281576640633],
                                                 std=[0.19376948189891913, 0.1938536818511742, 0.19386376721220433])
                                                                          ]))
        test_dataset_fold = NetworkDataset(test_img_patch_fold, 
                                test_label_patch_fold,
                                transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.7350986935007351, 0.735139274726245, 0.7351281576640633],
                                                 std=[0.19376948189891913, 0.1938536818511742, 0.19386376721220433])
                                                                          ]))
        train_dataset.append(train_dataset_fold)
        val_dataset.append(val_dataset_fold)
        test_dataset.append(test_img_patch_fold)
        num+=1
    return train_dataset,val_dataset,test_dataset
train_dataset,val_dataset,test_dataset = splitdataset(4)
train_dataloader = DataLoader(train_dataset[0], batch_size=8, shuffle=True,num_workers=0)
val_dataloader = DataLoader(val_dataset[0], batch_size=1, shuffle=False)
