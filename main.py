import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import sys
import pickle
import random
import logging
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import time
############################################
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import transforms
from albumentations import Compose, ShiftScaleRotate, Resize, CenterCrop, HorizontalFlip, RandomBrightnessContrast,Normalize
from albumentations.pytorch import ToTensor
############################################
from Unet import U_Net,R2U_Net,AttU_Net,R2AttU_Net
from modeling.deeplab import *
from unet import UNet
from dataset import DatasetMR, DatasetCOCO
from MultiClassDiceLoss import *
from cfg import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--validate', dest='validate', action="store_true",
                        help='validate or train')
    parser.add_argument('-m', '--checkpoint_path', dest='checkpoint_path', type=str, default="./x.pth",
                        help='validate from this checkpoint')

    return parser.parse_args()


def make_one_hot(gt, num_classes):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [B, 1, H, W]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [B,num_classes, H, W]
    """
    #print(gt.shape)
    #gt = gt.unsqueeze(1)
    gt_onehot = torch.zeros((gt.shape[0], num_classes, gt.shape[2], gt.shape[3])).cuda()
    gt_onehot.scatter_(1, gt, 1) # dim = 1 src = 1
    #print(gt_onehot.shape)
    return gt_onehot

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs), targets)

def validate(val_loader, net, criterion, device, validate_args):
    n_classes = validate_args["n_classes"]
    net.eval()
    n_val = len(val_loader)
    ############################################
    loss_all, gts_all, predictions_all = [], [], []
    tot_dice = np.zeros(n_classes)
    tot_iou = np.zeros(n_classes)
    count = np.zeros(n_classes)
    ############################################
    with tqdm(total=n_val, desc='Validation round', unit='img', leave=False) as pbar:
        for batch in loader:
            imgs = batch["image"]
            imgs = imgs.to(device=device, dtype=torch.float32) # [B,1,H,W]
            masks = batch['mask']
            masks = masks.to(device=device, dtype=torch.long).unsqueeze(1) # [B,1,H,W]
            masks_onehot = make_one_hot(masks,n_classes) # (B,n_classes,H,W)
            gts_all.append(masks_onehot)
            ############################################
            with torch.no_grad():
                masks_pred = net(imgs) # (B,n_classes,H,W)
                loss = criterion(masks_pred, masks)
                loss_all.append(loss.item())
                predictions_all.append(masks_pred)
            ############################################
            for c in range(n_classes):
                mask_c = masks_onehot[0][c]
                mask_pred_c = masks_pred[0][c]
                if torch.max(mask_c) > 0:
                    count[c] += 1
                    dice, iou = dice_score(mask_c, mask_pred_c)
                    tot_dice[c] += dice
                    tot_iou[c] += iou
            ############################################
            pbar.update(1)

    ############################################
    for i in range(n_classes):
        tot_dice[i] /= count[i]
        tot_iou[i] /= count[i]

    return np.around(tot_dice, 8), np.around(tot_iou, 8)



def train(train_loader, net, criterion, optimizer, device, train_args):

    epochs, checkpoint_dir, checkpoint_space = train_args["epochs"], train_args["checkpoint_dir"], train_args["checkpoint_space"]
    ############################################
    net.train()
    global_step = 1
    writer = SummaryWriter(comment='Train')
    ############################################
    with tqdm(total=len(train_loader) * epochs, unit='batch', leave = False) as pbar:
        for epoch in range(epochs):
            epoch_loss = 0
            for batch in train_loader:
                imgs = batch["image"]
                masks = batch['mask']
                imgs = imgs.to(device=device, dtype=torch.float32)
                masks = masks.to(device=device, dtype=torch.long)
                ############################################
                masks_pred = net(imgs)
                loss = criterion(masks_pred, masks.squeeze(1))
                epoch_loss += loss.item()
                ############################################
                optimizer.zero_grad()
                #loss.backward(retain_graph=True)
                loss.backward()
                optimizer.step()
                ############################################
                writer.add_scalar('Loss/train', loss.item(), global_step)
                pbar.set_postfix(**{'loss': loss.item()})
                pbar.set_description("%d / %d Epochs" % (epoch, global_step % epochs))
                pbar.update(1)
                ############################################
                if global_step % checkpoint_space == 0:
                    torch.save(net.state_dict(),
                               checkpoint_dir + f'Train_Step_{global_step + 1}.pth')
                    logging.info(f'Checkpoint Step {global_step + 1} saved !')
    ############################################
    writer.close()

def main(args, cfgs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    ############################################
    class_names = cfgs["class_names"]
    n_classes = cfgs["n_classes"]
    multi_gpu = cfgs["gpus"]
    Net = cfgs["net_name"]
    image_root = cfgs["image_root"]
    epochs = cfgs["epochs"]
    batch_size_train = cfgs["batch_size_train"]
    checkpoint_dir = cfgs["checkpoint_dir"]
    checkpoint_space = cfgs["checkpoint_space"]
    ############################################
    validate_flag = args.validate
    model_load_path = args.checkpoint_path
    ############################################
    try:
        if Net == "deeplab":
            net = DeepLab(num_classes = n_classes)
        elif Net == "unet":
            net = UNet(n_channels = 3, n_classes = n_classes)
        elif Net == "R2U_Net":
            net = R2U_Net(img_ch=3,output_ch=n_classes,t=3)
        elif Net == "AttU_Net":
            net = AttU_Net(img_ch=3,output_ch=n_classes)
        elif Net == "R2AttU_Net":
            net = R2AttU_Net(img_ch=3,output_ch=n_classes,t=3)
        elif Net == "U_Net":
            net = U_Net(img_ch=3,output_ch=n_classes)
        net.to(device = device)
        if multi_gpu > 1:
            net = nn.parallel.DataParallel(net)
        ############################################
        optimizer = optim.Adam(net.parameters())
        if n_classes > 1:
            criterion = nn.CrossEntropyLoss()
            # criterion = MultiClassDiceLoss()
        else:
            criterion = nn.BCEWithLogitsLoss()
        ############################################
        # todo transformer
        transform_train = transforms.Compose([
            #transforms.Resize((307, 409)),
            transforms.ToTensor(),
        ])
        transform_val = transforms.Compose([
            #transforms.Resize((307, 409)),
            transforms.ToTensor(),
        ])
        ############################################
        spliteRate = 0.8
        imageDir = os.path.join(image_root, "digestpath_img_patch")
        #maskDir = os.path.join(image_root, "digestpath_mask_patch")
        typeNames = ["normal", "low level", "high level"]
        trainImagePaths = []
        testImagePaths = []
        for i in range(len(typeNames)):
            subDir = os.path.join(imageDir, typeNames[i])
            subjectIds = os.listdir(subDir)
            tmpIndex = int( len(subjectIds) * spliteRate)
            for subjectId in subjectIds[:tmpIndex]:
                subjectDir = os.path.join(subDir, subjectId)
                for fileNames in os.listdir(subjectDir):
                    filePath = os.path.join(subjectDir, fileName)
                    trainImagePaths.append(filePath)
            for subjectId in subjectIds[tmpIndex:]:
                subjectDir = os.path.join(subDir, subjectId)
                for fileNames in os.listdir(subjectDir):
                    filePath = os.path.join(subjectDir, fileName)
                    testImagePaths.append(filePath)
        trainMaskPaths = [p.replace("/digestpath_img_patch/", "/digestpath_mask_patch/") for p in trainImagePaths]
        testMaskPaths = [p.replace("/digestpath_img_patch/", "/digestpath_mask_patch/") for p in testImagePaths]
        ############################################
        train_set = wsiDataset(trainImagePaths, trainMaskPaths, transform_train)
        train_loader = DataLoader(train_set, batch_size=batch_size_train, num_workers=4, shuffle=True)
        val_set = wsiDataset(testImagePaths, testMaskPaths, transform_val)
        val_loader = DataLoader(val_set, batch_size=1, num_workers=4, shuffle=False)
        ############################################
        if validate_flag:
            if os.path.exists(model_load_path):
                net.load_state_dict(torch.load(model_load_path, map_location = device))
                logging.info(f'Checkpoint loaded from {model_load_path}')
                validate_args = {"n_classes":n_classes, "checkpoint_dir":checkpoint_dir, "checkpoint_space":checkpoint_space,}
                validate(val_loader, net, criterion, device, validate_args)
            else:
                logging.info(f'No such checkpoint !')
        else:
            # todo lr_scheduler
            train_args = {"epochs":epochs, "checkpoint_dir":checkpoint_dir, "checkpoint_space":checkpoint_space,}
            train(train_loader, net, criterion, optimizer, device, train_args)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), os.path.join(checkpoint_dir,'INTERRUPTED.pth'))
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)



def main_coco(args, cfgs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    ############################################
    class_names = cfgs["class_names"]
    n_classes = cfgs["n_classes"]
    multi_gpu = cfgs["gpus"]
    Net = cfgs["net_name"]
    image_root = cfgs["image_root"]
    json_file_val = cfgs["json_file_val"]
    json_file_train = cfgs["json_file_train"]
    epochs = cfgs["epochs"]
    batch_size_train = cfgs["batch_size_train"]
    checkpoint_dir = cfgs["checkpoint_dir"]
    checkpoint_space = cfgs["checkpoint_space"]
    ############################################
    validate_flag = args.validate
    model_load_path = args.checkpoint_path
    ############################################
    try:
        if Net == "deeplab":
            net = DeepLab(num_classes = n_classes)
        elif Net == "unet":
            net = UNet(n_channels = 3, n_classes = n_classes)
        elif Net == "R2U_Net":
            net = R2U_Net(img_ch=3,output_ch=n_classes,t=3)
        elif Net == "AttU_Net":
            net = AttU_Net(img_ch=3,output_ch=n_classes)
        elif Net == "R2AttU_Net":
            net = R2AttU_Net(img_ch=3,output_ch=n_classes,t=3)
        elif Net == "U_Net":
            net = U_Net(img_ch=3,output_ch=n_classes)
        net.to(device = device)
        if multi_gpu > 1:
            net = nn.parallel.DataParallel(net)
        ############################################
        optimizer = optim.Adam(net.parameters())
        if n_classes > 1:
            criterion = nn.CrossEntropyLoss()
            # criterion = MultiClassDiceLoss()
        else:
            criterion = nn.BCEWithLogitsLoss()
        ############################################
        # todo transformer
        transform_train = transforms.Compose([
            transforms.Resize((307, 409)),
            transforms.ToTensor(),
        ])
        transform_val = transforms.Compose([
            transforms.Resize((307, 409)),
            transforms.ToTensor(),
        ])
        train_set = DatasetCOCO(json_file_train, image_root, transform_train)
        train_loader = DataLoader(train_set, batch_size=batch_size_train, num_workers=4, shuffle=False)
        val_set = DatasetCOCO(json_file_val, image_root, transform_val)
        val_loader = DataLoader(val_set, batch_size=1, num_workers=4, shuffle=False)
        ############################################
        if validate_flag:
            if os.path.exists(model_load_path):
                net.load_state_dict(torch.load(model_load_path, map_location = device))
                logging.info(f'Checkpoint loaded from {model_load_path}')
                validate_args = {"n_classes":n_classes, "checkpoint_dir":checkpoint_dir, "checkpoint_space":checkpoint_space,}
                validate(val_loader, net, criterion, device, validate_args)
            else:
                logging.info(f'No such checkpoint !')
        else:
            # todo lr_scheduler
            train_args = {"epochs":epochs, "checkpoint_dir":checkpoint_dir, "checkpoint_space":checkpoint_space,}
            train(train_loader, net, criterion, optimizer, device, train_args)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), os.path.join(checkpoint_dir,'INTERRUPTED.pth'))
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    main(args, cfgs)

