import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
from modeling.resunet import ResUNet
from modeling.clsResUnet import clsResUnet
from modeling.deeplab import *
from unet import UNet
from dataset import *
from MultiClassDiceLoss import *
from cfg_double import *


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
    return F.one_hot(gt.squeeze(1),num_classes=num_classes).permute(0,3,1,2)
    """
    print(gt.shape)
    #gt = gt.unsqueeze(1)
    gt_onehot = torch.zeros((gt.shape[0], num_classes, gt.shape[2], gt.shape[3])).cuda()
    gt_onehot.scatter_(1, gt, 1) # dim = 1 src = 1
    print(gt_onehot.shape)
    return gt_onehot
    """

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs), targets)


def validate(loader, net, criterion, device, validate_args):
    n_classes = validate_args["n_classes"]
    net.eval()
    n_val = len(loader)
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
            masks = masks.to(device=device, dtype=torch.long) # [B,1,H,W]
            masks_onehot = make_one_hot(masks,n_classes) # (B,n_classes,H,W)
            #gts_all.append(masks_onehot)
            ############################################
            with torch.no_grad():
                masks_pred, _ = net(imgs) # (B,n_classes,H,W)
                #loss = criterion(masks_pred, masks.squeeze(1))
                #loss_all.append(loss.item())
                #predictions_all.append(masks_pred)
            ############################################
            for c in range(n_classes):
                mask_c = masks_onehot[0][c]
                mask_pred_c = masks_pred[0][c]
                if torch.max(mask_c) > 0:
                    count[c] += 1
                    dice, iou = dice_score(mask_pred_c, mask_c)
                    tot_dice[c] += dice
                    tot_iou[c] += iou
            ############################################
            pbar.update(1)

    ############################################
    for i in range(n_classes):
        tot_dice[i] /= count[i]
        tot_iou[i] /= count[i]

    diceRes, iouRes = np.around(tot_dice, 8), np.around(tot_iou, 8)
    return diceRes, iouRes



def train(train_loader, val_loader, net, criterion, optimizer, device, train_args):
    n_classes, epochs, checkpoint_dir, checkpoint_space = train_args["n_classes"], train_args["epochs"], train_args["checkpoint_dir"], train_args["checkpoint_space"]
    ############################################
    net.train()
    global_step = 1
    ############################################
    with tqdm(total=len(train_loader) * epochs, unit='batch', leave = False) as pbar:
        for epoch in range(epochs):
            pbar.set_description("%d / %d Epochs" % (epoch, global_step % epochs))
            epoch_loss = 0
            for batch in train_loader:
                imgs = batch["image"]
                masks = batch['mask']
                label = batch['label']
                imgs = imgs.to(device=device, dtype=torch.float32)
                masks = masks.to(device=device, dtype=torch.long)
                label = label.to(device=device, dtype=torch.long)
                ############################################
                masks_pred, label_pred = net(imgs)
                loss_masks = criterion(masks_pred, masks.squeeze(1))

                loss_labels = criterion(label_pred, label.squeeze(1))
                loss = 0.2*loss_labels + loss_masks
                epoch_loss += loss.item()
                ############################################
                optimizer.zero_grad()
                #loss.backward(retain_graph=True)
                loss.backward()
                optimizer.step()
                ############################################
                writer.add_scalar('Loss/train_step', loss.item(), global_step)
                pbar.set_postfix(**{'loss': loss.item()})
                pbar.update(1)
                ############################################
                global_step+=1
                if global_step % checkpoint_space == 0:
                    diceRes, iouRes = validate(val_loader, net, criterion, device, train_args)
                    for i in range(n_classes):
                        writer.add_scalar(f'Dice/valid_{i}', diceRes[i], global_step)
                        writer.add_scalar(f'iou/valid_{i}', iouRes[i], global_step)
                    logging.info(f'dice: {diceRes} !')
                    logging.info(f'iou: {iouRes} !')
                    torch.save(net.state_dict(),
                               checkpoint_dir + f'Train_Step_{global_step + 1}.pth')
                    logging.info(f'Checkpoint Step {global_step + 1} saved !')
            writer.add_scalar('Loss/train_epoch', epoch_loss/len(train_loader), epoch)
    ############################################
    writer.close()

def weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        torch.nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)

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
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    ############################################
    validate_flag = args.validate
    model_load_path = args.checkpoint_path
    ############################################
    try:
        if Net == "clsresunet":
            net = clsResUnet(3, n_classes)
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
        n_folds = 4
        imageDir = os.path.join(image_root, "digestpath_img_patch")
        #maskDir = os.path.join(image_root, "digestpath_mask_patch")
        typeNames = ["normal", "low level", "high level"]
        #typeNames = [ "high level"]
        trainImagePaths = []
        validImagePaths = []
        testImagePaths = []
        for i in range(len(typeNames)):
            print(typeNames[i])
            subDir = os.path.join(imageDir, typeNames[i])
            subjectIds = os.listdir(subDir)
            tmpIndex1 = len(subjectIds) // n_folds
            tmpIndex2 = len(subjectIds) // n_folds * 2

            for subjectId in subjectIds[tmpIndex2:]:
                subjectDir = os.path.join(subDir, subjectId)
                for fileName in os.listdir(subjectDir):
                    filePath = os.path.join(subjectDir, fileName)
                    trainImagePaths.append(filePath)
            for subjectId in subjectIds[:tmpIndex1]:
                subjectDir = os.path.join(subDir, subjectId)
                for fileName in os.listdir(subjectDir):
                    filePath = os.path.join(subjectDir, fileName)
                    validImagePaths.append(filePath)
            for subjectId in subjectIds[tmpIndex1:tmpIndex2]:
            #for subjectId in subjectIds:
                subjectDir = os.path.join(subDir, subjectId)
                for fileName in os.listdir(subjectDir):
                    filePath = os.path.join(subjectDir, fileName)
                    testImagePaths.append(filePath)
        trainMaskPaths = [p.replace("/digestpath_img_patch/", "/digestpath_mask_patch/")[:-4]+".npy" for p in trainImagePaths]
        validMaskPaths = [p.replace("/digestpath_img_patch/", "/digestpath_mask_patch/")[:-4]+".npy" for p in validImagePaths]
        testMaskPaths = [p.replace("/digestpath_img_patch/", "/digestpath_mask_patch/")[:-4]+".npy" for p in testImagePaths]
        print("train-set #", len(trainMaskPaths))
        print("valid-set #", len(validMaskPaths))
        print("test-set #", len(testMaskPaths))
        ############################################
        train_set = wsiDataset2(trainImagePaths, trainMaskPaths, transform_train)
        train_loader = DataLoader(train_set, batch_size=batch_size_train, num_workers=4, shuffle=True)
        val_set = wsiDataset2(validImagePaths, validMaskPaths, transform_val)
        val_loader = DataLoader(val_set, batch_size=1, num_workers=4, shuffle=False)
        test_set = wsiDataset2(testImagePaths, testMaskPaths, transform_val)
        test_loader = DataLoader(test_set, batch_size=1, num_workers=4, shuffle=False)
        ############################################
        if validate_flag:
            if os.path.exists(model_load_path):
                net.load_state_dict(torch.load(model_load_path, map_location = device))
                logging.info(f'Checkpoint loaded from {model_load_path}')
                validate_args = {"n_classes":n_classes, "checkpoint_dir":checkpoint_dir, "checkpoint_space":checkpoint_space,}
                diceRes, iouRes = validate(test_loader, net, criterion, device, validate_args)
                logging.info(f'Test-dataset dice: {diceRes} !')
                logging.info(f'Test-dataset iou: {iouRes} !')
            else:
                logging.info(f'No such checkpoint !')
        else:
            # todo lr_scheduler
            train_args = {"n_classes":n_classes, "epochs":epochs, "checkpoint_dir":checkpoint_dir, "checkpoint_space":checkpoint_space,}
            train(train_loader, val_loader, net, criterion, optimizer, device, train_args)
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
    writer = SummaryWriter(comment=f"Train_{cfgs['net_name']}")
    main(args, cfgs)

