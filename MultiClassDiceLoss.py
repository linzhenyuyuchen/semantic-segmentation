import torch
from torch import nn
from cfg import *


def dice_score(input,target): # validation
    eps = 0.0001
    input = input.contiguous()
    target = target.contiguous()
    input = torch.sigmoid(input)
    input[input >= 0.5] = 1
    input[input < 0.5] = 0
    inter = torch.dot(input.view(-1), target.view(-1))
    union = torch.sum(input) + torch.sum(target) + eps
    union2 = torch.max(input,target).sum() + eps

    iou = inter.float() / union2.float()
    dice = (2 * inter.float()) / union.float()
    return dice.item(), iou.item()

def dice_loss(input,target): # train
    eps = 0.0001
    input = input.contiguous()
    target = target.contiguous()
    input = torch.sigmoid(input)
    inter = torch.dot(input.view(-1), target.view(-1))

    union = torch.sum(input) + torch.sum(target) + eps

    dice = 1 - (2 * inter.float() + eps) / union.float()

    return dice

class MultiClassDiceLoss(nn.Module): # training
    """
    input (B, n_classes, H, W)
    target (B, H , W)
    """

    def __init__(self):
        super(MultiClassDiceLoss, self).__init__()

    def forward(self, inputs, targets, weights=None):
        # targets (B, H, W)
        targets = make_one_hot(targets,n_classes) # (B,n_classes, H, W)
        totalLoss = 0
        for i in range(n_classes):
            diceLoss = dice_loss(inputs[:, i], targets[:, i])
            if weights is not None:
                diceLoss *= weights[i]
            totalLoss += diceLoss
        totalLoss /= n_classes
        return totalLoss
