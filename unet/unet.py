import torch
import torch.nn as nn
import numpy as np

class down(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(down, self).__init__()

        self.downs = nn.Sequential(
            nn.Conv2d(n_channels, n_classes, 3, padding = 1),
            nn.Conv2d(n_classes, n_classes, 3, padding = 1),
            nn.BatchNorm2d(n_classes),
            nn.ReLU(inplace = True)
        )

    def forward(self,x):
        x = self.downs(x)
        x = self.downs(x)
        return x

class up(nn.Module):
    def __init__(self,n_channels, n_classes, y):
        super(up, self).__init__()

        self.upconv =nn.Sequential(
            nn.ConvTranspose2d(n_channels, n_classes, 2, stride = 2),
        )
        self.ups = nn.Sequential(
            nn.Conv2d(n_channels, n_classes, 3, padding = 1),
            nn.Conv2d(n_classes, n_classes, 3, padding = 1),
            nn.BatchNorm2d(n_classes),
            nn.ReLU6(inplace = True)
        )
        self.y = y
    def forward(self,x):
        x = self.upconv(x)
        # 按维数1拼接（横着拼）
        x = torch.cat((x,self.y), dim=1)
        x = self.ups(x)
        x = self.ups(x)
        return x

class unet(nn.Module):
    def __init__(self,n_channels,n_classes):
        super(unet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.mp = nn.MaxPool2d(2)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self,x):
        # down sample
        block1 = down(self.n_channels,64)
        x1_use = block1(x)
        x1_mp = self.mp(x1_use)

        block2 - down(64,128)
        x2_use = block2(x1_mp)
        x2_mp = self.mp(x2_use)

        block3 = down(128,256)
        x3_use = block3(x2_mp)
        x3_mp = self.mp(x3_use)

        block4 = down(256,512)
        x4_use = block4(x3_mp)
        x4_mp = self.mp(x4_use)

        block5 = down(512,1024)
        x5_use = block5(x4_mp)

        ## up sample

        block6 = up(1024,512,x4_use)
        x6 = block6(x5_use)

        block7 = up(512,256)
        x7 = block7(x6)

        block8 = up(256,128)
        x8 = block8(x7)

        block9 = up(128,64)
        x9 = block9(x8)

        out = nn.Conv2d(64,self.n_classes,1)
        x10 = out(x9)

        return x10
