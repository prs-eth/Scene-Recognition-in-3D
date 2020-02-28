import torch
import torch.nn as nn

import MinkowskiEngine as ME
from MinkowskiEngine.modules.resnet_block import BasicBlock, Bottleneck


class VGGNet(nn.Module):
    BLOCK = None
    LAYERS = ()
    INIT_DIM = 64
    PLANES = (64, 128, 256, 512)

    def __init__(self, in_channels, out_channels, D=3):
        nn.Module.__init__(self)
        self.D = D

        self.network_initialization(in_channels, out_channels, D)
        self.weight_initialization()

    def network_initialization(self, in_channels, out_channels, D):
        self.conv1=ME.MinkowskiConvolution(in_channels,8,kernel_size=3,dimension=D)
        self.conv2=ME.MinkowskiConvolution(8,8,kernel_size=3,dimension=D)
        
        self.pool=ME.MinkowskiMaxPooling(kernel_size=3,stride=2,dimension=D)
        
        self.conv3=ME.MinkowskiConvolution(8,16,kernel_size=3,dimension=D)
        self.conv4=ME.MinkowskiConvolution(16,16,kernel_size=3,dimension=D)
        
        
        
        
    

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode='fan_out', nonlinearity='relu')

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def _conv_block(self,m,n):
        

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)

        x = self.glob_avg(x)
        return self.final(x)
