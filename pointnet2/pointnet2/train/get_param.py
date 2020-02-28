from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import numpy as np
import time
import glob
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import etw_pytorch_utils as pt_utils
import pprint
import os.path as osp
import os,sys
import argparse

sys.path.append('../..')
from pointnet2.models import Pointnet2ClsMSG as Pointnet
from pointnet2.models.pointnet2_msg_cls import model_fn_decorator
from pointnet2.data import ScanNet
import pointnet2.data.data_utils as d_utils

if __name__ == "__main__":
    # model
    model = Pointnet(input_channels=3, num_classes=21, use_xyz=True)
    print('#parameters %d' % sum([x.nelement() for x in model.parameters()]))