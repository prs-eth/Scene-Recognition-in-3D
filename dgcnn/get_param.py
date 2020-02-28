#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: main.py
@Time: 2018/10/13 10:39 PM
"""

from __future__ import print_function
import os,sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from ScannetLoader import ScanNet
from model import PointNet, DGCNN
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream
from torchvision import transforms
import data_utils as d_utils
import glob
import time
    

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--local',type=bool,default=False)
    parser.add_argument('--exp_name', type=str, default='color', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--batch_size', type=int, default=18, metavar='batch_size',
                        help='Size of batch)')
    
    parser.add_argument('--add_color',type=bool,default=True)  
    parser.add_argument('--input_plane',type=int,default=6)
    
    
    parser.add_argument('--output_plane',type=int,default=21)
    parser.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='number of episode to train ')
  
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['pointnet', 'dgcnn'],
                        help='Model to use, [pointnet, dgcnn]')
    parser.add_argument('--use_sgd', type=bool, default=False,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--num_points', type=int, default=4096,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=40, metavar='N',
                        help='Num of nearest neighbors to use')
    args = parser.parse_args()
    # load models
    if args.model == 'pointnet':
        model = PointNet(args)
    elif args.model == 'dgcnn':
        model = DGCNN(args)
        
    print('#parameters %d' % sum([x.nelement() for x in model.parameters()]))