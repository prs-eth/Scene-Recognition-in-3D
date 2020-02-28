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

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()

EPSILON=sys.float_info.epsilon
TYPE_VALID_CLASS_IDS=[1,2,3,4,8,9,13,14,15,16,18,20,21]
TYPE_VALID_CLASS_IDS=[ele-1 for ele in TYPE_VALID_CLASS_IDS]
def get_IoU(gt,pred,num_labels):
    assert pred.shape == gt.shape
    idxs=gt<num_labels
    
    n_correct=(gt[idxs]==pred[idxs]).sum()
    n_samples=idxs.sum()
    acc=round(n_correct*1.0/n_samples,3)
    
    confusion_matrix=np.bincount(pred[idxs]*num_labels+gt[idxs],
                                 minlength=num_labels**2).reshape((num_labels,num_labels)).astype(np.ulonglong)+EPSILON
    IoU=np.around(
        confusion_matrix.diagonal()/(confusion_matrix.sum(0)+confusion_matrix.sum(1)-confusion_matrix.diagonal()),decimals=4)
    recall=np.around(
        confusion_matrix.diagonal()/confusion_matrix.sum(0),decimals=4)
    
    recall=recall[TYPE_VALID_CLASS_IDS]
    IoU=IoU[TYPE_VALID_CLASS_IDS]
    return IoU

def get_IoU_Recall_Acc(gt,pred,num_labels):
    assert pred.shape == gt.shape
    idxs=gt<num_labels
    
    n_correct=(gt[idxs]==pred[idxs]).sum()
    n_samples=idxs.sum()
    acc=round(n_correct*1.0/n_samples,3)
    
    confusion_matrix=np.bincount(pred[idxs]*num_labels+gt[idxs],
                                 minlength=num_labels**2).reshape((num_labels,num_labels)).astype(np.ulonglong)+EPSILON
    IoU=np.around(
        confusion_matrix.diagonal()/(confusion_matrix.sum(0)+confusion_matrix.sum(1)-confusion_matrix.diagonal()),decimals=4)
    recall=np.around(
        confusion_matrix.diagonal()/confusion_matrix.sum(0),decimals=4)
    
    recall=recall[TYPE_VALID_CLASS_IDS]
    IoU=IoU[TYPE_VALID_CLASS_IDS]
    return round(IoU.mean(),3),round(recall.mean(),3), acc

def get_recall(gt,pred,num_labels):
    assert pred.shape == gt.shape
    idxs=gt<num_labels
    
    n_correct=(gt[idxs]==pred[idxs]).sum()
    n_samples=idxs.sum()
    acc=round(n_correct*1.0/n_samples,3)
    
    confusion_matrix=np.bincount(pred[idxs]*num_labels+gt[idxs],
                                 minlength=num_labels**2).reshape((num_labels,num_labels)).astype(np.ulonglong)+EPSILON
    recall=np.around(
        confusion_matrix.diagonal()/confusion_matrix.sum(0),decimals=4)
    
    recall=recall[TYPE_VALID_CLASS_IDS]
    return recall

def get_recall(gt,pred,num_labels):
    assert pred.shape == gt.shape
    idxs=gt<num_labels
    
    n_correct=(gt[idxs]==pred[idxs]).sum()
    n_samples=idxs.sum()
    acc=round(n_correct*1.0/n_samples,3)
    
    confusion_matrix=np.bincount(pred[idxs]*num_labels+gt[idxs],
                                 minlength=num_labels**2).reshape((num_labels,num_labels)).astype(np.ulonglong)+EPSILON
    recall=np.around(
        confusion_matrix.diagonal()/confusion_matrix.sum(0),decimals=4)
    
    recall=recall[TYPE_VALID_CLASS_IDS]
    return recall



def parse_args():
    parser = argparse.ArgumentParser(
        description="Arguments for cls training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Training settings
    parser.add_argument('--local',type=bool,default=True)
    parser.add_argument('--exp_name', type=str, default='0/xyz', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--batch_size', type=int, default=20, metavar='batch_size',
                        help='Size of batch)')
    
    parser.add_argument('--add_color',type=bool,default=False)  
    parser.add_argument('--input_plane',type=int,default=0)

    
    parser.add_argument('--output_plane',type=int,default=21)  

    parser.add_argument(
        "--num_points", type=int, default=0, help="Number of points to train with"
    )
    parser.add_argument(
        "-weight_decay", type=float, default=1e-5, help="L2 regularization coeff"
    )
    parser.add_argument("-lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument(
        "-lr_decay", type=float, default=0.7, help="Learning rate decay gamma"
    )
    parser.add_argument(
        "-decay_step", type=float, default=2e5, help="Learning rate decay step"
    )
    parser.add_argument(
        "-bn_momentum", type=float, default=0.5, help="Initial batch norm momentum"
    )
    parser.add_argument(
        "-bnm_decay", type=float, default=0.5, help="Batch norm momentum decay gamma"
    )
    parser.add_argument(
        "-epochs", type=int, default=300, help="Number of epochs to train for"
    )

    return parser.parse_args()


lr_clip = 1e-5
bnm_clip = 1e-2

if __name__ == "__main__":
    args = parse_args()
    args.cuda = torch.cuda.is_available()
    
    # load data
    if(args.local):
        base_train='/net/pf-pc27/scratch3/scannet/dataset/train'
        base_val='/net/pf-pc27/scratch3/scannet/dataset/val'

    else:
        base_train='/cluster/work/igp_psr/shenhuan/data/dataset/train'
        base_val='/cluster/work/igp_psr/shenhuan/data/dataset/val'
    
    train_files=glob.glob(base_train+'/*.pth')
    val_files=glob.glob(base_val+'/*.pth')
    
    if(args.add_color):
        val_set = ScanNet(val_files,args.num_points,color=True)
    else:
        val_set = ScanNet(val_files,args.num_points)

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False
    )

    # model
    device = torch.device("cuda" if args.cuda else "cpu")
    model = Pointnet(input_channels=args.input_plane, num_classes=args.output_plane, use_xyz=True)
    model.to(device)
    model=nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    
    # restore model
    path_checkpoint=os.path.join('logs',args.exp_name,'model.t7')
    checkpoint = torch.load(path_checkpoint)  # checkpoint
    model.load_state_dict(checkpoint) 
    
    ####################
    # Validate
    ####################
    model.eval()
    test_pred = []
    test_true = []
    
    with torch.no_grad():  # avoid out of memory problem
        for data, label in val_loader:
            data, label = data.to(device), label.to(device).squeeze()
            logits = model(data)
            logits=logits.detach()
            preds =torch.argmax(logits,1)
            
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.cpu().numpy())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    
    iou,recall,acc=get_IoU_Recall_Acc(test_true,test_pred,args.output_plane)
    recall=get_recall(test_true,test_pred,args.output_plane)
    IOU=get_IoU(test_true,test_pred,args.output_plane)
    print(acc)
    print(IOU)
