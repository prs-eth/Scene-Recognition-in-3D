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



def parse_args():
    parser = argparse.ArgumentParser(
        description="Arguments for cls training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Training settings
    parser.add_argument('--local',type=bool,default=False)
    parser.add_argument('--exp_name', type=str, default='xyz', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--batch_size', type=int, default=18, metavar='batch_size',
                        help='Size of batch)')
    
    parser.add_argument('--add_color',type=bool,default=False)  
    parser.add_argument('--input_plane',type=int,default=0)

    
    parser.add_argument('--output_plane',type=int,default=21)  

    parser.add_argument(
        "--num_points", type=int, default=4096, help="Number of points to train with"
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
    # ios 
    if not os.path.exists('logs'):
        os.makedirs('logs')
    if not os.path.exists('logs/'+args.exp_name):
        os.makedirs('logs/'+args.exp_name)
    io = IOStream('logs/' + args.exp_name + '/run.log')
    io.cprint(str(args))
    
    # data augmentations
    transforms = transforms.Compose(
        [
            d_utils.PointcloudToTensor(),
            d_utils.PointcloudScale(),
            d_utils.PointcloudRotate(),
            d_utils.PointcloudRotatePerturbation(),
            d_utils.PointcloudTranslate(),
            d_utils.PointcloudJitter(),
            d_utils.PointcloudRandomInputDropout()
        ]
    )
    
    # load data
    if(args.local):
        base_train=''
        base_val=''

    else:
        base_train=''
        base_val=''
    
    train_files=glob.glob(base_train+'/*.pth')
    val_files=glob.glob(base_val+'/*.pth')
    
    if(args.add_color):
        train_set=ScanNet(train_files,args.num_points,transforms=transforms,color=True)
        val_set = ScanNet(val_files,args.num_points,color=True)
    else:
        train_set=ScanNet(train_files,args.num_points,transforms=transforms)
        val_set = ScanNet(val_files,args.num_points)
        
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True
    )
    

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
    
    
    # optimizer
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    
    # scheduler
    lr_lbmd = lambda it: max(
        args.lr_decay ** (int(it * args.batch_size / args.decay_step)),
        lr_clip / args.lr,
    )
    bn_lbmd = lambda it: max(
        args.bn_momentum
        * args.bnm_decay ** (int(it * args.batch_size / args.decay_step)),
        bnm_clip,
    )
    lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lambda=lr_lbmd, last_epoch=-1)
    bnm_scheduler = pt_utils.BNMomentumScheduler(
        model, bn_lambda=bn_lbmd, last_epoch=-1
    )
    
    # loss    
    criterion = torch.nn.CrossEntropyLoss()

    # training
    it=0
    best_test_acc = 0.0
    for epoch in range(args.epochs):
        start=time.time()
        lr_scheduler.step()
        bnm_scheduler.step()
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        for data, label in train_loader:
            data, label = data.to(device), label.to(device).squeeze()
            batch_size = data.size()[0]
            optimizer.zero_grad()
            logits = model(data)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            
            logits=logits.detach()
            preds =torch.argmax(logits,1)
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.cpu().numpy())
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        iou,recall,acc=get_IoU_Recall_Acc(train_true,train_pred,args.output_plane)
        loss=train_loss/count
        
        
        outstr = 'Train %d, loss: %.3f, iou: %.3f, recall: %.3f, acc: %.3f' % (
            epoch,loss,iou,recall,acc
        )
        io.cprint(outstr)
        ####################
        # Validate
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        with torch.no_grad():  # avoid out of memory problem
            for data, label in val_loader:
                data, label = data.to(device), label.to(device).squeeze()
                batch_size = data.size()[0]
                logits = model(data)
                loss = criterion(logits, label)
                
                count += batch_size
                test_loss += loss.item() * batch_size
                
                logits=logits.detach()
                preds =torch.argmax(logits,1)
                
                test_true.append(label.cpu().numpy())
                test_pred.append(preds.cpu().numpy())
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        
        iou,recall,acc=get_IoU_Recall_Acc(test_true,test_pred,args.output_plane)
        loss=test_loss/count
        
        end=time.time()
        outstr = 'Val %d, loss: %.3f, iou: %.3f, recall: %.3f, acc: %.3f' % (
            epoch,loss,iou,recall,acc
        )
        
        io.cprint(outstr)
        
        outstr='Time: %d s' % (end-start)
        io.cprint(outstr)
        
        if acc >= best_test_acc:
            io.cprint('Better model!')
            best_test_acc = acc
            torch.save(model.state_dict(), 'logs/%s/model.t7' % args.exp_name)
