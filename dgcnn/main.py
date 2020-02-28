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

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
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


def _init_():
    if not os.path.exists('logs'):
        os.makedirs('logs')
    if not os.path.exists('logs/pointnet/'+args.exp_name):
        os.makedirs('logs/pointnet/'+args.exp_name)

def train(args, io):
    # transformations to preprocess the sample
    transform = transforms.Compose(
            [
                d_utils.PointcloudToTensor(),
                d_utils.PointcloudScale(),
                d_utils.PointcloudRotate(),
                d_utils.PointcloudRotatePerturbation(),
                d_utils.PointcloudTranslate(),
                d_utils.PointcloudJitter(),
                d_utils.PointcloudRandomInputDropout(),
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
        train_set=ScanNet(train_files,transforms=transform,color=True)
        val_set = ScanNet(val_files,color=True)
    else:
        train_set=ScanNet(train_files,transforms=transform)
        val_set = ScanNet(val_files)
        
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )


    device = torch.device("cuda" if args.cuda else "cpu")

    # load models
    if args.model == 'pointnet':
        model = PointNet(args).to(device)
    elif args.model == 'dgcnn':
        model = DGCNN(args).to(device)
    else:
        raise Exception("Not implemented")
    model=nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    
    
    # optimizer 
    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        
        
    # scheduler and loss
    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    # training
    best_test_acc = 0.0
    for epoch in range(args.epochs):
        start=time.time()
        scheduler.step()
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
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()
            logits = model(data)
            loss = criterion(logits, label)
            loss.backward()
            opt.step()
            
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
                data = data.permute(0, 2, 1)
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
            best_test_acc = acc
            torch.save(model.state_dict(), 'logs/pointnet/%s/model.t7' % args.exp_name)


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
    
    
    parser.add_argument('--model', type=str, default='pointnet', metavar='N',
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

    _init_()

    io = IOStream('logs/dgcnn/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')
    
    train(args,io)