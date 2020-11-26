import torch
import torch.utils.data
import torch.optim as optim
from tensorboardX import SummaryWriter
import os,sys
import MinkowskiEngine as ME
import time
import numpy as np

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



def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

# adjust learning rate by 
def adjust_learning_rate(optimizer):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    c_lr = get_lr(optimizer)
    for param_group in optimizer.param_groups:
        param_group['lr'] = c_lr*0.1

def train(net, loaders, device, logger, config):
    # optimizer
    if(config['optimizer']=='SGD'):
        optimizer = optim.SGD(
            net.parameters(),
            lr=config['sgd_lr'],
            momentum=config['momentum'],
            weight_decay=config['weight_decay'])
    elif(config['optimizer']=='Adam'):
        optimizer=optim.Adam(net.parameters(),lr=config['adam_lr'])
    
    # loss
    criterion = torch.nn.CrossEntropyLoss()
    path_checkpoint=os.path.join(config['dump_dir'],'model.pth')
    
    # train and val loader
    train_loader=iter(loaders['train'])
    val_loader=loaders['val']
    curr_val_acc=0
    ###########################
    ## training
    ###########################
    net.train()
    train_loss = 0.0
    train_pred = []
    train_true = []
    start=time.time()
    for curr_iter in range(1,config['max_iter']):
        # train on one batch and optimize
        data_dict = train_loader.next()
        optimizer.zero_grad()
        sin = ME.SparseTensor(data_dict['feats'],
                              data_dict['coords'].int()).to(device)
        sout = net(sin)
        loss = criterion(sout.F, data_dict['labels'].to(device))
        loss.backward()
        optimizer.step()
        
        # update train stats
        train_loss+=loss.item()
        preds=torch.argmax(sout.F.detach(), 1).cpu().numpy()
        gt=data_dict['labels'].cpu().numpy()
        train_true.append(gt)
        train_pred.append(preds)
        
        torch.cuda.empty_cache()
        ###########################
        ## validation
        ###########################
        if curr_iter % config['val_freq']==0:
            end=time.time()
            ### evaluate
            net.eval()
            test_loss = 0.0
            test_pred = []
            test_true = []
            n_iters=0
            for data_dict in val_loader:
                sin = ME.SparseTensor(data_dict['feats'],
                              data_dict['coords'].int()).to(device)
                sout = net(sin)
                loss = criterion(sout.F, data_dict['labels'].to(device))
                
                # update val stats
                preds=torch.argmax(sout.F.detach(), 1).cpu().numpy()
                gt=data_dict['labels'].cpu().numpy()
                test_true.append(gt)
                test_pred.append(preds)
                test_loss+=loss.item()
                n_iters+=1
                
            train_true = np.concatenate(train_true)
            train_pred=np.concatenate(train_pred)
            iou,recall,acc=get_IoU_Recall_Acc(train_true,train_pred,config['num_labels'])
            loss=train_loss/config['val_freq']
            c_epoch=curr_iter//config['val_freq']
            outstr = 'Train %d, loss: %.3f, iou: %.3f, recall: %.3f, acc: %.3f' % (
                c_epoch,loss,iou,recall,acc
            )
            logger.info(outstr)
            
            test_true = np.concatenate(test_true)
            test_pred = np.concatenate(test_pred)
            iou,recall,acc=get_IoU_Recall_Acc(test_true,test_pred,config['num_labels'])
            loss=test_loss/n_iters
            outstr = 'Val %d, loss: %.3f, iou: %.3f, recall: %.3f, acc: %.3f' % (
                c_epoch,loss,iou,recall,acc
            )
            logger.info(outstr)
            logger.info('Time: %d s' % (end-start))
            
            ### update checkpoint
            if(acc>curr_val_acc):
                curr_val_acc=acc
                torch.save({
                    'state_dict': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'restore_iter': curr_iter,
                    'curr_val_acc':curr_val_acc
                }, path_checkpoint)
                logger.info('model updated, validation acc %.2f' % curr_val_acc)
            
            train_loss = 0.0
            train_pred = []
            train_true = []
            start=time.time()
            net.train()