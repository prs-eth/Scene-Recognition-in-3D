import torch
import torch.utils.data
import torch.optim as optim
from tensorboardX import SummaryWriter
import os,sys
import MinkowskiEngine as ME
import time
from sklearn.metrics import jaccard_score,accuracy_score
import numpy as np
import pandas as pd
sys.path.append('..')
from libs.solvers import PolyLR
from libs.metrics import get_IoU, get_Recall,get_Acc
from tqdm import tqdm

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


def init_stats():
    '''
    stats used for evaluation
    '''
    stats={}
    stats['sem_iou']=0
    stats['sem_loss']=0
    stats['sem_k_iou']=[]
    stats['sem_acc']=0
    
    stats['clf_correct']=0
    stats['clf_loss']=0
    stats['num_samples']=0
    return stats

# this works for one parameter group
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
    ######################
    # optimizer
    ######################
    if(config['optimizer']=='SGD'):
        optimizer= optim.SGD(net.parameters(),lr=config['sgd_lr'],momentum=config['sgd_momentum'],
                       dampening=config['sgd_dampening'],weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'Adam':
        optimizer= optim.Adam(net.parameters(),lr=config['adam_lr'],
                        betas=(config['adam_beta1'], config['adam_beta2']),weight_decay=config['weight_decay'])

    ######################
    # loss
    ######################
    sem_criterion = torch.nn.CrossEntropyLoss(ignore_index=config['sem_num_labels'])
    clf_criterion=torch.nn.CrossEntropyLoss()
    
    ######################
    # restore model
    ######################
    writer = SummaryWriter(log_dir=config['dump_dir'])
    restore_iter = 1
    curr_best_metric=0
    path_checkpoint=config['pretrained_weights']
    path_new_model=os.path.join(config['dump_dir'],config['checkpoint'])
    if os.path.exists(path_checkpoint) and config['restore']:
        checkpoint = torch.load(path_checkpoint)  # checkpoint
        pretrained_dict=checkpoint['state_dict']
        # update pretrained layers
        model_dict=net.state_dict()
        
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        net.load_state_dict(model_dict)
        
    #######################
    # train and val loader
    #######################
    train_loader=iter(loaders['train'])
    val_loader=loaders['val']
    
    ###########################
    ## training
    ###########################
    net.train()    
    start=time.time()
    stats_train=init_stats()
    for curr_iter in range(restore_iter, config['max_iter']):  
        try:
            # train on one batch and optimize
            data_dict = train_loader.next()
            optimizer.zero_grad()
            sin = ME.SparseTensor(data_dict['feats'],
                                data_dict['coords'].int()).to(device)
            clf_out = net(sin)
            writer.add_scalar('lr',get_lr(optimizer),curr_iter)
            
            ###########################
            ## scene classification part
            ###########################
            loss=clf_criterion(clf_out.F,data_dict['clf_labels'].to(device))
            loss.backward()
            optimizer.step()
            # if(curr_iter == 1500):
            #     adjust_learning_rate(optimizer)
                
            
            writer.add_scalar('train/iter_loss',loss.item(),curr_iter)
            
            stats_train['clf_loss']+=loss.item()
            is_correct = data_dict['clf_labels'] == torch.argmax(clf_out.F, 1).cpu()
            stats_train['clf_correct']+=is_correct.sum().item()
            stats_train['num_samples']+=data_dict['clf_labels'].size()[0]
            
            ###########################
            ## validation
            ###########################
            if curr_iter % config['val_freq']==0:
                end=time.time()
                ### evaluate
                net.eval()
                stats_val=init_stats()
                n_iters=0
                with torch.no_grad():  # avoid out of memory problem
                    for data_dict in val_loader:
                        sin = ME.SparseTensor(data_dict['feats'],data_dict['coords'].int()).to(device)
                        clf_out = net(sin)
                        
                        ###########################
                        ## scene classification part
                        ###########################
                        loss=clf_criterion(clf_out.F,data_dict['clf_labels'].to(device))
                        stats_val['clf_loss']+=loss.item()
                        
                        is_correct = data_dict['clf_labels'] == torch.argmax(clf_out.F, 1).cpu()
                        stats_val['clf_correct']+=is_correct.sum().item()
                        stats_val['num_samples']+=data_dict['clf_labels'].size()[0]
                        
                        n_iters+=1
                
                ###########################
                ## scene stats
                ###########################
                writer.add_scalar('train/clf_loss',stats_train['clf_loss']/config['val_freq'],curr_iter)
                writer.add_scalar('train/clf_acc',stats_train['clf_correct']/stats_train['num_samples'],curr_iter)
                writer.add_scalar('validate/clf_loss',stats_val['clf_loss']/n_iters,curr_iter)
                writer.add_scalar('validate/clf_acc',stats_val['clf_correct']/stats_val['num_samples'],curr_iter)
                
                
                logger.info('Iter: %d, time: %d s' % (curr_iter,end-start))   
                
                logger.info('Train: clf_acc: %.3f, clf_loss: %.3f'%
                            (stats_train['clf_correct']/stats_train['num_samples'],
                                stats_train['clf_loss']/config['val_freq']))
                
                logger.info('Val  : clf_acc: %.3f, clf_loss: %.3f' % 
                            (stats_val['clf_correct']/stats_val['num_samples'],
                                stats_val['clf_loss']/n_iters))
                
                ### update checkpoint
                val_acc=stats_val['clf_correct']/stats_val['num_samples']
                c_metric=val_acc
                if(c_metric>curr_best_metric):
                    curr_best_metric=c_metric
                    torch.save({
                        'state_dict': net.state_dict()
                    }, path_new_model)
                    logger.info('---------- model updated, best metric: %.3f ----------' % curr_best_metric)
                
                stats_train=init_stats()
                start=time.time()
                net.train()
        except:
            print('sth wrong')
            

def test(net, loader, device,config):    
    ######################
    # restore model
    ######################
    path_checkpoint=os.path.join(config['dump_dir'],config['checkpoint'])
    if os.path.exists(path_checkpoint) and config['restore']:
        checkpoint = torch.load(path_checkpoint)  # checkpoint
        net.load_state_dict(checkpoint['state_dict'])
    
    
    ###########################
    ## test
    ###########################
    predictions,scene_names=[],[]
    net.eval()
    with torch.no_grad():  # avoid out of memory problem
        for data_dict in loader:
            sin = ME.SparseTensor(data_dict['feats'],data_dict['coords'].int()).to(device)
            clf_out = net(sin)
            
            ###########################
            ## scene classification part
            ###########################
            preds = torch.argmax(clf_out.F, 1).cpu().tolist()
            scene_name=data_dict['scene_names']
            predictions.extend(preds)
            scene_names.extend(scene_name)
            print(preds)
    
    remapper=np.ones(14)*(100)
    for i,x in enumerate([1,2,3,4,8,9,13,14,15,16,18,20,21]):
        remapper[i]=x
    # write to the prediction file
    f=open(os.path.join(config['dump_dir'],'prediction.txt'),'w')
    for i in range(len(scene_names)):
        scene_name=scene_names[i]
        predict=remapper[predictions[i]]
        f.write('%s %d\n' % (scene_name,predict))
    f.close()
            

def validate(net,loader,device,config):
    ######################
    # restore model
    ######################
    path_checkpoint=os.path.join(config['dump_dir'],config['checkpoint'])
    if os.path.exists(path_checkpoint) and config['restore']:
        checkpoint = torch.load(path_checkpoint)  # checkpoint
        net.load_state_dict(checkpoint['state_dict'])
    
    
    ###########################
    ## validate
    ###########################
    predictions,scene_names=[],[]
    gts=[]
    pred_histos=[]
    net.eval()
    with torch.no_grad():  # avoid out of memory problem
        for data_dict in tqdm(loader):
            sin = ME.SparseTensor(data_dict['feats'],data_dict['coords'].int()).to(device)
            clf_out = net(sin)
            
            ###########################
            ## scene classification part
            ###########################
            pred_histos.append(np.array(clf_out.F.cpu()))
            preds = torch.argmax(clf_out.F, 1).cpu().tolist()
            scene_name=data_dict['scene_names']
            predictions.extend(preds)
            scene_names.extend(scene_name)
            gts.extend(data_dict['clf_labels'].tolist())
    
    # iou,recall,acc=get_IoU_Recall_Acc(np.array(gts),np.array(predictions),21)
    # print(iou,recall,acc)
    df=pd.DataFrame(columns=['scene_names','prediction','ground_truth'],
                    data=np.array([scene_names,predictions,gts]).T)
    df.to_csv(os.path.join(config['dump_dir'],config['savefile']+'.csv'),index=False)
    # np.savez(os.path.join(config['dump_dir'],config['savefile']+'.npz'),
    #          scene_names=scene_names,histos=np.array(pred_histos))
