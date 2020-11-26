import torch
import torch.utils.data
import torch.optim as optim
from tensorboardX import SummaryWriter
import os,sys
import MinkowskiEngine as ME
import time
from sklearn.metrics import jaccard_score
import numpy as np
from libs.solvers import PolyLR
from tqdm import tqdm


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

def validate(net,val_loader,device,config): 
    ### evaluate
    net.eval()
    results=dict()
    scene_histograms=[]
    scene_types=[]
    with torch.no_grad():  # avoid out of memory problem
        for data_dict in tqdm(val_loader):        
            sin = ME.SparseTensor(data_dict['feats'],
                        data_dict['coords'].int()).to(device)
            sout = net(sin)
            predictions=torch.argmax(sout.F, 1).cpu().numpy()
            
            sem_histogram=np.bincount(predictions.astype('int'),minlength=config['num_labels'])
            scene_histograms.append(sem_histogram/sem_histogram.sum())
            
            scene_types.append(data_dict['labels'])
    
    x=np.array(scene_histograms)
    y=np.array(scene_types)
    np.savez('results_val.npz',x=x,y=y)
    

def train(net, loaders, device, logger, config):
    # optimizer
    if(config['optimizer']=='SGD'):
        optimizer= optim.SGD(net.parameters(),lr=config['sgd_lr'],momentum=config['sgd_momentum'],
                       dampening=config['sgd_dampening'],weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'Adam':
        optimizer= optim.Adam(net.parameters(),lr=config['adam_lr'],
                        betas=(config['adam_beta1'], config['adam_beta2']),weight_decay=config['weight_decay'])
    
    scheduler=PolyLR(optimizer, max_iter=config['max_iter'], power=config['poly_power'], last_step=-1)
    
    # loss
    criterion = torch.nn.CrossEntropyLoss(ignore_index=20)
    
    # restore model
    restore_iter = 1
    curr_val_iou=0
    path_checkpoint=os.path.join(config['dump_dir'],'model.pth')
    if os.path.exists(path_checkpoint) and config['restore']:
        checkpoint = torch.load(path_checkpoint)  # checkpoint
        net.load_state_dict(checkpoint['state_dict'])
        restore_iter = checkpoint['restore_iter'] + 1
        curr_val_iou=checkpoint['curr_val_iou']
        logger.info('Restore from %d' % restore_iter)
        optimizer.load_state_dict(checkpoint['optimizer']) # restore optimizer
        scheduler.load_state_dict(checkpoint['scheduler'])
    
    # summary writer
    writer = SummaryWriter(log_dir=config['dump_dir'])
    # train and val loader
    train_loader=iter(loaders['train'])
    val_loader=loaders['val']
    
    ###########################
    ## training
    ###########################
    net.train()
    train_loss,train_score=0,0
    train_ind_score=[]
    start=time.time()
    optimizer.zero_grad()
    for curr_iter in tqdm(range(restore_iter, config['max_iter'])):   
        try:         
            # train on one batch and optimize
            data_dict = train_loader.next()
            
            coords_batch0, feats_batch0 = ME.utils.sparse_collate(data_dict['coords'], data_dict['feats'])
            sin = ME.SparseTensor(feats_batch0, coords=coords_batch0).to(device)
            sout = net(sin)
            loss = criterion(sout.F, data_dict['labels'].to(device))
            
            loss.backward()
            if(curr_iter % config['iter_size'] == 0):
                optimizer.step()
                optimizer.zero_grad()

            scheduler.step()
            # update train stats
            train_loss+=loss.item()
            gt=data_dict['labels'].cpu()
            preds=torch.argmax(sout.F, 1).cpu()
            c_score=jaccard_score(gt,preds,
                                labels=np.arange(config['num_labels']-1).tolist(),average='micro')
            train_score+=c_score
            train_ind_score.append(
                jaccard_score(gt,preds,labels=np.arange(config['num_labels']-1).tolist(),average=None))
            writer.add_scalar('train/loss', loss.item(), curr_iter)
            writer.add_scalar('train/iou',c_score,curr_iter)
            writer.add_scalar('lr',get_lr(optimizer),curr_iter)
            
            ###########################
            ## validation
            ###########################
            if curr_iter % config['val_freq']==0:
                end=time.time()
                ### evaluate
                net.eval()
                optimizer.zero_grad()
                val_loss,val_score=0,0
                val_ind_score=[]
                n_iters=0
                with torch.no_grad():  # avoid out of memory problem
                    for data_dict in val_loader:
                        coords_batch0, feats_batch0 = ME.utils.sparse_collate(data_dict['coords'], data_dict['feats'])
                        sin = ME.SparseTensor(feats_batch0, coords=coords_batch0).to(device)
                        sout = net(sin)
                        loss = criterion(sout.F, data_dict['labels'].to(device))
                            
                        # update val stats
                        val_loss+=loss.item()
                        gt=data_dict['labels'].cpu()
                        preds=torch.argmax(sout.F, 1).cpu()
                        c_score=jaccard_score(gt,preds,
                                labels=np.arange(config['num_labels']-1).tolist(),average='micro')
                        val_ind_score.append(
                            jaccard_score(gt,preds,labels=np.arange(config['num_labels']-1).tolist(),average=None))
                        val_score+=c_score
                        n_iters+=1
                    
                writer.add_scalar('validate/loss', val_loss/n_iters, curr_iter)
                writer.add_scalar('validate/iou',val_score/n_iters,curr_iter)
                logger.info('Iter: %d, train iou: %.2f, train loss: %.3f, time: %d s' % 
                            (curr_iter,train_score/config['val_freq'],train_loss/config['val_freq'],end-start))
                logger.info('Iter: %d, val   iou: %.2f, val   loss: %.3f' % 
                            (curr_iter,val_score/n_iters,val_loss/n_iters))
                
                mean_train_ind_score=np.around(np.array(train_ind_score).mean(0),decimals=2)
                mean_val_ind_score=np.around(np.array(val_ind_score).mean(0),decimals=2)
                logger.info('Train iou: \n%s' % str(mean_train_ind_score.tolist()))
                logger.info('Val   iou: \n%s' % str(mean_val_ind_score.tolist()))   
                                
                ### update checkpoint
                val_iou=val_score/n_iters
                if(val_iou>curr_val_iou):
                    curr_val_iou=val_iou
                    torch.save({
                        'state_dict': net.state_dict(),
                        'restore_iter': curr_iter,
                        'curr_val_iou':curr_val_iou,
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                    }, path_checkpoint)
                    logger.info('---------- model updated, validation iou %.2f ----------' % curr_val_iou)
                
                # scheduler.step()
                train_loss,train_score=0,0
                train_ind_score=[]
                start=time.time()
                net.train() 
        except Exception as e:
            print(e)
            torch.cuda.empty_cache()