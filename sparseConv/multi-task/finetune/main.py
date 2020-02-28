import os,sys
from data_loader import get_iterators,get_testdataset,get_valdataset
sys.path.append('../..')
from libs.utils import getLogger
from trainVal import train,test,validate
import torch
import argparse
from models.res16unet import Res16UNet34C

if __name__=='__main__':
    path_train='' 
    path_val='' 
    path_test=''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ############## config ############## 
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--voxel_size', type=float, default=0.02)
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--crop_rate',type=float,default=None)
    parser.add_argument('--leave_rate',type=float,default=None)
    parser.add_argument('--skip_rate',type=int,default=1)
    parser.add_argument('--ind_remove',type=int,default=None)
    
    # model
    parser.add_argument('--dimension',default=3,type=int)
    parser.add_argument('--in_plane',default=3,type=int)
    parser.add_argument('--sem_num_labels',default=20,type=int)
    parser.add_argument('--clf_num_labels',default=21,type=int)
    parser.add_argument('--train_batch_size', default=32, type=int)
    parser.add_argument('--val_batch_size',default=20,type=int)
    parser.add_argument('--test_batch_size',default=2,type=int)
    parser.add_argument('--val_freq', type=int, default=20)
    parser.add_argument('--train_limit_numpoint',default=650000,type=int)
    
    # optimizer
    parser.add_argument('--optimizer',type=str,default='Adam')
    parser.add_argument('--sgd_lr', default=5e-2, type=float)
    parser.add_argument('--adam_lr', default=1e-5, type=float)
    parser.add_argument('--sgd_momentum', type=float, default=0.9)
    parser.add_argument('--sgd_dampening', type=float, default=0.1)
    parser.add_argument('--adam_beta1', type=float, default=0.9)
    parser.add_argument('--adam_beta2', type=float, default=0.999)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--poly_power', type=float, default=0.9)

    parser.add_argument('--freq_step_lr',default=3e3,type=int)
    parser.add_argument('--lr_scheduler',type=str,default='PolyLR')
    parser.add_argument('--bn_momentum', type=float, default=0.02)
    parser.add_argument('--max_iter', type=int, default=120000)
    parser.add_argument('--conv1_kernel_size', type=int, default=3, help='First layer conv kernel size')
    
    
    # logs
    parser.add_argument('--savefile',type=str,default='best')
    parser.add_argument('--dump_dir',type=str,default='logs')
    parser.add_argument('--checkpoint',type=str,default='Mink16UNet34C_MultiTask.pth')
    parser.add_argument('--pretrained_weights', type=str, default='../../model_zoo/multi-task.pth')
    parser.add_argument('--restore', type=bool, default=True)
    
    config = vars(parser.parse_args())
    
    
    if not os.path.exists(config['dump_dir']):
        os.makedirs(config['dump_dir'])
    
    # store settings
    path_log=os.path.join(config['dump_dir'],'train.log')
    logger=getLogger(path_log)
    logger.info(config)
    
    # model 
    net = Res16UNet34C(config['in_plane'],config['sem_num_labels'],config,D=config['dimension'])
    net.to(device)
    logger.info(net)
    logger.info('#parameters %d' % sum([x.nelement() for x in net.parameters()]))
    
    # train/val loader
    loaders=get_iterators(path_train,path_val,config)
    train(net,loaders,device,logger,config)
    
    # val_loader=get_valdataset(path_val,config)
    # validate(net,val_loader,device,config)
    
