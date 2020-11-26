import os,sys
from libs.seg_data_loader import get_iterators,get_val_loader
from libs.utils import getLogger
from trainVal import train,validate
import torch
import argparse
from models.res16unet import Res16UNet34C

if __name__=='__main__':
    path_train='/net/pf-pc27/scratch3/scannet/dataset/train/*.pth' 
    path_val='/net/pf-pc27/scratch3/scannet/dataset/val/*.pth' 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ############## config ############## 
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--voxel_size', type=float, default=0.02)
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--num_points',type=int,default=99999)
    parser.add_argument('--use_color',type=bool,default=True)
    
    # model
    parser.add_argument('--num_labels',default=21,type=int)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--val_batch_size',default=1,type=int)
    parser.add_argument('--iter_size', type = int, default=4)
    
    # optimizer
    parser.add_argument('--optimizer',type=str,default='SGD')
    parser.add_argument('--sgd_lr', default=1e-1, type=float)
    parser.add_argument('--adam_lr', default=5e-4, type=float)
    parser.add_argument('--sgd_momentum', type=float, default=0.9)
    parser.add_argument('--sgd_dampening', type=float, default=0.1)
    parser.add_argument('--adam_beta1', type=float, default=0.9)
    parser.add_argument('--adam_beta2', type=float, default=0.999)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--poly_power', type=float, default=0.9)

    parser.add_argument('--freq_step_lr',default=2e3,type=int)
    parser.add_argument('--lr_scheduler',type=str,default='PolyLR')
    parser.add_argument('--bn_momentum', type=float, default=0.02)
    parser.add_argument('--max_iter', type=int, default=120000)
    parser.add_argument('--conv1_kernel_size', type=int, default=3, help='First layer conv kernel size')
   
    # logs
    parser.add_argument('--dump_dir',type=str,default='logs')
    parser.add_argument('--val_freq', type=int, default=1e3)
    parser.add_argument('--restore', type=bool, default=True)
    
    config = vars(parser.parse_args())
    
    ##################################################
    # train
    ##################################################
    # store settings
    config['dump_dir']=os.path.join('logs',str(config['num_points']))
    if not os.path.exists(config['dump_dir']):
        os.makedirs(config['dump_dir'])
    path_log=os.path.join(config['dump_dir'],'train.log')
    logger=getLogger(path_log)
    logger.info(config)
    
    # model 
    net = Res16UNet34C(3,config['num_labels'],config,D=3)
    net.to(device)
    logger.info(net)
    logger.info('#parameters %d' % sum([x.nelement() for x in net.parameters()]))
    
    # train/val loader
    loaders=get_iterators(path_train,path_val,config)
    
    # train
    train(net,loaders,device,logger,config)