import os,sys
sys.path.append('..')
sys.path.append('../libs')
from libs.cls_data_loader import get_iterators
from libs.utils import getLogger
from trainVal import train
import torch
import argparse
from models.resnet import ResNet50,ResNet18,ResNet34,ResNet14
from datetime import datetime

if __name__=='__main__':    
    ############## config ############## 
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--num_points',type=int,default=4096)
    parser.add_argument('--voxel_size', type=float, default=0.02)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--use_color',type=bool,default=False)
    parser.add_argument('--local',type=bool,default=True)
    
    # model
    parser.add_argument('--num_labels',default=21,type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--sgd_lr', default=1e-1, type=float)
    parser.add_argument('--adam_lr', default=1e-3, type=float)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--optimizer',type=str,default='Adam')
    parser.add_argument('--freq_lr',type=int,default=3e3)
    
    # logs
    parser.add_argument('--dump_dir',type=str,default='logs/res34')
    parser.add_argument('--max_iter', type=int, default=9000)
    parser.add_argument('--val_freq', type=int, default=30)
    parser.add_argument('--weights', type=str, default='model.pth')
    parser.add_argument('--restore', type=bool, default=True)
    
    config = vars(parser.parse_args())
    c_time=datetime.now()
    c_time=c_time.strftime('%m%d%H%M%S')
    input_plane=3 if config['use_color'] else 1
    config['dump_dir']=os.path.join('logs',str(config['num_points'])+'_'+str(input_plane))
    if not os.path.exists(config['dump_dir']):
        os.makedirs(config['dump_dir'])
    
    torch.backends.cudnn.benchmark=True
    # dataset
    if(config['local']):
        path_train='' 
        path_val='' 
    else:
        path_train=''
        path_val=''
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # store settings
    path_log=os.path.join(config['dump_dir'],'train.log')
    logger=getLogger(path_log)
    logger.info(config)
    
    # model 
    net=ResNet14(input_plane,config['num_labels'], D=3)        
    net.to(device)
    logger.info(net)
    logger.info('#parameters %d' % sum([x.nelement() for x in net.parameters()]))
    
    # train/val loader
    loaders=get_iterators(path_train,path_val,config)
    
    # train
    train(net,loaders,device,logger,config)