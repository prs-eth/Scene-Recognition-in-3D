import torch, numpy as np, glob, math, torch.utils.data, scipy.ndimage, multiprocessing as mp
import math
import MinkowskiEngine as ME
from torch.utils.data.sampler import Sampler
import os,sys
from libs.pc_transform import get_transform

# segmantic lable remapper
SEM_CLASS_LABELS = ('wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window','bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator','shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture')
SEM_VALID_CLASS_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
SEM_REMAPPER=np.ones(150)*(20)
for i,x in enumerate(SEM_VALID_CLASS_IDS):
    SEM_REMAPPER[x]=i

    
'''
ScanNet dataset, load numpy arrays
'''
class ScanNetDataset(torch.utils.data.Dataset):
    def __init__(self,path,num_points,use_color=False,transform=None,voxel_size=0.02):
        torch.utils.data.Dataset.__init__(self)
        
        self.voxel_size=voxel_size
        self.transform=transform
        self.num_point=num_points
        self.color=use_color
        self.files=glob.glob(path)
    
    def __getitem__(self,idx):
         # load data
        crn_path=self.files[idx]
        x=torch.load(crn_path)
        if(self.num_point!=99999):
            feats=x['%d_feats' % self.num_point]/255
            feats-=0.5
            xyz=x['%d_coords' % self.num_point]
            label=SEM_REMAPPER[x['%d_sem_label' % self.num_point].astype('int')]
        else:
            feats=x['feats']/255
            feats-=0.5
            xyz=x['coords']
            label=SEM_REMAPPER[x['sem_label'].astype('int')]
        
        scene_name=x['scene_name']
        
        # augment data
        if(self.transform is not None):
            xyz,feats,label=self.transform(xyz,feats,label)
            xyz,feats=xyz.numpy(),feats.numpy()
            
        # voxelization
        sel = ME.utils.sparse_quantize(xyz / self.voxel_size, return_index=True)
        down_xyz, down_feats,down_labels= xyz[sel], feats[sel],label[sel]
        
        # Get coords, shift to center
        coords = np.floor(down_xyz / self.voxel_size)
        coords-=coords.min(0)
        
        if(self.color):
            feats=down_feats
        else:
            feats=np.ones((down_feats.shape[0],1))
        
        return (coords,feats,down_labels,scene_name)
    
    def __len__(self):
        return len(self.files)


'''
collate data for each batch
'''
def collate_fn(list_data):
    coords, feats, labels,scene_names = list(zip(*list_data))
    labels_batch=torch.from_numpy(np.hstack(labels)).long()
    # Concatenate all lists
    return {
        'coords': coords,
        'feats': feats,
        'labels': labels_batch,
        'scene_names':scene_names
    }

class InfSampler(Sampler):
    """Samples elements randomly, without replacement.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source, shuffle=True):
        self.data_source = data_source
        self.shuffle = shuffle
        self.reset_permutation()

    def reset_permutation(self):
        perm = len(self.data_source)
        if self.shuffle:
            perm = torch.randperm(perm)
        self._perm = perm.tolist()

    def __iter__(self):
        return self

    def __next__(self):
        if len(self._perm) == 0:
            self.reset_permutation()
        return self._perm.pop()

    def __len__(self):
        return len(self.data_source)


def get_iterators(path_train,path_val,config):
    tsfm=get_transform()
    # train loader
    train_set=ScanNetDataset(path_train,config['num_points'],
                             use_color=config['use_color'],
                             transform=tsfm,
                             voxel_size=config['voxel_size'])
    train_args = {
        'batch_size': config['batch_size'],
        'num_workers': config['num_workers'],
        'collate_fn': collate_fn,
        'sampler':InfSampler(train_set),
        'pin_memory': False,
        'drop_last': False
    }
    train_loader = torch.utils.data.DataLoader(train_set, **train_args)
    
    # val loader
    val_set=ScanNetDataset(path_val,config['num_points'],
                           use_color=config['use_color'],
                           transform=None,
                           voxel_size=config['voxel_size'])
    val_args = {
        'batch_size': config['batch_size'],
        'num_workers': config['num_workers'],
        'collate_fn': collate_fn,
        'pin_memory': False,
        'drop_last': False
    }
    val_loader = torch.utils.data.DataLoader(val_set,**val_args)
    
    # return two loaders
    return {
        'train': train_loader,
        'val': val_loader
    }

def get_val_loader(path_val,config):
    # val loader
    val_set=ScanNetDataset(path_val,config['num_points'],
                           use_color=config['use_color'],
                           transform=None,
                           voxel_size=config['voxel_size'])
    val_args = {
        'batch_size': config['val_batch_size'],
        'num_workers': config['num_workers'],
        'collate_fn': collate_fn,
        'pin_memory': True,
        'drop_last': False
    }
    val_loader = torch.utils.data.DataLoader(val_set,**val_args)
    
    return val_loader