import torch, numpy as np, glob, math, torch.utils.data, scipy.ndimage, multiprocessing as mp
import math
import MinkowskiEngine as ME
from torch.utils.data.sampler import Sampler
import os,sys

MAX_POINTS=3000000

SEM_COLOR_MAP = {
    0: (0., 0., 0.),
    1: (174., 199., 232.),
    2: (152., 223., 138.),
    3: (31., 119., 180.),
    4: (255., 187., 120.),
    5: (188., 189., 34.),
    6: (140., 86., 75.),
    7: (255., 152., 150.),
    8: (214., 39., 40.),
    9: (197., 176., 213.),
    10: (148., 103., 189.),
    11: (196., 156., 148.),
    12: (23., 190., 207.),
    14: (247., 182., 210.),
    15: (66., 188., 102.),
    16: (219., 219., 141.),
    17: (140., 57., 197.),
    18: (202., 185., 52.),
    19: (51., 176., 203.),
    20: (200., 54., 131.),
    21: (92., 193., 61.),
    22: (78., 71., 183.),
    23: (172., 114., 82.),
    24: (255., 127., 14.),
    25: (91., 163., 138.),
    26: (153., 98., 156.),
    27: (140., 153., 101.),
    28: (158., 218., 229.),
    29: (100., 125., 154.),
    30: (178., 127., 135.),
    32: (146., 111., 194.),
    33: (44., 160., 44.),
    34: (112., 128., 144.),
    35: (96., 207., 209.),
    36: (227., 119., 194.),
    37: (213., 92., 176.),
    38: (94., 106., 211.),
    39: (82., 84., 163.),
    40: (100., 85., 144.),
}

# segmantic lable remapper
SEM_CLASS_LABELS = ('wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
                'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator',
                'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture')
SEM_VALID_CLASS_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
SEM_REMAPPER=np.ones(150)*(20)
for i,x in enumerate(SEM_VALID_CLASS_IDS):
    SEM_REMAPPER[x]=i

# scene type remapper
TYPE_CLASS_LABELS=('aparment','bathroom','bedroom','conference room','copy','hallway','kitchen','laundry room','living room','office','storage','misc')
TYPE_VALID_CLASS_IDS=[1,2,3,4,8,9,13,14,15,16,18,20,21]
TYPE_REMAPPER=np.ones(22)*(12)
for i,x in enumerate(TYPE_VALID_CLASS_IDS):
    TYPE_REMAPPER[x]=i
    
'''
ScanNet dataset
'''
class ScanNetDataset(torch.utils.data.Dataset):
    def __init__(self,path,augment=False,voxel_size=0.02,leave_rate=None,
                 crop_rate=None,skip_rate=1,ind_remove=None):
        torch.utils.data.Dataset.__init__(self)
        
        self.voxel_size=voxel_size
        self.augment=augment
        self.leave_rate=leave_rate
        self.crop_rate=crop_rate
        self.skip_rate=skip_rate
        self.ind_remove=ind_remove

        # load data
        self.data=[]
        for x in torch.utils.data.DataLoader(
            glob.glob(path), collate_fn=lambda x: torch.load(x[0]),num_workers=mp.cpu_count()):
            self.data.append(x)
        
        # preprocess data on train/val/test data
        for i in range(len(self.data)):
            # normalize colors
            self.data[i]['feats']/=255
            self.data[i]['feats']-=0.5
            
            # scene type label
            # self.data[i]['scene_label']=TYPE_REMAPPER[self.data[i]['scene_label']]
            self.data[i]['scene_label']-=1
            
            # semantic label
            self.data[i]['sem_label']=SEM_REMAPPER[self.data[i]['sem_label'].astype('int')]
    
    def __getitem__(self,n):
        crn_sample=self.data[n]
        xyz=crn_sample['coords']
        feats=crn_sample['feats']
        sem_labels=crn_sample['sem_label']
        scene_type=crn_sample['scene_label']
        scene_name=crn_sample['scene_name']
        
        # filter by semantic index
        ind_left=sem_labels!=self.ind_remove
        xyz,feats,sem_labels=xyz[ind_left],feats[ind_left],sem_labels[ind_left]
        
        
        # voxelization
        sel = ME.utils.sparse_quantize(xyz / self.voxel_size, return_index=True)
        down_xyz, down_feat,down_labels = xyz[sel],feats[sel],sem_labels[sel]

        # Get coords, shift to center
        coords = np.floor(down_xyz / self.voxel_size)
        coords-=coords.min(0)
        
        return (coords,down_feat,down_labels,scene_type,scene_name)
    
    def __len__(self):
        return len(self.data)


'''
collate data for each batch
'''
def collate_fn(list_data):
    new_list_data = []
    num_removed = 0
    for data in list_data:
        if data is not None:
            new_list_data.append(data)
        else:
            num_removed += 1

    list_data = new_list_data

    if len(list_data) == 0:
        raise ValueError('No data in the batch')

    coords, feats, labels,scene_types,scene_names = list(zip(*list_data))

    eff_num_batch = len(coords)
    assert len(labels) == eff_num_batch

    lens = [len(c) for c in coords]
    
    # filter samples
    cum_len=np.cumsum(lens)
    n_samples=(cum_len<MAX_POINTS).sum()
    feats=feats[:n_samples]
    labels=labels[:n_samples]
    coords=coords[:n_samples]
    scene_types=scene_types[:n_samples]
    scene_names=scene_names[:n_samples]
    
    
    # Concatenate all lists
    curr_ptr = 0
    num_tot_pts = sum(lens[:n_samples])
    coords_batch = torch.zeros(num_tot_pts, 4)
    feats_batch = torch.from_numpy(np.vstack(feats)).float()
    labels_batch=torch.from_numpy(np.hstack(labels)).long()
    scene_types_batch=torch.from_numpy(np.hstack(scene_types)).long()
    

    for batch_id in range(n_samples):
        coords_batch[curr_ptr:curr_ptr + lens[batch_id], :3] = torch.from_numpy(
            coords[batch_id])
        coords_batch[curr_ptr:curr_ptr + lens[batch_id], 3] = batch_id
        curr_ptr += len(coords[batch_id])
    
    
    return {
        'coords': coords_batch,
        'feats': feats_batch,
        'sem_labels': labels_batch,
        'clf_labels':scene_types_batch,
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
    # train loader
    train_set=ScanNetDataset(path_train,augment=True,voxel_size=config['voxel_size'])
    train_args = {
        'batch_size': config['train_batch_size'],
        'num_workers': config['num_workers'],
        'collate_fn': collate_fn,
        'sampler':InfSampler(train_set),
        'pin_memory': False,
        'drop_last': False
    }
    train_loader = torch.utils.data.DataLoader(train_set, **train_args)
    
    # val loader
    val_set=ScanNetDataset(path_val,augment=False,voxel_size=config['voxel_size'])
    val_args = {
        'batch_size': config['val_batch_size'],
        'num_workers': config['num_workers'],
        'collate_fn': collate_fn,
        'pin_memory': False,
        'drop_last': False
    }
    val_loader = torch.utils.data.DataLoader(val_set,**val_args)
    
    return {
        'train': train_loader,
        'val': val_loader
    }

    

def get_testdataset(path_test,config):
    test_set=ScanNetDataset(path_test,augment=False,voxel_size=config['voxel_size'])
    val_args = {
        'batch_size': config['test_batch_size'],
        'num_workers': config['num_workers'],
        'collate_fn': collate_fn,
        'pin_memory': False,
        'drop_last': False
    }
    test_loader = torch.utils.data.DataLoader(test_set,**val_args)
    return test_loader


def get_valdataset(path_val,config):
     # val loader
    val_set=ScanNetDataset(path_val,augment=False,voxel_size=config['voxel_size'],
                           leave_rate=config['leave_rate'],crop_rate=config['crop_rate'],
                           skip_rate=config['skip_rate'],ind_remove=config['ind_remove'])
    val_args = {
        'batch_size': config['val_batch_size'],
        'num_workers': config['num_workers'],
        'collate_fn': collate_fn,
        'pin_memory': False,
        'drop_last': False
    }
    val_loader = torch.utils.data.DataLoader(val_set,**val_args)
    return val_loader
    