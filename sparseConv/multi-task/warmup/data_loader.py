import torch, numpy as np, glob, math, torch.utils.data, scipy.ndimage, multiprocessing as mp
import math
import MinkowskiEngine as ME
from torch.utils.data.sampler import Sampler
import os,sys

sys.path.append('../..')
import libs.data_utils as d_utils
from torchvision import transforms

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
TYPE_REMAPPER=np.ones(22)*(13)
for i,x in enumerate(TYPE_VALID_CLASS_IDS):
    TYPE_REMAPPER[x]=i
    
'''
ScanNet dataset
'''
class ScanNetDataset(torch.utils.data.Dataset):
    def __init__(self,path,voxel_size=0.02,transform=None,use_color=True):
        torch.utils.data.Dataset.__init__(self)
        
        self.voxel_size=voxel_size
        self.transform=transform
        self.use_color=use_color
        self.coords=[]
        self.feats=[]
        self.scene_label=[]
        
        # preprocess data on train/val/test data
        files=glob.glob(path)
        for eachfile in files:
            # normalize colors
            data=torch.load(eachfile)
            data['feats']/=255 
            data['feats']-=0.5  
            
            # remap labels
            data['scene_label']-=1

            self.coords.append(data['coords'])
            self.feats.append(data['feats'])
            self.scene_label.append(data['scene_label'])
            
    def __getitem__(self,n):
        xyz=self.coords[n]
        feats=self.feats[n]
        scene_type=self.scene_label[n]
        
        # augment data by random rotation along z axis
        if self.transform is not None:
            xyz = self.transform(xyz)
            xyz=xyz.numpy()
            
        # voxelization
        sel = ME.utils.sparse_quantize(xyz / self.voxel_size, return_index=True)
        down_xyz, down_feat = xyz[sel], feats[sel]

        # Get coords, shift to center
        coords = np.floor(down_xyz / self.voxel_size)
        coords-=coords.min(0)
        
        if(not self.use_color):
            down_feat=np.ones((down_feat.shape[0],1))
        
        return (coords,down_feat,scene_type)
    
    def __len__(self):
        return len(self.coords)


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

    coords, feats, labels = list(zip(*list_data))

    eff_num_batch = len(coords)
    assert len(labels) == eff_num_batch

    lens = [len(c) for c in coords]
    curr_ptr = 0
    num_tot_pts = sum(lens)
    coords_batch = torch.zeros(num_tot_pts, 4)
    feats_batch = torch.from_numpy(np.vstack(feats)).float()
    

    for batch_id in range(eff_num_batch):
        coords_batch[curr_ptr:curr_ptr + lens[batch_id], :3] = torch.from_numpy(
            coords[batch_id])
        coords_batch[curr_ptr:curr_ptr + lens[batch_id], 3] = batch_id
        curr_ptr += len(coords[batch_id])

    # Concatenate all lists
    return {
        'coords': coords_batch,
        'feats': feats_batch,
        'labels': torch.LongTensor(labels),
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
    transform = transforms.Compose(
        [
            d_utils.PointcloudToTensor(),
            d_utils.PointcloudRotate(),
            d_utils.PointcloudRotatePerturbation(),
            d_utils.PointcloudTranslate(),
            d_utils.PointcloudJitter(),
            d_utils.PointcloudRandomInputDropout()
        ]
    )
    
    # train loader
    train_set=ScanNetDataset(path_train,voxel_size=config['voxel_size'],transform=transform,use_color=config['use_color'])
    train_args = {
        'batch_size': config['train_batch_size'],
        'num_workers': config['num_workers'],
        'collate_fn': collate_fn,
        'sampler':InfSampler(train_set),
        'pin_memory': True,
        'drop_last': False
    }
    train_loader = torch.utils.data.DataLoader(train_set, **train_args)
    
    # val loader
    val_set=ScanNetDataset(path_val,voxel_size=config['voxel_size'],use_color=config['use_color'])
    val_args = {
        'batch_size': config['val_batch_size'],
        'num_workers': config['num_workers'],
        'collate_fn': collate_fn,
        'pin_memory': True,
        'drop_last': False
    }
    val_loader = torch.utils.data.DataLoader(val_set,**val_args)
    
    # return two loaders
    return {
        'train': train_loader,
        'val': val_loader
    }
