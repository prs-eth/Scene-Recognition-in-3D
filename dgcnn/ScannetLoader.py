import torch, numpy as np, glob, math, torch.utils.data, scipy.ndimage, multiprocessing as mp
import os
import plyfile

class ScanNet(torch.utils.data.Dataset):
    def __init__(self,files,transforms=None,color=False):
        torch.utils.data.Dataset.__init__(self)
        point_list,label_list=[],[]
        for eachfile in files:
            # get label
            data=torch.load(eachfile)
            label_list.append(data['scene_label']-1)
            
            # get data
            coords=data['4096_coords']
            coords-=coords.mean(0)
            coords/=4
            
            # add color
            if(color):
                colors=data['4096_feats']
                colors/=127.5
                colors-=1
                points=np.hstack((coords,colors))
            else:
                points=coords
            point_list.append(np.expand_dims(points,0))
        
        self.points=np.concatenate(point_list,0)
        self.labels=np.expand_dims(np.array(label_list),1)
        self.transforms=transforms
    
    def __getitem__(self,idx):
        current_points = self.points[idx].copy()
        label = torch.from_numpy(self.labels[idx]).type(torch.LongTensor)
        if self.transforms is not None:
            current_points = self.transforms(current_points)

        return current_points, label
    
    def __len__(self):
        return self.points.shape[0]