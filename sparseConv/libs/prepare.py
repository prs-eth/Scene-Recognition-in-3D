# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import glob, plyfile, numpy as np, multiprocessing as mp, torch
from utils import *
import sys

# get scene class
path_scene_types_all='/net/pf-pc27/scratch3/scannet/tasks/scene_types_all.txt'
scene_type_mapping = read_scene_types_mapping(path_scene_types_all, remove_spaces=True)

# for train and val dataset
def parse_ply_1(fn):
    # get input coordinates and true colors
    path_data=fn
    a=plyfile.PlyData().read(path_data)
    v=np.array([list(x) for x in a.elements[0]])
    coords=v[:,:3]
    colors=v[:,3:6]

    # get semantic lables
    path_label=fn[:-3]+'labels.ply'
    a=plyfile.PlyData().read(path_label)
    v=np.array([list(x) for x in a.elements[0]])
    labels=v[:,-1]
    
    assert coords.shape[0]==labels.shape[0]
    
    # get scene type label
    path_info_file=fn[:-15]+'.txt'
    scene_name = os.path.splitext(os.path.basename(path_info_file))[0]
    type_id = get_scene_type_id(get_field_from_info_file(path_info_file, 'sceneType'), scene_type_mapping)

    torch.save({'coords':coords,'feats':colors,'sem_label':labels,
    'scene_label': type_id,'scene_name':scene_name},fn[:-4]+'.pth')
    print(fn)

# for test dataset
def parse_ply_2(fn):
    # get input coordinates and colors
    a=plyfile.PlyData().read(fn)
    v=np.array([list(x) for x in a.elements[0]])
    coords=v[:,:3]
    colors=v[:,3:6]
    sem_labels=np.ones(coords.shape[0])
    scene_label=1
    torch.save({'coords':coords,'feats':colors,'scene_name': fn[41:53],
                'sem_label':sem_labels,'scene_label':scene_label},fn[:-4]+'.pth')
    print(fn)
    

if(sys.argv[1] in ['train','val']):
    if(sys.argv[1]=='val'):
        base_dir='/net/pf-pc27/scratch3/scannet/val/'
    else:
        base_dir='/net/pf-pc27/scratch3/scannet/train/'

    files=sorted(glob.glob(base_dir+'*/*_vh_clean_2.ply'))   # only use xyz and raw RGB
    files2=sorted(glob.glob(base_dir+'*/*.txt'))  # we can access the scene type from here
    assert len(files) == len(files2)
    p = mp.Pool(processes=mp.cpu_count())
    p.map(parse_ply_1,files)
    p.close()
    p.join()
 
elif(sys.argv[1]=='test'):
    base_dir='/net/pf-pc27/scratch3/scannet/scans_test/'
    files=sorted(glob.glob(base_dir+'*/*_vh_clean_2.ply'))
    print(len(files))
    p = mp.Pool(processes=mp.cpu_count())
    p.map(parse_ply_2,files)
    p.close()
    p.join()