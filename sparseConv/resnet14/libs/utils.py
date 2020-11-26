import time
import os,sys
import math
import numpy as np
import logging


def getLogger(file_name):
    logger = logging.getLogger('train')  # set logger name
    logger.setLevel(logging.INFO)  # logger level

    ch = logging.StreamHandler()  
    ch.setLevel(logging.INFO)  

    fh = logging.FileHandler(file_name, mode='a')  
    fh.setLevel(logging.INFO)  

    formatter = logging.Formatter("%(asctime)s - %(message)s","%Y-%m-%d %H:%M:%S") # use formatter to set
    ch.setFormatter(formatter)  
    fh.setFormatter(formatter)
    logger.addHandler(fh)  
    logger.addHandler(ch)
    return logger

def get_scene_type_id(type_name, type_mapping):
    name = type_name.strip().lower()
    name=name.replace(' ','')
    if name in type_mapping:
        return type_mapping[name]
    return -1


def get_field_from_info_file(filename, field_name):
    lines = open(filename).read().splitlines()
    lines = [line.split(' = ') for line in lines]
    mapping = { x[0]:x[1] for x in lines }
    if field_name in mapping:
        return mapping[field_name]
    else:
        logger.info('Failed to find %s in info file %s' % (field_name, filename))

# input: scene_types.txt or scene_types_all.txt
def read_scene_types_mapping(filename, remove_spaces=True):
    assert os.path.isfile(filename)
    mapping = dict()
    lines = open(filename).read().splitlines()
    lines = [line.split('\t') for line in lines]
    if remove_spaces:
        mapping = { x[1].strip().replace(' ',''):int(x[0]) for x in lines }
    else:
        mapping = { x[1]:int(x[0]) for x in lines }        
    return mapping